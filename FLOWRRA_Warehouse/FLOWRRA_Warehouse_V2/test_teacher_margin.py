"""
test_teacher_margin.py -- the policy learns from being overridden by RULES.

THE PROBLEM. When the orchestrator's rules override a fleet, the replay buffer
stores the action the RULES executed (correctly), and the TD loss trains the
value of that action. The action the network itself preferred was never
executed, so no transition carries its consequence and its value is never
corrected. The network keeps preferring the vetoed move; the rules keep vetoing
it; nothing tells the network. cold_run25: in warning zones the network proposed
waiting 12-17% of the time while 61-70% of executed actions were waits, flat
across all 60 episodes.

THE FIX. core_warehouse.py records, per fleet, whether the rules chose the
executed action (teacher_mask) and the structural validity of each action in the
state acted from (valid_mask). learn() adds a large-margin term (DQfD, Hester et
al. 2018) on the COMBINED value:

    relu( max over valid a != a_rules of Q(s,a) + margin - Q(s,a_rules) )

with Q(s,a_rules) detached, so it pushes the preferred alternative DOWN and
never pulls the rules' action up. Switch: CONFIG training.teacher_margin.enabled.

WHAT THIS CHECKS
  1. Old transitions (11 and 12 fields) still pad and sample; new ones pad the
     teacher fields correctly.
  2. A batch mixing old and new transitions reads the old ones as "not taught".
  3. The loss on hand-built values: the number, the agreement, and that the
     gradient lands ONLY on each taught fleet's best valid alternative.
  4. Switched off, one learning step is IDENTICAL to a step on transitions that
     carry no teacher signal at all -- loss and every weight.
  5. Switched on, repeated learning on taught transitions shrinks the margin
     loss and does not lower agreement.

Run:  python test_teacher_margin.py      (ends with ALL PASS)
"""

import random

import numpy as np
import torch

from agent_warehouse import GNNAgent, GraphReplayBuffer, teacher_margin_terms

FAIL = []

A = 7          # 6 moves + idle
N = 3          # fleets per transition
HEADS = ["safety", "delivery", "efficiency"]
WEIGHTS = [6.0, 4.0, 1.0]


def check(name, got, want):
    ok = got == want
    print(f"{'PASS' if ok else 'FAIL'}  {name}: got {got!r} want {want!r}")
    if not ok:
        FAIL.append(name)


# ------------------------------------------------------------------ helpers
def transition(rng, teacher=True, actions=(0, 2, 1), taught=(1.0, 0.0, 1.0)):
    """One stored transition as core_warehouse.py pushes it."""
    valid = np.ones((N, A), dtype=bool)
    valid[2, 3] = False
    kw = dict(
        node_features=rng.normal(size=(N, 8)).astype(np.float32),
        adj_matrix=np.ones((N, N), dtype=np.float32),
        actions=np.array(actions, dtype=np.int64),
        rewards=np.zeros((N, len(HEADS)), dtype=np.float32),
        next_node_features=rng.normal(size=(N, 8)).astype(np.float32),
        next_adj_matrix=np.ones((N, N), dtype=np.float32),
        done=False,
        integrity=1.0,
        active_mask=np.ones(N, dtype=np.float32),
        recovery_action=0,
        next_valid_mask=np.ones((N, A), dtype=bool),
        next_active_mask=np.ones(N, dtype=np.float32),
    )
    if teacher:
        kw["valid_mask"] = valid
        kw["teacher_mask"] = np.array(taught, dtype=np.float32)
    return kw


def small_agent(weight, margin=0.2, lr=3e-4):
    return GNNAgent(
        node_feature_dim=8, edge_feature_dim=0, action_size=A,
        hidden_dim=16, num_layers=1, n_heads=2,
        reward_heads=HEADS, head_weights=WEIGHTS,
        teacher_margin=margin, teacher_weight=weight,
        batch_size=4, dropout=0.0, lr=lr, seed=0,
    )


# ------------------------------------------------------------------ 1. padding
def test_padding_old_and_new_layouts():
    """Every stored layout pads to the 14-field form."""
    rng = np.random.default_rng(0)
    buf = GraphReplayBuffer(10)
    buf.push(**transition(rng, teacher=True))
    new = buf.buffer[0]
    check("new_layout_has_14_fields", len(new), 14)

    out = GraphReplayBuffer._pad_transition(new, 5)
    check("valid_mask_padded_shape", out[12].shape, (5, A))
    check("padded_fleet_valid_idle_only",
          out[12][4].tolist(), [True] + [False] * (A - 1))
    check("teacher_mask_padded_with_zeros", out[13].tolist(), [1.0, 0.0, 1.0, 0.0, 0.0])
    check("original_validity_kept", bool(out[12][2, 3]), False)

    same = GraphReplayBuffer._pad_transition(new, N)
    check("no_padding_needed_keeps_14", len(same), 14)

    for n_fields in (11, 12):
        old = tuple(new[:n_fields])
        o = GraphReplayBuffer._pad_transition(old, 5)
        check(f"old_{n_fields}_fields_pads_to_14", len(o), 14)
        check(f"old_{n_fields}_fields_has_no_teacher", (o[12], o[13]), (None, None))


# ------------------------------------------------------------------ 2. sampling
def test_sample_mixes_old_and_new():
    """An old transition in the batch reads as 'not taught', not as an error."""
    rng = np.random.default_rng(1)
    buf = GraphReplayBuffer(10)
    buf.push(**transition(rng, teacher=False))
    buf.push(**transition(rng, teacher=True))
    random.seed(0)
    batch = buf.sample(2)
    # 16 since masked pooling added the real-row masks (test_graph_pooling.py).
    check("sample_returns_16", len(batch), 16)
    valid_t, teacher_t = batch[12], batch[13]
    check("valid_tensor_present", valid_t is not None, True)
    check("teacher_tensor_shape", tuple(teacher_t.shape), (2, N))
    check("only_the_new_transition_taught", float(teacher_t.sum().item()), 2.0)

    buf_old = GraphReplayBuffer(10)
    buf_old.push(**transition(rng, teacher=False))
    buf_old.push(**transition(rng, teacher=False))
    b = buf_old.sample(2)
    check("all_old_batch_gives_no_teacher", (b[12], b[13]), (None, None))


# ------------------------------------------------------------------ 3. the loss
def test_margin_math_and_where_the_gradient_lands():
    """
    Five fleets, margin 0.2:
      0 taught, rules=2, Q_rules 1.0, best alt 0.5 -> leads by 0.5: hinge 0, agrees
      1 taught, rules=0, Q_rules 0.3, best alt 0.6 -> hinge 0.6+0.2-0.3 = 0.5
      2 taught, rules=1, Q_rules 0.3, INVALID alt 9.0 ignored, best valid 0.2
                -> hinge 0.1, agrees
      3 not taught (huge alternative) -> nothing
      4 taught but inactive (padding) -> nothing
    loss = (0 + 0.5 + 0.1) / 3 = 0.2 ; agree = 2/3 ; share = 3 of 4 active
    """
    q = torch.zeros(1, 5, A)
    q[0, 0] = torch.tensor([0.5, 0.0, 1.0, 0.1, 0.2, 0.3, 0.4])
    q[0, 1] = torch.tensor([0.3, 0.6, 0.1, 0.0, 0.0, 0.0, 0.0])
    q[0, 2] = torch.tensor([0.0, 0.3, 0.2, 9.0, 0.0, 0.0, 0.0])
    q[0, 3] = torch.tensor([0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    q[0, 4] = torch.tensor([0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    q.requires_grad_(True)
    actions = torch.tensor([[2, 0, 1, 0, 0]])
    valid = torch.ones(1, 5, A, dtype=torch.bool)
    valid[0, 2, 3] = False
    teacher = torch.tensor([[1.0, 1.0, 1.0, 0.0, 1.0]])
    active = torch.tensor([[1.0, 1.0, 1.0, 1.0, 0.0]])

    loss, agree, share, n = teacher_margin_terms(q, actions, valid, teacher, active, 0.2)
    check("loss_value", round(float(loss.item()), 5), 0.2)
    check("agreement", round(agree, 5), round(2 / 3, 5))
    check("share_of_active", round(share, 5), 0.75)
    check("taught_count", n, 3.0)

    loss.backward()
    g = q.grad[0]
    check("leading_fleet_untouched", float(g[0].abs().sum().item()), 0.0)
    check("pushes_best_alternative_down", round(float(g[1, 1].item()), 5), round(1 / 3, 5))
    check("never_pulls_rules_action_up", float(g[1, 0].item()), 0.0)
    check("invalid_action_untouched", float(g[2, 3].item()), 0.0)
    check("best_valid_alternative_pushed", round(float(g[2, 2].item()), 5), round(1 / 3, 5))
    check("untaught_fleet_untouched", float(g[3].abs().sum().item()), 0.0)
    check("inactive_fleet_untouched", float(g[4].abs().sum().item()), 0.0)

    z, za, zs, zn = teacher_margin_terms(q, actions, valid,
                                         torch.zeros(1, 5), active, 0.2)
    check("nothing_taught_zero_loss", float(z.item()), 0.0)
    check("nothing_taught_no_graph", z.requires_grad, False)


# ------------------------------------------------------------------ 4. off = identical
def test_switched_off_is_identical():
    """
    Weight 0: one learning step on transitions WITH the teacher signal must equal
    one step on the same transitions WITHOUT it (an old caller) -- same loss,
    same weights afterwards. The diagnostics are still measured.
    """
    rng_a = np.random.default_rng(7)
    rng_b = np.random.default_rng(7)
    a, b = small_agent(0.0), small_agent(0.0)
    for _ in range(6):
        a.memory.push(**transition(rng_a, teacher=True))
        b.memory.push(**transition(rng_b, teacher=False))

    random.seed(3); torch.manual_seed(3)
    la = a.learn()
    random.seed(3); torch.manual_seed(3)
    lb = b.learn()
    # To 1e-6 rather than bit-for-bit: on a GPU some kernels are not
    # deterministic, so two identical steps can differ in the last float bits.
    check("same_loss", abs(la - lb) < 1e-6, True)
    same = all(torch.allclose(p, q, rtol=0.0, atol=1e-6) for p, q in
               zip(a.policy_net.parameters(), b.policy_net.parameters()))
    check("same_weights_after_step", same, True)
    check("still_measured_when_off", a.last_teacher_n > 0, True)
    check("old_transitions_measure_nothing", b.last_teacher_n, 0.0)


# ------------------------------------------------------------------ 5. it teaches
def test_switched_on_teaches():
    """
    Weight 1, a deliberately large margin so the term is active from the start:
    repeated learning on taught transitions must shrink the margin loss, and
    agreement with the rules must not fall.
    """
    rng = np.random.default_rng(11)
    agent = small_agent(1.0, margin=5.0, lr=1e-2)
    for _ in range(8):
        agent.memory.push(**transition(rng, teacher=True,
                                       actions=(0, 0, 0), taught=(1.0, 1.0, 1.0)))
    random.seed(5); torch.manual_seed(5)
    agent.learn()
    first_loss, first_agree = agent.last_teacher_loss, agent.last_teacher_agree
    for _ in range(150):
        agent.learn()
    last_loss, last_agree = agent.last_teacher_loss, agent.last_teacher_agree
    print(f"      margin loss {first_loss:.3f} -> {last_loss:.3f} | "
          f"agreement {first_agree:.2f} -> {last_agree:.2f}")
    check("term_active_at_start", first_loss > 0, True)
    check("margin_loss_shrinks", last_loss < first_loss, True)
    check("agreement_does_not_fall", last_agree >= first_agree, True)


if __name__ == "__main__":
    for fn in list(globals().values()):
        if callable(fn) and getattr(fn, "__name__", "").startswith("test_"):
            print(f"\n--- {fn.__name__} ---")
            fn()
    print("\n" + ("ALL PASS" if not FAIL else f"FAILURES: {FAIL}"))