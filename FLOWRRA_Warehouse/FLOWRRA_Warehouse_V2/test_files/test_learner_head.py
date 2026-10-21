"""
test_learner_head.py -- step 3a: the Learner head.

WHAT 3a ADDS (config training.learner; off = no head built)
  A small head on top of the shared trunk learns RULES' judgement: on every step
  RULES decided (the teacher_mask the buffer already carries), it is trained to
  predict the action RULES chose. Two rules keep it from repeating the margin
  teacher's failure:
    * it READS the trunk but never TRAINS it (h.detach()), so its lessons can
      never move a Q-value -- design principle 7, "values stay honest";
    * at choice time it speaks ONLY for fleets inside a conflict:
      score = Q + beta x its log-probability; free traffic is untouched.

WHAT THIS CHECKS
  1. Off by default: no head is built, so older checkpoints still load.
  2. Its training never touches the trunk or the three consequence heads.
  3. It learns: on taught transitions, agreement with RULES rises and loss falls.
  4. It nudges only conflicted fleets, and the counters record it.
  5. It never picks an invalid action.

Run:  python test_learner_head.py      (ends with ALL PASS; needs PyTorch)
"""

import random

import numpy as np
import torch

from agent_warehouse import GNNAgent

FAIL = []
A = 7
HEADS = ["safety", "delivery", "efficiency"]


def check(name, got, want):
    ok = got == want
    print(f"{'PASS' if ok else 'FAIL'}  {name}: got {got!r} want {want!r}")
    if not ok:
        FAIL.append(name)


def agent(beta=0.0, weight=0.0, lr=3e-4):
    return GNNAgent(node_feature_dim=8, edge_feature_dim=0, action_size=A,
                    hidden_dim=16, num_layers=1, n_heads=2,
                    reward_heads=HEADS, head_weights=[6.0, 4.0, 1.0],
                    batch_size=4, dropout=0.0, lr=lr, seed=0,
                    learner_beta=beta, learner_weight=weight)


def transition(rng, n=4, rules_action=2):
    return dict(
        node_features=rng.normal(size=(n, 8)).astype(np.float32),
        adj_matrix=np.ones((n, n), dtype=np.float32),
        actions=np.full(n, rules_action, dtype=np.int64),
        rewards=rng.normal(scale=0.1, size=(n, len(HEADS))).astype(np.float32),
        next_node_features=rng.normal(size=(n, 8)).astype(np.float32),
        next_adj_matrix=np.ones((n, n), dtype=np.float32),
        done=False, integrity=1.0, active_mask=np.ones(n, dtype=np.float32),
        recovery_action=0,
        next_valid_mask=np.ones((n, A), dtype=bool),
        next_active_mask=np.ones(n, dtype=np.float32),
        valid_mask=np.ones((n, A), dtype=bool),
        teacher_mask=np.ones(n, dtype=np.float32),       # RULES decided every fleet
    )


# ------------------------------------------------------------------ 1. off
def test_off_builds_no_head():
    a = agent()
    check("no_learner_head", a.policy_net.learner_head is None, True)
    keys = list(a.policy_net.state_dict())
    check("no_learner_weights_in_checkpoint", any("learner" in k for k in keys), False)


# ------------------------------------------------------------------ 2. never trains the trunk
def test_never_trains_the_trunk():
    rng_a, rng_b = np.random.default_rng(1), np.random.default_rng(1)
    on, off = agent(beta=0.3, weight=1.0, lr=1e-2), agent(beta=0.3, weight=0.0, lr=1e-2)
    for _ in range(6):
        on.memory.push(**transition(rng_a))
        off.memory.push(**transition(rng_b))
    random.seed(2); torch.manual_seed(2)
    on.learn()
    random.seed(2); torch.manual_seed(2)
    off.learn()
    lid_on = {id(p) for p in on.policy_net.learner_head.parameters()}
    lid_off = {id(p) for p in off.policy_net.learner_head.parameters()}
    rest_on = [p for p in on.policy_net.parameters() if id(p) not in lid_on]
    rest_off = [p for p in off.policy_net.parameters() if id(p) not in lid_off]
    same = all(torch.allclose(x, y, rtol=0.0, atol=1e-6) for x, y in zip(rest_on, rest_off))
    check("trunk_and_q_heads_identical_with_or_without_learner_loss", same, True)
    moved = any(not torch.allclose(x, y) for x, y in zip(on.policy_net.learner_head.parameters(),
                                                        off.policy_net.learner_head.parameters()))
    check("learner_head_itself_learned", moved, True)


# ------------------------------------------------------------------ 3. it learns
def test_learns_rules_choice():
    rng = np.random.default_rng(3)
    a = agent(beta=0.3, weight=1.0, lr=1e-2)
    for _ in range(8):
        a.memory.push(**transition(rng, rules_action=5))
    random.seed(4); torch.manual_seed(4)
    a.learn()
    first_loss, first_agree = a.last_learner_loss, a.last_learner_agree
    for _ in range(150):
        a.learn()
    print(f"      learner loss {first_loss:.3f} -> {a.last_learner_loss:.3f} | "
          f"agreement {first_agree:.2f} -> {a.last_learner_agree:.2f}")
    check("loss_falls", a.last_learner_loss < first_loss, True)
    check("agrees_with_rules", a.last_learner_agree, 1.0)


# ------------------------------------------------------------------ 4-5. nudge
def _greedy(a, feats, adj, valid, conflict):
    return a.choose_actions(node_features=feats, adj_matrix=adj, episode_number=0,
                            total_episodes=1, eps_min=0.0, eps_peak=0.0,
                            valid_action_masks=valid, conflict_mask=conflict)


def test_nudges_only_conflicted_fleets():
    rng = np.random.default_rng(5)
    a = agent(beta=50.0, weight=1.0)
    n = 4
    feats = rng.normal(size=(n, 8)).astype(np.float32)
    adj = np.ones((n, n), dtype=np.float32)
    valid = np.ones((n, A), dtype=bool)
    base = _greedy(a, feats, adj, valid, None)
    k = (int(base[0]) + 1) % A                         # an action fleet 0 would NOT pick
    last = a.policy_net.learner_head[-1]
    with torch.no_grad():
        last.weight.zero_()
        last.bias.zero_()
        last.bias[k] = 20.0                            # the Learner strongly prefers k
    conflict = np.array([True, False, False, False])
    used0, changed0 = a.learner_used, a.learner_changed
    nudged = _greedy(a, feats, adj, valid, conflict)
    check("conflicted_fleet_follows_the_learner", int(nudged[0]), k)
    check("free_fleets_untouched", list(nudged[1:]), list(base[1:]))
    check("counted_as_used_and_changed",
          (a.learner_used - used0, a.learner_changed - changed0), (1, 1))

    valid2 = valid.copy()
    valid2[0, k] = False                               # k is not allowed here
    chosen = _greedy(a, feats, adj, valid2, conflict)
    check("never_picks_an_invalid_action", int(chosen[0]) != k, True)


# ------------------------------------------------------------------ 6. bounded nudge
def test_prob_nudge_is_bounded_and_records_gaps():
    """learner_run1: an unbounded (log-probability) nudge let a confident Learner
    override 82-89% of conflicted choices. With "prob" its pull is at most beta.

    Calibrated from the network itself, so it cannot depend on luck (the first
    version used a fixed beta that was too small for a fresh network's value
    gaps, and neither nudge could flip anything):
      * beta = half the SMALLEST best-vs-second-best gap among the fleets, so a
        pull of at most beta can never flip a clear preference;
      * the Learner is made certain of action k, by enough that the UNBOUNDED
        pull (beta x the log-probability gap) exceeds every fleet's distance
        from its best action to k.
    Expected: bounded flips nobody; unbounded flips every fleet not already on k."""
    rng = np.random.default_rng(6)
    n = 6
    feats = rng.normal(size=(n, 8)).astype(np.float32)
    adj = np.ones((n, n), dtype=np.float32)
    valid = np.ones((n, A), dtype=bool)
    conflict = np.ones(n, dtype=bool)

    probe = agent(beta=1.0, weight=1.0)
    base = _greedy(probe, feats, adj, valid, None)       # no nudge: the heads alone
    q = np.array(probe.last_q_values, dtype=np.float64)
    k = (int(base[0]) + 1) % A
    movable = int((base != k).sum())
    qs = np.sort(q, axis=1)
    min_gap = float((qs[:, -1] - qs[:, -2]).min())
    beta = 0.5 * min_gap
    to_k = float((q.max(axis=1) - q[:, k]).max())
    certainty = 2.0 * to_k / beta + 10.0
    print(f"      smallest heads' gap {min_gap:.5f} -> beta {beta:.5f} | "
          f"Learner certainty (logit) {certainty:.1f} | fleets not on k: {movable}")
    check("heads_have_a_clear_preference", min_gap > 1e-6, True)

    changed, gaps = {}, []
    for mode in ("logprob", "prob"):
        a = agent(beta=beta, weight=1.0)                 # same seed: same network
        a.learner_nudge = mode
        last = a.policy_net.learner_head[-1]
        with torch.no_grad():
            last.weight.zero_()
            last.bias.zero_()
            last.bias[k] = certainty                     # a CERTAIN Learner
        a.learner_q_gaps.clear()
        c0 = a.learner_changed
        _greedy(a, feats, adj, valid, conflict)
        changed[mode] = a.learner_changed - c0
        if mode == "prob":
            gaps = list(a.learner_q_gaps)
    print(f"      changed: logprob {changed['logprob']} | prob {changed['prob']} | of {movable} movable")
    check("unbounded_lets_a_certain_learner_override_every_fleet", changed["logprob"], movable)
    check("bounded_never_overrides_a_clear_preference", changed["prob"], 0)
    check("heads_gap_recorded_for_every_conflicted_choice", len(gaps), n)
    check("gaps_are_non_negative", all(g >= 0 for g in gaps), True)


# ------------------------------------------------------------------ 7. beta follows the gap
def test_beta_follows_the_heads_gap():
    """learner_run2: a fixed beta (0.1, calibrated on short runs) was twice a
    trained network's typical gap (~0.05), so the Learner flipped 25-37% of
    conflicted choices. Mode "gap": beta = scale x median of recent gaps."""
    a = agent(beta=0.1, weight=1.0)
    check("fixed_mode_uses_fixed_beta", a._effective_beta(), 0.1)
    a.learner_beta_mode, a.learner_beta_scale, a.learner_gap_min = "gap", 1.0, 3
    a._gap_window.extend([0.04, 0.06])
    check("warm_start_until_enough_gaps", a._effective_beta(), 0.1)
    a._gap_window.append(0.05)
    check("beta_is_the_median_gap", round(a._effective_beta(), 6), 0.05)
    a.learner_beta_scale = 2.0
    check("scale_multiplies_it", round(a._effective_beta(), 6), 0.1)

    # Behaviour: the same CERTAIN Learner flips everyone when the heads' typical
    # gap is huge, and nobody when it is tiny -- beta really follows the gap.
    rng = np.random.default_rng(7)
    n = 6
    feats = rng.normal(size=(n, 8)).astype(np.float32)
    adj = np.ones((n, n), dtype=np.float32)
    valid = np.ones((n, A), dtype=bool)
    probe = agent(beta=0.1, weight=1.0)
    base = _greedy(probe, feats, adj, valid, None)
    k = (int(base[0]) + 1) % A
    movable = int((base != k).sum())
    flips = {}
    for label, g in (("huge", 1e6), ("tiny", 1e-9)):
        b = agent(beta=0.1, weight=1.0)                  # same seed: same network
        b.learner_nudge, b.learner_beta_mode, b.learner_gap_min = "prob", "gap", 1
        b._gap_window.extend([g] * b._gap_window.maxlen)
        last = b.policy_net.learner_head[-1]
        with torch.no_grad():
            last.weight.zero_()
            last.bias.zero_()
            last.bias[k] = 50.0
        c0 = b.learner_changed
        _greedy(b, feats, adj, valid, np.ones(n, dtype=bool))
        flips[label] = b.learner_changed - c0
    print(f"      flips with huge gap {flips['huge']} | tiny gap {flips['tiny']} | of {movable} movable")
    check("huge_gap_lets_a_certain_learner_flip_every_fleet", flips["huge"], movable)
    check("tiny_gap_flips_nobody", flips["tiny"], 0)


if __name__ == "__main__":
    for fn in list(globals().values()):
        if callable(fn) and getattr(fn, "__name__", "").startswith("test_"):
            print(f"\n--- {fn.__name__} ---")
            fn()
    print("\n" + ("ALL PASS" if not FAIL else f"FAILURES: {FAIL}"))