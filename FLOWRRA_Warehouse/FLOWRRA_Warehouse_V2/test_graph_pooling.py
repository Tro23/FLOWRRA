"""
test_graph_pooling.py -- the graph-level heads see the same floor in training as
when choosing.

THE BUG. The recovery and stability heads read one summary of the floor: the
mean of every fleet's embedding. Training batches mix fleet counts (25/40/60),
so smaller states are padded with dummy fleets -- and the plain mean averaged
those dummies in. A 25-fleet state padded to 60 was summarised as 25 fleets +
35 dummies in training, but as 25 fleets when the same state was seen live.
The recovery head trained on one picture and acted on another.

THE FIX. sample() records which rows are real (before padding) for the state
and the next state; GNNPolicy.forward(node_mask=...) averages real rows only.
A live state is never padded, so choosing is unchanged (node_mask=None).

WHY ONLY THE MEAN NEEDED FIXING. Attention masks non-neighbours with -inf and
nothing normalises across fleets, so a padded fleet cannot touch a real fleet's
embedding. Only the floor-wide average could see it.

WHAT THIS CHECKS
  1. sample() marks the real rows of each state, for mixed fleet counts.
  2. A padded state, masked, gives EXACTLY the recovery Q and stability of the
     same state seen live -- and the old unmasked mean does not (the bug).
  3. With every row real, the masked mean is the old mean exactly.
  4. learn() runs on a mixed-size batch and stays finite.

Run:  python test_graph_pooling.py      (ends with ALL PASS)
"""

import random

import numpy as np
import torch

from agent_warehouse import DEVICE, GNNAgent, GraphReplayBuffer

FAIL = []

A = 7
HEADS = ["safety", "delivery", "efficiency"]
WEIGHTS = [6.0, 4.0, 1.0]


def check(name, got, want):
    ok = got == want
    print(f"{'PASS' if ok else 'FAIL'}  {name}: got {got!r} want {want!r}")
    if not ok:
        FAIL.append(name)


def transition(rng, n):
    """One stored transition with n fleets, as core_warehouse.py pushes it."""
    return dict(
        node_features=rng.normal(size=(n, 8)).astype(np.float32),
        adj_matrix=np.ones((n, n), dtype=np.float32),
        actions=rng.integers(0, A, size=n).astype(np.int64),
        rewards=rng.normal(scale=0.1, size=(n, len(HEADS))).astype(np.float32),
        next_node_features=rng.normal(size=(n, 8)).astype(np.float32),
        next_adj_matrix=np.ones((n, n), dtype=np.float32),
        done=False,
        integrity=1.0,
        active_mask=np.ones(n, dtype=np.float32),
        recovery_action=int(rng.integers(0, 3)),
        next_valid_mask=np.ones((n, A), dtype=bool),
        next_active_mask=np.ones(n, dtype=np.float32),
        valid_mask=np.ones((n, A), dtype=bool),
        teacher_mask=np.zeros(n, dtype=np.float32),
    )


def small_agent():
    return GNNAgent(
        node_feature_dim=8, edge_feature_dim=0, action_size=A,
        hidden_dim=16, num_layers=1, n_heads=2,
        reward_heads=HEADS, head_weights=WEIGHTS,
        batch_size=4, dropout=0.0, lr=3e-4, seed=0,
    )


def close(a, b, tol=1e-5):
    return bool(torch.allclose(a, b, atol=tol, rtol=0.0))


# ------------------------------------------------------------------ 1. real rows
def test_sample_marks_real_rows():
    rng = np.random.default_rng(0)
    buf = GraphReplayBuffer(10)
    buf.push(**transition(rng, 3))
    buf.push(**transition(rng, 5))
    random.seed(0)
    b = buf.sample(2)
    check("sample_returns_16", len(b), 16)
    real, next_real = b[14], b[15]
    check("mask_shape", tuple(real.shape), (2, 5))
    check("real_rows_per_state", sorted(real.sum(dim=1).tolist()), [3.0, 5.0])
    check("next_state_rows_too", sorted(next_real.sum(dim=1).tolist()), [3.0, 5.0])
    i3 = int((real.sum(dim=1) == 3).nonzero()[0])
    check("padding_rows_are_zero", real[i3, 3:].tolist(), [0.0, 0.0])


# ------------------------------------------------------------------ 2. train == live
def test_masked_pooling_matches_the_live_state():
    rng = np.random.default_rng(1)
    agent = small_agent()
    agent.policy_net.eval()
    t3, t5 = transition(rng, 3), transition(rng, 5)
    agent.memory.push(**t3)
    agent.memory.push(**t5)
    random.seed(0)
    b = agent.memory.sample(2)
    feats, adjs, real = b[0], b[1], b[14]
    i3 = int((real.sum(dim=1) == 3).nonzero()[0])
    with torch.no_grad():
        _, st_masked, _ = agent.policy_net(feats, adjs, node_mask=real)
        rq_masked = agent.policy_net.last_recovery_q[i3].clone()
        st_masked = st_masked[i3].clone()
        _, st_old, _ = agent.policy_net(feats, adjs)
        rq_old = agent.policy_net.last_recovery_q[i3].clone()
        st_old = st_old[i3].clone()
        nf = torch.from_numpy(t3["node_features"])[None].to(DEVICE)
        adj = torch.from_numpy(t3["adj_matrix"])[None].to(DEVICE)
        _, st_live, _ = agent.policy_net(nf, adj)
        rq_live = agent.policy_net.last_recovery_q[0].clone()
        st_live = st_live[0].clone()
    check("recovery_q_masked_equals_live", close(rq_masked, rq_live), True)
    check("stability_masked_equals_live", close(st_masked, st_live), True)
    check("old_mean_differs_from_live (the bug)", close(rq_old, rq_live), False)
    print(f"      recovery Q live {rq_live.tolist()}\n"
          f"      masked          {rq_masked.tolist()}\n"
          f"      old (padded)    {rq_old.tolist()}")


# ------------------------------------------------------------------ 3. no padding = old mean
def test_all_real_is_the_old_mean():
    rng = np.random.default_rng(2)
    agent = small_agent()
    agent.policy_net.eval()
    t = transition(rng, 4)
    nf = torch.from_numpy(t["node_features"])[None].to(DEVICE)
    adj = torch.from_numpy(t["adj_matrix"])[None].to(DEVICE)
    with torch.no_grad():
        _, st_none, _ = agent.policy_net(nf, adj)
        rq_none = agent.policy_net.last_recovery_q.clone()
        _, st_ones, _ = agent.policy_net(nf, adj, node_mask=torch.ones(1, 4, device=DEVICE))
        rq_ones = agent.policy_net.last_recovery_q.clone()
    check("all_ones_mask_equals_no_mask_recovery", close(rq_none, rq_ones, 1e-6), True)
    check("all_ones_mask_equals_no_mask_stability", close(st_none, st_ones, 1e-6), True)


# ------------------------------------------------------------------ 4. learn() runs
def test_learn_runs_on_mixed_sizes():
    rng = np.random.default_rng(3)
    agent = small_agent()
    for n in (3, 5, 2, 5, 4, 3):
        agent.memory.push(**transition(rng, n))
    random.seed(4)
    torch.manual_seed(4)
    before = [p.detach().clone() for p in agent.policy_net.parameters()]
    loss = agent.learn()
    after = list(agent.policy_net.parameters())
    check("learn_returns_finite_loss", bool(np.isfinite(loss)), True)
    check("recovery_q_finite", bool(np.isfinite(agent.last_recovery_q)), True)
    check("weights_updated",
          any(not torch.equal(a, b) for a, b in zip(before, after)), True)


if __name__ == "__main__":
    for fn in list(globals().values()):
        if callable(fn) and getattr(fn, "__name__", "").startswith("test_"):
            print(f"\n--- {fn.__name__} ---")
            fn()
    print("\n" + ("ALL PASS" if not FAIL else f"FAILURES: {FAIL}"))