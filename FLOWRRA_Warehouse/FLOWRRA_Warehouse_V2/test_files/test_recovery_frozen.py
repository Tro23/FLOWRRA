"""
test_recovery_frozen.py -- under the conflict ladder, the learned recovery head
is parked: frozen, not trained.

WHY. The ladder never asks it (L3 is a fixed rule), and its values drifted with
or without the teacher. Left training, its error signal would keep flowing into
the trunk the fleet heads share -- the leak that amplified teacher_run1. Frozen,
it cannot disturb Safety, Delivery or Efficiency. It stays in the network, so it
can be unfrozen later with a narrower job if L3 proves common.

HOW. GNNAgent(train_recovery=False): the recovery loss is left out of the total,
so the head's weights get no gradient. main_runner_warehouse.py passes
train_recovery = not conflict.ladder.enabled.

WHAT THIS CHECKS
  1. Frozen: one learning step leaves every recovery-head weight unchanged,
     while the rest of the network still learns.
  2. Not frozen (ladder off): the recovery head's weights do change.

Run:  python test_recovery_frozen.py      (ends with ALL PASS)
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


def transition(rng, n=4):
    return dict(
        node_features=rng.normal(size=(n, 8)).astype(np.float32),
        adj_matrix=np.ones((n, n), dtype=np.float32),
        actions=rng.integers(0, A, size=n).astype(np.int64),
        rewards=rng.normal(scale=0.5, size=(n, len(HEADS))).astype(np.float32),
        next_node_features=rng.normal(size=(n, 8)).astype(np.float32),
        next_adj_matrix=np.ones((n, n), dtype=np.float32),
        done=False, integrity=0.5, active_mask=np.ones(n, dtype=np.float32),
        recovery_action=int(rng.integers(0, 3)),
        next_valid_mask=np.ones((n, A), dtype=bool),
        next_active_mask=np.ones(n, dtype=np.float32),
        valid_mask=np.ones((n, A), dtype=bool),
        teacher_mask=np.zeros(n, dtype=np.float32),
    )


def one_step(train_recovery):
    agent = GNNAgent(node_feature_dim=8, edge_feature_dim=0, action_size=A,
                     hidden_dim=16, num_layers=1, n_heads=2,
                     reward_heads=HEADS, head_weights=[6.0, 4.0, 1.0],
                     batch_size=4, dropout=0.0, lr=1e-2, seed=0,
                     train_recovery=train_recovery)
    rng = np.random.default_rng(0)
    for _ in range(6):
        agent.memory.push(**transition(rng))
    rec_before = [p.detach().clone() for p in agent.policy_net.recovery_head.parameters()]
    rec_ids = {id(p) for p in agent.policy_net.recovery_head.parameters()}
    rest_before = [p.detach().clone() for p in agent.policy_net.parameters()
                   if id(p) not in rec_ids]
    random.seed(1)
    torch.manual_seed(1)
    agent.learn()
    rec_changed = any(not torch.equal(b, p) for b, p in
                      zip(rec_before, agent.policy_net.recovery_head.parameters()))
    rest_after = [p for p in agent.policy_net.parameters() if id(p) not in rec_ids]
    rest_changed = any(not torch.equal(b, p) for b, p in zip(rest_before, rest_after))
    return rec_changed, rest_changed


def test_frozen_under_the_ladder():
    rec_changed, rest_changed = one_step(train_recovery=False)
    check("recovery_head_unchanged", rec_changed, False)
    check("fleet_heads_and_trunk_still_learn", rest_changed, True)


def test_trains_when_ladder_off():
    rec_changed, _ = one_step(train_recovery=True)
    check("recovery_head_learns_without_the_ladder", rec_changed, True)


if __name__ == "__main__":
    for fn in list(globals().values()):
        if callable(fn) and getattr(fn, "__name__", "").startswith("test_"):
            print(f"\n--- {fn.__name__} ---")
            fn()
    print("\n" + ("ALL PASS" if not FAIL else f"FAILURES: {FAIL}"))