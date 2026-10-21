"""
check_identity.py -- does a change leave today's behaviour EXACTLY as it was?

    python check_identity.py --old ../flowrra_before --new .
    python check_identity.py --old ../flowrra_before --new . --steps 150 --expect-changed risk_steps

Runs the same seeded episodes, with a LEARNING agent, once in each source tree
(each in its own subprocess, so the two trees never share an import), and
compares:

  * every fleet's position at every step, and every step's reward
  * every head loss at every learning step, and the recovery loss
  * the final network weights (hash of every parameter)
  * every statistic in get_error_statistics(), key by key

under two configurations: BENCHMARK (stream off, fixed missions) and STREAM
(order stream on, docks, an order bank). CONFLICT_DESIGN.md asks for exactly
this after every component, with its switch off.

Statistics a change is MEANT to alter are named with --expect-changed (and new
keys are listed, never compared). Everything else must match to the last digit.
A measurement-only change must also leave positions, losses and weights
byte-identical: if they move, the "measurement" is feeding a decision.

The fixture is the smoke-test warehouse (test_smoke_integration.build_instance)
with every Phase-1/2 flag on, 24 fleets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import subprocess
import sys
import tempfile

# ---------------------------------------------------------------- the worker
WORKER = r'''
import sys, os, pickle, random, hashlib, io, contextlib
sys.path.insert(0, os.getcwd())
import numpy as np
import torch
torch.set_num_threads(1)
from config_warehouse import CONFIG
import test_smoke_integration as smoke

mode, steps, seed, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
smoke.set_flags(True)
CONFIG["stream"]["enabled"] = (mode == "stream")
CONFIG["training"]["batch_size"] = 16          # learning starts early

from core_warehouse import FLOWRRA
from agent_warehouse import GNNAgent
from node_warehouse import precompute_goal_distances
from main_runner_warehouse import _recovery_value_bound

G, grid, miss, gdm, gp = smoke.build_instance(24, seed)
kw = dict(mode="training", goal_distance_maps=gdm, shared_pool_mode=True, goal_pool=gp)
if mode == "stream":
    rng = np.random.default_rng(seed + 100)
    nodes = sorted(G.nodes())
    bank = sorted({nodes[int(i)] for i in rng.integers(0, len(nodes), 60)} | set(gp))
    gdm = precompute_goal_distances(G, [{"goal_node": g} for g in bank])
    kw.update(goal_distance_maps=gdm, order_bank=bank, order_seed=seed * 1000003 + 1,
              order_floor_window=None)
with contextlib.redirect_stdout(io.StringIO()):
    env = FLOWRRA(G, grid, miss, **kw)
base = len(env.nodes[0].get_state_vector(env.nodes))
rd = CONFIG["reward_decomposition"]; g = CONFIG["gnn"]
agent = GNNAgent(
    node_feature_dim=base + env.density.output_dim,
    edge_feature_dim=g["edge_feature_dim"], action_size=g["action_size"],
    hidden_dim=g["hidden_dim"], num_layers=g["num_layers"], n_heads=g["num_heads"],
    reward_heads=rd["heads"], head_weights=rd["weights"],
    dropout=g["dropout"], lr=g["learning_rate"], gamma=CONFIG["training"]["gamma"],
    buffer_capacity=CONFIG["training"]["buffer_capacity"],
    batch_size=CONFIG["training"]["batch_size"], stability_coef=g["stability_coef"],
    encoder_mode=g["encoder"], base_feature_dim=base,
    density_diamond_mask=env.density._diamond_mask,
    # As main_runner_warehouse builds it, so the learning path is the real one.
    double_dqn=bool(CONFIG["training"].get("double_dqn", False)),
    target_tau=float(CONFIG["training"].get("target_tau", 0.0)),
    recovery_double_dqn=bool(CONFIG["training"].get("recovery_double_dqn", False)),
    recovery_value_bound=_recovery_value_bound(),
    per_fleet_terminal=bool(CONFIG["training"].get("per_fleet_terminal", False)))
env.gnn = agent
agent.reset_episode_state()

pos, rew, losses = [], [], []
for t in range(steps):
    with contextlib.redirect_stdout(io.StringIO()):
        r = env.step(episode_step=3, total_episodes=10)
        if len(agent.memory) >= agent.batch_size:
            agent.learn(node_ids=[n.id for n in env.nodes])
            losses.append((sorted(agent.last_head_losses.items()),
                           agent.last_recovery_loss))
    rew.append(r)
    pos.append(sorted((n.id, tuple(np.round(n.current_pos, 4).tolist())) for n in env.nodes))
    if env.is_episode_over():
        break

h = hashlib.sha256()
for k, v in sorted(agent.policy_net.state_dict().items()):
    h.update(k.encode()); h.update(v.detach().cpu().numpy().tobytes())
stats = env.get_error_statistics()
pickle.dump({"pos": pos, "rew": rew, "losses": losses, "weights": h.hexdigest(),
             "stats": stats, "steps": t + 1, "collisions": env.loop.total_collisions},
            open(out, "wb"))
'''


def run_tree(tree: str, mode: str, steps: int, seed: int) -> dict:
    with tempfile.TemporaryDirectory() as td:
        script = os.path.join(td, "worker.py")
        out = os.path.join(td, "out.pkl")
        open(script, "w").write(WORKER)
        r = subprocess.run([sys.executable, script, mode, str(steps), str(seed), out],
                           cwd=tree, capture_output=True, text=True)
        if r.returncode != 0:
            raise SystemExit(f"worker failed in {tree} ({mode}):\n{r.stderr[-3000:]}")
        return pickle.load(open(out, "rb"))


def same(a, b) -> bool:
    if isinstance(a, float) and isinstance(b, float):
        return (math.isnan(a) and math.isnan(b)) or a == b
    return a == b


def compare(label: str, old: dict, new: dict, expect_changed: set) -> list:
    fails = []
    print(f"\n[{label}] {old['steps']} steps, collisions {old['collisions']} vs {new['collisions']}")

    first = next((i for i, (a, b) in enumerate(zip(old["pos"], new["pos"])) if a != b), None)
    ok = first is None and len(old["pos"]) == len(new["pos"])
    print(f"   positions every step    {'identical' if ok else f'DIVERGE at step {first}'}")
    if not ok:
        fails.append(f"{label}: positions")
    ok = old["rew"] == new["rew"]
    print(f"   reward every step       {'identical' if ok else 'DIFFER'}")
    if not ok:
        fails.append(f"{label}: rewards")
    ok = old["losses"] == new["losses"]
    print(f"   losses ({len(old['losses'])} learn steps) {'identical' if ok else 'DIFFER'}")
    if not ok:
        fails.append(f"{label}: losses")
    ok = old["weights"] == new["weights"]
    print(f"   final weights           {'identical' if ok else 'DIFFER'}")
    if not ok:
        fails.append(f"{label}: weights")

    so, sn = old["stats"], new["stats"]
    shared = sorted(set(so) & set(sn))
    added = sorted(set(sn) - set(so))
    removed = sorted(set(so) - set(sn))
    changed = [k for k in shared if not same(so[k], sn[k])]
    unexpected = [k for k in changed if k not in expect_changed]
    print(f"   statistics              {len(shared)} shared, {len(shared) - len(changed)} identical, "
          f"{len(changed)} changed, {len(added)} new, {len(removed)} removed")
    for k in changed:
        tag = "expected" if k in expect_changed else "UNEXPECTED"
        print(f"      {tag:<10} {k}: {so[k]!r} -> {sn[k]!r}")
    if added:
        print(f"      new: {', '.join(added)}")
    if removed:
        print(f"      REMOVED: {', '.join(removed)}")
        fails.append(f"{label}: statistics removed {removed}")
    if unexpected:
        fails.append(f"{label}: unexpected statistic changes {unexpected}")
    return fails


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--old", required=True, help="source tree before the change")
    ap.add_argument("--new", default=".", help="source tree after the change")
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--modes", default="benchmark,stream")
    ap.add_argument("--expect-changed", default="",
                    help="comma-separated statistics the change is MEANT to alter")
    args = ap.parse_args()

    expect = {k.strip() for k in args.expect_changed.split(",") if k.strip()}
    fails = []
    for mode in [m.strip() for m in args.modes.split(",") if m.strip()]:
        for seed in [int(s) for s in args.seeds.split(",") if s.strip()]:
            old = run_tree(os.path.abspath(args.old), mode, args.steps, seed)
            new = run_tree(os.path.abspath(args.new), mode, args.steps, seed)
            fails += compare(f"{mode} seed {seed}", old, new, expect)

    print()
    if fails:
        print("NOT IDENTICAL:\n  " + "\n  ".join(fails))
        sys.exit(1)
    print("IDENTICAL -- behaviour, learning and every unexpected statistic unchanged.")


if __name__ == "__main__":
    main()
