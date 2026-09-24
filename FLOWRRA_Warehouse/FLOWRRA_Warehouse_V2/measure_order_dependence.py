"""
measure_order_dependence.py -- does the step depend on the order fleets are processed in?

    python measure_order_dependence.py
    python measure_order_dependence.py --fleets 40 --horizons 1 5 40

A SIMULTANEOUS step -- every fleet takes one step together -- cannot depend on
where a fleet sits in env.nodes. This runs the identical scenario twice, once
with the fleet list forwards and once reversed, and counts how many fleets end
up somewhere different.

To isolate the ORDER from everything else, every fleet is given a fixed,
order-independent action: one step down its own goal gradient. So the actions
are identical in both runs; only the processing order differs.

  0 / N at every horizon   -> the step is simultaneous
  anything else            -> the outcome depends on list position

Measured before the simultaneous step existed: 2 / 40 after one step, settling
at 5 / 40. This is acceptance test 1 in SIMULTANEOUS_STEP.md.

CAVEAT. The smoke-test fixture draws goals with replacement, so it has fewer
unique goals than fleets (35 for 40), which manufactures goal contention the
real scenarios do not have. Treat the magnitude as an upper bound; the
invariant being tested -- zero divergence -- does not depend on it.
"""

import argparse
import random
import sys

import numpy as np
import torch

from config_warehouse import CONFIG
import test_smoke_integration as smoke
from core_warehouse import FLOWRRA
from agent_warehouse import GNNAgent


def run(reverse: bool, steps: int, n_fleets: int, seed: int, simultaneous: bool):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    smoke.set_flags(True)
    CONFIG.setdefault("step", {})["simultaneous"] = simultaneous
    G, grid, miss, gdm, gp = smoke.build_instance(n_fleets, seed)
    env = FLOWRRA(G, grid, miss, mode="training", goal_distance_maps=gdm,
                  shared_pool_mode=True, goal_pool=gp)
    base = len(env.nodes[0].get_state_vector(env.nodes))
    rd = CONFIG["reward_decomposition"]
    g = CONFIG["gnn"]
    env.gnn = GNNAgent(
        node_feature_dim=base + env.density.output_dim,
        edge_feature_dim=g["edge_feature_dim"], action_size=7,
        hidden_dim=g["hidden_dim"], num_layers=g["num_layers"],
        n_heads=g["num_heads"], reward_heads=rd["heads"],
        head_weights=rd["weights"], encoder_mode="conv",
        base_feature_dim=base,
        density_diamond_mask=env.density._diamond_mask)

    # Order-independent actions: each fleet steps down its OWN goal gradient.
    def fixed_actions(**kw):
        out = []
        for node, mask in zip(env.nodes, kw["valid_action_masks"]):
            grad = node.get_goal_gradient()
            cands = [(grad[a - 1], a) for a in range(1, 7) if mask[a]]
            best = max(cands) if cands else (0.0, 0)
            out.append(best[1] if best[0] > 0 else 0)
        return np.array(out)

    env.gnn.choose_actions = fixed_actions
    env.gnn.learn = lambda *a, **k: None     # no weight updates between runs

    if reverse:
        env.nodes = env.nodes[::-1]
    for _ in range(steps):
        env.step(episode_step=40, total_episodes=40)

    positions = {n.id: tuple(np.round(n.current_pos, 2)) for n in env.nodes}
    smoke.set_flags(False)
    return positions, sorted(env.claimed_goals), env.loop.total_collisions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fleets", type=int, default=40)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--horizons", type=int, nargs="+", default=[1, 5, 40])
    ap.add_argument("--simultaneous", action="store_true",
                    help="turn step.simultaneous on; must then report 0 at every horizon")
    args = ap.parse_args()

    worst = 0
    for h in args.horizons:
        fwd, claims_f, coll_f = run(False, h, args.fleets, args.seed, args.simultaneous)
        rev, claims_r, coll_r = run(True, h, args.fleets, args.seed, args.simultaneous)
        differ = [fid for fid in fwd if fwd[fid] != rev[fid]]
        worst = max(worst, len(differ))
        print(f"after {h:>3} step(s): fleets in a DIFFERENT place "
              f"{len(differ):>3}/{args.fleets}   "
              f"same goals claimed: {claims_f == claims_r}   "
              f"collisions {coll_f} vs {coll_r}")

    print()
    if worst == 0:
        print("SIMULTANEOUS -- no fleet depends on its position in the list.")
        sys.exit(0)
    print(f"ORDER-DEPENDENT -- up to {worst} fleet(s) diverged. "
          f"See SIMULTANEOUS_STEP.md.")
    sys.exit(1)


if __name__ == "__main__":
    main()