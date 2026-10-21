"""
lesion_harness.py

Measures how much an ALREADY TRAINED checkpoint actually reads each block of its
state vector, by zeroing that block at inference time and re-running the same
instances.

    python lesion_harness.py --resume checkpoints/ckpt_pilot3.pth \
        --maps-dir all_maps --scens-dir all_scens_v2 \
        --episodes 30 --agent-sets 25 \
        --arms control,rays_all,goal_gradient,density

WHAT THIS IS AND IS NOT
-----------------------
THIS IS A LESION. It answers: does the trained policy read this block?
IT IS NOT AN ABLATION. It does not answer: could a policy trained without this
block have done as well?

Those come apart in both directions and the distinction is the whole point of
running this before committing 24 hours to a retrain:

  * A block can lesion CATASTROPHICALLY and still be redundant. The network
    leaned on it because it was there; trained without it, it would have learned
    to read something else that carries the same information. The six BFS
    goal-gradient dims and the ray distances overlap heavily -- both are
    recomputed per map at runtime, and both encode "which way is clear".

  * A block can lesion HARMLESSLY and still be necessary. Removing it at
    inference leaves the network's other features carrying a representation that
    was only learnable BECAUSE the block was present during training.

So: a large lesion effect licenses spending the retrain. A near-zero lesion
effect is much stronger evidence -- if zeroing all 42 ray dims barely moves
behaviour, the trained policy is not reading its rays, and improving ray
fidelity is not where the next gain lives.

DESIGN NOTES
------------
PAIRED INSTANCES. Every arm sees the identical sequence of (map, scenario,
fleet count) drawn from one seeded RNG re-created per arm. Differences between
arms are therefore attributable to the lesion and not to instance sampling. An
unpaired version of this test on 30 episodes would be swamped by the ~25%
step-count variance tail already documented for this benchmark.

GREEDY. agent.epsilon_gaussian is overridden to return 0.0 for the whole run.
Under the default Gaussian schedule, epsilon at an arbitrary episode index is
whatever the training curve says, which would inject a different amount of
random action into each arm and is not what is being measured.

FROZEN. agent.learn() is never called and the checkpoint is reloaded from disk
before every arm, so no arm can contaminate the next. Transitions still land in
the replay buffer; nothing reads them.

METRICS. Reports completion rate, collisions, gradient agreement, brake duty
cycle and override rate per arm, with a paired per-episode delta against the
control arm. mean_peer_gap is a CENSORED mean (at proximity search_radius) --
see proximity_warehouse.GraphProximity.nearest_censored.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import statistics
import sys
import time
from typing import Any, Dict, List

import numpy as np

from config_warehouse import CONFIG


def _paired_delta(control: List[float], arm: List[float]) -> Dict[str, float]:
    """
    Paired per-episode difference, with a t-like ratio. Not a p-value: with 20-40
    paired episodes and no normality assumption checked, a reported p here would
    be more confident than the design supports. The ratio is for ranking arms.
    """
    pairs = [(a, c) for a, c in zip(arm, control)
             if a is not None and c is not None
             and np.isfinite(a) and np.isfinite(c)]
    if len(pairs) < 2:
        return {"mean_delta": float("nan"), "sd": float("nan"),
                "ratio": float("nan"), "n": len(pairs)}
    diffs = [a - c for a, c in pairs]
    mean = statistics.fmean(diffs)
    sd = statistics.pstdev(diffs)
    se = sd / (len(diffs) ** 0.5) if sd > 0 else 0.0
    return {"mean_delta": mean, "sd": sd,
            "ratio": (mean / se) if se > 0 else float("inf") if mean else 0.0,
            "n": len(diffs)}


def run_arm(arm: str, args, names, cache, sample_instance, FLOWRRA, agent) -> List[Dict[str, Any]]:
    """Run `args.episodes` episodes with one lesion applied. Returns per-episode rows."""
    # The lesion is read by FLOWRRA.__init__ from CONFIG, so it has to be set
    # before each environment is constructed, not after.
    CONFIG["lesion"]["zero_blocks"] = [] if arm == "control" else arm.split("+")

    # Re-seeded per arm so every arm draws the SAME instance sequence.
    rng = random.Random(args.seed)
    agent_sets = [int(v) for v in args.agent_sets.split(",") if v.strip()]

    rows = []
    for ep in range(1, args.episodes + 1):
        map_name = rng.choice(names)
        k = (rng.choice(agent_sets) if agent_sets
             else rng.randint(args.agents_min, args.agents_max))
        inst = sample_instance(cache, map_name, k, rng)
        if inst is None:
            continue

        env = FLOWRRA(inst["G"], inst["pos_dict"], inst["missions"], mode="training",
                      goal_distance_maps=inst["gdm"], shared_pool_mode=True,
                      goal_pool=inst["goal_pool"])
        env.gnn = agent
        agent.reset_episode_state()

        steps = 0
        for _ in range(args.max_steps):
            env.step(episode_step=ep, total_episodes=args.episodes)
            steps += 1
            if env.is_episode_over():
                break

        est = env.get_error_statistics()
        total = max(1, len(env.goal_pool))
        rows.append({
            "arm": arm, "episode": ep, "map": map_name, "scen": inst["scen"],
            "agents": len(env.nodes), "steps": steps,
            "completion_rate": len(env.claimed_goals) / total,
            "collisions": env.loop.total_collisions,
            "collisions_per_step": env.loop.total_collisions / max(1, steps),
            "gradient_agreement": env.get_gradient_agreement(),
            "brake_duty_cycle": est["brake_duty_cycle"],
            "mean_peer_gap_censored": est["mean_peer_gap"],
            "action_override_rate": est["action_override_rate"],
            "tabu_overrides": est.get("tabu_overrides", 0),
            "tabu_overrides_stuck": est.get("tabu_overrides_stuck", 0),
            "projection_fallbacks": est.get("projection_fallbacks", 0),
            "phantom_pairs_rejected": est.get("phantom_pairs_rejected", 0),
            # Did idle caching actually ENGAGE? Without this, a null result is
            # ambiguous: it could mean staleness does not matter, or it could
            # mean almost nothing was ever reused. The 76% reuse that motivated
            # this was measured at 200 fleets over 200 steps; these episodes are
            # 25 fleets with a median around 150 steps, so the regime may simply
            # not arise. A reuse rate near zero makes the comparison vacuous.
            "occupancy_pct": 100.0 * len(env.nodes) / max(1, len(inst["pos_dict"])),
            "map_nodes": len(inst["pos_dict"]),
            "ray_mean_cells": est.get("ray_mean_cells", 0.0),
            "ray_saturated_rate": est.get("ray_saturated_rate", 0.0),
            "ray_blocked_rate": est.get("ray_blocked_rate", 0.0),
            "ray_peer_hit_rate": est.get("ray_peer_hit_rate", 0.0),
            "idle_mode": CONFIG.get("perception", {}).get("idle_mode", "full"),
            "idle_reused": est.get("idle_perception_reused", 0),
            "idle_computed": est.get("idle_perception_computed", 0),
            "idle_reuse_rate": est.get("idle_perception_reuse_rate", 0.0),
            "frozen_at_end": len(getattr(env, "frozen_nodes", ())),
            "handovers_completed": est["handovers_completed"],
            "recovery_wasted": est["recovery_wasted"],
        })
        print(f"  [{arm:<18}] ep {ep:03d} {map_name:<20} k={len(env.nodes):<3} "
              f"done={rows[-1]['completion_rate']:.3f} coll={rows[-1]['collisions']:<4} "
              f"grad={rows[-1]['gradient_agreement']:.3f} steps={steps}")
    return rows


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--resume", required=True, help="checkpoint to lesion")
    ap.add_argument("--maps-dir", default="all_maps")
    ap.add_argument("--scens-dir", default="all_scens_v2")
    ap.add_argument("--maps", default="")
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--agents-min", type=int, default=25)
    ap.add_argument("--agents-max", type=int, default=25)
    ap.add_argument("--agent-sets", default="25")
    ap.add_argument("--max-steps", type=int,
                    default=CONFIG["training"]["max_steps_per_episode"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="lesion_results")
    ap.add_argument(
        "--arms", default="control,rays_all,goal_gradient,density",
        help="comma-separated lesion arms. 'control' = no lesion. Combine blocks "
             "with '+', e.g. rays_all+goal_gradient. Block names come from "
             "FleetNode.state_layout() plus 'density'.")
    args = ap.parse_args()

    # Imported here, after argparse, so --help works without torch installed.
    import torch
    from main_runner_warehouse import MapCache, sample_instance
    from core_warehouse import FLOWRRA
    from agent_warehouse import GNNAgent

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    only = [m.strip() for m in args.maps.split(",") if m.strip()] or None
    cache = MapCache(args.maps_dir, args.scens_dir)
    names = cache.discover(only)
    if not names:
        raise SystemExit(f"no usable maps in {args.maps_dir} / {args.scens_dir}")

    rd = CONFIG["reward_decomposition"]
    heads, weights = rd["heads"], rd["weights"]

    # Probe env to size the state vector. MUST be built with NO lesion: the
    # lesion zeroes dimensions, it does not remove them, so input_dim is the
    # same across arms -- but building the probe under a lesion would still be
    # confusing to read in the log.
    CONFIG["lesion"]["zero_blocks"] = []
    probe_rng = random.Random(args.seed)
    agent_sets = [int(v) for v in args.agent_sets.split(",") if v.strip()]
    probe = sample_instance(cache, names[0],
                            min(agent_sets) if agent_sets else args.agents_min,
                            probe_rng)
    penv = FLOWRRA(probe["G"], probe["pos_dict"], probe["missions"], mode="init",
                   goal_distance_maps=probe["gdm"], shared_pool_mode=True,
                   goal_pool=probe["goal_pool"])
    n0 = penv.nodes[0]
    base_len = len(n0.get_state_vector(penv.nodes))
    density_dim = len(penv.density.get_local_affordance(n0.current_pos, penv.nodes, set()))
    input_dim = base_len + density_dim

    layout = dict(n0.state_layout())
    layout["density"] = (base_len, input_dim)
    print(f"\ninput_dim={input_dim}  (base {base_len} + density {density_dim})")
    print("state layout:")
    for name, (lo, hi) in sorted(layout.items(), key=lambda kv: kv[1]):
        if name.startswith("_"):
            continue
        print(f"   {name:<22} [{lo:>4}, {hi:>4})  {hi-lo:>4} dims "
              f"({100*(hi-lo)/input_dim:4.1f}%)")
    print()

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    for a in arms:
        if a == "control":
            continue
        for blk in a.split("+"):
            if blk not in layout:
                raise SystemExit(
                    f"unknown lesion block {blk!r}. Known: "
                    f"{sorted(k for k in layout if not k.startswith('_'))}")

    all_rows: List[Dict[str, Any]] = []
    t0 = time.time()
    for arm in arms:
        print(f"\n=== arm: {arm} ===")
        # Rebuilt and reloaded per arm so nothing carries over. The buffer fills
        # during a run and learn() is never called, but a fresh agent removes any
        # doubt about that.
        agent = GNNAgent(
            node_feature_dim=input_dim,
        edge_feature_dim=CONFIG["gnn"].get("edge_feature_dim", 0),
            action_size=CONFIG["gnn"]["action_size"],
            hidden_dim=CONFIG["gnn"]["hidden_dim"],
            num_layers=CONFIG["gnn"]["num_layers"],
            n_heads=CONFIG["gnn"]["num_heads"],
            reward_heads=heads, head_weights=weights,
            dropout=CONFIG["gnn"]["dropout"], lr=CONFIG["gnn"]["learning_rate"],
            gamma=CONFIG["training"]["gamma"],
            buffer_capacity=CONFIG["training"]["buffer_capacity"],
            batch_size=CONFIG["training"]["batch_size"],
            stability_coef=CONFIG["gnn"]["stability_coef"],
        )
        agent.load(args.resume)
        # Pure greedy for every arm. Without this each arm inherits whatever the
        # Gaussian schedule returns at that episode index, which is exploration
        # noise varying across arms and would be read as a lesion effect.
        agent.epsilon_gaussian = lambda *a, **k: 0.0
        all_rows.extend(run_arm(arm, args, names, cache, sample_instance, FLOWRRA, agent))

    CONFIG["lesion"]["zero_blocks"] = []
    os.makedirs(args.out, exist_ok=True)

    try:
        import pandas as pd
        pd.DataFrame(all_rows).to_csv(os.path.join(args.out, "lesion_rows.csv"), index=False)
    except Exception as e:
        with open(os.path.join(args.out, "lesion_rows.json"), "w") as f:
            json.dump(all_rows, f, indent=2)
        print(f"(pandas unavailable: {e}; wrote JSON instead)")

    # gradient_agreement and brake_duty_cycle lead because they are the only two
    # with usable resolution at 25 fleets. Measured on the 2026-09-14 control
    # arm: collisions were 1 in 6,991 steps (floor), and completion took three
    # distinct values with 21 of 30 episodes at exactly 1.00 (ceiling). Both
    # headline metrics are pinned against their bounds and cannot register a
    # small degradation.
    metrics = ["gradient_agreement", "brake_duty_cycle", "completion_rate",
               "collisions_per_step", "action_override_rate", "idle_reuse_rate",
               "ray_saturated_rate", "ray_peer_hit_rate", "occupancy_pct"]
    by_arm: Dict[str, List[Dict[str, Any]]] = {}
    for r in all_rows:
        by_arm.setdefault(r["arm"], []).append(r)

    print("\n" + "=" * 78)
    print(f"LESION RESULTS  ({args.episodes} paired episodes/arm, "
          f"{(time.time()-t0)/60:.1f} min)")
    print("=" * 78)
    print("A lesion says the trained policy READS a block. It does NOT say a policy")
    print("trained without it would be worse. Do not report these as ablations.\n")

    _coll = sum(r["collisions"] for r in all_rows)
    _steps = sum(r["steps"] for r in all_rows)
    _ceil = sum(1 for r in all_rows if r["completion_rate"] >= 1.0)
    print(f"RESOLUTION CHECK: {_coll} collisions in {_steps} steps; "
          f"{_ceil}/{len(all_rows)} episodes at completion 1.00.")
    if _coll < 10:
        print("  collisions are on the FLOOR -- they cannot detect a small change here.")
    if _ceil > 0.5 * len(all_rows):
        print("  completion is on the CEILING -- read gradient_agreement instead.")
    print()

    control = by_arm.get("control", [])
    summary = {}
    for arm, rows in by_arm.items():
        print(f"-- {arm}")
        entry = {}
        for m in metrics:
            vals = [r[m] for r in rows if np.isfinite(r[m])]
            mean = statistics.fmean(vals) if vals else float("nan")
            line = f"   {m:<22} {mean:8.4f}"
            if arm != "control" and control:
                d = _paired_delta([r[m] for r in control], [r[m] for r in rows])
                line += (f"   delta {d['mean_delta']:+8.4f}  "
                         f"ratio {d['ratio']:+6.2f}  n={d['n']}")
                entry[m] = {"mean": mean, **d}
            else:
                entry[m] = {"mean": mean}
            print(line)
        summary[arm] = entry

    with open(os.path.join(args.out, "lesion_summary.json"), "w") as f:
        json.dump({"args": vars(args), "layout":
                   {k: list(v) for k, v in layout.items()},
                   "summary": summary}, f, indent=2)
    print(f"\nwrote {args.out}/lesion_rows.csv and {args.out}/lesion_summary.json")


if __name__ == "__main__":
    main()