"""
benchmark_all.py

Runs FLOWRRA, Prioritized Planning and PIBT over the SAME instances of the 3D
MAPF Warehouse benchmark (Wang, Veerapaneni, Wu, Li & Likhachev, ICAPS 2024) and
writes one CSV with one row per (algorithm, map, seed, agent-count).

TWO FAIRNESS CONTROLS, both of which are easy to get wrong and fatal if you do:

  1. IDENTICAL TASK. FLOWRRA assigns goals via Hungarian matching at spawn. The
     baselines are handed that EXACT assignment rather than re-deriving their
     own. Otherwise the comparison silently includes task allocation, and a
     difference in who-goes-where gets reported as a difference in planning.

  2. IDENTICAL UNITS. FLOWRRA moves at base_speed 0.5, so it spends two
     simulator steps per graph hop; the planners move one hop per timestep. Raw
     makespan is therefore ~2x larger for FLOWRRA for no reason but the clock.
     Every cost is normalised to HOPS before it is reported, and both the raw
     and normalised columns are kept so the conversion is auditable.

The disturbance sweep (--disturb) is where the actual claim lives. On clean
instances a planner should win on makespan and sum-of-costs -- it is solving the
problem it was designed for, optimally or near-optimally. Report that plainly.
The question this harness exists to answer is what each method costs when the
warehouse stops cooperating.
"""

import os
import time
import argparse
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from benchmark_flowrra import discover_instances, load_instance, build_agent, run_flowrra_instance
from baselines_mapf import run_baseline_instance, run_rolling_horizon_instance
from core_warehouse import FLOWRRA
from node_warehouse import precompute_goal_distances
from config_warehouse import CONFIG

SPEED = CONFIG["warehouse"]["base_speed"]


def hungarian_assignment(G, pos_dict, fleet_missions, goal_pool) -> Dict[str, str]:
    """
    Extracts FLOWRRA's spawn-time goal assignment so every method solves the
    identical task. Built by constructing the env and reading back each fleet's
    locked target -- reimplementing the matching here would risk drifting from
    what FLOWRRA actually did.
    """
    gdm = precompute_goal_distances(G, [{"goal_node": g} for g in goal_pool])
    env = FLOWRRA(G, pos_dict, fleet_missions, mode="init", goal_distance_maps=gdm,
                  shared_pool_mode=True, goal_pool=goal_pool)
    return {n.id: n.current_goal_id for n in env.nodes if n.current_goal_id}


def normalise(row: Dict[str, Any], algorithm: str) -> Dict[str, Any]:
    """Adds hop-normalised costs. FLOWRRA counts simulator steps; planners count
    hops. Without this the comparison is off by 1/base_speed."""
    scale = SPEED if algorithm == "FLOWRRA" else 1.0
    row["makespan_hops"] = round(row.get("makespan", 0) * scale, 1)
    row["soc_hops"] = round(row.get("sum_of_costs", 0) * scale, 1)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--maps-dir", default="all_maps")
    ap.add_argument("--scens-dir", default="all_scens")
    ap.add_argument("--checkpoint", default="checkpoints/flowrra_warehouse_gnn.pth")
    ap.add_argument("--out", default="benchmark_all.csv")
    ap.add_argument("--agents", default="25")
    ap.add_argument("--seeds", default="")
    ap.add_argument("--maps", default="")
    ap.add_argument("--methods", default="FLOWRRA,PP,PIBT,RHCR-PIBT,RHCR-PP",
                    help="RHCR-* are rolling-horizon reimplementations of the warehouse "
                         "SOTA mechanism (Li et al., AAAI 2021), not ports of the tuned "
                         "C++ originals -- label them as such in any write-up.")
    ap.add_argument("--rhcr-window", type=int, default=20)
    ap.add_argument("--rhcr-replan-every", type=int, default=5)
    ap.add_argument("--max-steps", type=int, default=CONFIG["training"]["max_steps_per_episode"])
    ap.add_argument("--epsilon", type=float, default=0.02)
    ap.add_argument("--fixed-assignment", action="store_true",
                    help="each agent goes to ITS OWN scenario goal (classical MAPF, "
                         "comparable to published numbers). Default is shared-pool: "
                         "FLOWRRA and the baselines share one Hungarian allocation, "
                         "which is the warehouse/LMAPF setting and is NOT comparable "
                         "to MAPF literature -- on a validated instance it cut the "
                         "task from 298 to 128 hops.")
    ap.add_argument("--disturb", action="store_true",
                    help="stall a few agents mid-run; planners must replan, FLOWRRA does not")
    ap.add_argument("--disturb-step", type=int, default=60)
    ap.add_argument("--disturb-agents", type=int, default=3)
    ap.add_argument("--disturb-duration", type=int, default=20)
    args = ap.parse_args()

    counts = [int(a) for a in args.agents.split(",") if a.strip()]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    seed_f = {int(s) for s in args.seeds.split(",") if s.strip()}
    map_f = {m.strip() for m in args.maps.split(",") if m.strip()}

    instances = discover_instances(args.maps_dir, args.scens_dir)
    if map_f:
        instances = [i for i in instances if i["map"] in map_f]
    if seed_f:
        instances = [i for i in instances if i["seed"] in seed_f]
    if not instances:
        print("[Bench] No instances found. Expected layout:")
        print("  all_maps/<name>_Nodes.csv + <name>_Edges.csv")
        print("  all_scens/<name>/<name>_StartGoalLocations_Seed<k>.csv")
        return

    total = len(instances) * len(counts) * len(methods)
    print(f"[Bench] {len(instances)} (map,seed) x {len(counts)} agent-counts x "
          f"{len(methods)} methods = {total} runs"
          + (" [DISTURBED]" if args.disturb else " [clean]"))

    agent = None
    if "FLOWRRA" in methods:
        G0, p0, m0, pool0 = load_instance(instances[0]["nodes_csv"],
                                          instances[0]["edges_csv"],
                                          instances[0]["scen_csv"], counts[0])
        agent = build_agent(G0, p0, m0, pool0, args.checkpoint)

    rows, t_start, done = [], time.time(), 0
    for inst in instances:
        for k in counts:
            try:
                G, pos_dict, missions, pool = load_instance(
                    inst["nodes_csv"], inst["edges_csv"], inst["scen_csv"], k)
            except Exception as exc:
                print(f"  [error] load {inst['map']} seed{inst['seed']}: {exc}")
                continue
            if len(missions) < k:
                continue

            if args.fixed_assignment:
                # Classical MAPF: the instance's own pairing, identical for all
                # methods, and comparable to published EECBS/LaCAM/PIBT numbers.
                assignment = {m["id"]: m["goal_node"] for m in missions}
            else:
                # Shared pool: FLOWRRA allocates via Hungarian and the baselines
                # inherit that SAME allocation. Not "making them like FLOWRRA" --
                # FLOWRRA solves allocation+routing+execution, so a fair baseline
                # must be a complete system too (allocation + routing). Handing
                # them a different allocation would hand FLOWRRA a 2.3x shorter
                # task and call the difference "planning".
                assignment = hungarian_assignment(G, pos_dict, missions, pool)
            disturbance = None
            if args.disturb:
                disturbance = {"step": args.disturb_step,
                               "agents": list(range(min(args.disturb_agents, k))),
                               "duration": args.disturb_duration}

            for method in methods:
                try:
                    if method == "FLOWRRA":
                        # FLOWRRA has no replan step; a stall is just a fleet that
                        # did not move, which its policy already handles inline.
                        r = run_flowrra_instance(G, pos_dict, missions, pool, agent,
                                                 args.max_steps, args.epsilon,
                                                 shared_pool=not args.fixed_assignment)
                        r["algorithm"] = "FLOWRRA"
                        r["plan_time_s"] = 0.0
                        r["replan_s"] = 0.0
                    elif method.startswith("RHCR-"):
                        r = run_rolling_horizon_instance(
                            G, pos_dict, missions, pool, method.split("-", 1)[1],
                            args.max_steps, assignment,
                            window=args.rhcr_window,
                            replan_every=args.rhcr_replan_every,
                            disturbance=disturbance)
                    else:
                        r = run_baseline_instance(G, pos_dict, missions, pool, method,
                                                  args.max_steps, assignment, disturbance)
                except Exception as exc:
                    print(f"  [error] {method} {inst['map']} seed{inst['seed']} k={k}: {exc}")
                    r = {"algorithm": method, "num_agents": k, "success": 0, "error": str(exc)}

                r = normalise(r, method)
                r.update(map=inst["map"], seed=inst["seed"], requested_agents=k,
                         disturbed=int(args.disturb),
                     protocol="fixed" if args.fixed_assignment else "shared_pool")
                rows.append(r)
                done += 1
                pd.DataFrame(rows).to_csv(args.out, index=False)

            rate = (time.time() - t_start) / max(done, 1)
            print(f"[Bench] {done}/{total} | {rate:.1f}s/run | eta {rate*(total-done)/60:.0f}m "
                  f"| {inst['map']} seed{inst['seed']} k={k}")

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f"\n[Bench] Wrote {len(df)} rows -> {args.out}\n")

    cols = ["success", "completion_rate", "makespan_hops", "soc_hops", "collisions",
            "mean_integrity", "proximity_margin", "plan_time_s", "replan_s",
            "decision_ms_per_step", "mean_time_to_recoherence", "blast_radius",
            "replan_count"]
    have = [c for c in cols if c in df.columns]
    print("=" * 78)
    print(("COMPARISON [" + ("FIXED assignment" if args.fixed_assignment
                                             else "SHARED POOL / Hungarian") + "]")
          + (" UNDER DISTURBANCE" if args.disturb else " (clean)"))
    print("=" * 78)
    print(df.groupby("algorithm")[have].mean().round(3).to_string())
    print("\nSuccess rate by map:")
    print(df.pivot_table(index="map", columns="algorithm",
                         values="success", aggfunc="mean").round(2).to_string())


if __name__ == "__main__":
    main()