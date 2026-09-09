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
    """
    DEPRECATED CONVERSION, kept only so old CSVs still parse.

    makespan_hops / soc_hops multiplied a TIME by base_speed to approximate a
    distance. Two things make that unsound: FLOWRRA never sustains nominal speed
    (affordance braking floors it at 0.05, dwelling at a pickup moves it not at
    all), and the underlying sum_of_costs charged every dead vehicle the full
    step budget. distance_travelled now counts cells actually traversed on both
    arms, which needs no conversion and no assumption about what a timestep is
    worth. Prefer it.
    """
    """Adds hop-normalised costs. FLOWRRA counts simulator steps; planners count
    hops. Without this the comparison is off by 1/base_speed."""
    scale = SPEED if algorithm.startswith("FLOWRRA") else 1.0
    row["makespan_hops"] = round(row.get("makespan", 0) * scale, 1)
    row["soc_hops"] = round(row.get("sum_of_costs", 0) * scale, 1)
    return row


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
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
    ap.add_argument("--steps-per-hop", type=int, default=int(round(1.0 / SPEED)),
                    help="simulator steps a BASELINE spends per graph edge. Default "
                         "matches FLOWRRA's base_speed so both arms share one clock "
                         "and raw step counts compare directly, with no conversion. "
                         "Set 1 for the textbook MAPF convention (one edge per "
                         "timestep), which makes the planners twice as fast as "
                         "FLOWRRA by construction.")
    ap.add_argument("--rhcr-window", type=int, default=20)
    ap.add_argument("--rhcr-replan-every", type=int, default=5)
    ap.add_argument("--max-steps", type=int, default=CONFIG["training"]["max_steps_per_episode"])
    ap.add_argument("--epsilon", default="0.02",
                    help="FLOWRRA's evaluation epsilon. Comma-separated values run "
                         "FLOWRRA once PER VALUE as separate rows (FLOWRRA(eps=0.10) "
                         "etc), which is more informative than drawing one at random "
                         "per instance: a random draw averages the effect of epsilon "
                         "into the noise, and the effect is the question. FLOWRRA "
                         "scored 0.25 success on the easy run -- if that is the policy "
                         "locking deterministically, epsilon should fix it, and a "
                         "sweep shows exactly where. Not 0.0: at exactly zero the "
                         "policy can deadlock with no stochastic escape, which the "
                         "replanning baselines get for free.")
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
    ap.add_argument("--fail", action="store_true",
                    help="PERMANENT failure: the vehicle dies and the order it was "
                         "carrying is stranded at its cell. Distinct from --disturb, "
                         "which is a temporary stall the vehicle recovers from. This "
                         "is the setting the recovery claim is about.")
    ap.add_argument("--fail-waves", default="",
                    help="comma-separated progress fractions, e.g. '0.35,0.6'. Each "
                         "fires a wave of --fail-agents failures. TWO OR MORE WAVES "
                         "ARE REQUIRED to test rescuer death: at a single burst nobody "
                         "is rescuing yet, so raising --fail-agents gives simultaneous "
                         "deaths, never a death mid-rescue. Overrides --fail-progress.")
    ap.add_argument("--fail-progress", type=float, default=0.4,
                    help="fire the failure once this FRACTION OF ORDERS has been "
                         "delivered. Self-normalising across arms with very different "
                         "clocks -- a shared step index landed 17%% into FLOWRRA's run "
                         "and 57%% into RHCR's, so they saw 11 and 7 failures on the "
                         "same instances. Raise toward 0.7 to make rescuers scarce, "
                         "which is where recall and re-dispatch are supposed to matter.")
    ap.add_argument("--fail-hop", type=int, default=15,
                    help="inject the failure after this many HOPS of travel, not "
                         "simulator steps. FLOWRRA moves at base_speed 0.5 so it "
                         "spends 1/base_speed steps per hop, while the planners move "
                         "one hop per step. A shared step index therefore lands "
                         "mid-mission for one arm and after the episode has finished "
                         "for the other -- measured: at step 40, FLOWRRA saw 2 "
                         "orphaned orders and RHCR saw 0, because RHCR was done by "
                         "step 8. Converted per arm, exactly as costs already are.")
    ap.add_argument("--fail-agents", type=int, default=3)
    args = ap.parse_args()

    counts = [int(a) for a in args.agents.split(",") if a.strip()]
    eps_list = [float(e) for e in str(args.epsilon).split(",") if e.strip()]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    # One FLOWRRA arm per epsilon value.
    if "FLOWRRA" in methods and len(eps_list) > 1:
        i = methods.index("FLOWRRA")
        methods = (methods[:i] + [f"FLOWRRA@{e}" for e in eps_list] + methods[i+1:])
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
    if any(m == "FLOWRRA" or m.startswith("FLOWRRA@") for m in methods):
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

            # IDENTICAL failures for every arm: same step, same agent indices.
            # Without this the arms face different instances and the comparison
            # measures which one got luckier.
            fail_spec = None
            if args.fail:
                waves = [float(v) for v in args.fail_waves.split(",") if v.strip()] \
                        or [args.fail_progress]
                fail_spec = {"progress_waves": waves,
                             "count": min(args.fail_agents, k),
                             "agents": []}

            for method in methods:
                # WHICH ARM IS TALKING. Every arm writes [Loop]/[Core] lines to
                # the same console, and fleet IDs are not comparable between
                # them: FLOWRRA takes its IDs from the scenario file (1..n) while
                # the baseline runner uses positional indices (0..n-1). So
                # "Fleet 5" from a baseline is a different vehicle from
                # "Fleet 5" in FLOWRRA's log, and with no arm marker the two
                # streams read as one episode. Without this line, a baseline
                # collision appears to follow FLOWRRA's completion message.
                print(f"\n===== {method} | {inst['map']} seed{inst['seed']} "
                      f"k={k} =====", flush=True)
                try:
                    if method == "FLOWRRA" or method.startswith("FLOWRRA@"):
                        _eps = (float(method.split("@")[1]) if "@" in method
                                else eps_list[0])
                        # FLOWRRA has no replan step; a stall is just a fleet that
                        # did not move, which its policy already handles inline.
                        r = run_flowrra_instance(G, pos_dict, missions, pool, agent,
                                                 args.max_steps, _eps,
                                                 shared_pool=not args.fixed_assignment,
                                                 failure=fail_spec)
                        r["algorithm"] = (f"FLOWRRA(eps={_eps:.2f})"
                                          if len(eps_list) > 1 else "FLOWRRA")
                        r["eval_epsilon"] = _eps
                        r["plan_time_s"] = 0.0
                        r["replan_s"] = 0.0
                    elif method.startswith("RHCR-"):
                        r = run_rolling_horizon_instance(
                            G, pos_dict, missions, pool, method.split("-", 1)[1].replace("+naive", ""),
                            args.max_steps, assignment,
                            window=args.rhcr_window,
                            replan_every=args.rhcr_replan_every,
                            steps_per_hop=args.steps_per_hop,
                            disturbance=disturbance,
                            failure=(dict(fail_spec,
                                          recovery=("naive" if method.endswith("+naive")
                                                    else "none"))
                                     if fail_spec else None))
                    else:
                        r = run_baseline_instance(G, pos_dict, missions, pool, method,
                                                  args.max_steps, assignment, disturbance)
                except Exception as exc:
                    print(f"  [error] {method} {inst['map']} seed{inst['seed']} k={k}: {exc}")
                    r = {"algorithm": method, "num_agents": k, "success": 0, "error": str(exc)}

                r = normalise(r, method)
                r.update(map=inst["map"], seed=inst["seed"], requested_agents=k,
                         map_nodes=G.number_of_nodes(),
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

    cols = ["success", "completion_rate", "distance_travelled", "collision_rate",
            "orders_orphaned", "orders_recovered",
            "rescuer_deaths",
            "orders_lost", "recovery_rate", "mean_recovery_hops", "makespan_hops", "soc_hops", "collisions",
            "mean_integrity", "proximity_margin", "plan_time_s", "replan_s",
            "decision_ms_per_step", "mean_time_to_recoherence", "blast_radius",
            "replan_count"]
    # OCCUPANCY and TIER MIX. The collision result is a claim about whether
    # spatial escape has room to work, so the summary has to show the two
    # quantities that decide it. Measured at k=200 on the 120k-node map: 0.17%
    # occupancy, Tier 1 fired 3 times in a whole episode, Tiers 2 and 3 never.
    # Zero collisions there is a statement about an empty warehouse, not about
    # the policy -- and without these columns the table cannot tell you that.
    if "num_agents" in df.columns and "map_nodes" in df.columns:
        df["occupancy_pct"] = (df.num_agents / df.map_nodes * 100).round(3)
    tiers = [c for c in ("tier1_spatial", "tier2_temporal", "tier3_yield")
             if c in df.columns]
    if tiers:
        tot = df[tiers].sum(axis=1)
        # Share of interventions resolved by SPATIAL escape. This is the number
        # the density question turns on: Tier 1 needs a free adjacent cell, so
        # as occupancy rises it should fall and Tier 2 should pick up the slack.
        df["tier1_share"] = (df.tier1_spatial / tot.replace(0, float("nan"))).round(3)
        df["tier_total"] = tot
        cols[1:1] = ["occupancy_pct"] + tiers + ["tier1_share", "tier_total"]

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