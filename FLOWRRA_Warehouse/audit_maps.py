"""
audit_maps.py

Per-map sanity report to run BEFORE committing to a long curriculum.

Answers three questions the training log will not tell you until it is too late:

  1. How much of each axis did the OLD hardcoded (50,50,10) bounds destroy?
     Useful for the write-up: it quantifies, per map, how much of the
     goal-direction feature was saturated before the per-map fix.

  2. How long are the journeys in the generated scenarios, in hops?

  3. Is max_steps_per_episode actually big enough for them? At base_speed 0.5 a
     fleet covers one hop per two simulator steps AT BEST -- before any
     congestion, detours, yields or affordance braking. If the step budget is
     below the straight-line requirement, large-map episodes time out by
     construction and the curriculum feeds the policy nothing but failure from
     exactly the maps the scaling argument depends on.

Usage:
    python audit_maps.py --maps-dir all_maps --scens-dir all_scens_fixed
"""

import argparse
import os
import random

import numpy as np
import pandas as pd
import networkx as nx

try:
    from config_warehouse import CONFIG
    MAX_STEPS = CONFIG["training"]["max_steps_per_episode"]
    BASE_SPEED = CONFIG["warehouse"]["base_speed"]
    OLD_BOUNDS = (50.0, 50.0, 10.0)
except Exception:
    MAX_STEPS, BASE_SPEED, OLD_BOUNDS = 780, 0.5, (50.0, 50.0, 10.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--maps-dir", default="all_maps")
    ap.add_argument("--scens-dir", default="all_scens_fixed")
    ap.add_argument("--sample", type=int, default=40,
                    help="start/goal pairs to measure per map")
    ap.add_argument("--max-steps", type=int, default=MAX_STEPS)
    args = ap.parse_args()

    names = sorted({f[:-len("_Nodes.csv")]
                    for f in os.listdir(args.maps_dir) if f.endswith("_Nodes.csv")})

    print(f"\nstep budget {args.max_steps} | base_speed {BASE_SPEED} "
          f"-> {args.max_steps * BASE_SPEED:.0f} hops of straight-line travel per fleet\n")
    header = (f"{'map':<22}{'nodes':>8}{'spanX':>7}{'spanY':>7}{'spanZ':>7}"
              f"{'satY':>7}{'satZ':>7}{'hops p50':>10}{'hops p95':>10}{'budget':>9}")
    print(header)
    print("-" * len(header))

    flagged = []
    for name in names:
        nodes_csv = os.path.join(args.maps_dir, f"{name}_Nodes.csv")
        edges_csv = os.path.join(args.maps_dir, f"{name}_Edges.csv")
        if not os.path.exists(edges_csv):
            continue

        nd = pd.read_csv(nodes_csv, index_col=False)
        nd["NodeId"] = nd["NodeId"].astype(str).str.strip()
        spans = [float(nd[c].max() - nd[c].min()) for c in ("X", "Y", "Z")]

        # fraction of each axis that the old hardcoded bounds clipped to 1.0
        sat = [1.0 - min(o / s, 1.0) if s > 0 else 0.0
               for s, o in zip(spans, OLD_BOUNDS)]

        ed = pd.read_csv(edges_csv, index_col=False)
        ed["nodeFrom"] = ed["nodeFrom"].astype(str).str.strip()
        ed["nodeTo"] = ed["nodeTo"].astype(str).str.strip()
        G = nx.Graph()
        G.add_nodes_from(nd["NodeId"])
        G.add_edges_from(zip(ed["nodeFrom"], ed["nodeTo"]))

        scen_dir = os.path.join(args.scens_dir, name)
        hops = []
        if os.path.isdir(scen_dir):
            files = [f for f in os.listdir(scen_dir) if "StartGoalLocations" in f]
            rng = random.Random(0)
            for f in rng.sample(files, min(3, len(files))):
                df = pd.read_csv(os.path.join(scen_dir, f), index_col=False)
                df["startNodeId"] = df["startNodeId"].astype(str).str.strip()
                df["goalNodeId"] = df["goalNodeId"].astype(str).str.strip()
                for _, r in df.head(args.sample // 3 + 1).iterrows():
                    s, g = r["startNodeId"], r["goalNodeId"]
                    if s in G and g in G:
                        try:
                            hops.append(nx.shortest_path_length(G, s, g))
                        except nx.NetworkXNoPath:
                            pass

        p50 = float(np.percentile(hops, 50)) if hops else float("nan")
        p95 = float(np.percentile(hops, 95)) if hops else float("nan")
        reach = args.max_steps * BASE_SPEED
        # ratio < ~2 means even an unobstructed fleet has little slack
        ratio = reach / p95 if hops and p95 > 0 else float("inf")
        verdict = "OK" if ratio >= 2.5 else ("TIGHT" if ratio >= 1.2 else "TOO SHORT")
        if verdict != "OK":
            flagged.append((name, p95, ratio, verdict))

        print(f"{name:<22}{G.number_of_nodes():>8}"
              f"{spans[0]:>7.0f}{spans[1]:>7.0f}{spans[2]:>7.0f}"
              f"{sat[1]*100:>6.0f}%{sat[2]*100:>6.0f}%"
              f"{p50:>10.0f}{p95:>10.0f}{verdict:>9}")

    print()
    if flagged:
        print("ATTENTION -- step budget vs journey length:")
        for name, p95, ratio, verdict in flagged:
            print(f"  {name}: p95 journey {p95:.0f} hops needs "
                  f"{p95/BASE_SPEED:.0f} steps of PERFECT straight-line travel; "
                  f"budget gives {ratio:.1f}x slack ({verdict}).")
        print("\n  A fleet never travels straight -- it detours around racks, yields,")
        print("  and brakes near peers. Below ~2.5x slack, episodes on these maps")
        print("  time out by construction, and the curriculum then trains mostly on")
        print("  failure from exactly the large maps the scaling argument needs.")
        print("  Either raise --max-steps for large maps or lower --min-hops-range")
        print("  when generating their scenarios.")
    else:
        print("Step budget looks adequate on every map.")

    print("\nsatY/satZ = fraction of that axis the OLD hardcoded (50,50,10) bounds")
    print("clipped to exactly 1.0. High values mean the goal-direction feature on")
    print("that axis was a sign bit with no magnitude, before the per-map fix.\n")


if __name__ == "__main__":
    main()