"""
validate_instance.py

Diagnostic for the 3D MAPF benchmark instances.

WHY THIS EXISTS
A full 450-run sweep reported soc_lower_bound = 0.0 on EVERY instance, a median
steps_run of 4, and completion_rate 0.999. Those are not results -- they say
every fleet believed it was already standing on its goal. Any benchmark number
computed on top of that is meaningless, so this script finds out where the
distance signal dies before another sweep is run.

It checks, in order:

  1. Graph sanity        -- nodes, edges, connected components.
  2. COORDINATE COLLISIONS -- the prime suspect. FleetNode maps a POSITION back
     to a node id via grid_pos_dict[(x, y, z)]. That dict is keyed by integer
     coordinates, so if two distinct nodes share one (X, Y, Z) the later one
     silently wins and a fleet standing on the first resolves to the second.
     If the survivor happens to sit near the goal, get_graph_distance_to_goal()
     returns ~0 and the fleet "arrives" without moving. A 3D warehouse with
     elevator shafts is exactly the topology where duplicate coordinates are
     plausible.
  3. Ground-truth distances -- networkx shortest_path_length from each start to
     its own scenario goal. This is the number the benchmark intends.
  4. FLOWRRA's view      -- initial_graph_distance after Hungarian assignment.
     Comparing 3 against 4 localises the fault: if networkx says 30 hops and
     FLOWRRA says 0, the loss is in the position->node lookup, not the data.

Usage:
    python validate_instance.py --map 50_5_5_10_5_10 --seed 0 --agents 25
"""

import argparse
import io
import contextlib
from collections import Counter, defaultdict

import numpy as np
import networkx as nx

from benchmark_flowrra import load_instance
from node_warehouse import precompute_goal_distances
from core_warehouse import FLOWRRA


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--maps-dir", default="all_maps")
    ap.add_argument("--scens-dir", default="all_scens")
    ap.add_argument("--map", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--agents", type=int, default=25)
    args = ap.parse_args()

    nodes_csv = f"{args.maps_dir}/{args.map}_Nodes.csv"
    edges_csv = f"{args.maps_dir}/{args.map}_Edges.csv"
    scen_csv = (f"{args.scens_dir}/{args.map}/"
                f"{args.map}_StartGoalLocations_Seed{args.seed}.csv")

    import pandas as pd
    raw_scen = pd.read_csv(scen_csv, index_col=False)
    print("=" * 78)
    print(f"INSTANCE  {args.map}  seed {args.seed}  k={args.agents}")
    print("=" * 78)
    print(f"scenario columns : {list(raw_scen.columns)}")
    print(f"scenario rows    : {len(raw_scen)}")
    print(raw_scen.head(3).to_string(index=False))

    G, pos_dict, missions, pool = load_instance(nodes_csv, edges_csv, scen_csv, args.agents)

    # ---- 1. graph sanity ---------------------------------------------------
    comps = list(nx.connected_components(G))
    comps.sort(key=len, reverse=True)
    print(f"\n1. GRAPH   nodes={G.number_of_nodes()}  edges={G.number_of_edges()}  "
          f"components={len(comps)}  largest={len(comps[0])}")
    if len(comps) > 1:
        print(f"   NOTE: {len(comps)-1} extra component(s); sizes "
              f"{[len(c) for c in comps[1:6]]}. Cross-component start/goal pairs "
              f"are unreachable and will read as infinite or fall back to Manhattan.")

    # ---- 2. coordinate collisions -----------------------------------------
    coord_to_nodes = defaultdict(list)
    for nid, p in pos_dict.items():
        coord_to_nodes[(int(round(p["X"])), int(round(p["Y"])), int(round(p["Z"])))].append(nid)
    dupes = {c: ns for c, ns in coord_to_nodes.items() if len(ns) > 1}
    lost = sum(len(ns) - 1 for ns in dupes.values())
    print(f"\n2. COORDINATE COLLISIONS   distinct (X,Y,Z) = {len(coord_to_nodes)} "
          f"for {len(pos_dict)} nodes")
    if dupes:
        print(f"   *** {len(dupes)} coordinates hold >1 node; {lost} nodes "
              f"({lost/len(pos_dict)*100:.1f}%) are UNREACHABLE via grid_pos_dict ***")
        for c, ns in list(dupes.items())[:5]:
            print(f"     {c} -> {ns[:6]}")
        print("   This is sufficient on its own to produce distance 0: a fleet")
        print("   standing on one node resolves to whichever node won the dict.")
    else:
        print("   none - every node has a unique integer coordinate.")

    # ---- 3. ground truth ---------------------------------------------------
    scen = raw_scen.copy()
    scen["startNodeId"] = scen["startNodeId"].astype(str).str.strip()
    scen["goalNodeId"] = scen["goalNodeId"].astype(str).str.strip()
    scen = scen.iloc[:args.agents]

    true_d, unreachable, same = [], 0, 0
    for _, r in scen.iterrows():
        s, g = r["startNodeId"], r["goalNodeId"]
        if s == g:
            same += 1
        if s in G and g in G:
            try:
                true_d.append(nx.shortest_path_length(G, s, g))
            except nx.NetworkXNoPath:
                unreachable += 1
        else:
            unreachable += 1
    print(f"\n3. GROUND TRUTH (networkx, start -> its OWN scenario goal)")
    print(f"   pairs with start == goal : {same}")
    print(f"   unreachable pairs        : {unreachable}")
    if true_d:
        print(f"   hops: mean {np.mean(true_d):.1f}  min {min(true_d)}  max {max(true_d)}  "
              f"sum {sum(true_d)}")
        print(f"   -> a correct soc_lower_bound for k={args.agents} is about "
              f"{sum(true_d)/0.5:.0f} simulator steps at base_speed 0.5")

    # ---- 4. what FLOWRRA actually sees ------------------------------------
    gdm = precompute_goal_distances(G, [{"goal_node": gid} for gid in pool])
    with contextlib.redirect_stdout(io.StringIO()):
        env = FLOWRRA(G, pos_dict, missions, mode="eval", goal_distance_maps=gdm,
                      shared_pool_mode=True, goal_pool=pool)
    starts = {m["id"]: m["start_node"] for m in missions}
    flow_d = [n.initial_graph_distance for n in env.nodes]
    print(f"\n4. FLOWRRA'S VIEW (after Hungarian assignment)")
    print(f"   sum initial_graph_distance = {sum(flow_d):.1f}   "
          f"(zero here is the bug the sweep reported)")
    print(f"   fleets assigned            = {sum(1 for n in env.nodes if n.current_goal_id)}"
          f"/{len(env.nodes)}")
    print("\n   per-fleet (first 8):")
    print(f"   {'fleet':<7}{'start':<12}{'assigned goal':<15}{'map?':<7}"
          f"{'BFS(start)':<12}{'initial_dist'}")
    for n in env.nodes[:8]:
        gm = n.goal_distance_map
        s = starts[n.id]
        # what the position->node lookup actually resolves to
        resolved = env.grid_pos_dict.get(tuple(int(round(v)) for v in n.current_pos))
        flag = "" if resolved == s else f"  <-- resolves to {resolved}, not {s}!"
        print(f"   {n.id:<7}{s:<12}{str(n.current_goal_id):<15}"
              f"{'yes' if gm else 'NONE':<7}"
              f"{str(gm.get(s)) if gm else '-':<12}{n.initial_graph_distance}{flag}")

    print("\n" + "=" * 78)
    if sum(flow_d) == 0 and true_d and sum(true_d) > 0:
        print("VERDICT: the data is fine (networkx finds real distances) but FLOWRRA")
        print("         reads zero. The loss is in the position -> node id lookup.")
        if dupes:
            print("         Coordinate collisions above are the cause.")
    elif same > args.agents * 0.5:
        print("VERDICT: the SCENARIO file pairs each agent's start with its own start.")
        print("         Check the column meanings before anything else.")
    else:
        print("VERDICT: distances look sane; the earlier sweep may have used a")
        print("         different loader path. Re-run the sweep and re-check.")
    print("=" * 78)


if __name__ == "__main__":
    main()