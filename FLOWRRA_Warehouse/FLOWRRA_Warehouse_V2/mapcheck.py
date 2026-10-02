"""
mapcheck.py -- how hard is a map, structurally, before any fleet moves?

    python mapcheck.py                        # every map in all_maps
    python mapcheck.py 25_5_2_5_2_1           # one
    python mapcheck.py 25_5_2_5_2_1 50_5_5_10_5_2
    python mapcheck.py --fleets 50 --out map_topology.csv

WHY. Occupancy is a bad proxy for difficulty. A 50-fleet run on 1,435 nodes is
3.5% occupied, which sounds mild -- and it produced a 2.03-cell mean peer gap,
74% brake duty and recovery firing on 779 of 780 steps. That is not what 3.5%
occupancy looks like on an open floor, so the floor is probably not open.

WHAT TO LOOK AT, in order of how much it explains a tangle:

  articulation_points   Cells whose removal DISCONNECTS the graph. A fleet
                        stopped on one does not merely obstruct, it severs the
                        map: everything behind it becomes unreachable, so the
                        BFS gradient for every fleet on the far side points
                        through a cell nobody can pass. Recovery cannot fix
                        this by relocating anyone one cell.

  vertical_edges        Links between z-levels. If goals are spread across
                        levels and only a handful of links exist, every
                        cross-level fleet funnels through the same few cells --
                        by construction, not by bad luck.

  deg1_pct              Dead ends. A fleet that enters one can only leave the
                        way it came, so two fleets in a dead end is an
                        unresolvable swap.

  bottleneck_ratio      fleets / articulation_points. Above ~1 there are more
                        fleets than severable cells, and contention for them is
                        guaranteed rather than probable.

  mean_degree           Route choice per cell. ~2 is corridor; the local
                        neighbourhood is a line, and there is nowhere to yield.

A map can be sparse in occupancy and still be a funnel. That is the case worth
detecting before blaming a policy for what the geometry made inevitable.
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import sys
from collections import Counter

import networkx as nx
import pandas as pd


def load(maps_dir: str, name: str):
    nodes = pd.read_csv(os.path.join(maps_dir, f"{name}_Nodes.csv"))
    edges = pd.read_csv(os.path.join(maps_dir, f"{name}_Edges.csv"))
    pos = {int(r.NodeId): (int(r.X), int(r.Y), int(r.Z)) for r in nodes.itertuples()}
    G = nx.Graph()
    G.add_nodes_from(pos)
    for r in edges.itertuples():
        a, b = int(r.nodeFrom), int(r.nodeTo)
        if a in pos and b in pos:
            G.add_edge(a, b)
    return G, pos


def analyse(name: str, G, pos, fleets: int) -> dict:
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    deg = dict(G.degree())
    dist = Counter(deg.values())

    vert = [(a, b) for a, b in G.edges() if pos[a][2] != pos[b][2]]
    levels = sorted({p[2] for p in pos.values()})
    vert_cells = {pos[a] for a, _ in vert} | {pos[b] for _, b in vert}

    # Articulation points and components are computed per component: a graph
    # that is already disconnected has goals that are simply unreachable, which
    # is a different and worse problem than a bottleneck.
    comps = list(nx.connected_components(G))
    arts = list(nx.articulation_points(G)) if n_nodes else []

    # Largest biconnected component: the biggest region with NO single point of
    # failure. If this is a small fraction of the map, most of the floor is
    # strung together by cut cells.
    bicomps = list(nx.biconnected_components(G)) if n_nodes else []
    largest_bi = max((len(b) for b in bicomps), default=0)

    return {
        "map": name,
        "nodes": n_nodes,
        "edges": n_edges,
        "mean_degree": round(2 * n_edges / n_nodes, 3) if n_nodes else 0,
        "deg1": dist.get(1, 0),
        "deg1_pct": round(100 * dist.get(1, 0) / n_nodes, 2) if n_nodes else 0,
        "deg2_pct": round(100 * dist.get(2, 0) / n_nodes, 2) if n_nodes else 0,
        "deg3plus_pct": round(
            100 * sum(v for k, v in dist.items() if k >= 3) / n_nodes, 2)
        if n_nodes else 0,
        "levels": len(levels),
        "vertical_edges": len(vert),
        "vertical_cells": len(vert_cells),
        "vertical_cells_pct": round(100 * len(vert_cells) / n_nodes, 2) if n_nodes else 0,
        "components": len(comps),
        "largest_component_pct": round(
            100 * max((len(c) for c in comps), default=0) / n_nodes, 2) if n_nodes else 0,
        "articulation_points": len(arts),
        "articulation_pct": round(100 * len(arts) / n_nodes, 2) if n_nodes else 0,
        "largest_biconnected_pct": round(100 * largest_bi / n_nodes, 2) if n_nodes else 0,
        "fleets": fleets,
        "occupancy_pct": round(100 * fleets / n_nodes, 2) if n_nodes else 0,
        "bottleneck_ratio": round(fleets / len(arts), 2) if arts else float("inf"),
    }


def verdict(r: dict) -> str:
    """One line on whether the geometry alone explains a tangle."""
    flags = []
    if r["components"] > 1:
        flags.append("DISCONNECTED -- some goals are unreachable")
    if r["articulation_pct"] > 20:
        flags.append("many cut cells")
    if r["bottleneck_ratio"] != float("inf") and r["bottleneck_ratio"] > 1:
        flags.append("more fleets than cut cells")
    if r["levels"] > 1 and r["vertical_cells_pct"] < 5:
        flags.append("cross-level traffic funnels through <5% of cells")
    if r["mean_degree"] < 2.3:
        flags.append("corridor-like, little room to yield")
    if r["deg1_pct"] > 5:
        flags.append("many dead ends")
    return "; ".join(flags) if flags else "no structural red flags"


def main():
    ap = argparse.ArgumentParser(
        description="Structural difficulty of a warehouse map.")
    ap.add_argument("maps", nargs="*", default=[],
                    help="map names; blank = every map in --maps-dir")
    ap.add_argument("--maps-dir", default="all_maps")
    ap.add_argument("--fleets", type=int, default=50,
                    help="fleet count to compute occupancy and bottleneck_ratio against")
    ap.add_argument("--out", default="map_topology.csv")
    args = ap.parse_args()

    names = args.maps
    if not names:
        names = sorted(
            os.path.basename(p)[: -len("_Nodes.csv")]
            for p in glob.glob(os.path.join(args.maps_dir, "*_Nodes.csv")))
    if not names:
        sys.exit(f"no maps found in {args.maps_dir}")

    rows = []
    for name in names:
        try:
            G, pos = load(args.maps_dir, name)
        except FileNotFoundError as exc:
            print(f"  skip {name}: {exc}")
            continue
        r = analyse(name, G, pos, args.fleets)
        r["verdict"] = verdict(r)
        rows.append(r)

        print(f"\n{name}")
        print(f"   {r['nodes']} nodes, {r['edges']} edges, "
              f"mean degree {r['mean_degree']}")
        print(f"   degree mix: {r['deg1_pct']}% dead-end, {r['deg2_pct']}% corridor, "
              f"{r['deg3plus_pct']}% junction")
        print(f"   {r['levels']} level(s), {r['vertical_edges']} vertical edges "
              f"through {r['vertical_cells']} cells ({r['vertical_cells_pct']}%)")
        print(f"   articulation points: {r['articulation_points']} "
              f"({r['articulation_pct']}% of cells)")
        print(f"   largest biconnected region: {r['largest_biconnected_pct']}% of the map")
        print(f"   at {r['fleets']} fleets: {r['occupancy_pct']}% occupancy, "
              f"bottleneck ratio {r['bottleneck_ratio']}")
        print(f"   -> {r['verdict']}")

    if rows:
        with open(args.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {args.out} ({len(rows)} map(s))")

        rank = sorted(rows, key=lambda r: (-r["articulation_pct"], r["mean_degree"]))
        print(f"\nhardest first, by share of cells that sever the map:")
        print(f"   {'map':<26}{'nodes':>8}{'art%':>8}{'deg':>7}{'occ%':>7}")
        for r in rank[:12]:
            print(f"   {r['map']:<26}{r['nodes']:>8}{r['articulation_pct']:>8}"
                  f"{r['mean_degree']:>7}{r['occupancy_pct']:>7}")


if __name__ == "__main__":
    main()


"""
"
import pandas as pd, networkx as nx
from collections import Counter
n=pd.read_csv('all_maps/25_5_2_5_2_1_Nodes.csv')
e=pd.read_csv('all_maps/25_5_2_5_2_1_Edges.csv')
pos={int(r.NodeId):(int(r.X),int(r.Y),int(r.Z)) for r in n.itertuples()}
G=nx.Graph(); G.add_nodes_from(pos)
for r in e.itertuples(): G.add_edge(int(r.nodeFrom),int(r.nodeTo))
s=pd.read_csv('all_scens_v3/25_5_2_5_2_1/25_5_2_5_2_1_StartGoalLocations_Seed8.csv')
vert=Counter(); tot=vc=0
for r in s.head(50).itertuples():
    try: p=nx.shortest_path(G,int(r.startNodeId),int(r.goalNodeId))
    except Exception: continue
    for a,b in zip(p,p[1:]):
        tot+=1
        if pos[a][2]!=pos[b][2]:
            vc+=1; vert[(pos[a][0],pos[a][1])]+=1
print(f'path hops that are vertical: {100*vc/max(1,tot):.0f}%')
print(f'distinct shafts used by 50 fleets: {len(vert)}')
print('busiest shafts (x,y): traversals', vert.most_common(6))

"
"""