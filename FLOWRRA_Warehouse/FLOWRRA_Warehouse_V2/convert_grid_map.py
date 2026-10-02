"""
convert_grid_map.py -- a published 2D grid map into FLOWRRA's format, exactly as
published or stacked into floors joined by lift shafts.

    # kiva exactly as RHCR publishes it (one level): the published comparison
    python convert_grid_map.py kiva.map --name kiva

    # the extended benchmark: 3 floors, 10 single-lane shafts, 5 cells between floors
    python convert_grid_map.py kiva.map --name kiva_3f --floors 3 --shafts 10

Reads both formats: RHCR's (first line "rows,cols", then #endpoints, #agents,
maxtime, then the grid) and MovingAI's ("type octile / height / width / map").
Cells: '@', 'T', 'O' are obstacles; everything else is a cell. RHCR's 'e'
(endpoints, beside the shelves) become the goal bank; 'r' (robot homes) are
where fleets start. A MovingAI map has neither, so every cell is both.

Writes, in FLOWRRA's layout:
    <maps-dir>/<name>_Nodes.csv      NodeId,X,Y,Z
    <maps-dir>/<name>_Edges.csv      nodeFrom,nodeTo,bidirectional
    <scens-dir>/<name>/<name>_GoalBank.csv
    <scens-dir>/<name>/<name>_StartGoalLocations_Seed<k>.csv

THE SINGLE-LEVEL CONVERSION IS EXACT: same cells, same 4-connected moves, X =
column, Y = row counted from the bottom, Z = 0. Nothing is added or removed, or
the published RHCR and Follower numbers stop applying.

STACKING (--floors > 1) is ours, not published: copies of the grid as floors,
(shaft_len + 1) apart in Z like the 50_ map, joined by --shafts single-lane lift
shafts of --shaft-len cells, placed evenly around the outer ring of open cells
so they sit where aisles meet the edge, as on 50_.

THE SCENARIOS ARE PLACEHOLDERS for the one-shot harness: starts at distinct
homes, goals at random endpoints. The lifelong comparison with RHCR needs its
own task generation ("next goal on arrival", matched to RHCR's code) -- that is
the next step, not this file.
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Tuple

import numpy as np

OBSTACLES = set("@TO")


def read_grid(path: str) -> Tuple[List[str], str]:
    lines = [l.rstrip("\r\n") for l in open(path)]
    if lines[0].startswith("type"):                       # MovingAI
        i = next(k for k, l in enumerate(lines) if l.strip() == "map") + 1
        return lines[i:], "movingai"
    rows, cols = (int(v) for v in lines[0].split(","))    # RHCR
    grid = lines[4:4 + rows]
    assert all(len(r) == cols for r in grid), "RHCR header does not match the grid"
    return grid, "rhcr"


def ring(grid: List[str]) -> List[Tuple[int, int]]:
    """Open cells on the outermost open ring, clockwise from the top-left."""
    R, C = len(grid), len(grid[0])
    top = [(0, c) for c in range(C)]
    right = [(r, C - 1) for r in range(1, R)]
    bottom = [(R - 1, c) for c in range(C - 2, -1, -1)]
    left = [(r, 0) for r in range(R - 2, 0, -1)]
    return [(r, c) for r, c in top + right + bottom + left if grid[r][c] not in OBSTACLES]


def build(grid: List[str], floors: int, shafts: int, shaft_len: int):
    R, C = len(grid), len(grid[0])
    gap = shaft_len + 1
    ids: Dict[Tuple[int, int, int], int] = {}
    kind: Dict[int, str] = {}
    for f in range(floors):
        for r in range(R):
            for c in range(C):
                ch = grid[r][c]
                if ch in OBSTACLES:
                    continue
                nid = len(ids)
                ids[(c, R - 1 - r, f * gap)] = nid
                kind[nid] = ch
    edges = set()
    for (x, y, z), a in ids.items():
        for dx, dy in ((1, 0), (0, 1)):
            b = ids.get((x + dx, y + dy, z))
            if b is not None:
                edges.add((a, b))
    shaft_at: List[Tuple[int, int]] = []
    if floors > 1 and shafts > 0:
        rg = ring(grid)
        picks = np.linspace(0, len(rg), shafts, endpoint=False).astype(int)
        shaft_at = [(c, R - 1 - r) for r, c in (rg[i] for i in picks)]
        for f in range(floors - 1):
            for (x, y) in shaft_at:
                prev = ids[(x, y, f * gap)]
                for k in range(1, gap):
                    nid = len(ids)
                    ids[(x, y, f * gap + k)] = nid
                    kind[nid] = "s"
                    edges.add((prev, nid))
                    prev = nid
                edges.add((prev, ids[(x, y, (f + 1) * gap)]))
    return ids, kind, sorted(edges), shaft_at


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("map_file")
    ap.add_argument("--name", required=True)
    ap.add_argument("--floors", type=int, default=1)
    ap.add_argument("--shafts", type=int, default=10, help="shafts between each pair of floors")
    ap.add_argument("--shaft-len", type=int, default=5, help="cells inside a shaft between floors")
    ap.add_argument("--agents", type=int, default=100)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--maps-dir", default="all_maps")
    ap.add_argument("--scens-dir", default="all_scens_v4")
    args = ap.parse_args()

    grid, fmt = read_grid(args.map_file)
    ids, kind, edges, shaft_at = build(grid, args.floors, args.shafts, args.shaft_len)
    os.makedirs(args.maps_dir, exist_ok=True)
    sdir = os.path.join(args.scens_dir, args.name)
    os.makedirs(sdir, exist_ok=True)

    with open(os.path.join(args.maps_dir, f"{args.name}_Nodes.csv"), "w") as fh:
        fh.write("NodeId,X,Y,Z\n")
        for (x, y, z), nid in sorted(ids.items(), key=lambda kv: kv[1]):
            fh.write(f"{nid},{x},{y},{z}\n")
    with open(os.path.join(args.maps_dir, f"{args.name}_Edges.csv"), "w") as fh:
        fh.write("nodeFrom,nodeTo,bidirectional\n")
        for a, b in edges:
            fh.write(f"{a},{b},true\n")

    endpoints = [n for n, k in kind.items() if k == "e"] or [n for n, k in kind.items() if k != "s"]
    homes = [n for n, k in kind.items() if k == "r"] or [n for n, k in kind.items() if k != "s"]
    with open(os.path.join(sdir, f"{args.name}_GoalBank.csv"), "w") as fh:
        fh.write("goalNodeId\n" + "".join(f"{n}\n" for n in endpoints))
    for seed in range(args.seeds):
        rng = np.random.default_rng(seed)
        k = min(args.agents, len(homes))
        starts = rng.choice(homes, k, replace=False)
        goals = rng.choice(endpoints, k, replace=len(endpoints) < k)
        with open(os.path.join(sdir, f"{args.name}_StartGoalLocations_Seed{seed}.csv"), "w") as fh:
            fh.write("agentId,startNodeId,goalNodeId\n")
            for i, (s, g) in enumerate(zip(starts, goals), 1):
                fh.write(f"{i},{s},{g}\n")

    n_shaft = sum(1 for k in kind.values() if k == "s")
    print(f"{args.name}: {fmt} grid {len(grid)}x{len(grid[0])}, {args.floors} floor(s), "
          f"{len(ids)} cells ({n_shaft} in shafts), {len(edges)} edges; "
          f"{len(endpoints)} endpoints (goal bank), {len(homes)} homes; "
          f"{args.seeds} scenario(s) of {min(args.agents, len(homes))} fleets"
          + (f"; shafts at {shaft_at}" if shaft_at else ""))


if __name__ == "__main__":
    main()
