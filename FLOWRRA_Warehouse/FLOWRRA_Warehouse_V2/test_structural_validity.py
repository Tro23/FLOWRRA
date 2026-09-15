"""
test_structural_validity.py

is_structurally_valid() is the single hottest function in FLOWRRA: the
2026-09-13 profile at 200 fleets / 120k nodes put it at 68.5% of cumulative step
time. Two changes were made to it, and both are the kind that are easy to get
subtly wrong and hard to notice, because a wrong answer here does not crash --
it silently lets a fleet drive through a shelf, or silently refuses a legal move.

  1. The redundant per-call re-sort and list copy were removed. The aisle line
     is already sorted by build_spatial_indices() and is never mutated.
  2. The O(n) bounding-node scan became two O(log n) bisects.

So this file does ONE thing: reimplement the ORIGINAL logic verbatim, and assert
the new implementation agrees with it on every input, exhaustively, including
every awkward case around the 0.1 tolerance band.
"""

import itertools
import numpy as np
import networkx as nx

from node_warehouse import FleetNode, build_spatial_indices

FAIL = []


def check(name, got, want):
    ok = got == want
    print(f"{'PASS' if ok else 'FAIL'}  {name}: got {got!r} want {want!r}")
    if not ok:
        FAIL.append(name)


def original_bounds(aligned_nodes, p_val):
    """The pre-2026-09-13 scan, verbatim, as the reference implementation."""
    line = sorted(aligned_nodes, key=lambda x: x[0])
    node_behind = None
    node_ahead = None
    for val, nid in line:
        if val <= p_val + 0.1:
            node_behind = nid
        if val >= p_val - 0.1 and node_ahead is None:
            node_ahead = nid
    return node_behind, node_ahead


def new_bounds(aligned_nodes, p_val):
    """What the rewritten function now computes."""
    from bisect import bisect_left, bisect_right
    from operator import itemgetter
    first = itemgetter(0)
    line = sorted(aligned_nodes, key=first)
    i_b = bisect_right(line, p_val + 0.1, key=first) - 1
    i_a = bisect_left(line, p_val - 0.1, key=first)
    behind = line[i_b][1] if i_b >= 0 else None
    ahead = line[i_a][1] if i_a < len(line) else None
    return behind, ahead


def test_bounds_equivalence_exhaustive():
    """
    Every gap pattern over an 8-cell aisle, crossed with positions on a fine
    grid that deliberately straddles the 0.1 tolerance: exactly on a node, just
    inside the band, exactly on the band edge, just outside, and mid-edge.
    """
    positions = []
    for base in range(-1, 9):
        for off in (-0.2, -0.11, -0.1, -0.09, 0.0, 0.09, 0.1, 0.11, 0.2,
                    0.25, 0.5, 0.75):
            positions.append(round(base + off, 4))

    mismatches = 0
    cases = 0
    # All subsets of {0..7} of size >= 1, as gap patterns.
    for r in range(1, 9):
        for present in itertools.combinations(range(8), r):
            line = [(v, f"n{v}") for v in present]
            for p in positions:
                cases += 1
                if original_bounds(line, p) != new_bounds(line, p):
                    mismatches += 1
                    if mismatches <= 5:
                        print(f"      mismatch: line={present} p={p} "
                              f"old={original_bounds(line, p)} new={new_bounds(line, p)}")
    print(f"      ({cases:,} cases checked)")
    check("bounds_equivalence", mismatches, 0)


def build_env():
    """Racked warehouse: aisles at even y, cross-aisles at x=0 and x=9."""
    G = nx.Graph()
    grid = {}
    for y in range(0, 10, 2):
        for x in range(10):
            grid[(x, y, 0)] = f"n_{x}_{y}"
            G.add_node(f"n_{x}_{y}")
        for x in range(9):
            G.add_edge(f"n_{x}_{y}", f"n_{x+1}_{y}")
    for x in (0, 9):
        for y in range(10):
            if (x, y, 0) not in grid:
                grid[(x, y, 0)] = f"n_{x}_{y}"
                G.add_node(f"n_{x}_{y}")
        for y in range(9):
            G.add_edge(f"n_{x}_{y}", f"n_{x}_{y+1}")
    aisle, coords = build_spatial_indices(grid)
    return G, grid, aisle


def test_index_is_already_sorted():
    """
    The premise of removing the per-call sort. If build_spatial_indices ever
    stops sorting, the bisects become silently wrong rather than slow, so this
    has to be asserted, not assumed.
    """
    G, grid, aisle = build_env()
    unsorted = 0
    for axis in (0, 1, 2):
        for line in aisle[axis].values():
            vals = [v for v, _ in line]
            if vals != sorted(vals):
                unsorted += 1
    check("index_presorted", unsorted, 0)


def test_real_moves_on_a_racked_map():
    """End-to-end through the real function on a real graph."""
    G, grid, aisle = build_env()

    def node(pos):
        n = FleetNode(id="a", current_pos=np.array(pos, dtype=np.float64),
                      goal_pos=np.array([0.0, 0.0, 0.0]),
                      grid_pos_dict=grid, aisle_index=aisle)
        n.G = G
        return n

    # Along a clear aisle: legal.
    check("along_aisle", node((3.0, 0.0, 0.0)).is_structurally_valid(
        np.array([3.5, 0.0, 0.0]), 1), True)
    # Into the rack from mid-aisle: illegal.
    check("into_rack", node((3.0, 0.0, 0.0)).is_structurally_valid(
        np.array([3.0, 0.5, 0.0]), 3), False)
    # Up the cross-aisle at x=0: legal.
    check("up_cross_aisle", node((0.0, 0.0, 0.0)).is_structurally_valid(
        np.array([0.0, 0.5, 0.0]), 3), True)
    # Off the end of the map: illegal.
    check("off_map_end", node((9.0, 0.0, 0.0)).is_structurally_valid(
        np.array([9.5, 0.0, 0.0]), 1), False)
    # Idle is always valid.
    check("idle_always_valid", node((3.0, 0.0, 0.0)).is_structurally_valid(
        np.array([3.0, 0.0, 0.0]), 0), True)


def test_full_function_matches_a_brute_force_reference():
    """
    Sweep every fleet position and every action against a brute-force checker
    built from the graph itself, with no index and no bisect.
    """
    G, grid, aisle = build_env()
    deltas = {1: (1, 0, 0), 2: (-1, 0, 0), 3: (0, 1, 0),
              4: (0, -1, 0), 5: (0, 0, 1), 6: (0, 0, -1)}

    def brute(proposed, action):
        """Original algorithm, no index: full grid scan + linear bounds."""
        move_axis = 0 if action in (1, 2) else 1 if action in (3, 4) else 2
        ax1, ax2 = [i for i in (0, 1, 2) if i != move_axis]
        aligned = [(c[move_axis], nid) for c, nid in grid.items()
                   if abs(c[ax1] - proposed[ax1]) < 0.1
                   and abs(c[ax2] - proposed[ax2]) < 0.1]
        if not aligned:
            return False
        b, a = original_bounds(aligned, proposed[move_axis])
        if b is None or a is None:
            return False
        if b != a and not G.has_edge(b, a):
            return False
        return True

    mismatches = 0
    cases = 0
    for (x, y, z) in sorted(grid.keys()):
        for frac in (0.0, 0.5):
            for action, d in deltas.items():
                pos = np.array([x, y, z], dtype=np.float64)
                pos[0 if d[0] else (1 if d[1] else 2)] += frac * (d[0] or d[1] or d[2])
                n = FleetNode(id="a", current_pos=pos,
                              goal_pos=np.array([0.0, 0.0, 0.0]),
                              grid_pos_dict=grid, aisle_index=aisle)
                n.G = G
                proposed = pos + np.array(d, dtype=np.float64) * 0.5
                cases += 1
                if n.is_structurally_valid(proposed, action) != brute(proposed, action):
                    mismatches += 1
                    if mismatches <= 5:
                        print(f"      mismatch at pos={pos} action={action}")
    print(f"      ({cases:,} cases checked)")
    check("full_function_equivalence", mismatches, 0)


def test_index_not_mutated():
    """
    Removing list() means the function now holds a REFERENCE to the shared
    index. If anything ever sorts or appends to it in place, every fleet on the
    map is affected. Assert the function leaves it untouched.
    """
    G, grid, aisle = build_env()
    before = {ax: {k: list(v) for k, v in aisle[ax].items()} for ax in (0, 1, 2)}
    n = FleetNode(id="a", current_pos=np.array([3.0, 0.0, 0.0]),
                  goal_pos=np.array([0.0, 0.0, 0.0]),
                  grid_pos_dict=grid, aisle_index=aisle)
    n.G = G
    for action in range(7):
        for frac in (0.0, 0.25, 0.5, 0.75):
            n.is_structurally_valid(np.array([3.0 + frac, 0.0, 0.0]), action)
    same = all(before[ax][k] == list(v)
               for ax in (0, 1, 2) for k, v in aisle[ax].items())
    check("index_untouched", same, True)


if __name__ == "__main__":
    for fn in list(globals().values()):
        if callable(fn) and getattr(fn, "__name__", "").startswith("test_"):
            print(f"\n--- {fn.__name__} ---")
            fn()
    print("\n" + ("ALL PASS" if not FAIL else f"FAILURES: {FAIL}"))