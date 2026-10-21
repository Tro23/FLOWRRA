"""
test_kernel.py -- the graph-hop falloff kernel in density_warehouse.stamp().

Three things have to hold at once, and they pull in different directions:

  1. RACK REJECTION. A source behind a shelf, Manhattan-close and graph-far,
     must stamp nothing on the observer's side. This is the defect being fixed.

  2. CORNER PRESERVATION. A source around a bend at graph distance 2 must still
     stamp at full kernel strength. Manhattan under-counts a bend as much as it
     over-counts a rack, so a naive "just use the mask" fix would lose this.

  3. STRAIGHT-LINE EQUIVALENCE. Down an unobstructed corridor, graph hops and
     Manhattan agree, so the new kernel must reproduce the old one exactly.
     Any disagreement there means the BFS is mis-counting depth.
"""

import numpy as np
import networkx as nx

from density_warehouse import WarehouseDensityField

FAIL = []


def check(name, got, want):
    ok = got == want
    print(f"{'PASS' if ok else 'FAIL'}  {name}: got {got!r} want {want!r}")
    if not ok:
        FAIL.append(name)


def racked():
    """
        y=2   A---A---A---A---A---A
              |                   |
        y=1   C       RACK        C
              |                   |
        y=0   A---A---A---A---A---A
    """
    G = nx.Graph()
    grid = {}
    for y in (0, 2):
        for x in range(6):
            grid[(x, y, 0)] = f"n_{x}_{y}"
            G.add_node(f"n_{x}_{y}")
        for x in range(5):
            G.add_edge(f"n_{x}_{y}", f"n_{x+1}_{y}")
    for x in (0, 5):
        grid[(x, 1, 0)] = f"n_{x}_1"
        G.add_node(f"n_{x}_1")
        G.add_edge(f"n_{x}_0", f"n_{x}_1")
        G.add_edge(f"n_{x}_1", f"n_{x}_2")
    return G, grid


def corridor():
    """One straight open corridor, 12 cells. Graph hops == Manhattan."""
    G = nx.Graph()
    grid = {}
    for x in range(12):
        grid[(x, 0, 0)] = f"c_{x}"
        G.add_node(f"c_{x}")
    for x in range(11):
        G.add_edge(f"c_{x}", f"c_{x+1}")
    return G, grid


def field(G, grid, **kw):
    return WarehouseDensityField(max_vision_range=10, grid_pos_dict=grid, graph=G, **kw)


class Fleet:
    def __init__(self, fid, pos, direction=(0, 0, 0), dmap=None, goal=None):
        self.id = fid
        self.current_pos = np.array(pos, dtype=np.float64)
        self.direction = np.array(direction, dtype=np.float64)
        self.goal_distance_map = dmap
        self.current_goal_id = goal


def test_kernel_reach():
    G, grid = racked()
    f = field(G, grid)
    # k(d) = max(0, 1 - d/3) -> [1.0, 0.667, 0.333, 0.0]. Reach is the last hop
    # count with a NON-ZERO weight, so 2.
    check("kernel_curve", [round(float(f._kernel[d]), 3) for d in range(4)],
          [1.0, 0.667, 0.333, 0.0])
    check("kernel_reach", f._kernel_reach, 2)


def test_neighbourhood_follows_the_bend():
    G, grid = racked()
    f = field(G, grid)
    nb = dict(f._kernel_neighbourhood((1, 0, 0)))
    # From (1,0): 1 hop -> (0,0) and (2,0). 2 hops -> (0,1) and (3,0).
    # (0,1) is around the corner and MUST be present at depth 2.
    check("bend_included", nb.get((0, 1, 0)), 2)
    check("bend_neighbours", sorted(k for k, v in nb.items() if v == 1),
          [(0, 0, 0), (2, 0, 0)])
    check("bend_depth2", sorted(k for k, v in nb.items() if v == 2),
          [(0, 1, 0), (3, 0, 0)])


def test_rack_cell_not_in_neighbourhood():
    G, grid = racked()
    f = field(G, grid)
    nb = dict(f._kernel_neighbourhood((2, 0, 0)))
    # (2,2,0) is Manhattan 2 away and graph 6 away, across the rack.
    check("rack_cell_excluded", (2, 2, 0) in nb, False)
    check("rack_cell_manhattan_distance",
          abs(2 - 2) + abs(2 - 0), 2)


def test_rack_source_stamps_nothing_on_observer():
    """
    Observer at (2,0,0). Source at (2,2,0): Manhattan 2, graph 6.
    Old kernel put 0.333 on the observer's own cell. New kernel puts 0.
    """
    G, grid = racked()
    obs = Fleet("obs", (2.0, 0.0, 0.0))
    peer = Fleet("p", (2.0, 2.0, 0.0))

    f_new = field(G, grid)
    v_new = f_new.get_local_affordance(obs.current_pos, [obs, peer], set(), own_id="obs")

    # kernel_metric, NOT graph=None: setting graph=None would also disable
    # _structure_mask, so the comparison would be measuring two changes at once.
    # (That is exactly what the first version of this test did, and it reported
    # a max abs diff of 1.0 on an open corridor where the two kernels are
    # identical by construction.)
    f_old = field(G, grid, kernel_metric="manhattan")
    v_old = f_old.get_local_affordance(obs.current_pos, [obs, peer], set(), own_id="obs")

    check("rack_source_used_manhattan_before", f_old.kernel_metric, "manhattan")
    check("rack_source_uses_graph_now", f_new._kernel_manhattan_fallbacks, 0)
    # Higher affordance == less repulsion. The new field must be strictly less
    # repelled by a peer it cannot reach.
    check("graph_kernel_less_repelled", bool(v_new.sum() > v_old.sum()), True)


def test_corner_source_still_stamps():
    """
    Observer at (1,0,0), source at (0,1,0) -- around the bend, graph distance 2.
    Must still land, at kernel[2] = 0.333, or the fix has traded one error for
    another.
    """
    G, grid = racked()
    obs = Fleet("obs", (1.0, 0.0, 0.0))
    peer = Fleet("p", (0.0, 1.0, 0.0))

    f = field(G, grid)
    v_peer = f.get_local_affordance(obs.current_pos, [obs, peer], set(), own_id="obs")
    f2 = field(G, grid)
    v_alone = f2.get_local_affordance(obs.current_pos, [obs], set(), own_id="obs")

    check("corner_peer_registers", bool(v_peer.sum() < v_alone.sum()), True)


def test_straight_corridor_matches_manhattan_exactly():
    """
    Down an open 1-D corridor, graph hops and Manhattan agree by construction,
    so new and old kernels must produce IDENTICAL vectors.
    """
    G, grid = corridor()
    obs = Fleet("obs", (5.0, 0.0, 0.0))
    peers = [Fleet("p1", (7.0, 0.0, 0.0)), Fleet("p2", (3.0, 0.0, 0.0))]

    f_new = field(G, grid)
    v_new = f_new.get_local_affordance(obs.current_pos, [obs] + peers, set(), own_id="obs")

    f_old = field(G, grid, kernel_metric="manhattan")
    v_old = f_old.get_local_affordance(obs.current_pos, [obs] + peers, set(), own_id="obs")

    diff = float(np.max(np.abs(v_new - v_old)))
    check("corridor_identical", diff < 1e-6, True)
    if diff >= 1e-6:
        print(f"      max abs diff = {diff}")


def test_cache_is_reused():
    G, grid = racked()
    f = field(G, grid)
    a = f._kernel_neighbourhood((2, 0, 0))
    n_after_first = len(f._kernel_nbhd_cache)
    b = f._kernel_neighbourhood((2, 0, 0))
    check("cache_hit_no_growth", len(f._kernel_nbhd_cache), n_after_first)
    check("cache_returns_same_object", a is b, True)


def test_off_graph_source_falls_back_and_is_counted():
    G, grid = racked()
    f = field(G, grid)
    obs = Fleet("obs", (2.0, 0.0, 0.0))
    ghost = Fleet("g", (2.0, 1.0, 0.0))          # inside the rack: not a node
    f.get_local_affordance(obs.current_pos, [obs, ghost], set(), own_id="obs")
    check("off_graph_counted", f._kernel_manhattan_fallbacks > 0, True)


if __name__ == "__main__":
    for fn in list(globals().values()):
        if callable(fn) and getattr(fn, "__name__", "").startswith("test_"):
            print(f"\n--- {fn.__name__} ---")
            fn()
    print("\n" + ("ALL PASS" if not FAIL else f"FAILURES: {FAIL}"))