"""
test_projection.py -- the BFS-descent peer projection in density_warehouse.py.

Same two-aisle racked warehouse as test_proximity.py:

    y=2   A---A---A---A---A---A        (x = 0..5)
          |                   |
    y=1   C       RACK        C
          |                   |
    y=0   A---A---A---A---A---A

A fleet at (3,0,0) whose goal is (0,2,0) must turn at x=0. Dead reckoning along
+X projects it to 4,5,6 -- away from its goal, into the far cross-aisle and off
the map. Descent projects 2,1,0 along y=0.
"""

import numpy as np
import networkx as nx
from collections import deque

from density_warehouse import WarehouseDensityField


def build_graph():
    G = nx.Graph()
    grid = {}

    def add(x, y):
        nid = f"n_{x}_{y}"
        grid[(x, y, 0)] = nid
        G.add_node(nid)
        return nid

    for y in (0, 2):
        for x in range(6):
            add(x, y)
        for x in range(5):
            G.add_edge(f"n_{x}_{y}", f"n_{x+1}_{y}")
    for x in (0, 5):
        add(x, 1)
        G.add_edge(f"n_{x}_0", f"n_{x}_1")
        G.add_edge(f"n_{x}_1", f"n_{x}_2")
    return G, grid


def bfs_from(G, goal_id):
    d = {goal_id: 0}
    q = deque([goal_id])
    while q:
        n = q.popleft()
        for nb in G.neighbors(n):
            if nb not in d:
                d[nb] = d[n] + 1
                q.append(nb)
    return d


class Fleet:
    def __init__(self, fid, pos, direction, dmap=None, goal_id=None):
        self.id = fid
        self.current_pos = np.array(pos, dtype=np.float64)
        self.direction = np.array(direction, dtype=np.float64)
        self.goal_distance_map = dmap
        self.current_goal_id = goal_id


FAIL = []


def check(name, got, want):
    ok = got == want
    print(f"{'PASS' if ok else 'FAIL'}  {name}: got {got!r} want {want!r}")
    if not ok:
        FAIL.append(name)


def field():
    G, grid = build_graph()
    f = WarehouseDensityField(
        max_vision_range=10, projection_steps=3, grid_pos_dict=grid, graph=G
    )
    return f, G, grid


def path_cells(f, fleet):
    p = f._intended_path(fleet)
    if p is None:
        return None
    return [[c for c, _ in step] for step in p]


# --------------------------------------------------------------------------
def test_straight_corridor():
    f, G, grid = field()
    dmap = bfs_from(G, "n_0_0")
    fl = Fleet("a", (3.0, 0.0, 0.0), (-1, 0, 0), dmap, "n_0_0")
    check("straight", path_cells(f, fl), [[(2, 0, 0)], [(1, 0, 0)], [(0, 0, 0)]])  # own cell (3) dropped


def test_turning_peer():
    """
    THE CASE THE OLD CODE DISCARDED. Fleet at (3,0,0) heading +X, goal (0,2,0).
    Its true route runs -X to the cross-aisle. Dead reckoning would stamp
    (4,0,0),(5,0,0),(6,0,0); (6,0,0) is not even a node.
    """
    f, G, grid = field()
    dmap = bfs_from(G, "n_0_2")
    fl = Fleet("a", (3.0, 0.0, 0.0), (1, 0, 0), dmap, "n_0_2")
    check("turning", path_cells(f, fl), [[(2, 0, 0)], [(1, 0, 0)], [(0, 0, 0)]])  # own cell (3) dropped


def test_weights_sum_to_one():
    f, G, grid = field()
    dmap = bfs_from(G, "n_0_2")
    fl = Fleet("a", (3.0, 0.0, 0.0), (1, 0, 0), dmap, "n_0_2")
    p = f._intended_path(fleet=fl)
    sums = [round(sum(w for _, w in step), 9) for step in p]
    check("weights_sum", sums, [1.0, 1.0, 1.0])


def test_tie_splits_weight():
    """
    From (3,0,0) with goal (3,2,0): both cross-aisles are equidistant (x=0 is
    3+2+3=8, x=5 is 2+2+2=6) -- not tied. Build an explicit tie instead: a fleet
    exactly midway on a symmetric ring.
    """
    G = nx.Graph()
    grid = {}
    # A 6-cell ring: (0,0) (1,0) (2,0) (2,1) (1,1) (0,1), closed.
    ring = [(0, 0), (1, 0), (2, 0), (2, 1), (1, 1), (0, 1)]
    for (x, y) in ring:
        grid[(x, y, 0)] = f"r_{x}_{y}"
        G.add_node(f"r_{x}_{y}")
    for i in range(len(ring)):
        a, b = ring[i], ring[(i + 1) % len(ring)]
        G.add_edge(f"r_{a[0]}_{a[1]}", f"r_{b[0]}_{b[1]}")

    f = WarehouseDensityField(
        max_vision_range=10, projection_steps=1, grid_pos_dict=grid, graph=G
    )
    dmap = bfs_from(G, "r_1_1")           # goal opposite the start
    fl = Fleet("a", (1.0, 0.0, 0.0), (0, 0, 0), dmap, "r_1_1")
    p = f._intended_path(fl)
    # The fleet is ON r_1_0, so that cell is dropped (already stamped) and the
    # first trail level is the tie itself.
    step0 = sorted(p[0])
    check("tie_branches", [c for c, _ in step0], [(0, 0, 0), (2, 0, 0)])
    check("tie_weights", [round(w, 6) for _, w in step0], [0.5, 0.5])


def test_at_goal_terminates():
    f, G, grid = field()
    dmap = bfs_from(G, "n_3_0")
    fl = Fleet("a", (3.0, 0.0, 0.0), (0, 0, 0), dmap, "n_3_0")
    # [] not None: the descent RAN and correctly found no future cells. None
    # would mean "could not compute" and would trigger dead reckoning, stamping
    # a trail for a fleet that is not going anywhere. See _intended_path's
    # return contract.
    check("at_goal", path_cells(f, fl), [])
    check("at_goal_not_none", f._intended_path(fl) is None, False)


def test_mid_edge_arrival_node():
    """
    At x=2.5 heading +X the arrival cell is 3, not round(2.5)=2. At x=3.5
    heading +X it is 4, not round(3.5)=4 by luck -- check both parities.
    """
    f, G, grid = field()
    dmap = bfs_from(G, "n_5_0")
    a = Fleet("a", (2.5, 0.0, 0.0), (1, 0, 0), dmap, "n_5_0")
    b = Fleet("b", (3.5, 0.0, 0.0), (1, 0, 0), dmap, "n_5_0")
    check("arrival_2p5", f._arrival_node(a), "n_3_0")
    check("arrival_3p5", f._arrival_node(b), "n_4_0")
    # And moving the other way.
    c = Fleet("c", (2.5, 0.0, 0.0), (-1, 0, 0), dmap, "n_5_0")
    check("arrival_2p5_back", f._arrival_node(c), "n_2_0")


def test_no_dmap_returns_none():
    f, G, grid = field()
    fl = Fleet("a", (3.0, 0.0, 0.0), (1, 0, 0), None, None)
    check("no_dmap", f._intended_path(fl), None)


def test_off_graph_returns_none():
    f, G, grid = field()
    dmap = bfs_from(G, "n_0_0")
    fl = Fleet("a", (2.0, 1.0, 0.0), (1, 0, 0), dmap, "n_0_0")  # inside the rack
    check("off_graph", f._intended_path(fl), None)


def test_cache_key_invalidates_on_move():
    f, G, grid = field()
    dmap = bfs_from(G, "n_0_0")
    fl = Fleet("a", (3.0, 0.0, 0.0), (-1, 0, 0), dmap, "n_0_0")
    p1 = f._cached_intended_path(fl)
    check("memo_lives_on_the_fleet", hasattr(fl, "_ip_memo"), True)
    fl.current_pos = np.array([2.0, 0.0, 0.0])
    p2 = f._cached_intended_path(fl)
    check("cache_updated", [c for c, _ in p2[0]], [(1, 0, 0)])
    check("cache_first_unchanged", [c for c, _ in p1[0]], [(2, 0, 0)])
    # Mid-edge heading -X toward node 2: stamp(current_pos) rounds 2.5 -> 2,
    # so cell 2 is already marked at FULL severity and the trail must not
    # re-stamp it at falloff severity on top.
    fl.current_pos = np.array([2.5, 0.0, 0.0])
    p3 = f._cached_intended_path(fl)
    check("mid_edge_no_double_stamp", [c for c, _ in p3[0]], [(1, 0, 0)])
    # Heading +X the arrival cell is 3, which stamp() did NOT mark, so keep it.
    fl.direction = np.array([1.0, 0.0, 0.0])
    p4 = f._cached_intended_path(fl)
    check("mid_edge_keeps_forward_cell", [c for c, _ in p4[0]], [(3, 0, 0)])


def test_no_trail_cell_ever_double_stamps():
    """
    INVARIANT: the caller stamps round(current_pos) at full peer_severity, so
    the LEADING trail level may not contain that cell -- that would be a pure
    double-count artifact of how the arrival node is resolved.

    LATER levels legitimately may. A fleet mid-edge at 2.5 heading +X with its
    goal at x=0 is moving AWAY: it finishes its move into cell 3, then descends
    back through cell 2. Re-occupying its current cell two steps out is a real
    prediction, not an artifact, and must not be suppressed.
    """
    f, G, grid = field()
    bad = 0
    for goal in ("n_0_0", "n_5_0", "n_0_2", "n_5_2"):
        dmap = bfs_from(G, goal)
        for x10 in range(0, 51, 5):
            x = x10 / 10.0
            for d in ((1, 0, 0), (-1, 0, 0), (0, 0, 0)):
                fl = Fleet("s", (x, 0.0, 0.0), d, dmap, goal)
                p = f._intended_path(fl)
                if not p:          # None (uncomputable) or [] (at goal)
                    continue
                own = tuple(int(v) for v in np.round(fl.current_pos))
                for c, _ in p[0]:
                    if c == own:
                        bad += 1
                        print(f"      x={x} dir={d} goal={goal} restamps {own}")
    check("no_double_stamp_leading_level", bad, 0)


def test_all_trail_cells_are_real_nodes():
    """The old dead reckoning could stamp cells that are not graph nodes."""
    f, G, grid = field()
    bad = 0
    for goal in ("n_0_2", "n_5_2"):
        dmap = bfs_from(G, goal)
        for x10 in range(0, 51, 5):
            for d in ((1, 0, 0), (-1, 0, 0)):
                fl = Fleet("s", (x10 / 10.0, 0.0, 0.0), d, dmap, goal)
                p = f._intended_path(fl)
                if not p:
                    continue
                for level in p:
                    for c, _ in level:
                        if c not in grid:
                            bad += 1
    check("all_trail_cells_real", bad, 0)


def test_memo_does_not_leak_between_fields():
    """
    The memo lives on the FLEET, so it must be keyed by which field computed it.
    Two fields over the same fleet objects -- an A/B on kernel metric, or the
    lesion harness -- must not serve each other's cached paths.
    """
    f, G, grid = field()
    dmap = bfs_from(G, "n_0_0")
    fl = Fleet("a", (3.0, 0.0, 0.0), (-1, 0, 0), dmap, "n_0_0")
    p1 = f._cached_intended_path(fl)
    check("first_field_computes", [c for c, _ in p1[0]], [(2, 0, 0)])

    g = WarehouseDensityField(max_vision_range=10, projection_steps=3,
                              grid_pos_dict=grid, graph=None)
    p2 = g._cached_intended_path(fl)
    check("second_field_recomputes", p2, None)
    check("second_field_counted_fallback", g._projection_fallbacks, 1)


def test_field_stamps_real_cells_only():
    """
    End to end: the affordance vector of an observer must show repulsion along
    the peer's real route. With dead reckoning the turning peer stamped into the
    rack, where _structure_mask zeroes it -- so the observer saw NOTHING.
    """
    f, G, grid = field()
    dmap = bfs_from(G, "n_0_2")
    observer = Fleet("obs", (1.0, 0.0, 0.0), (0, 0, 0), bfs_from(G, "n_5_0"), "n_5_0")
    peer = Fleet("p", (3.0, 0.0, 0.0), (1, 0, 0), dmap, "n_0_2")

    v_new = f.get_local_affordance(
        observer.current_pos, [observer, peer], frozen_node_ids=set(),
        own_goal_pos=np.array([5.0, 0.0, 0.0]), own_id="obs",
    )

    f2 = WarehouseDensityField(
        max_vision_range=10, projection_steps=3, grid_pos_dict=grid, graph=G,
        project_stationary=False,
    )
    f2.graph = None  # force the dead-reckoning fallback
    f2.grid_pos_dict = grid
    v_old = f2.get_local_affordance(
        observer.current_pos, [observer, peer], frozen_node_ids=set(),
        own_goal_pos=np.array([5.0, 0.0, 0.0]), own_id="obs",
    )

    # The observer sits at x=1 and the peer's true route comes toward it.
    # The new field must be MORE repulsive (lower affordance) on average.
    check("new_more_repulsive", bool(v_new.sum() < v_old.sum()), True)
    check("fallback_counted", f2._projection_fallbacks > 0, True)


if __name__ == "__main__":
    for fn in list(globals().values()):
        if callable(fn) and getattr(fn, "__name__", "").startswith("test_"):
            print(f"\n--- {fn.__name__} ---")
            fn()
    print("\n" + ("ALL PASS" if not FAIL else f"FAILURES: {FAIL}"))