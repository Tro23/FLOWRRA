"""
test_phase1.py -- the three Phase-1 design changes.

All three ship behind flags defaulting to CURRENT behaviour, so the first thing
each test asserts is that the default path is untouched. A design change that
silently moves the baseline makes every earlier measurement incomparable.

  1. CHANNEL SPLIT     get_local_volume() returns (2, S, S, S): mask and
                       repulsion separately, never multiplied. The multiply in
                       get_local_affordance() is what makes 0 mean both "no
                       track here" and "infinitely contested here".

  2. GRAPH ADJACENCY   the GAT's attention graph on graph distance, not
                       Manhattan. Fourth and last instance of the phantom-pair
                       defect, and the only one that corrupted a TOPOLOGY rather
                       than a number.

  3. RAY TRANSFORM     d/(d+c) instead of min(d/25, 1). Same saturating-encoding
                       defect density fixed in August and the rays never got.
"""

import numpy as np
import networkx as nx

from node_warehouse import FleetNode, build_spatial_indices
from density_warehouse import WarehouseDensityField
from proximity_warehouse import GraphProximity

FAIL = []


def check(name, got, want):
    ok = got == want
    print(f"{'PASS' if ok else 'FAIL'}  {name}: got {got!r} want {want!r}")
    if not ok:
        FAIL.append(name)


def racked(width=14, aisles=4):
    """Aisles at even y, cross-aisles only at the two ends."""
    G = nx.Graph()
    grid = {}
    ys = list(range(0, aisles * 2, 2))
    for y in ys:
        for x in range(width):
            grid[(x, y, 0)] = f"n_{x}_{y}"
            G.add_node(f"n_{x}_{y}")
        for x in range(width - 1):
            G.add_edge(f"n_{x}_{y}", f"n_{x+1}_{y}")
    for x in (0, width - 1):
        for y in range(ys[0], ys[-1] + 1):
            if (x, y, 0) not in grid:
                grid[(x, y, 0)] = f"n_{x}_{y}"
                G.add_node(f"n_{x}_{y}")
        for y in range(ys[0], ys[-1]):
            G.add_edge(f"n_{x}_{y}", f"n_{x}_{y+1}")
    return G, grid, build_spatial_indices(grid)[0]


def mk(G, grid, aisle, fid, pos, **kw):
    n = FleetNode(id=fid, current_pos=np.array(pos, dtype=np.float64),
                  goal_pos=np.array([0.0, 0.0, 0.0]),
                  grid_pos_dict=grid, aisle_index=aisle, **kw)
    n.G = G
    return n


# ====================================================================== 1
def test_default_affordance_path_unchanged():
    G, grid, aisle = racked()
    f = WarehouseDensityField(max_vision_range=10, grid_pos_dict=grid, graph=G)
    a = mk(G, grid, aisle, "a", (5.0, 0.0, 0.0))
    b = mk(G, grid, aisle, "b", (7.0, 0.0, 0.0))
    v = f.get_local_affordance(a.current_pos, [a, b], set(), own_id="a")
    check("default_shape_still_231", v.shape, (231,))
    check("default_flag_off_after_call", f._return_volume, False)


def test_volume_has_two_channels():
    G, grid, aisle = racked()
    f = WarehouseDensityField(max_vision_range=10, grid_pos_dict=grid, graph=G)
    a = mk(G, grid, aisle, "a", (5.0, 0.0, 0.0))
    b = mk(G, grid, aisle, "b", (6.0, 0.0, 0.0))
    vol = f.get_local_volume(a.current_pos, [a, b], set(), own_id="a")
    S = 2 * f.local_radius + 1
    check("volume_shape", vol.shape, (2, S, S, S))
    check("channel0_is_binary",
          bool(np.all((vol[0] == 0.0) | (vol[0] == 1.0))), True)
    check("channel1_has_repulsion", bool(vol[1].max() > 0), True)
    check("flag_reset_after_call", f._return_volume, False)


def test_split_preserves_what_the_multiply_destroys():
    """
    The whole point. In the flat encoding, an UNREACHABLE cell and a MAXIMALLY
    CONTESTED cell both read 0. In the volume they are distinguishable.
    """
    G, grid, aisle = racked()
    f = WarehouseDensityField(max_vision_range=10, grid_pos_dict=grid, graph=G)
    a = mk(G, grid, aisle, "a", (5.0, 0.0, 0.0))
    peers = [a] + [mk(G, grid, aisle, f"p{i}", (5.0 + i, 0.0, 0.0))
                   for i in range(1, 4)]
    vol = f.get_local_volume(a.current_pos, peers, set(), own_id="a")
    mask, rep = vol[0], vol[1]

    unreachable = (mask == 0)
    contested = (mask == 1) & (rep > 0)
    check("some_cells_unreachable", bool(unreachable.any()), True)
    check("some_cells_contested", bool(contested.any()), True)
    # In the volume the two sets are disjoint and separable. In the flat
    # affordance both would tend to 0.
    check("unreachable_and_contested_are_distinguishable",
          bool(not (unreachable & contested).any()), True)


def test_volume_and_affordance_agree_where_they_should():
    """mask/(1+R) recomputed from the volume must reproduce the flat vector."""
    G, grid, aisle = racked()
    f = WarehouseDensityField(max_vision_range=10, grid_pos_dict=grid, graph=G)
    a = mk(G, grid, aisle, "a", (5.0, 0.0, 0.0))
    peers = [a, mk(G, grid, aisle, "p", (6.0, 0.0, 0.0))]
    flat = f.get_local_affordance(a.current_pos, peers, set(), own_id="a")
    vol = f.get_local_volume(a.current_pos, peers, set(), own_id="a")
    rebuilt = (vol[0] / (1.0 + vol[1]))[f._diamond_mask]
    check("volume_reproduces_affordance",
          bool(np.allclose(rebuilt, flat, atol=1e-6)), True)


# ====================================================================== 2
def test_graph_adjacency_rejects_phantom_neighbours():
    """
    Two fleets in adjacent aisles behind a rack: Manhattan-2 apart, graph-far.
    Manhattan links them in the attention graph; graph distance must not.
    """
    G, grid, aisle = racked(width=14, aisles=4)
    a = mk(G, grid, aisle, "a", (6.0, 0.0, 0.0))
    b = mk(G, grid, aisle, "b", (6.0, 2.0, 0.0))

    manhattan = float(np.sum(np.abs(a.current_pos - b.current_pos)))
    graph_d = nx.shortest_path_length(G, grid[(6, 0, 0)], grid[(6, 2, 0)])
    check("manhattan_would_link_them", bool(manhattan <= 10.0), True)
    check("graph_distance_is_far", bool(graph_d > 10.0), True)

    prox = GraphProximity(G, grid, search_radius=10.0, metric="graph")
    prox.refresh([a, b], excluded_ids=set())
    linked = [p for p, _ in prox.peers_within("a", 10.0)]
    check("graph_adjacency_excludes_phantom", linked, [])


def test_graph_adjacency_keeps_real_neighbours():
    G, grid, aisle = racked()
    a = mk(G, grid, aisle, "a", (5.0, 0.0, 0.0))
    b = mk(G, grid, aisle, "b", (9.0, 0.0, 0.0))     # same aisle, 4 hops
    prox = GraphProximity(G, grid, search_radius=10.0, metric="graph")
    prox.refresh([a, b], excluded_ids=set())
    check("real_neighbour_kept", [p for p, _ in prox.peers_within("a", 10.0)], ["b"])


# ====================================================================== 3
def test_ray_transform_default_unchanged():
    G, grid, aisle = racked(width=40, aisles=4)
    a = mk(G, grid, aisle, "a", (2.0, 0.0, 0.0))
    d, _, _ = a.sense_6_axis_rays([a])
    check("default_transform", a.ray_transform, "clip25")
    check("default_saturates_at_one", float(d[0]), 1.0)


def test_smooth_transform_never_saturates():
    G, grid, aisle = racked(width=40, aisles=4)
    vals = []
    for x in (2.0, 10.0, 20.0):
        a = mk(G, grid, aisle, "a", (x, 0.0, 0.0),
               ray_transform="smooth", ray_softness=8.0)
        a.ray_range = 50
        d, _, _ = a.sense_6_axis_rays([a])
        vals.append(round(float(d[0]), 4))
    check("smooth_values_all_below_one", bool(all(v < 1.0 for v in vals)), True)
    check("smooth_is_strictly_decreasing_with_distance_to_wall",
          vals == sorted(vals, reverse=True), True)
    print(f"      clearances 37/29/19 cells -> {vals}")

    # Under clip25 the first two positions -- 37 and 29 cells of clear aisle,
    # genuinely different -- BOTH read exactly 1.0. The distinction is gone. The
    # third (19 cells) is under the threshold so it still varies; saturation is
    # a ceiling effect, not a total loss, and the ceiling is exactly where long
    # open aisles live.
    flat = []
    for x in (2.0, 10.0, 20.0):
        a = mk(G, grid, aisle, "a", (x, 0.0, 0.0))
        a.ray_range = 50
        d, _, _ = a.sense_6_axis_rays([a])
        flat.append(round(float(d[0]), 4))
    print(f"      clip25 on the same three    -> {flat}")
    check("clip25_collapses_37_and_29_cells", flat[0] == flat[1] == 1.0, True)
    check("smooth_keeps_them_apart", vals[0] != vals[1], True)


if __name__ == "__main__":
    for fn in list(globals().values()):
        if callable(fn) and getattr(fn, "__name__", "").startswith("test_"):
            print(f"\n--- {fn.__name__} ---")
            fn()
    print("\n" + ("ALL PASS" if not FAIL else f"FAILURES: {FAIL}"))