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
    d = a.sense_6_axis_rays([a])[0]
    check("default_transform", a.ray_transform, "clip25")
    check("default_saturates_at_one", float(d[0]), 1.0)


def test_smooth_transform_never_saturates():
    G, grid, aisle = racked(width=40, aisles=4)
    vals = []
    for x in (2.0, 10.0, 20.0):
        a = mk(G, grid, aisle, "a", (x, 0.0, 0.0),
               ray_transform="smooth", ray_softness=8.0)
        a.ray_range = 50
        d = a.sense_6_axis_rays([a])[0]
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
        d = a.sense_6_axis_rays([a])[0]
        flat.append(round(float(d[0]), 4))
    print(f"      clip25 on the same three    -> {flat}")
    check("clip25_collapses_37_and_29_cells", flat[0] == flat[1] == 1.0, True)
    check("smooth_keeps_them_apart", vals[0] != vals[1], True)


# ====================================================================== 4
def test_situation_features_now_eight():
    """
    Waiting adds sf_is_waiting and sf_wait_steps. The layout must follow, or
    every lesion index downstream silently points at the wrong dimensions --
    which is exactly why state_layout() is derived and not hardcoded.
    """
    G, grid, aisle = racked()
    a = mk(G, grid, aisle, "a", (5.0, 0.0, 0.0))
    b = mk(G, grid, aisle, "b", (7.0, 0.0, 0.0))
    sit = a.get_situation_features()
    check("situation_width", sit.shape, (8,))

    layout = a.state_layout()
    check("layout_situation_width",
          layout["situation_features"][1] - layout["situation_features"][0], 8)
    check("layout_total_matches_vector",
          layout["_base_len"][1], int(len(a.get_state_vector([a, b]))))

    # Blocks must still tile the vector with no gap or overlap.
    # rays_all and ray_semantics are overlapping CONVENIENCE GROUPS, not
    # blocks: they span dims that the real blocks already cover. Only the
    # disjoint blocks should tile the vector.
    _groups = {"rays_all", "ray_semantics"}
    named = sorted(((k, v) for k, v in layout.items()
                    if not k.startswith("_") and k not in _groups),
                   key=lambda kv: kv[1][0])
    cursor, gaps = 0, 0
    for k, (lo, hi) in named:
        if lo != cursor:
            gaps += 1
        cursor = hi
    check("no_gaps_after_widening", gaps, 0)


def test_wait_features_land_in_their_own_slice():
    G, grid, aisle = racked()
    a = mk(G, grid, aisle, "a", (5.0, 0.0, 0.0))
    b = mk(G, grid, aisle, "b", (7.0, 0.0, 0.0))
    before = a.get_state_vector([a, b]).copy()
    a.sf_is_waiting = 1.0
    a.sf_wait_steps = 0.5
    after = a.get_state_vector([a, b])

    changed = sorted(np.flatnonzero(np.abs(after - before) > 1e-9).tolist())
    lo, _hi = a.state_layout()["situation_features"]
    check("wait_dims_are_the_last_two_of_situation", changed, [lo + 6, lo + 7])


def test_wait_defaults_to_zero():
    """Default OFF means a fleet that has never waited reads exactly 0."""
    G, grid, aisle = racked()
    a = mk(G, grid, aisle, "a", (5.0, 0.0, 0.0))
    sit = a.get_situation_features()
    check("is_waiting_defaults_zero", float(sit[6]), 0.0)
    check("wait_steps_defaults_zero", float(sit[7]), 0.0)


# ====================================================================== 5
def test_ray_semantics_channels():
    """
    A WAITING fleet, a DEAD fleet, a PARKED fleet and a WALL all present as
    direction 0. Before these channels they were indistinguishable through a
    ray, which makes waiting a coin flip: "will move when clear" and "will never
    move again" looked identical.
    """
    G, grid, aisle = racked(width=20, aisles=3)
    a = mk(G, grid, aisle, "a", (5.0, 0.0, 0.0))

    # +X: a waiting peer.  -X: a permanently immobile one.
    w = mk(G, grid, aisle, "w", (9.0, 0.0, 0.0)); w.sf_is_waiting = 1.0
    d = mk(G, grid, aisle, "d", (2.0, 0.0, 0.0)); d.sf_is_immobile = 1.0
    _dist, _vel, _disp, hw, hp, hu = a.sense_6_axis_rays([a, w, d])

    check("plus_x_sees_waiting", float(hw[0]), 1.0)
    check("plus_x_not_permanent", float(hp[0]), 0.0)
    check("minus_x_sees_permanent", float(hp[1]), 1.0)
    check("minus_x_not_waiting", float(hw[1]), 0.0)
    check("no_unknown_obstacles", float(hu.sum()), 0.0)

    # A wall is the all-zero case, and stays distinguishable from both.
    check("y_ray_is_a_wall",
          (float(hw[2]), float(hp[2]), float(hu[2])), (0.0, 0.0, 0.0))


def test_unregistered_obstacle_is_detected_only_by_ray():
    """
    Humans and debris are one category. They are NOT in the fleet registry, so
    they have no position report, no velocity and no goal -- and no transparency
    rule, because nobody's goal is ever a human.
    """
    G, grid, aisle = racked(width=20, aisles=3)
    a = mk(G, grid, aisle, "a", (5.0, 0.0, 0.0))

    _d0, _v0, _p0, _w0, _pm0, hu0 = a.sense_6_axis_rays([a])
    check("clear_before", float(hu0.sum()), 0.0)
    clear_dist = float(_d0[0])

    a.static_obstacles = {(8, 0, 0)}
    d1, _v1, _p1, _w1, _pm1, hu1 = a.sense_6_axis_rays([a])
    check("obstacle_flags_the_ray", float(hu1[0]), 1.0)
    check("obstacle_stops_the_ray", bool(float(d1[0]) < clear_dist), True)
    check("obstacle_has_no_velocity", float(np.abs(_v1[0]).sum()), 0.0)
    check("only_that_ray_flagged", float(hu1.sum()), 1.0)


def test_rays_all_and_ray_semantics_slices():
    G, grid, aisle = racked()
    a = mk(G, grid, aisle, "a", (5.0, 0.0, 0.0))
    L = a.state_layout()
    check("rays_all_now_60", L["rays_all"][1] - L["rays_all"][0], 60)
    check("ray_semantics_is_18",
          L["ray_semantics"][1] - L["ray_semantics"][0], 18)
    check("base_len_82", L["_base_len"][1], 82)


# ====================================================================== 6
def test_action_entropy_is_degree_independent():
    """
    Normalised by log(LIVE neighbours), not log(6). Against log(6) the feature
    would mostly measure DEGREE -- a corridor could never exceed 0.387 however
    undecided it was -- which the mask already tells the network.
    """
    G, grid, aisle = racked(width=16, aisles=3)
    # Add a mid-map cross-aisle so a junction exists.
    grid[(8, 1, 0)] = "n_8_1"
    G.add_node("n_8_1")
    G.add_edge("n_8_0", "n_8_1")
    G.add_edge("n_8_1", "n_8_2")
    aisle2, _ = build_spatial_indices(grid)
    f = WarehouseDensityField(max_vision_range=10, grid_pos_dict=grid, graph=G)

    corridor = mk(G, grid, aisle2, "a", (5.0, 0.0, 0.0))
    junction = mk(G, grid, aisle2, "b", (8.0, 0.0, 0.0))
    hc = f.action_entropy(f.get_local_affordance(
        corridor.current_pos, [corridor], set(), own_id="a"))
    hj = f.action_entropy(f.get_local_affordance(
        junction.current_pos, [junction], set(), own_id="b"))
    check("corridor_clear_is_one", round(hc, 4), 1.0)
    check("junction_clear_is_also_one", round(hj, 4), 1.0)


def test_action_entropy_falls_with_asymmetric_contention():
    G, grid, aisle = racked(width=16, aisles=3)
    f = WarehouseDensityField(max_vision_range=10, grid_pos_dict=grid, graph=G)
    vals = []
    for k in range(4):
        a = mk(G, grid, aisle, "a", (7.0, 0.0, 0.0))
        fl = [a] + [mk(G, grid, aisle, f"p{j}", (8.0 + j, 0.0, 0.0))
                    for j in range(k)]
        vals.append(round(f.action_entropy(f.get_local_affordance(
            a.current_pos, fl, set(), own_id="a")), 4))
    print(f"      queue of 0/1/2/3 ahead -> {vals}")
    check("entropy_decreases_monotonically", vals == sorted(vals, reverse=True), True)
    check("clear_case_is_one", vals[0], 1.0)

    # SYMMETRIC contention must NOT drop it: two equally bad options is still
    # undecided, which is the correct reading for a decisiveness measure.
    a = mk(G, grid, aisle, "a", (7.0, 0.0, 0.0))
    sym = [a, mk(G, grid, aisle, "l", (6.0, 0.0, 0.0)),
           mk(G, grid, aisle, "r", (8.0, 0.0, 0.0))]
    hs = f.action_entropy(f.get_local_affordance(
        a.current_pos, sym, set(), own_id="a"))
    check("symmetric_contention_stays_undecided", round(hs, 4), 1.0)


def test_entropy_edge_cases():
    G, grid, aisle = racked(width=16, aisles=3)
    f = WarehouseDensityField(max_vision_range=10, grid_pos_dict=grid, graph=G)
    check("all_zero_affordance_is_zero",
          f.action_entropy(np.zeros(231, dtype=np.float32)), 0.0)
    # A dead end has one live neighbour: no choice, so no entropy.
    corner = mk(G, grid, aisle, "c", (0.0, 0.0, 0.0))
    h = f.action_entropy(f.get_local_affordance(
        corner.current_pos, [corner], set(), own_id="c"))
    check("entropy_is_bounded", bool(0.0 <= h <= 1.0), True)


def test_gibbs_block_in_layout():
    G, grid, aisle = racked()
    a = mk(G, grid, aisle, "a", (5.0, 0.0, 0.0))
    b = mk(G, grid, aisle, "b", (7.0, 0.0, 0.0))
    L = a.state_layout()
    check("gibbs_is_two_dims", L["gibbs_state"][1] - L["gibbs_state"][0], 2)
    check("base_len_82", L["_base_len"][1], 82)
    check("layout_matches_vector", L["_base_len"][1],
          int(len(a.get_state_vector([a, b]))))

    before = a.get_state_vector([a, b]).copy()
    a.sf_local_entropy = 0.4
    a.sf_throughput_t = 0.6
    after = a.get_state_vector([a, b])
    changed = sorted(np.flatnonzero(np.abs(after - before) > 1e-9).tolist())
    lo, _ = L["gibbs_state"]
    check("gibbs_dims_land_in_their_slice", changed, [lo, lo + 1])


# ====================================================================== 7
def test_obstacle_is_a_hard_veto_not_a_preference():
    """
    A human is not a fleet. You can push through a stopped fleet -- that is how
    a rescuer reaches a pickup -- but repulsion is a PREFERENCE, and a large
    enough reward can outbid a preference. So the obstacle appears twice: high
    severity in the field, and an absolute veto at the action mask.
    """
    G, grid, aisle = racked(width=16, aisles=3)
    clear = mk(G, grid, aisle, "a", (5.0, 0.0, 0.0))
    blocked = mk(G, grid, aisle, "b", (5.0, 0.0, 0.0),
                 static_obstacles={(6, 0, 0)})

    m_clear = clear.get_valid_action_mask()
    m_block = blocked.get_valid_action_mask()
    check("plus_x_allowed_when_clear", bool(m_clear[1]), True)
    check("plus_x_vetoed_by_obstacle", bool(m_block[1]), False)
    check("minus_x_untouched", bool(m_block[2]), True)
    check("idle_always_valid", bool(m_block[0]), True)


def test_obstacle_raises_repulsion_harder_than_any_fleet():
    G, grid, aisle = racked(width=16, aisles=3)
    f = WarehouseDensityField(max_vision_range=10, grid_pos_dict=grid, graph=G)
    a = mk(G, grid, aisle, "a", (5.0, 0.0, 0.0))

    v_clear = f.get_local_affordance(a.current_pos, [a], set(), own_id="a")
    v_peer = f.get_local_affordance(
        a.current_pos, [a, mk(G, grid, aisle, "p", (6.0, 0.0, 0.0))],
        set(), own_id="a")
    v_obst = f.get_local_affordance(
        a.current_pos, [a], set(), own_id="a", static_obstacles={(6, 0, 0)})

    h_clear = f.action_entropy(v_clear)
    h_peer = f.action_entropy(v_peer)
    h_obst = f.action_entropy(v_obst)
    print(f"      entropy  clear {h_clear:.4f}  peer {h_peer:.4f}  obstacle {h_obst:.4f}")
    check("obstacle_moves_entropy_more_than_a_peer", bool(h_obst < h_peer), True)
    check("both_below_clear", bool(h_peer < h_clear), True)


def test_obstacle_gets_no_near_goal_transparency():
    """
    The near-goal discount makes a corpse transparent to its own rescuer. It
    must NOT do that for a human -- nobody's goal is ever a person.
    """
    G, grid, aisle = racked(width=16, aisles=3)
    f = WarehouseDensityField(max_vision_range=10, grid_pos_dict=grid, graph=G)
    a = mk(G, grid, aisle, "a", (5.0, 0.0, 0.0))
    a.goal_pos = np.array([6.0, 0.0, 0.0])      # goal IS the obstacle cell

    v = f.get_local_affordance(a.current_pos, [a], set(),
                               own_goal_pos=a.goal_pos, own_id="a",
                               static_obstacles={(6, 0, 0)})
    v_clear = f.get_local_affordance(a.current_pos, [a], set(),
                                     own_goal_pos=a.goal_pos, own_id="a")
    check("obstacle_on_own_goal_still_repels",
          bool(f.action_entropy(v) < f.action_entropy(v_clear)), True)
    # And the veto holds regardless of where the fleet wants to go.
    b = mk(G, grid, aisle, "b", (5.0, 0.0, 0.0), static_obstacles={(6, 0, 0)})
    b.goal_pos = np.array([6.0, 0.0, 0.0])
    check("veto_holds_even_toward_own_goal",
          bool(b.get_valid_action_mask()[1]), False)


if __name__ == "__main__":
    for fn in list(globals().values()):
        if callable(fn) and getattr(fn, "__name__", "").startswith("test_"):
            print(f"\n--- {fn.__name__} ---")
            fn()
    print("\n" + ("ALL PASS" if not FAIL else f"FAILURES: {FAIL}"))