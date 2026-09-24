"""
test_obstacles.py

The unregistered-obstacle channel end to end, plus a range audit of every
feature the Phase-1 and Phase-2 work added.

TWO THINGS ARE BEING CHECKED

1. OBSTACLES FLOW ALL THE WAY THROUGH, costing no state dimensions. They enter
   as cells, are perceived by RAY CAST alone -- the only channel in which
   something nobody registered can exist -- and act in three places:

       ray_hit_unknown          the fleet knows something is there
       get_valid_action_mask    HARD VETO, the move is impossible
       density mask + R         (mask 0, R > 0): not traversable, and not a wall

   Four states are distinguishable in two channels:
       free      mask 1, R 0        wall      mask 0, R 0
       congested mask 1, R > 0      obstacle  mask 0, R > 0

2. EVERY FEATURE STAYS IN ITS RANGE across real episodes. A feature that leaves
   its range does not crash -- it quietly dominates the encoder, which is the
   failure this whole redesign exists to undo.
"""

import numpy as np
import networkx as nx

from config_warehouse import CONFIG
from density_warehouse import WarehouseDensityField
from node_warehouse import FleetNode, build_spatial_indices
from obstacles_warehouse import ObstacleField, from_config

FAIL = []


def check(name, got, want):
    ok = got == want
    print(f"{'PASS' if ok else 'FAIL'}  {name}: got {got!r} want {want!r}")
    if not ok:
        FAIL.append(name)


def corridor(width=14):
    G = nx.Graph()
    grid = {}
    for x in range(width):
        grid[(x, 0, 0)] = f"n{x}"
        G.add_node(f"n{x}")
    for x in range(width - 1):
        G.add_edge(f"n{x}", f"n{x+1}")
    return G, grid, build_spatial_indices(grid)[0]


def junction():
    """A cross: four arms meeting at (5,5,0)."""
    G = nx.Graph()
    grid = {}
    for x in range(11):
        grid[(x, 5, 0)] = f"h{x}"
        G.add_node(f"h{x}")
    for x in range(10):
        G.add_edge(f"h{x}", f"h{x+1}")
    for y in range(11):
        if (5, y, 0) in grid:
            continue
        grid[(5, y, 0)] = f"v{y}"
        G.add_node(f"v{y}")
    for y in range(10):
        a = grid[(5, y, 0)]
        b = grid[(5, y + 1, 0)]
        G.add_edge(a, b)
    return G, grid, build_spatial_indices(grid)[0]


def mk(G, grid, aisle, fid, pos, **kw):
    n = FleetNode(id=fid, current_pos=np.array(pos, dtype=np.float64),
                  goal_pos=np.array([0.0, 0.0, 0.0]),
                  grid_pos_dict=grid, aisle_index=aisle, **kw)
    n.G = G
    return n


# ====================================================== the four-state encoding
def test_four_states_are_distinguishable():
    G, grid, aisle = corridor()
    f = WarehouseDensityField(max_vision_range=10, grid_pos_dict=grid, graph=G)
    a = mk(G, grid, aisle, "a", (5.0, 0.0, 0.0))
    peer = mk(G, grid, aisle, "p", (6.0, 0.0, 0.0))
    L = f.local_radius

    vol = f.get_local_volume(a.current_pos, [a, peer], set(), own_id="a",
                             static_obstacles={(4, 0, 0)})
    mask, rep = vol[0], vol[1]

    def at(c):
        i = (int(c[0]) - 5 + L, int(c[1]) + L, int(c[2]) + L)
        return float(mask[i]), float(rep[i])

    # Three hops away: the falloff kernel reaches 2, so this is genuinely clear.
    # (7,0,0) is only 1 hop from the peer and picks up repulsion -- "free" has to
    # mean outside the kernel's reach, not merely "no peer standing on it".
    free_m, free_r = at((9, 0, 0))
    cong_m, cong_r = at((6, 0, 0))            # a peer is here
    wall_m, wall_r = at((5, 1, 0))            # no track at all
    obst_m, obst_r = at((4, 0, 0))            # a human is here

    print(f"      free     mask={free_m:.0f} R={free_r:.2f}")
    print(f"      congested mask={cong_m:.0f} R={cong_r:.2f}")
    print(f"      wall     mask={wall_m:.0f} R={wall_r:.2f}")
    print(f"      obstacle mask={obst_m:.0f} R={obst_r:.2f}")

    check("free_is_1_0", (free_m, free_r > 0), (1.0, False))
    check("congested_is_1_pos", (cong_m, cong_r > 0), (1.0, True))
    check("wall_is_0_0", (wall_m, wall_r > 0), (0.0, False))
    check("obstacle_is_0_pos", (obst_m, obst_r > 0), (0.0, True))


def test_entropy_collapses_when_an_option_is_removed():
    """
    The check you wanted: does entropy actually reach 0?

    It does, and only because the obstacle zeroes the MASK. Leaving mask=1 and
    only raising R would make the cell "possible but unattractive" -- affordance
    1/(1+3) = 0.25 against 1.0, entropy ~0.72 -- while get_valid_action_mask()
    vetoes the move outright. Entropy would be counting an option the fleet
    cannot take.
    """
    G, grid, aisle = corridor()
    f = WarehouseDensityField(max_vision_range=10, grid_pos_dict=grid, graph=G)
    p = np.array([5.0, 0.0, 0.0])

    h_clear = f.action_entropy(f.get_local_affordance(p, [], set()))
    h_block = f.action_entropy(f.get_local_affordance(
        p, [], set(), static_obstacles={(6, 0, 0)}))
    print(f"      corridor clear {h_clear:.4f} -> one side blocked {h_block:.4f}")
    check("clear_corridor_is_undecided", round(h_clear, 4), 1.0)
    check("blocked_corridor_has_no_choice", round(h_block, 4), 0.0)

    # A junction loses one of four arms: entropy falls but does not vanish.
    G2, grid2, aisle2 = junction()
    f2 = WarehouseDensityField(max_vision_range=10, grid_pos_dict=grid2, graph=G2)
    q = np.array([5.0, 5.0, 0.0])
    j_clear = f2.action_entropy(f2.get_local_affordance(q, [], set()))
    j_block = f2.action_entropy(f2.get_local_affordance(
        q, [], set(), static_obstacles={(6, 5, 0)}))
    print(f"      junction clear {j_clear:.4f} -> one arm blocked {j_block:.4f}")

    # BOTH read 1.0, and that is correct rather than a bug. Entropy is
    # normalised by log(LIVE neighbours), so four equally-good options and three
    # equally-good options are both maximally UNDECIDED. Losing an arm changes
    # how many choices exist, not how hard the choice is -- and how many exist is
    # what the action mask says, not what entropy measures.
    check("junction_stays_undecided", round(j_block, 4), round(j_clear, 4))

    j = mk(G2, grid2, aisle2, "j", (5.0, 5.0, 0.0),
           static_obstacles={(6, 5, 0)})
    j_free = mk(G2, grid2, aisle2, "j2", (5.0, 5.0, 0.0))
    check("junction_loses_an_option_at_the_mask",
          (int(j_free.get_valid_action_mask().sum())
           - int(j.get_valid_action_mask().sum())), 1)


def test_cache_is_not_poisoned_by_a_transient_human():
    """
    _structure_mask returns a CACHED array keyed by centre cell. Zeroing it in
    place would make one fleet's transient view of where a human stood permanent
    for every later fleet on that cell.
    """
    G, grid, aisle = corridor()
    f = WarehouseDensityField(max_vision_range=10, grid_pos_dict=grid, graph=G)
    p = np.array([5.0, 0.0, 0.0])
    before = f.action_entropy(f.get_local_affordance(p, [], set()))
    f.get_local_affordance(p, [], set(), static_obstacles={(6, 0, 0)})
    after = f.action_entropy(f.get_local_affordance(p, [], set()))
    check("mask_cache_unpoisoned", round(after, 4), round(before, 4))


# ====================================================== the obstacle field
def test_field_places_and_moves():
    G, grid, aisle = corridor(width=20)
    fld = ObstacleField(G, grid, n_humans=3, n_debris=2, human_move_period=2,
                        seed=1)
    fld.reset()
    check("humans_placed", len(fld.humans), 3)
    check("debris_placed", len(fld.debris), 2)
    check("cells_reported", len(fld.occupied_cells()), 5)

    debris_before = set(fld.debris)
    seen = {tuple(sorted(fld.occupied_cells()))}
    for _ in range(12):
        fld.step()
        seen.add(tuple(sorted(fld.occupied_cells())))
    check("debris_never_moves", set(fld.debris), debris_before)
    check("humans_did_move", len(seen) > 1, True)
    check("human_moves_counted", fld.human_moves > 0, True)


def test_field_avoids_reserved_cells():
    """An episode must not begin with a fleet inside a human, or with an
    unreachable goal."""
    G, grid, aisle = corridor(width=20)
    reserved = {(x, 0, 0) for x in range(0, 15)}
    fld = ObstacleField(G, grid, n_humans=2, n_debris=2, seed=2)
    fld.reset(avoid=reserved)
    check("nothing_placed_on_reserved_cells",
          sorted(fld.occupied_cells() & reserved), [])


def test_fixed_cells_place_exactly():
    G, grid, aisle = corridor(width=20)
    fld = ObstacleField(G, grid, n_humans=0, n_debris=0,
                        fixed_cells=[(7, 0, 0), (11, 0, 0)], seed=0)
    fld.reset()
    check("fixed_cells_honoured",
          sorted(fld.occupied_cells()), [(7, 0, 0), (11, 0, 0)])


def test_from_config_respects_the_flag():
    G, grid, aisle = corridor()
    cfg = {"obstacles": {"enabled": False, "n_humans": 3}}
    check("disabled_returns_none", from_config(G, grid, cfg), None)
    cfg["obstacles"]["enabled"] = True
    fld = from_config(G, grid, cfg, seed=0)
    check("enabled_returns_field", fld is not None, True)


# ====================================================== range audit
def test_feature_ranges_over_a_real_episode():
    """
    Every feature added in Phase 1 and 2, audited across a real episode with
    obstacles live. Out-of-range does not crash -- it quietly dominates the
    encoder, which is the failure this redesign exists to undo.
    """
    import test_smoke_integration as smoke

    CONFIG["obstacles"].update({"enabled": True, "n_humans": 4, "n_debris": 4,
                                "human_move_period": 3})
    try:
        env, agent, base, dens, nf, mem, steps = smoke.run_episode(True, steps=45)
        est = env.get_error_statistics()
        print(f"      obstacles: {est.get('obstacle_humans')} humans, "
              f"{est.get('obstacle_debris')} debris, "
              f"{est.get('obstacle_human_moves')} moves")
        check("obstacles_active", est["static_obstacles"] > 0, True)
        check("humans_actually_moved", est.get("obstacle_human_moves", 0) > 0, True)

        # The same health counters the smoke test asserts.
        #
        # This file passed throughout the off-grid-memory-splat hunt -- not
        # because it was unaffected, but because it never LOOKED. It runs a real
        # episode through the same machinery, so a silent 158 fallbacks would
        # have sailed straight through. And it runs a configuration the smoke
        # test does not (obstacles enabled), so it can catch regressions on a
        # path nothing else covers.
        kf = est["kernel_manhattan_fallbacks"]
        if kf:
            print(f"      kernel fallbacks {kf} from "
                  f"{est.get('kernel_fallback_cells')}")
        check("no_kernel_fallbacks", kf, 0)
        check("no_ray_origin_recovery", est["ray_origin_recovered"], 0)
        print(f"      off-grid memory splats refused: "
              f"{est.get('memory_offgrid_rejected', 0)}   "
              f"projection fallbacks: {est['projection_fallbacks']}")

        L = env.nodes[0].state_layout()
        V = np.stack([n.get_state_vector(env.nodes) for n in env.nodes])
        check("all_finite", bool(np.all(np.isfinite(V))), True)

        bounds = {
            "ray_distances": (0.0, 1.0),
            "ray_hit_waiting": (0.0, 1.0),
            "ray_hit_permanent": (0.0, 1.0),
            "ray_hit_unknown": (0.0, 1.0),
            # Signed: a direction that INCREASES graph distance to the goal is
            # negative. [-1, 1], not [0, 1].
            "goal_gradient": (-1.0, 1.0),
            "situation_features": (0.0, 1.0),
            "gibbs_state": (0.0, 1.0),
        }
        print(f"      {'block':<20}{'min':>8}{'max':>8}{'mean':>8}")
        for blk, (lo_b, hi_b) in bounds.items():
            lo, hi = L[blk]
            sl = V[:, lo:hi]
            print(f"      {blk:<20}{sl.min():>8.3f}{sl.max():>8.3f}{sl.mean():>8.3f}")
            check(f"{blk}_in_range",
                  bool(sl.min() >= lo_b - 1e-6 and sl.max() <= hi_b + 1e-6), True)

        lo, hi = L["ray_hit_unknown"]
        hits = int(V[:, lo:hi].sum())
        print(f"      rays that hit an unregistered obstacle: {hits}")
        if hits == 0:
            print("      NOTE: no ray happened to face one this step -- the "
                  "channel is wired, but this run did not exercise it")
    finally:
        CONFIG["obstacles"].update({"enabled": False, "n_humans": 0,
                                    "n_debris": 0})
        smoke.set_flags(False)


if __name__ == "__main__":
    for fn in list(globals().values()):
        if callable(fn) and getattr(fn, "__name__", "").startswith("test_"):
            print(f"\n--- {fn.__name__} ---")
            fn()
    print("\n" + ("ALL PASS" if not FAIL else f"FAILURES: {FAIL}"))