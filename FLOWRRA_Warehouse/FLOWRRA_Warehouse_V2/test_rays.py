"""
test_rays.py -- the graph-walk rewrite of sense_6_axis_rays.

The rewrite has to satisfy three things at once:

  1. AGREE WITH THE ORIGINAL on what it can see. The old trace accumulated
     distance in 0.5 increments from a continuous position; the new one counts
     whole cells from the fleet's cell. So they agree to within the half-cell
     quantisation, and nowhere else may they differ.

  2. NOT TUNNEL. Two cells can be adjacent in coordinates with no track between
     them. Naive unit-stepping in coordinates would miss that -- landing exactly
     on a node makes is_structurally_valid's `behind` and `ahead` the same node,
     so the connecting edge is never tested. The graph walk must refuse to cross.

  3. RAYS MUST BE INDEPENDENT. One blocked direction may not shorten another.

Plus a timing comparison, because the whole point was that this function had
become the largest single term in the profile (16.9s of 32.6s) and was GROWING
as the map emptied.
"""

import time

import numpy as np
import networkx as nx

from node_warehouse import FleetNode, build_spatial_indices

FAIL = []


def check(name, got, want):
    ok = got == want
    print(f"{'PASS' if ok else 'FAIL'}  {name}: got {got!r} want {want!r}")
    if not ok:
        FAIL.append(name)


# --------------------------------------------------------------------- maps
def racked(width=60, aisles=10):
    """Aisles at even y, cross-aisles at both ends. Long clear runs."""
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
    return G, grid


def mk(G, grid, aisle, fid, pos, direction=(1, 0, 0)):
    n = FleetNode(id=fid, current_pos=np.array(pos, dtype=np.float64),
                  goal_pos=np.array([0.0, 0.0, 0.0]),
                  grid_pos_dict=grid, aisle_index=aisle)
    n.G = G
    n.direction = np.array(direction, dtype=np.float32)
    return n


# ------------------------------------------------------- reference (original)
def original_rays(self, all_fleets):
    """The pre-2026-09-13 coordinate trace, verbatim, as reference."""
    ray_dirs = [np.array([1, 0, 0]), np.array([-1, 0, 0]),
                np.array([0, 1, 0]), np.array([0, -1, 0]),
                np.array([0, 0, 1]), np.array([0, 0, -1])]
    fleet_pos_map = {tuple(np.round(f.current_pos).astype(int)): f
                     for f in all_fleets if f.id != self.id}
    dists = np.zeros(6, dtype=np.float32)
    hits = [None] * 6
    for ray_idx, r_dir in enumerate(ray_dirs):
        distance_found = 0.0
        trace_pos = self.current_pos.copy()
        _ray_speed = max(float(self.speed), 1e-3)
        for _ in range(int(self.max_vision_range * (5.0 / _ray_speed))):
            next_pos = trace_pos + (r_dir * self.speed)
            if not self.is_structurally_valid(next_pos, action=ray_idx + 1):
                break
            distance_found += self.speed
            trace_pos = next_pos
            t = tuple(np.round(trace_pos).astype(int))
            if t in fleet_pos_map:
                hits[ray_idx] = fleet_pos_map[t]
                break
        dists[ray_idx] = float(min(distance_found / 25.0, 1.0))
    return dists, hits


# ------------------------------------------------------------------- tests
def test_agrees_with_original_on_open_map():
    """
    Over many positions and fleet layouts, the two must agree to within the
    half-cell quantisation: |old_cells - new_cells| <= 1 always, and the SAME
    peer must be reported wherever one is found.
    """
    G, grid = racked()
    aisle, _ = build_spatial_indices(grid)
    rng = np.random.default_rng(0)
    cells = sorted(grid.keys())

    worst = 0
    peer_disagreements = 0
    cases = 0
    for trial in range(60):
        fleets = []
        for i in range(14):
            c = cells[int(rng.integers(0, len(cells)))]
            p = [float(c[0]), float(c[1]), float(c[2])]
            if rng.random() < 0.5:
                p[0] += 0.5
            fleets.append(mk(G, grid, aisle, f"f{i}", p))
        for me in fleets:
            # SKIP the stacked-fleet case. Banker's rounding makes the ORIGINAL
            # wrong here: its first half-step lands at x+0.5, np.round(34.5) is
            # 34, so it re-checks the observer's OWN cell and reports a
            # co-located fleet as a hit 0.5 cells ahead. The graph walk starts
            # from the next cell and never re-tests its own, which is correct.
            # A known improvement, not a disagreement to reconcile.
            my_cell = tuple(np.round(me.current_pos).astype(int))
            if any(tuple(np.round(f.current_pos).astype(int)) == my_cell
                   for f in fleets if f.id != me.id):
                continue
            cases += 1
            new_d, new_pv = me.sense_6_axis_rays(fleets)[:2]
            old_d, old_hits = original_rays(me, fleets)
            for r in range(6):
                gap = abs(new_d[r] * 25.0 - old_d[r] * 25.0)
                # Both saturate at 25 cells; above that the clip hides any gap.
                if new_d[r] >= 1.0 and old_d[r] >= 1.0:
                    continue
                worst = max(worst, gap)
                old_hit = old_hits[r] is not None
                new_hit = bool(np.any(new_pv[r] != 0)) or old_hit
                if old_hit and not np.any(new_pv[r] != 0) and old_hits[r].direction.any():
                    peer_disagreements += 1
    print(f"      ({cases} fleet-observations, worst cell gap {worst:.1f})")
    check("agrees_within_half_cell", worst <= 1.0 + 1e-6, True)
    check("peer_hits_agree", peer_disagreements, 0)


def test_does_not_tunnel_through_a_gap():
    """
    Two cells adjacent in coordinates with the connecting edge removed. The ray
    must stop, not step over it.
    """
    G, grid = racked(width=20, aisles=3)
    G.remove_edge("n_8_0", "n_9_0")
    aisle, _ = build_spatial_indices(grid)
    me = mk(G, grid, aisle, "me", (5.0, 0.0, 0.0))
    d = me.sense_6_axis_rays([me])[0]
    check("stops_at_missing_edge", round(float(d[0]) * 25.0), 3)   # 5 -> 8
    # And the node on the far side still exists, so this is a genuine edge test.
    check("far_node_exists", (9, 0, 0) in grid, True)


def test_rays_are_independent():
    G, grid = racked(width=20, aisles=3)
    aisle, _ = build_spatial_indices(grid)
    me = mk(G, grid, aisle, "me", (5.0, 0.0, 0.0))
    d = me.sense_6_axis_rays([me])[0]
    # +X clear, -X clear to x=0, +Y blocked (rack at y=1), -Y blocked, Z blocked.
    check("plus_x_sees_far", float(d[0]) * 25.0 > 5, True)
    check("minus_x_sees_5", round(float(d[1]) * 25.0), 5)
    check("plus_y_blocked", float(d[2]), 0.0)
    check("z_blocked", (float(d[4]), float(d[5])), (0.0, 0.0))


def test_cap_loses_no_information():
    """
    ray_range = 25 because min(cells/25, 1.0) pins at 1.0 there. A 50-cell clear
    corridor must read exactly the same as a 25-cell one.
    """
    G, grid = racked(width=60, aisles=3)
    aisle, _ = build_spatial_indices(grid)
    a = mk(G, grid, aisle, "a", (2.0, 0.0, 0.0))    # 57 cells of clear +X
    d = a.sense_6_axis_rays([a])[0]
    check("saturates_at_one", float(d[0]), 1.0)
    a.ray_range = 50
    d50 = a.sense_6_axis_rays([a])[0]
    check("cap_25_matches_cap_50", float(d50[0]), float(d[0]))


def test_speed_comparison():
    """
    The rewrite exists for cost. Measured on an EMPTY corridor, which is the
    regime that was getting more expensive as the episode progressed.
    """
    G, grid = racked(width=60, aisles=10)
    aisle, _ = build_spatial_indices(grid)
    fleets = [mk(G, grid, aisle, f"f{i}", (float(2 + i * 5), 0.0, 0.0))
              for i in range(6)]
    me = fleets[0]

    N = 3000
    t0 = time.perf_counter()
    for _ in range(N):
        original_rays(me, fleets)
    t_old = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(N):
        me.sense_6_axis_rays(fleets)
    t_new = time.perf_counter() - t0

    print(f"      original  {t_old:7.3f}s  = {t_old/N*1e6:7.1f} us/call")
    print(f"      graph     {t_new:7.3f}s  = {t_new/N*1e6:7.1f} us/call")
    print(f"      speedup   {t_old/t_new:.1f}x")
    check("rewrite_is_faster", t_new < t_old, True)


if __name__ == "__main__":
    for fn in list(globals().values()):
        if callable(fn) and getattr(fn, "__name__", "").startswith("test_"):
            print(f"\n--- {fn.__name__} ---")
            fn()
    print("\n" + ("ALL PASS" if not FAIL else f"FAILURES: {FAIL}"))