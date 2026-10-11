"""
test_obstacles_cleanup.py -- humans who come to clear debris (obstacles "cleanup" mode).

WHAT "cleanup" MODE DOES (obstacles_warehouse.py)
  Debris appears during the episode; a human enters at the edge of that floor,
  walks a shortest path to it, cleans it, walks out to another edge cell and
  leaves. A human never steps onto a fleet: blocked, they wait, then look for a
  way around. Debris nobody can reach is cleared after max_debris_life steps.

WHAT THIS CHECKS
  1. "wander" mode (the default) still behaves as before.
  2. A full lifecycle: debris appears, a human enters at the floor's edge,
     cleans it, leaves; the counters record each stage.
  3. A human never stands on a fleet's cell, and walks around a fleet that
     blocks the way.
  4. Debris never appears on a fleet or on a cell the orchestrator keeps clear.
  5. Debris nobody can reach expires, so nothing is permanent.
  6. The same seed gives the same history; occupied_cells() holds both kinds.

Run:  python test_obstacles_cleanup.py      (ends with ALL PASS; needs networkx)
"""

import networkx as nx

from obstacles_warehouse import ObstacleField

FAIL = []


def check(name, got, want):
    ok = got == want
    print(f"{'PASS' if ok else 'FAIL'}  {name}: got {got!r} want {want!r}")
    if not ok:
        FAIL.append(name)


def warehouse(nx_=8, ny=6, floors=2, lifts=((0, 0), (7, 5))):
    """floors of nx_ x ny cells, joined by a few lifts. Ids 'x_y_z'."""
    G = nx.Graph()
    pos = {}
    for z in range(floors):
        for x in range(nx_):
            for y in range(ny):
                pos[(x, y, z)] = f"{x}_{y}_{z}"
                G.add_node(f"{x}_{y}_{z}")
        for x in range(nx_):
            for y in range(ny):
                if x + 1 < nx_:
                    G.add_edge(f"{x}_{y}_{z}", f"{x+1}_{y}_{z}")
                if y + 1 < ny:
                    G.add_edge(f"{x}_{y}_{z}", f"{x}_{y+1}_{z}")
    for z in range(floors - 1):
        for (x, y) in lifts:
            G.add_edge(f"{x}_{y}_{z}", f"{x}_{y}_{z+1}")
    return G, pos


def cleanup_field(G, pos, seed=1, **kw):
    args = dict(mode="cleanup", seed=seed, debris_interval=1, max_debris=1,
                response_delay=0, cleanup_steps=3, human_move_period=1,
                human_patience=2, max_debris_life=1000)
    args.update(kw)
    return ObstacleField(G, pos, **args)


# ------------------------------------------------------------------ 1. wander
def test_wander_mode_unchanged():
    G, pos = warehouse()
    f = ObstacleField(G, pos, n_humans=2, n_debris=3, human_move_period=2,
                      fixed_cells=[(4, 4, 0)], seed=3)
    f.reset()
    check("wander_counts", (len(f.humans), len(f.debris)), (2, 4))
    check("fixed_cell_is_debris", "4_4_0" in f.debris, True)
    debris0 = list(f.debris)
    humans0 = list(f.humans)
    f.step()
    check("humans_wait_between_moves", f.humans, humans0)
    f.step()
    check("humans_move_on_their_period", f.humans != humans0, True)
    for _ in range(50):
        f.step(fleet_cells={"1_1_0"})          # accepted, ignored in wander mode
    check("debris_never_moves", f.debris, debris0)
    check("wander_stats_keys", sorted(f.statistics()),
          ["debris", "human_moves", "humans", "obstacle_cells"])


# ------------------------------------------------------------------ 2. lifecycle
def test_full_lifecycle():
    G, pos = warehouse()
    f = cleanup_field(G, pos)
    f.reset()
    check("nothing_on_the_floor_at_start", f.occupied_cells(), set())
    entries, debris_seen = [], set()
    for _ in range(300):
        before = len(f.c_humans)
        f.step(fleet_cells=set())
        debris_seen |= set(f.c_debris)
        if len(f.c_humans) > before:
            entries.append(f.c_humans[-1]["entry"])
        s = f.statistics()
        if s["debris_cleaned"] >= 1 and s["humans_exited"] >= 1:
            break
    s = f.statistics()
    check("debris_appeared", s["debris_spawned"] >= 1, True)
    check("a_human_entered", s["humans_entered"] >= 1, True)
    check("debris_cleaned", s["debris_cleaned"] >= 1, True)
    check("human_left", s["humans_exited"] >= 1, True)
    floor_edges = {i for edge in f._edge_by_floor.values() for i in edge}
    check("humans_enter_at_a_floor_edge", all(e in floor_edges for e in entries), True)
    check("nothing_expired", s["debris_expired"], 0)
    check("lifetime_recorded", s["debris_life_mean"] > 0, True)


# ------------------------------------------------------------------ 3. fleets
def test_never_steps_onto_a_fleet():
    G, pos = warehouse(nx_=10, ny=7, floors=1, lifts=())
    f = cleanup_field(G, pos, seed=7, max_debris=2)
    f.reset()
    # A wall of parked fleets across the middle, with a single gap, plus a few
    # scattered ones: humans must wait and walk around.
    fleets = {f"5_{y}_0" for y in range(6)} | {"2_2_0", "7_4_0", "8_1_0"}
    clash = 0
    for _ in range(3000):
        f.step(fleet_cells=fleets)
        clash += sum(1 for h in f.c_humans if h["cell"] in fleets)
        clash += sum(1 for d in f.c_debris if d in fleets)
    s = f.statistics()
    check("no_human_or_debris_ever_on_a_fleet", clash, 0)
    check("humans_waited_somewhere", s["human_wait_steps"] > 0, True)
    check("and_found_ways_around", s["human_reroutes"] > 0, True)
    check("work_still_got_done", s["debris_cleaned"] > 0, True)
    check("near_misses_counted", s["near_miss_steps"] > 0, True)


# ------------------------------------------------------------------ 4. where debris appears
def test_debris_respects_keep_clear_cells():
    G, pos = warehouse()
    keep = {(x, y, 0) for x in range(4) for y in range(6)}       # half of floor 0
    fleets = {"6_3_0", "6_4_0", "1_1_1"}
    f = cleanup_field(G, pos, seed=11, max_debris=3, cleanup_steps=1)
    f.reset(avoid=keep)
    keep_ids = {pos[c] for c in keep}
    bad = 0
    for _ in range(2000):
        f.step(fleet_cells=fleets)
        bad += sum(1 for d in f.c_debris if d in keep_ids or d in fleets)
    check("debris_spawned_many_times", f.statistics()["debris_spawned"] > 20, True)
    check("never_on_a_kept_clear_cell_or_a_fleet", bad, 0)


# ------------------------------------------------------------------ 5. nothing permanent
def test_unreachable_debris_expires():
    G, pos = warehouse(nx_=6, ny=6, floors=1, lifts=())
    f = cleanup_field(G, pos, seed=5, max_debris_life=40, debris_interval=10 ** 9)
    f.reset()
    f.c_debris["3_3_0"] = {"born": 0, "human": False}          # placed by hand
    ring = {f"{x}_{y}_0" for x in (2, 3, 4) for y in (2, 3, 4)} - {"3_3_0"}
    for _ in range(60):
        f.step(fleet_cells=ring)                                 # fleets wall it in
    s = f.statistics()
    check("unreachable_debris_expired", s["debris_expired"], 1)
    check("it_is_gone", "3_3_0" in f.c_debris, False)


# ------------------------------------------------------------------ 6. determinism
def _history(seed):
    G, pos = warehouse()
    f = cleanup_field(G, pos, seed=seed, max_debris=2, debris_interval=20)
    f.reset()
    hist = []
    for _ in range(400):
        f.step(fleet_cells={"3_3_0", "4_2_1"})
        hist.append(tuple(sorted(f.occupied_cells())))
    return hist, f.statistics()


def test_same_seed_same_history():
    h1, s1 = _history(2)
    h2, s2 = _history(2)
    h3, _ = _history(3)
    check("same_seed_same_cells_every_step", h1 == h2, True)
    check("same_seed_same_counters", s1 == s2, True)
    check("different_seed_differs", h1 != h3, True)


def test_occupied_cells_hold_both_kinds():
    G, pos = warehouse()
    f = cleanup_field(G, pos, seed=4, cleanup_steps=50)
    f.reset()
    for _ in range(200):
        f.step(fleet_cells=set())
        if f.c_debris and f.c_humans:
            break
    cells = f.occupied_cells()
    d = next(iter(f.c_debris))
    h = f.c_humans[0]["cell"]
    check("debris_cell_reported", f.coords_by_id[d] in cells, True)
    check("human_cell_reported", f.coords_by_id[h] in cells, True)


if __name__ == "__main__":
    for fn in list(globals().values()):
        if callable(fn) and getattr(fn, "__name__", "").startswith("test_"):
            print(f"\n--- {fn.__name__} ---")
            fn()
    print("\n" + ("ALL PASS" if not FAIL else f"FAILURES: {FAIL}"))