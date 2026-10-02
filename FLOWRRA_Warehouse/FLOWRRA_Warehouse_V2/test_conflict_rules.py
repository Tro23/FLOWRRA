"""
test_conflict_rules.py -- CONFLICT_DESIGN.md components 1 and 2.

  1. PATH-BASED WARNINGS, on hand-built geometry where the right answer is
     obvious: a corridor (head-on, convoy, blocker, fleets parting), a crossing
     (contested), and the rules themselves (floor, follow gap, fallback).
     Then in a real episode: the loop's warned pairs are exactly the
     classifier's, and the preemptive filters see pairs beyond 2 hops.
  2. DIRECTION-AWARE BRAKING: a steady convoy at 1.5 hops is no longer
     throttled; a closing pair at the same distance still is; and the step
     stays order-independent (the brake reads a pre-move snapshot).
"""

import contextlib
import io
import sys
import types
from collections import deque

import numpy as np
import networkx as nx

from config_warehouse import CONFIG
from conflict_warehouse import ConflictSettings, PathConflicts
from density_warehouse import WarehouseDensityField
from proximity_warehouse import GraphProximity
import drive_shortest_path as drv

FAIL = []


def check(name, got, want):
    ok = got == want
    print(f"{'PASS' if ok else 'FAIL'}  {name}: got {got!r} want {want!r}")
    if not ok:
        FAIL.append(name)


# ------------------------------------------------------------ geometry
def corridor(n=12):
    G, grid = nx.Graph(), {}
    for x in range(n):
        grid[(x, 0, 0)] = f"c{x}"
        G.add_node(f"c{x}")
        if x:
            G.add_edge(f"c{x-1}", f"c{x}")
    return G, grid


def crossing():
    """Horizontal y=0, x=0..8, crossed by vertical x=4, y=-4..4."""
    G, grid = nx.Graph(), {}

    def add(x, y):
        nid = f"n{x}_{y}"
        grid[(x, y, 0)] = nid
        G.add_node(nid)
        return nid
    for x in range(9):
        add(x, 0)
        if x:
            G.add_edge(f"n{x-1}_0", f"n{x}_0")
    for y in range(-4, 5):
        if y != 0:
            add(4, y)
    for y in range(-4, 4):
        G.add_edge(f"n4_{y}", f"n4_{y+1}")
    return G, grid


def bfs(G, goal):
    d, q = {goal: 0}, deque([goal])
    while q:
        u = q.popleft()
        for v in G.neighbors(u):
            if v not in d:
                d[v] = d[u] + 1
                q.append(v)
    return d


class Fleet:
    def __init__(self, fid, pos, goal, G, direction=(0, 0, 0)):
        self.id = fid
        self.current_pos = np.array(pos, dtype=np.float64)
        self.direction = np.array(direction, dtype=np.float32)
        self.goal_distance_map = bfs(G, goal) if goal else None
        self.current_goal_id = goal


def judge(G, grid, fleets, stationary=()):
    dens = WarehouseDensityField(max_vision_range=10, grid_pos_dict=grid, graph=G)
    prox = GraphProximity(G, grid, search_radius=4.0, metric="graph")
    prox.refresh(fleets, excluded_ids=set())
    pc = PathConflicts(dens, prox, ConflictSettings())
    return {frozenset((v.a, v.b)): v for v in
            pc.evaluate({f.id: f for f in fleets}, set(stationary), 0.5)}


def one(G, grid, fleets, stationary=()):
    vs = list(judge(G, grid, fleets, stationary).values())
    return vs[0] if vs else None


# ================================================================ component 1
def test_head_on_three_hops_apart_warns():
    """Today: no warning until 2 hops. Now: warned at 3, splat at the meeting cell."""
    G, grid = corridor()
    v = one(G, grid, [Fleet("a", (2, 0, 0), "c11", G, (1, 0, 0)),
                      Fleet("b", (5, 0, 0), "c0", G, (-1, 0, 0))])
    check("kind", v.kind, "head_on")
    check("dist", v.dist, 3.0)
    check("warns", v.warn, True)
    check("meets_between_them", v.meet in {(3, 0, 0), (4, 0, 0)}, True)


def test_convoy_at_follow_gap_is_not_a_warning():
    """57% of cold_run24's warnings. Same route, leader moving: not a conflict."""
    G, grid = corridor()
    v = one(G, grid, [Fleet("a", (2, 0, 0), "c11", G, (1, 0, 0)),
                      Fleet("b", (3.5, 0, 0), "c11", G, (1, 0, 0))])
    check("kind", v.kind, "following")
    check("follower_is_behind", v.follower, "a")
    check("no_warning_at_1.5", v.warn, False)


def test_convoy_too_close_warns():
    G, grid = corridor()
    v = one(G, grid, [Fleet("a", (2, 0, 0), "c11", G, (1, 0, 0)),
                      Fleet("b", (3.25, 0, 0), "c11", G, (1, 0, 0))])
    check("kind", v.kind, "following")
    check("warns_below_gap", (v.warn, v.why), (True, "gap"))


def test_floor_always_warns():
    G, grid = corridor()
    v = one(G, grid, [Fleet("a", (2, 0, 0), "c11", G, (1, 0, 0)),
                      Fleet("b", (3, 0, 0), "c11", G, (1, 0, 0))])
    check("floor_warns_even_a_convoy", (v.warn, v.why), (True, "floor"))


def test_stationary_blocker_warns():
    G, grid = corridor()
    v = one(G, grid, [Fleet("a", (2, 0, 0), "c11", G, (1, 0, 0)),
                      Fleet("b", (5, 0, 0), "c11", G, (0, 0, 0))], stationary={"b"})
    check("kind", v.kind, "blocked")
    check("warns", v.warn, True)
    check("splat_on_the_blocker", v.meet, (5, 0, 0))


def test_fleets_parting_do_not_warn():
    """Back to back and driving apart at 1.5 hops. Today this warned."""
    G, grid = corridor()
    v = one(G, grid, [Fleet("a", (4, 0, 0), "c0", G, (-1, 0, 0)),
                      Fleet("b", (5.5, 0, 0), "c11", G, (1, 0, 0))])
    check("kind", v.kind, "none")
    check("no_warning", v.warn, False)


def test_crossing_contested_warns():
    """Both reach the junction within one step of each other."""
    G, grid = crossing()
    v = one(G, grid, [Fleet("a", (2, 0, 0), "n8_0", G, (1, 0, 0)),
                      Fleet("b", (4, -1, 0), "n4_4", G, (0, 1, 0))])
    check("kind", v.kind, "contested")
    check("warns", v.warn, True)
    check("meet_at_junction", v.meet, (4, 0, 0))


def test_unknown_route_falls_back_to_distance():
    G, grid = corridor()
    near = one(G, grid, [Fleet("a", (2, 0, 0), "c11", G, (1, 0, 0)),
                         Fleet("b", (4, 0, 0), None, G)])
    far = one(G, grid, [Fleet("a", (2, 0, 0), "c11", G, (1, 0, 0)),
                        Fleet("b", (5, 0, 0), None, G)])
    check("unknown_kind", near.kind, "unknown")
    check("within_2_warns_as_today", (near.warn, near.why), (True, "fallback"))
    check("at_3_does_not", far.warn, False)


def test_classify_precedence_and_sequential():
    """Direct on synthetic routes: the order of the rules, and sequential."""
    pc = PathConflicts(None, None, ConflictSettings())
    A, B, C, D, E = (0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0), (4, 0, 0)
    F, H = (5, 0, 0), (6, 0, 0)
    # only shared cell C: a at t=1, b at t=4 -- three steps apart: sequential
    check("sequential", pc.classify([{C}, {D}], [{E}, {F}, {H}, {C}], {A}, {B}, False, False)[0],
          "sequential")
    # the same, one step apart: contested
    check("contested_within_one", pc.classify([{C}, {D}], [{E}, {C}], {A}, {B}, False, False)[0],
          "contested")
    # both head for each other's cell: head-on even if one is stationary
    check("head_on_beats_blocked",
          pc.classify([{B}], [{A}], {A}, {B}, False, True)[0], "head_on")
    # same route, other side stationary: blocked, not following
    check("blocked_beats_following",
          pc.classify([{B}, {C}], [], {A}, {B}, False, True)[0], "blocked")
    check("nothing_shared", pc.classify([{C}], [{E}], {A}, {B}, False, False)[0], "none")
    s = ConflictSettings()
    check("decide_sequential_silent", pc.decide("sequential", 2.0), (False, ""))
    check("decide_following_at_gap", pc.decide("following", 1.5), (False, ""))


def test_rack_between_is_never_a_pair():
    """Two aisles, rack between: 2 apart on the grid, 6 by graph -- no pair."""
    G, grid = nx.Graph(), {}
    for y in (0, 2):
        for x in range(6):
            grid[(x, y, 0)] = f"r{x}_{y}"
            G.add_node(f"r{x}_{y}")
            if x:
                G.add_edge(f"r{x-1}_{y}", f"r{x}_{y}")
    for x in (0, 5):
        grid[(x, 1, 0)] = f"r{x}_1"
        G.add_edge(f"r{x}_0", f"r{x}_1")
        G.add_edge(f"r{x}_1", f"r{x}_2")
    got = judge(G, grid, [Fleet("a", (2, 0, 0), "r5_0", G, (1, 0, 0)),
                          Fleet("b", (2, 2, 0), "r0_2", G, (-1, 0, 0))])
    check("no_pair_across_rack", len(got), 0)


# ------------------------------------------------ in a real episode
def real_env(sets, agents=24, seed=0, recovery="always"):
    drv.apply_config(sets)
    from core_warehouse import FLOWRRA
    with contextlib.redirect_stdout(io.StringIO()):
        G, grid, miss, kw = drv.synthetic_instance(types.SimpleNamespace(agents=agents), seed)
        env = FLOWRRA(G, grid, miss, **kw)
    env.gnn = drv.ShortestPathDriver(env, recovery, 0.5, seed)
    return env


ON = {"conflict.path_warnings": True, "conflict.directional_braking": True}


def test_loop_uses_exactly_the_classified_pairs():
    env = real_env(ON)
    bad, beyond = 0, 0
    with contextlib.redirect_stdout(io.StringIO()):
        for _ in range(40):
            env.step(episode_step=1, total_episodes=1)
            warned = {frozenset((v.a, v.b)) for v in env._verdicts_start if v.warn}
            looped = {frozenset((a, b)) for a, b, _ in env.loop.warning_pairs}
            if warned != looped:
                bad += 1
            beyond += sum(1 for _a, _b, d in env.loop.warning_pairs
                          if d > env.loop.warning_threshold)
            # _step_warning, not loop.warning_nodes: force_repair() empties the
            # latter whenever recovery runs, and the driver invokes it here.
            nodes = {x for p in looped for x in p}
            if nodes != env._step_warning:
                bad += 1
    st = env.get_error_statistics()
    print(f"      warned by kind: " + ", ".join(
        f"{k} {st[f'conflict_warned_{k}']}/{st[f'conflict_pairs_{k}']}"
        for k in ("head_on", "blocked", "following", "contested", "sequential", "none", "unknown")))
    check("loop_set_equals_classifier", bad, 0)
    check("warnings_beyond_two_hops_exist", beyond > 0, True)
    check("convoys_not_warned_at_gap",
          st["conflict_warned_following"] < st["conflict_pairs_following"], True)


def test_preemptive_filter_sees_pairs_beyond_two_hops():
    """
    The filter used to re-walk proximity.pairs(radius=2.0). A fleet whose only
    warned partner is 2.5 hops away would have been dropped from preemption
    -- the new warnings would have been silently ignored.
    """
    env = real_env(ON)
    with contextlib.redirect_stdout(io.StringIO()):
        env.step(episode_step=1, total_episodes=1)
    a, b = env.nodes[0].id, env.nodes[1].id
    # head-on, so neither filter may drop them as a convoy
    env.nodes[0].direction = np.array([1, 0, 0], dtype=np.float32)
    env.nodes[1].direction = np.array([-1, 0, 0], dtype=np.float32)
    env._yield_until = {}
    env.loop.warning_pairs = [(a, b, 2.5)]
    kept = env._drop_following_pairs({a, b})
    check("far_warned_pair_kept", kept, {a, b})
    kept2 = env._drop_handled_pairs({a, b})
    check("far_warned_pair_kept_by_held_filter", kept2, {a, b})


def test_switches_off_leave_nothing_behind():
    env = real_env({})
    with contextlib.redirect_stdout(io.StringIO()):
        for _ in range(5):
            env.step(episode_step=1, total_episodes=1)
    check("no_classifier", env.conflicts, None)
    check("no_verdicts", env._verdicts_start, [])
    check("no_brake_exemptions", env.brake_convoy_exempt, 0)


# ================================================================ component 2
def convoy_env(sets, lead_dir=(1, 0, 0)):
    """Two fleets on the smoke warehouse's y=0 aisle, 1.5 hops apart, both
    having just moved +X with the gap steady -- a convoy -- or, with
    lead_dir=(-1,0,0), the leader driving back toward the follower."""
    # no preemptive recovery: it would move the pair and hide the brake
    env = real_env(sets, agents=2, recovery="never")
    a, b = env.nodes[0], env.nodes[1]
    for n, x, d in ((a, 3.0, (1, 0, 0)), (b, 4.5, lead_dir)):
        n.current_pos = np.array([x, 0.0, 0.0])
        n.direction = np.array(d, dtype=np.float32)
        n.last_pos = n.current_pos - 0.5 * np.array(d, dtype=np.float64)
        n.goal_pos = np.array([20.0 if d[0] > 0 else 0.0, 0.0, 0.0])
    goal = {1: "n_20_0", -1: "n_0_0"}
    for n, d in ((a, 1), (b, lead_dir[0])):
        n.current_goal_id = goal[d]
        n.goal_distance_map = env.goal_distance_maps.get(goal[d]) or bfs(env.G, goal[d])
    env.proximity.refresh(env.nodes, excluded_ids=env.immobile_nodes)
    return env, a, b


def moved(env, a, b):
    pa, pb = a.current_pos.copy(), b.current_pos.copy()
    with contextlib.redirect_stdout(io.StringIO()):
        env.step(episode_step=1, total_episodes=1)
    return (round(float(np.abs(a.current_pos - pa).sum()), 3),
            round(float(np.abs(b.current_pos - pb).sum()), 3))


def test_convoy_no_longer_throttled():
    off = moved(*convoy_env({}))
    on = moved(*convoy_env({"conflict.directional_braking": True}))
    print(f"      moved per step: braking by distance {off}   direction-aware {on}")
    check("off_throttles_the_convoy", off[0] < 0.5 and off[1] < 0.5, True)
    check("on_runs_at_full_speed", on, (0.5, 0.5))


def test_closing_pair_still_brakes():
    off = moved(*convoy_env({}, lead_dir=(-1, 0, 0)))
    on = moved(*convoy_env({"conflict.directional_braking": True}, lead_dir=(-1, 0, 0)))
    print(f"      head-on at 1.5: by distance {off}   direction-aware {on}")
    check("closing_pair_brakes_the_same", on, off)
    check("and_is_throttled", on[0] < 0.5, True)


def test_order_independent_with_both_switches():
    """Acceptance test 1 of SIMULTANEOUS_STEP.md, with components 1 and 2 on."""
    import measure_order_dependence as mod
    worst = 0
    for h in (1, 5, 20):
        res = []
        for rev in (False, True):
            drv.apply_config(ON)
            res.append(mod.run(rev, h, 24, 7, True)[0])
        worst = max(worst, sum(1 for k in res[0] if res[0][k] != res[1][k]))
    check("forwards_equals_reversed", worst, 0)


if __name__ == "__main__":
    for fn in list(globals().values()):
        if callable(fn) and getattr(fn, "__name__", "").startswith("test_"):
            print(f"\n--- {fn.__name__} ---")
            fn()
    drv.apply_config({})
    print("\n" + ("ALL PASS" if not FAIL else f"FAILURES: {FAIL}"))
    sys.exit(1 if FAIL else 0)
