"""
test_corridor_rules.py -- CONFLICT_DESIGN.md components 3 and 4, end to end.

Real FLOWRRA episodes on a hand-built map: a single-lane corridor between two
junction hubs, each hub with three side branches to pull over into. Two fleets
set off from opposite ends, each bound for the far side -- the head-on that
filled cold_run24's shafts. Driven by the shortest-path driver, which never
yields on its own: whatever keeps them apart is the rules.

  short corridor (6 cells, in sight): entry -- one waits, pulls over as the
      other arrives, then goes. No collision, both deliver.
  long corridor (30 cells, beyond sight): both enter, meet inside; priority
      sends the one further from its goal back out to pull over. No
      collision, both deliver, and it is the right one that backs out.
  crossing: two fleets claiming the same junction cell -- one waits a step.

Plus: the corridor index on the real 50_ geometry type, the priority key's
order (nearest goal, then seniority), and order-independence with all four
components on.
"""

import contextlib
import io
import sys

import numpy as np
import networkx as nx

from config_warehouse import CONFIG
import drive_shortest_path as drv

FAIL = []
ALL = {"conflict.path_warnings": True, "conflict.directional_braking": True,
       "conflict.node_aligned_moves": True,
       "conflict.corridor_entry": True, "conflict.priority": True,
       "stream.enabled": False}
OFF = {"stream.enabled": False}


def check(name, got, want):
    ok = got == want
    print(f"{'PASS' if ok else 'FAIL'}  {name}: got {got!r} want {want!r}")
    if not ok:
        FAIL.append(name)


def hubs(L):
    """Corridor x=1..L at y=0 between junction hubs at x=0 and x=L+1; each hub
    has branches up, down and outward (4 cells)."""
    G, grid = nx.Graph(), {}

    def add(x, y):
        nid = f"h{x}_{y}"
        grid[(x, y, 0)] = nid
        G.add_node(nid)
        return nid

    def line(pts):
        ids = [add(x, y) for x, y in pts]
        for a, b in zip(ids, ids[1:]):
            G.add_edge(a, b)
    line([(x, 0) for x in range(-4, L + 6)])
    for hx in (0, L + 1):
        line([(hx, y) for y in range(0, 4)])
        line([(hx, -y) for y in range(0, 4)])
    return G, grid


def episode(L, sets, goals=None, steps=400):
    drv.apply_config(sets)
    from core_warehouse import FLOWRRA
    G, grid = hubs(L)
    by = {v: k for k, v in grid.items()}
    goals = goals or {"A": f"h{L + 5}_0", "B": "h-4_0"}
    starts = {"A": "h-4_0", "B": f"h{L + 5}_0"}
    miss = [{"id": f, "start_node": starts[f], "goal_node": goals[f],
             "start_pos": np.array(by[starts[f]], dtype=np.float32),
             "goal_pos": np.array(by[goals[f]], dtype=np.float32)} for f in ("A", "B")]
    with contextlib.redirect_stdout(io.StringIO()):
        env = FLOWRRA(G, grid, miss, mode="training")
        env.gnn = drv.ShortestPathDriver(env, "never", 0.5, 0)
        for t in range(steps):
            env.step(episode_step=1, total_episodes=1)
            if all(n.id in env.immobile_nodes for n in env.nodes) or not env.nodes:
                break
    done = {n.id for n in env.nodes if n.id in env.immobile_nodes} | set(env.despawned_nodes)
    return env, done, t + 1


def summary(env):
    st = env.get_error_statistics()
    return {k[9:]: v for k, v in st.items() if k.startswith("corridor_")}


# ======================================================================
def test_corridor_index_finds_the_corridor():
    drv.apply_config(ALL)
    from corridor_warehouse import CorridorIndex
    G, grid = hubs(6)
    idx = CorridorIndex(G, {v: k for k, v in grid.items()})
    sid = idx.sid_of["h3_0"]
    check("corridor_length", idx.length(sid), 6)
    check("corridor_ends", set(idx.ends[sid]), {"h0_0", "h7_0"})
    check("straight", idx.straight[sid], True)
    check("hub_is_not_a_corridor_cell", "h0_0" in idx.sid_of, False)


def test_short_corridor_entry():
    off, done_off, t_off = episode(6, OFF)
    on, done_on, t_on = episode(6, ALL)
    s = summary(on)
    print(f"      off: collisions {off.loop.total_collisions}, delivered {sorted(done_off)} in {t_off} steps")
    print(f"      on : collisions {on.loop.total_collisions}, delivered {sorted(done_on)} in {t_on} steps  "
          f"waits {s['entry_waits']} pull-overs {s['entry_pullovers']} contests {s['entry_contests']}")
    check("off_collides", off.loop.total_collisions > 0, True)
    check("on_no_collision", on.loop.total_collisions, 0)
    check("on_both_deliver", done_on, {"A", "B"})
    check("entry_rule_fired", s["entry_waits"] + s["entry_contests"] > 0, True)


def test_long_corridor_meeting_backs_out_the_right_fleet():
    # B's goal is 2 cells past the far hub; A's is 5: B is nearer, A backs out.
    L = 30
    goals = {"A": f"h{L + 5}_0", "B": "h-2_0"}
    on, done, t = episode(L, ALL, goals=goals, steps=600)
    s = summary(on)
    backed = sorted({f for _st, f, ev, _sid in on.rules.log if ev == "retreat"})
    print(f"      collisions {on.loop.total_collisions}, delivered {sorted(done)} in {t} steps; "
          f"meetings {s['meetings']} retreats {s['retreaters']} released {s['released']} "
          f"timeouts {s['timeouts']}; backed out: {backed}")
    check("met_inside", s["meetings"] >= 1, True)
    check("the_farther_fleet_backed_out", backed, ["A"])
    check("released_not_timed_out", (s["released"] >= 1, s["timeouts"]), (True, 0))
    check("no_collision", on.loop.total_collisions, 0)
    check("both_deliver", done, {"A", "B"})


def test_priority_key_order():
    drv.apply_config(ALL)
    from corridor_warehouse import ConflictRules

    class N:
        def __init__(self, fid, hops):
            self.id, self._h = fid, hops

        def get_graph_distance_to_goal(self):
            return self._h

    r = ConflictRules.__new__(ConflictRules)
    r.aging, r.waited, r.entered = 0.25, {}, {}
    near, far = N("x", 5), N("y", 9)
    check("nearest_goal_first", min((near, far), key=r.key).id, "x")
    a, b = N("a", 7), N("b", 7)
    r.entered = {"a": (3, 40), "b": (3, 12)}
    check("then_seniority", min((a, b), key=r.key).id, "b")
    r.waited = {"y": 20}                       # 9 - 0.25*20 = 4 < 5
    check("aging_lifts_a_long_waiter", min((near, far), key=r.key).id, "y")


def test_node_aligned_moves():
    """A step that would cross a node stops on it; a turn snaps the old axis;
    with the switch off, movement is exactly as before."""
    import node_warehouse as nw
    from node_warehouse import FleetNode, build_spatial_indices
    G, grid = hubs(6)
    aisle, _ = build_spatial_indices(grid)

    def fleet(pos):
        n = FleetNode(id="m", current_pos=np.array(pos, dtype=np.float64),
                      goal_pos=np.zeros(3), grid_pos_dict=grid, aisle_index=aisle)
        n.G = G
        n.speed = 0.5
        return n

    got = {}
    for flag in (False, True):
        nw.NODE_ALIGNED = flag
        a = fleet((2.83, 0.0, 0.0)); a.apply_discrete_action(1)       # +X past node 3
        b = fleet((0.0, -0.08, 0.0)); b.apply_discrete_action(1)      # turn +X off-rail
        c = fleet((2.0, 0.0, 0.0)); c.apply_discrete_action(1)        # on phase
        got[flag] = (round(float(a.current_pos[0]), 3), round(float(b.current_pos[1]), 3),
                     round(float(c.current_pos[0]), 3))
    nw.NODE_ALIGNED = False
    check("off_overshoots_and_keeps_residual", got[False], (3.33, -0.08, 2.5))
    check("on_lands_on_node_and_snaps", got[True], (3.0, 0.0, 2.5))


def test_yield_to_stopped():
    """
    S is held by recovery ON the hub junction for 30 steps; M comes down the
    side branch and its route runs straight through S. Braking floors at 0.1,
    so without the rule M creeps into S. With it, M queues behind S, then both
    go on once the hold ends.
    """
    import drive_shortest_path as d
    res = {}
    for label, sets in (("off", dict(ALL)), ("on", dict(ALL, **{"conflict.yield_to_stopped": True}))):
        d.apply_config(sets)
        from core_warehouse import FLOWRRA
        G, grid = hubs(6)
        by = {v: k for k, v in grid.items()}
        spec = {"S": ("h0_0", "h-4_0"), "M": ("h0_3", "h11_0")}
        miss = [{"id": f, "start_node": a, "goal_node": g,
                 "start_pos": np.array(by[a], dtype=np.float32),
                 "goal_pos": np.array(by[g], dtype=np.float32)} for f, (a, g) in spec.items()]
        with contextlib.redirect_stdout(io.StringIO()):
            env = FLOWRRA(G, grid, miss, mode="training")
            env.gnn = d.ShortestPathDriver(env, "never", 0.5, 0)
            env._yield_until["S"] = 30
            for t in range(200):
                env.step(episode_step=1, total_episodes=1)
                if all(n.id in env.immobile_nodes for n in env.nodes):
                    break
        done = {n.id for n in env.nodes if n.id in env.immobile_nodes} | set(env.despawned_nodes)
        st = env.get_error_statistics()
        res[label] = (env.loop.total_collisions, done, st.get("corridor_stopped_waits", 0))
        print(f"      {label}: collisions {res[label][0]}, delivered {sorted(done)}, "
              f"queued-behind-stopped waits {res[label][2]}")
    check("off_drives_into_the_stopped_fleet", res["off"][0] > 0, True)
    check("on_no_collision", res["on"][0], 0)
    check("on_both_deliver", res["on"][1], {"S", "M"})
    check("on_queued", res["on"][2] > 0, True)


def test_switches_off_build_nothing():
    env, _, _ = episode(6, OFF, steps=3)
    check("no_rules_object", env.rules, None)
    check("no_overrides", env._rule_actions, {})


def test_order_independent_with_all_four():
    import measure_order_dependence as mod
    worst = 0
    for h in (1, 5, 20):
        res = []
        for rev in (False, True):
            drv.apply_config(ALL)
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
