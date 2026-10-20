"""
test_world_fixes.py -- build log entry 44's changes to the world, each against
the old world (every switch off reproduces it exactly).

  1. THE CONVOY CRAWL (safety.bubble_floor_fix). Two robots in a row, 1 hop
     apart, RULES with our full brain. Old: the leader is braked to 10% as well,
     so the pair crawls. Fixed: the leader drives at full speed; no collision,
     the gap never under 1 hop.
  2. THE FLOAT STALL (safety.node_snap_eps). The same pair driven to their
     goals. Old: they stop a millionth of a hop short of a node and wait for
     each other until the livelock escape (40 steps). Fixed: home without it.
  3. ORPHANS NEED A PICKUP (errors.orphans_need_pickup). An order waiting in an
     open pickup is reserved for everyone and is not "an order waiting" in the
     book; off, it is free to take.
  4. THE NAIVE RESCUE (errors.naive_rescue). A rescuer that dies while
     fetching: ours puts the pickup back on the board and sends another; naive
     abandons it. A robot that dies carrying a rescued load: ours opens a
     pickup; naive loses the order.

Run:  python test_world_fixes.py      (ends with ALL PASS)
"""

import contextlib
import io

import numpy as np

import node_warehouse as nw
from config_warehouse import CONFIG
from core_warehouse import FLOWRRA
from node_warehouse import precompute_goal_distances
from test_mapf_drivers import floor
from test_smoke_integration import build_instance

FAIL = []


def check(name, got, want):
    ok = got == want
    print(f"{'PASS' if ok else 'FAIL'}  {name}: got {got!r} want {want!r}")
    if not ok:
        FAIL.append(name)


def world(fixes: bool):
    CONFIG["safety"]["node_snap_eps"] = 1e-4 if fixes else 0.0
    CONFIG["safety"]["bubble_floor_fix"] = fixes


def pair_run(fixes: bool, steps=200):
    """B at (1,0) for (6,0), A behind at (0,0) for (5,0); RULES, full brain."""
    from drive_shortest_path import ShortestPathDriver
    world(fixes)
    G, pos, grid = floor(7, 1)
    pairs = [("B", "n_1_0", "n_6_0"), ("A", "n_0_0", "n_5_0")]
    missions = [{"id": f, "start_node": s, "goal_node": g,
                 "start_pos": np.array([pos[s]["X"], 0.0, 0.0], dtype=np.float32),
                 "goal_pos": np.array([pos[g]["X"], 0.0, 0.0], dtype=np.float32)}
                for f, s, g in pairs]
    CONFIG["stream"]["enabled"] = False
    CONFIG["errors"]["enabled"] = False
    CONFIG.setdefault("obstacles", {})["enabled"] = False
    env = FLOWRRA(G, grid, missions, mode="training",
                  goal_distance_maps=precompute_goal_distances(G, missions),
                  shared_pool_mode=False)
    env.max_steps_per_episode = steps
    drv = ShortestPathDriver(env, "never", 0.5, 0)
    drv.epsilon_gaussian = lambda *a, **k: 0.0
    env.gnn = drv
    b10, gap, home = None, float("inf"), {}
    for t in range(steps):
        with contextlib.redirect_stdout(io.StringIO()):
            env.step(episode_step=1, total_episodes=1)
        p = {n.id: n.current_pos.copy() for n in env.nodes}
        if len(p) == 2:
            gap = min(gap, float(np.abs(p["A"] - p["B"]).sum()))
        if t == 9:
            b10 = float(p["B"][0]) - 1.0
        for n in env.nodes:
            if n.id in env.frozen_nodes and n.id not in home:
                home[n.id] = t + 1
        if len(home) == 2:
            break
    return {"b_travel_10": round(b10, 3), "gap": gap, "home": home,
            "collisions": env.loop.total_collisions,
            "livelock": env.livelock_overrides}


def test_convoy_crawl_and_float_stall():
    old = pair_run(False)
    new = pair_run(True)
    world(False)
    print(f"      old world: {old}")
    print(f"      fixed:     {new}")
    check("1_old_leader_crawls", old["b_travel_10"] < 1.0, True)
    check("1_fixed_leader_full_speed", new["b_travel_10"] >= 2.5, True)
    check("1_fixed_no_collision", new["collisions"], 0)
    check("1_fixed_gap_kept", new["gap"] >= 1.0 - 1e-3, True)
    check("2_old_needed_the_livelock_escape", old["livelock"] > 0, True)
    check("2_fixed_home_without_it", (sorted(new["home"]), new["livelock"]), (["A", "B"], 0))
    check("2_fixed_faster", max(new["home"].values()) < max(old["home"].values()), True)


def pool_env(**errors):
    saved = {k: CONFIG["errors"].get(k) for k in errors}
    CONFIG["errors"].update(errors)
    CONFIG["stream"]["enabled"] = False
    try:
        G, grid_pos, missions, gdm, goal_pool = build_instance(8)
        with contextlib.redirect_stdout(io.StringIO()):
            env = FLOWRRA(G, grid_pos, missions, mode="training", goal_distance_maps=gdm,
                          shared_pool_mode=True, goal_pool=goal_pool)
    finally:
        CONFIG["errors"].update(saved)
    return env


def test_orphans_need_pickup():
    for on in (True, False):
        env = pool_env(orphans_need_pickup=on)
        g = sorted(env.goal_pool)[0]
        for n in env.nodes:                         # nobody holds g
            if n.current_goal_id == g:
                env._release_target(n)
        env.open_pickups["pickup_x"] = {"pos": env.nodes[0].current_pos.copy(),
                                        "goal_ids": [g], "source_fleet": "dead"}
        reserved = g in env._reserved_by_others(env.nodes[1])
        check(f"3_{'on' if on else 'off'}_pickup_order_reserved_for_everyone", reserved, on)
        env.claimed_goals = set(env.goal_pool) - {g}
        env.targeted_goals = {}
        check(f"3_{'on' if on else 'off'}_counts_as_waiting_in_the_book",
              env._book_order_waiting(), not on)


def test_naive_rescue():
    for naive in (False, True):
        env = pool_env(naive_rescue=naive, orphans_need_pickup=True)
        rescuer, dead, spare = env.nodes[0], env.nodes[1], env.nodes[2]
        env.open_pickups["pickup_d"] = {"pos": dead.current_pos.copy(),
                                        "goal_ids": [sorted(env.goal_pool)[0]],
                                        "source_fleet": dead.id}
        env._pickup_assignment[rescuer.id] = "pickup_d"
        with contextlib.redirect_stdout(io.StringIO()):
            env._stop_fleet(rescuer)                # dies while fetching
            env._dispatch_rescuers()
        resent = "pickup_d" in env._pickup_assignment.values()
        tag = "naive" if naive else "ours"
        check(f"4_{tag}_dead_fetcher_pickup_resent", resent, not naive)
        check(f"4_{tag}_pickup_abandoned", "pickup_d" in env._abandoned_pickups, naive)

        # A robot that dies carrying a rescued load.
        g2 = sorted(env.goal_pool)[1]
        env._inherited_orders.add(g2)
        spare.current_goal_id = g2
        with contextlib.redirect_stdout(io.StringIO()):
            env._stop_fleet(spare)
            env.step_count += env.stop_confirm_steps + 1
            env._confirm_stops_and_open_pickups()
        opened = any(p.get("source_fleet") == spare.id for p in env.open_pickups.values())
        check(f"4_{tag}_carrier_gets_a_pickup", opened, not naive)
        check(f"4_{tag}_carried_order_lost", g2 in env._naive_lost_goals, naive)


if __name__ == "__main__":
    test_convoy_crawl_and_float_stall()
    test_orphans_need_pickup()
    test_naive_rescue()
    nw.SNAP_EPS = 0.0
    print("\nALL PASS" if not FAIL else f"\nFAILED: {FAIL}")
