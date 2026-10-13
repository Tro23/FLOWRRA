"""
test_stream_no_work.py -- in the stream, a fleet with no order left drives out
instead of freezing mid-aisle (build log entries 33-34; stream.no_work_exit).

THE PROBLEM (entry 33). When a fleet's order is taken by a peer while it is en
route and nothing is left to retarget to, three places froze it where it stood.
A frozen fleet is outside the guard, the proximity index and the collision
check, and can later be recalled for a rescue: all three guard-era collisions
were a fleet driving onto one of these (fleets 27#2/37#2 in the full test,
192#2/70#2 and 15#2/86#2 in run 6). The delivery path had already replaced
freezing with a drive-out to the nearest dock; these three had not.

THE FIX. With stream.no_work_exit on, those three places send the fleet out at
the nearest dock, as the delivery path does, without counting a delivery. A
fleet that cannot reach any dock still parks, and is counted.

WHAT THIS CHECKS (RULES driver, synthetic floor, 24 fleets, the stream):
  1. Switch on, a fleet's order taken and nothing left: it is NOT frozen, its
     target becomes a dock, and it leaves the floor there; no delivery is
     credited to it; the counter says 1.
  2. No dock reachable: it parks as before, and the parked counter says 1.
  3. Switch off: it freezes exactly as before, and no counter moves.

Run:  python test_stream_no_work.py      (ends with ALL PASS; about a minute)
"""
import contextlib
import io
import random
import sys
import types

import numpy as np

import drive_shortest_path as drv

FAIL = []


def check(name, got, want):
    ok = got == want
    print(f"{'PASS' if ok else 'FAIL'}  {name}: got {got!r} want {want!r}")
    if not ok:
        FAIL.append(name)


def make_env(no_work_exit: bool, seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    drv.apply_config({"stream.enabled": True, "stream.no_work_exit": no_work_exit,
                      "errors.enabled": False, "obstacles.enabled": False})
    from core_warehouse import FLOWRRA
    with contextlib.redirect_stdout(io.StringIO()):
        G, grid, miss, kw = drv.synthetic_instance(types.SimpleNamespace(agents=24), seed)
        env = FLOWRRA(G, grid, miss, **kw)
    env.gnn = drv.ShortestPathDriver(env, "never", 0.5, seed)
    return env


def quiet_steps(env, n):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        for _ in range(n):
            env.step(episode_step=1, total_episodes=1)
    return out.getvalue()


def pick_fleet(env):
    """A mobile fleet chasing an order: not leaving, not rescuing, not stopped."""
    for n in env.nodes:
        if (n.id not in env.immobile_nodes and n.id not in env._despawning
                and n.id not in env._pickup_assignment and n.current_goal_id is not None):
            return n
    return None


def take_its_order(env, node, dock_reachable=True):
    """Its order is claimed by someone else, and nothing is left to retarget to."""
    env.claimed_goals.add(node.current_goal_id)
    _orig_retarget = env._retarget
    env._retarget = lambda n, _o=_orig_retarget: (False if n.id == node.id else _o(n))
    if not dock_reachable:
        _orig_exit = env._nearest_exit
        env._nearest_exit = lambda n, _o=_orig_exit: (None if n.id == node.id else _o(n))


def test_switch_on_drives_out():
    env = make_env(True)
    quiet_steps(env, 5)
    x = pick_fleet(env)
    check("found_a_fleet_to_test", x is not None, True)
    if x is None:
        return
    take_its_order(env, x)
    log = quiet_steps(env, 1)
    est = env.get_error_statistics()
    check("on_not_frozen", x.id in env.frozen_nodes, False)
    check("on_heading_out", x.id in env._despawning, True)
    check("on_target_is_its_dock", x.current_goal_id == env._despawning.get(x.id), True)
    check("on_counted", est.get("stream_no_work_exits"), 1)
    check("on_logged", f"Fleet {x.id} has no order left" in log, True)
    # Drive on until it has left the floor; it must never freeze or deliver.
    froze, delivered, left = False, False, False
    for _ in range(200):
        log = quiet_steps(env, 1)
        froze |= x.id in env.frozen_nodes
        delivered |= (f"fleet {x.id} at " in log and "Delivery #" in log)
        if x.id not in [n.id for n in env.nodes]:
            left = True
            break
    check("on_left_the_floor", left, True)
    check("on_never_frozen_on_the_way", froze, False)
    check("on_no_delivery_credited", delivered, False)
    check("on_nothing_parked", env.get_error_statistics().get("stream_parked_no_exit"), 0)


def test_no_dock_parks_and_counts():
    env = make_env(True)
    quiet_steps(env, 5)
    x = pick_fleet(env)
    if x is None:
        check("found_a_fleet_to_test_2", False, True)
        return
    take_its_order(env, x, dock_reachable=False)
    quiet_steps(env, 1)
    est = env.get_error_statistics()
    check("no_dock_parks", x.id in env.frozen_nodes, True)
    check("no_dock_counted", est.get("stream_parked_no_exit"), 1)
    check("no_dock_not_sent_out", est.get("stream_no_work_exits"), 0)


def test_switch_off_unchanged():
    env = make_env(False)
    quiet_steps(env, 5)
    x = pick_fleet(env)
    if x is None:
        check("found_a_fleet_to_test_3", False, True)
        return
    take_its_order(env, x)
    quiet_steps(env, 1)
    est = env.get_error_statistics()
    check("off_freezes_as_before", x.id in env.frozen_nodes, True)
    check("off_not_sent_out", x.id in env._despawning, False)
    check("off_counters_still", (est.get("stream_no_work_exits"), est.get("stream_parked_no_exit")), (0, 0))


if __name__ == "__main__":
    try:
        test_switch_on_drives_out()
        test_no_dock_parks_and_counts()
        test_switch_off_unchanged()
    finally:
        drv.apply_config({})
    print("\n" + ("ALL PASS" if not FAIL else f"FAILURES: {FAIL}"))
    sys.exit(1 if FAIL else 0)
