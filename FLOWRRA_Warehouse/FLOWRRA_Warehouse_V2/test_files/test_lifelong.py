"""
test_lifelong.py -- lifelong mode ("next goal on arrival") and the braking switch.

  1. LIFELONG. A fleet that reaches its goal is rewarded as before and takes the
     next goal of ITS OWN sequence at once -- in order, never frozen -- and the
     count and throughput are reported in POGEMA's units (goals per timestep of
     a one-cell-per-step agent). Without sequences, arrival freezes as before.
  2. BRAKING SWITCH. warehouse.braking=False: full speed even with a peer
     close by (as in POGEMA); True (the default): throttled as always.

Real FLOWRRA on the hand-built hub map from test_corridor_rules.py, driven by
the shortest-path driver.
"""

import contextlib
import io
import sys

import numpy as np

import drive_shortest_path as drv
from test_corridor_rules import hubs

FAIL = []


def check(name, got, want):
    ok = got == want
    print(f"{'PASS' if ok else 'FAIL'}  {name}: got {got!r} want {want!r}")
    if not ok:
        FAIL.append(name)


def env_on_hubs(sets, sequences=None, starts=None, L=6, lifelong=True):
    drv.apply_config(dict({"stream.enabled": False, "errors.enabled": False}, **sets))
    from core_warehouse import FLOWRRA
    G, grid = hubs(L)
    by = {v: k for k, v in grid.items()}
    starts = starts or {"A": "h-4_0"}
    sequences = sequences or {"A": [f"h{L + 5}_0"]}
    miss = [{"id": f, "start_node": starts[f], "goal_node": sequences[f][0],
             "start_pos": np.array(by[starts[f]], dtype=np.float32),
             "goal_pos": np.array(by[sequences[f][0]], dtype=np.float32)} for f in starts]
    with contextlib.redirect_stdout(io.StringIO()):
        env = FLOWRRA(G, grid, miss, mode="training",
                      lifelong_goals=(sequences if lifelong else None))
    env.gnn = drv.ShortestPathDriver(env, "never", 0.5, 0)
    return env


def run(env, steps):
    with contextlib.redirect_stdout(io.StringIO()):
        for _ in range(steps):
            env.step(episode_step=1, total_episodes=1)


def test_next_goal_on_arrival_in_order():
    # A shuttles between the two ends of the map, far end first. Long enough
    # never to run out (kiva sequences hold 300 goals per agent).
    seq = ["h11_0", "h-4_0"] * 10
    env = env_on_hubs({}, sequences={"A": seq})
    seen = []
    with contextlib.redirect_stdout(io.StringIO()):
        for _ in range(260):
            env.step(episode_step=1, total_episodes=1)
            g = env.nodes[0].current_goal_id
            if g is not None and (not seen or seen[-1] != g):
                seen.append(g)
    st = env.get_error_statistics()
    print(f"      goals reached {st['lifelong_goals_reached']} in {env.step_count} steps; "
          f"goals taken in turn: {seen}; throughput {st['lifelong_throughput']:.3f}/timestep")
    check("reached_several", st["lifelong_goals_reached"] >= 3, True)
    check("taken_in_sequence_order", seen, seq[1:1 + len(seen)])
    check("never_frozen", len(env.frozen_nodes), 0)
    check("did_not_run_out", st["lifelong_cycled"], 0)
    check("episode_not_over", env.is_episode_over(), False)
    speed = 0.5
    check("throughput_in_pogema_units", round(st["lifelong_throughput"], 6),
          round(st["lifelong_goals_reached"] / (env.step_count * speed), 6))


def test_running_out_cycles_and_is_counted():
    env = env_on_hubs({}, sequences={"A": ["h11_0", "h-4_0"]})
    run(env, 200)
    st = env.get_error_statistics()
    print(f"      2-goal sequence, {st['lifelong_goals_reached']} goals reached, cycled {st['lifelong_cycled']}")
    check("cycled_counted", st["lifelong_cycled"] > 0, True)
    check("still_moving", len(env.frozen_nodes), 0)


def test_without_sequences_arrival_freezes_as_before():
    env = env_on_hubs({}, lifelong=False)
    run(env, 80)
    check("frozen_on_arrival", env.nodes[0].id in env.frozen_nodes, True)
    check("no_lifelong_stats", "lifelong_goals_reached" in env.get_error_statistics(), False)


def closing_pair(braking):
    env = env_on_hubs({"warehouse.braking": braking}, starts={"A": "h2_0", "B": "h4_0"},
                      sequences={"A": ["h11_0"], "B": ["h-4_0"]})
    a, b = env.nodes
    b.current_pos = np.array([3.5, 0.0, 0.0])            # head-on, 1.5 hops apart
    env.proximity.refresh(env.nodes, excluded_ids=env.immobile_nodes)
    pa = a.current_pos.copy()
    run(env, 1)
    return round(float(np.abs(a.current_pos - pa).sum()), 3)


def test_braking_switch():
    on, off = closing_pair(True), closing_pair(False)
    print(f"      moved in one step, peer 1.5 hops ahead: braking on {on}, off {off}")
    check("braking_on_throttles", on < 0.5, True)
    check("braking_off_full_speed", off, 0.5)


if __name__ == "__main__":
    for fn in list(globals().values()):
        if callable(fn) and getattr(fn, "__name__", "").startswith("test_"):
            print(f"\n--- {fn.__name__} ---")
            fn()
    drv.apply_config({})
    print("\n" + ("ALL PASS" if not FAIL else f"FAILURES: {FAIL}"))
    sys.exit(1 if FAIL else 0)
