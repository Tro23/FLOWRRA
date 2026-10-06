"""
test_failure_waves.py -- several fleets stop AT ONCE, at random, a few times an
episode.

THE PROBLEM. The shock benchmark kills 9 vehicles in 3 simultaneous waves of 3,
and that is where the learned policy falls behind the rules. Training never
produced that regime: single random failures, one at a time, ~2.5 an episode
(cold_run23: mean 2.55, range 0-5). A policy cannot learn what it never sees.

THE FIX. core_warehouse.py, _maybe_inject_wave() (config errors.waves). Each
eligible step rolls prob_per_step; a wave stops size_min..size_max fleets under
way at once, through the same path a single failure uses (_stop_fleet). Eligible
while errors are on, fewer than max_waves have fired, min_gap_steps have passed
since the last wave, and the episode's progress (orders delivered or stranded,
the benchmark's own measure) is inside [progress_start, progress_end]. Drawn
from the wave's OWN random stream, seeded per episode.

WHAT THIS CHECKS
  1. Off: nothing happens, and not one draw is taken from the global random
     stream -- so every other random choice in training is exactly as before.
  2. errors_enabled False switches waves off too. That is how the shock
     benchmark keeps its scripted waves the only failures.
  3. A wave stops several fleets at once, all of them under way.
  4. The gap between waves and the per-episode cap hold.
  5. The progress window gates them.
  6. Single failures keep their own budget: wave failures do not use it up.
  7. A wave can catch a rescuer mid-rescue, and that pickup goes back on the
     board (the single-failure path's own fix, now shared).
  8. Schedules differ between episodes and repeat exactly on a rerun.
  9. The counters reach get_error_statistics().

No learning and no env.step(): the wave logic is driven directly on the smoke
test's synthetic warehouse, so this runs in seconds.

Run:  python test_failure_waves.py      (ends with ALL PASS)
"""

import random

from config_warehouse import CONFIG
import core_warehouse
from core_warehouse import FLOWRRA
from test_smoke_integration import build_instance

FAIL = []

BASE_WAVES = {
    "enabled": True, "prob_per_step": 1.0, "max_waves": 3,
    "size_min": 3, "size_max": 3, "progress_start": 0.0, "progress_end": 1.0,
    "min_gap_steps": 40, "seed": 0,
}


def check(name, got, want):
    ok = got == want
    print(f"{'PASS' if ok else 'FAIL'}  {name}: got {got!r} want {want!r}")
    if not ok:
        FAIL.append(name)


def make_env(**wave_overrides):
    """A fresh 24-fleet episode with the given wave settings, every fleet
    eligible (progress filter off), 100 steps in."""
    saved_waves = CONFIG["errors"].get("waves")
    saved_enabled = CONFIG["errors"].get("enabled")
    CONFIG["errors"]["waves"] = {**BASE_WAVES, **wave_overrides}
    CONFIG["errors"]["enabled"] = True
    try:
        G, grid_pos, missions, gdm, goal_pool = build_instance(24)
        env = FLOWRRA(G, grid_pos, missions, mode="training",
                      goal_distance_maps=gdm, shared_pool_mode=True,
                      goal_pool=goal_pool)
    finally:
        CONFIG["errors"]["waves"] = saved_waves
        CONFIG["errors"]["enabled"] = saved_enabled
    env.error_min_progress = 0.0     # fleets have not moved yet in this test
    env.step_count = 100
    return env


# ------------------------------------------------------------------ 1. off
def test_off_touches_nothing():
    env = make_env(enabled=False)
    state = random.getstate()
    for _ in range(50):
        env._maybe_inject_wave()
    check("no_global_random_draws", random.getstate() == state, True)
    check("nobody_stopped", len(env.stopped_nodes), 0)
    check("no_wave_stream_when_off", env._wave_rng is None, True)


# ------------------------------------------------------------------ 2. benchmark gate
def test_errors_off_switches_waves_off():
    env = make_env()
    env.errors_enabled = False       # exactly what the shock benchmark does
    for _ in range(10):
        env._maybe_inject_wave()
    check("no_wave_with_errors_off", env.waves_fired, 0)
    check("nobody_stopped", len(env.stopped_nodes), 0)


# ------------------------------------------------------------------ 3. a wave
def test_a_wave_stops_several_at_once():
    env = make_env()
    under_way = {n.id for n in env._error_candidates()}
    env._maybe_inject_wave()
    check("one_wave", env.waves_fired, 1)
    check("three_at_once", env.wave_failures, 3)
    check("three_stopped", len(env.stopped_nodes), 3)
    check("counted_as_errors", env.total_errors, 3)
    check("all_were_under_way", env.stopped_nodes <= under_way, True)
    check("all_stopped_at_this_step",
          {env._error_step[f] for f in env.stopped_nodes}, {100})


# ------------------------------------------------------------------ 4. gap and cap
def test_gap_and_cap():
    env = make_env()
    env._maybe_inject_wave()
    env._maybe_inject_wave()
    check("not_twice_in_one_step", env.waves_fired, 1)
    env.step_count += 39
    env._maybe_inject_wave()
    check("not_inside_the_gap", env.waves_fired, 1)
    env.step_count += 1
    env._maybe_inject_wave()
    check("fires_once_the_gap_has_passed", env.waves_fired, 2)
    env.step_count += 40
    env._maybe_inject_wave()
    env.step_count += 40
    env._maybe_inject_wave()
    check("capped_at_max_waves", env.waves_fired, 3)
    check("failures_match_waves", env.wave_failures, 9)


# ------------------------------------------------------------------ 5. progress
def test_progress_window_gates_waves():
    early = make_env(progress_start=1.1)
    early._maybe_inject_wave()
    check("not_before_progress_start", early.waves_fired, 0)
    late = make_env(progress_end=-0.1)
    late._maybe_inject_wave()
    check("not_after_progress_end", late.waves_fired, 0)
    check("progress_is_a_share", 0.0 <= late._wave_progress() <= 1.0, True)


# ------------------------------------------------------------------ 6. single budget
def test_singles_keep_their_own_budget():
    env = make_env()
    env._maybe_inject_wave()                  # 3 wave failures
    env.error_prob_per_step = 1.0
    env.error_max_per_episode = 1
    env.error_min_step = 0
    env._maybe_inject_error()
    check("single_still_fires_after_a_wave", env.total_errors, 4)
    env._maybe_inject_error()
    check("single_budget_then_used_up", env.total_errors, 4)


# ------------------------------------------------------------------ 7. rescuers
def test_wave_catches_a_rescuer_mid_rescue():
    env = make_env()
    cands = env._error_candidates()
    rescuer = cands[0].id
    env._pickup_assignment[rescuer] = "some_pickup"
    env.wave_size_min = env.wave_size_max = len(cands)   # hit everyone under way
    env._maybe_inject_wave()
    check("rescuer_counted", env.wave_rescuers_hit, 1)
    check("rescuer_stopped", rescuer in env.stopped_nodes, True)
    check("its_pickup_back_on_the_board", rescuer in env._pickup_assignment, False)


# ------------------------------------------------------------------ 8. randomness
def test_schedules_vary_and_reproduce():
    core_warehouse._WAVE_EPISODES = 0
    a = make_env(prob_per_step=0.5)
    b = make_env(prob_per_step=0.5)
    draws_a = [a._wave_rng.random() for _ in range(5)]
    draws_b = [b._wave_rng.random() for _ in range(5)]
    check("episodes_differ", draws_a != draws_b, True)
    core_warehouse._WAVE_EPISODES = 0
    a2 = make_env(prob_per_step=0.5)
    check("rerun_repeats", [a2._wave_rng.random() for _ in range(5)], draws_a)
    core_warehouse._WAVE_EPISODES = 0
    same_count = make_env(prob_per_step=0.5, seed=1)
    check("seed_changes_the_schedule",
          [same_count._wave_rng.random() for _ in range(5)] != draws_a, True)


# ------------------------------------------------------------------ 9. stats
def test_counters_reach_the_statistics():
    env = make_env()
    env._maybe_inject_wave()
    est = env.get_error_statistics()
    check("wave_fired", est["wave_fired"], 1)
    check("wave_failures", est["wave_failures"], 3)
    check("wave_rescuers_hit", est["wave_rescuers_hit"], 0)
    check("errors_injected_includes_waves", est["errors_injected"], 3)


if __name__ == "__main__":
    for fn in list(globals().values()):
        if callable(fn) and getattr(fn, "__name__", "").startswith("test_"):
            print(f"\n--- {fn.__name__} ---")
            fn()
    print("\n" + ("ALL PASS" if not FAIL else f"FAILURES: {FAIL}"))