"""
test_keyed_draws.py -- the order-leak fix (build log entry 43, step 3).

THE LEAK. Single failures drew from Python's shared `random`, waves and people
from their own sequential streams. Each draw therefore depended on every draw
before it: exploration moves, the order robots are listed in, how many steps a
wave was eligible, where people could walk. Two arms on the "same day" drift
onto different failure days as soon as their robots differ, which adds noise to
every paired comparison in the full condition.

THE FIX. config keyed_draws.seed. None (the default, and training): exactly the
old draws. An int: each draw keyed by (seed, what is drawn, the step, and the
robot, debris spot or person), hashed with crc32 (the same in every process).
Victims are the candidates with the lowest key this step, so list order does
not matter. deploy_eval sets the seed per instance (--keyed-draws on).

WHAT THIS CHECKS
  Off (seed None):
    1. A single failure uses the global stream exactly as before.
    2. A wave uses its own stream exactly as before.
    3. People and debris draw from the field's own stream (the same object).
  On (seed an int):
    4. A single failure's victim ignores the global stream and the list order.
    5. Off, the same change of history DOES change the victim: the leak was real.
    6. A wave fires and picks the same victims whatever came before.
    7. If one victim is not a candidate in another arm, the rest are still hit.
    8. Different seeds give different days; the same seed repeats exactly.
    9. A keyed roll is uniform: the firing rate matches prob_per_step.
   10. People start and walk the same whatever the field's own stream did.
   11. Cleanup mode: debris appears at the same steps and cells.

No learning and no env.step(): the draws are driven directly on the smoke
test's synthetic warehouse, so this runs in seconds.

Run:  python test_keyed_draws.py      (ends with ALL PASS)
"""

import random
import zlib

from config_warehouse import CONFIG
from core_warehouse import FLOWRRA
from obstacles_warehouse import ObstacleField
from test_smoke_integration import build_instance

FAIL = []

WAVES = {"enabled": True, "prob_per_step": 1.0, "max_waves": 3, "size_min": 3,
         "size_max": 3, "progress_start": 0.0, "progress_end": 1.0,
         "min_gap_steps": 40, "seed": 0}


def check(name, got, want):
    ok = got == want
    print(f"{'PASS' if ok else 'FAIL'}  {name}: got {got!r} want {want!r}")
    if not ok:
        FAIL.append(name)


def make_env(keyed, waves=False, reverse=False):
    """A fresh 24-fleet episode, 100 steps in, every fleet eligible, a single
    failure certain when rolled. keyed: the keyed_draws seed (None = off)."""
    saved = (CONFIG.get("keyed_draws"), CONFIG["errors"].get("waves"),
             CONFIG["errors"].get("enabled"))
    CONFIG["keyed_draws"] = {"seed": keyed}
    CONFIG["errors"]["waves"] = dict(WAVES) if waves else {"enabled": False}
    CONFIG["errors"]["enabled"] = True
    try:
        G, grid_pos, missions, gdm, goal_pool = build_instance(24)
        env = FLOWRRA(G, grid_pos, missions, mode="training",
                      goal_distance_maps=gdm, shared_pool_mode=True, goal_pool=goal_pool)
    finally:
        CONFIG["keyed_draws"], CONFIG["errors"]["waves"], CONFIG["errors"]["enabled"] = saved
    env.error_min_progress = 0.0
    env.error_prob_per_step = 1.0
    env.error_min_step = 0
    env.error_max_step_fraction = 1.0
    env.error_max_per_episode = 1000
    env.max_steps_per_episode = 10_000
    env.step_count = 100
    if reverse:
        env.nodes = list(reversed(env.nodes))
    return env


def single_victim(env):
    before = set(env.stopped_nodes)
    env._maybe_inject_error()
    return sorted(set(env.stopped_nodes) - before)


def wave_victims(env):
    before = set(env.stopped_nodes)
    env._maybe_inject_wave()
    return sorted(set(env.stopped_nodes) - before)


# ------------------------------------------------------------------ off: unchanged
def test_off_single_failure_unchanged():
    env = make_env(None)
    cands = env._error_candidates()
    random.seed(5)
    random.random()
    expected = random.choice(cands).id
    random.seed(5)
    check("1_off_single_failure_as_before", single_victim(env), [expected])


def test_off_wave_unchanged():
    env = make_env(None, waves=True)
    cands = env._error_candidates()
    replica = random.Random()
    replica.setstate(env._wave_rng.getstate())
    replica.random()
    k = min(len(cands), replica.randint(3, 3))
    expected = sorted(n.id for n in replica.sample(cands, k))
    check("2_off_wave_as_before", wave_victims(env), expected)


def test_off_people_same_stream():
    G, grid_pos, *_ = build_instance(24)
    fld = ObstacleField(G, grid_pos, n_humans=2, seed=3)
    check("3_off_people_use_their_own_stream", fld._r("move", 0) is fld.rng, True)


# ------------------------------------------------------------------ on: keyed
def test_single_failure_ignores_history():
    random.seed(1)
    a = single_victim(make_env(77))
    random.seed(999)
    for _ in range(1000):
        random.random()
    b = single_victim(make_env(77, reverse=True))
    check("4_keyed_victim_ignores_history_and_order", a, b)
    check("4_one_victim", len(a), 1)
    key = f"77|error_victim|100|{a[0]}"
    env = make_env(77)
    check("4_rank_is_crc32_of_the_key", env._keyed_rank("error_victim", a[0]),
          zlib.crc32(key.encode()))


def test_leak_was_real():
    random.seed(1)
    a = single_victim(make_env(None))
    random.seed(999)
    for _ in range(1000):
        random.random()
    b = single_victim(make_env(None))
    check("5_off_history_changes_the_victim", a != b, True)


def test_wave_ignores_history():
    a = wave_victims(make_env(77, waves=True))
    env = make_env(77, waves=True, reverse=True)
    for _ in range(100):
        env._wave_rng.random()             # a different history of the old stream
    random.seed(4242)
    b = wave_victims(env)
    check("6_keyed_wave_same_victims", a, b)
    check("6_three_at_once", len(a), 3)


def test_missing_victim_spares_the_rest():
    a = wave_victims(make_env(77, waves=True))
    env = make_env(77, waves=True)
    gone = a[0]
    for n in env.nodes:                    # not a candidate in this arm: idle
        if n.id == gone:
            n.current_goal_id = None
    b = wave_victims(env)
    check("7_rest_still_hit", set(a) - {gone} <= set(b), True)
    check("7_still_three", len(b), 3)
    check("7_absent_robot_spared", gone in b, False)


def test_seeds_differ_and_repeat():
    days = {}
    for seed in (77, 78):
        env = make_env(seed)
        hit = []
        for step in range(100, 110):
            env.step_count = step
            hit += single_victim(env)
        days[seed] = hit
    again = make_env(77)
    hit = []
    for step in range(100, 110):
        again.step_count = step
        hit += single_victim(again)
    check("8_different_seeds_different_days", days[77] != days[78], True)
    check("8_same_seed_repeats", hit, days[77])


def test_keyed_roll_is_uniform():
    env = make_env(77)
    p = 0.3
    fired = 0
    n = 4000
    for step in range(1, n + 1):
        env.step_count = step
        fired += env._keyed_rng("error").random() < p
    rate = fired / n
    check("9_rate_matches_prob", abs(rate - p) < 0.03, True)
    print(f"      (keyed firing rate {rate:.3f} against {p})")


def test_people_ignore_history():
    G, grid_pos, *_ = build_instance(24)
    a = ObstacleField(G, grid_pos, n_humans=3, n_debris=2, human_move_period=1,
                      seed=1, keyed_seed=9)
    b = ObstacleField(G, grid_pos, n_humans=3, n_debris=2, human_move_period=1,
                      seed=2, keyed_seed=9)
    a.reset()
    b.reset()
    same_start = (a.humans, a.debris) == (b.humans, b.debris)
    for _ in range(50):
        a.step()
        b.step()
    check("10_keyed_people_start_the_same", same_start, True)
    check("10_keyed_people_walk_the_same", a.humans, b.humans)
    c = ObstacleField(G, grid_pos, n_humans=3, n_debris=2, human_move_period=1, seed=1)
    d = ObstacleField(G, grid_pos, n_humans=3, n_debris=2, human_move_period=1, seed=2)
    c.reset()
    d.reset()
    for _ in range(50):
        c.step()
        d.step()
    check("10_off_different_streams_differ", c.humans != d.humans, True)


def test_cleanup_debris_ignores_history():
    G, grid_pos, *_ = build_instance(24)
    kw = dict(n_humans=0, n_debris=0, mode="cleanup", debris_interval=20, max_debris=3)
    a = ObstacleField(G, grid_pos, seed=1, keyed_seed=9, **kw)
    b = ObstacleField(G, grid_pos, seed=2, keyed_seed=9, **kw)
    a.reset()
    b.reset()
    trace_a, trace_b = [], []
    for _ in range(300):
        a.step(fleet_cells=set())
        b.step(fleet_cells=set())
        trace_a.append(tuple(sorted(a.c_debris)))
        trace_b.append(tuple(sorted(b.c_debris)))
    check("11_keyed_debris_same_steps_and_cells", trace_a == trace_b, True)
    check("11_debris_did_appear", a.statistics().get("debris_spawned", 0) > 0, True)


if __name__ == "__main__":
    test_off_single_failure_unchanged()
    test_off_wave_unchanged()
    test_off_people_same_stream()
    test_single_failure_ignores_history()
    test_leak_was_real()
    test_wave_ignores_history()
    test_missing_victim_spares_the_rest()
    test_seeds_differ_and_repeat()
    test_keyed_roll_is_uniform()
    test_people_ignore_history()
    test_cleanup_debris_ignores_history()
    print("\nALL PASS" if not FAIL else f"\nFAILED: {FAIL}")
