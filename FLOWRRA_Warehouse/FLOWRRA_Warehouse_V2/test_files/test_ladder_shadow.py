"""
test_ladder_shadow.py -- THE CONFLICT LADDER, step 2a-ii: shadow mode.

WHAT 2a-ii ADDS (config conflict.ladder.shadow; measurement only)
  Each step, the warned path verdicts become a wait-for graph W:
      head_on    a -> b and b -> a   (a 2-cycle: waiting can never solve it)
      blocked    mover -> blocker    (verdicts now record the blocker)
      following  follower -> leader
      contested  same group, no arrow (ordering solves it)
  and the episode counts what the ladder would face: conflict groups by size,
  cycles (2-fleet and multi-fleet), how many sit in single-lane corridors (where
  RULES' back-out already exists), and how long each lasts. A cycle seen at one
  check only would have cleared in the grace check; one seen at 2+ checks would
  have needed a retreat.

Also in 2a-ii: under the ladder the learned recovery head is frozen
(test_recovery_frozen.py, needs PyTorch).

WHAT THIS CHECKS
  1. Off: nothing is counted.
  2. A head-on pair is one 2-fleet group and one 2-cycle.
  3. A chain (following + blocked) is one 3-fleet group with no cycle.
  4. A ring of three is one multi-fleet cycle.
  5. Unwarned and sequential pairs are ignored.
  6. Persistence: a cycle lasting one check clears in grace; one lasting three
     needs a retreat; one still alive at the end is counted too.
  7. The blocker is recorded by the real classifier.

Run:  python test_ladder_shadow.py      (ends with ALL PASS)
"""

from config_warehouse import CONFIG
from conflict_warehouse import Verdict
from core_warehouse import FLOWRRA
from test_smoke_integration import build_instance

FAIL = []


def check(name, got, want):
    ok = got == want
    print(f"{'PASS' if ok else 'FAIL'}  {name}: got {got!r} want {want!r}")
    if not ok:
        FAIL.append(name)


def make_env(shadow: bool):
    saved = dict(CONFIG["conflict"].get("ladder", {}) or {})
    CONFIG["conflict"]["ladder"] = {"enabled": True, "shadow": shadow,
                                    "aging_reset_steps": 3, "l3_stuck_steps": 30}
    try:
        G, grid_pos, missions, gdm, goal_pool = build_instance(24)
        env = FLOWRRA(G, grid_pos, missions, mode="training",
                      goal_distance_maps=gdm, shared_pool_mode=True,
                      goal_pool=goal_pool)
    finally:
        CONFIG["conflict"]["ladder"] = saved
    return env


def V(a, b, kind, warn=True, follower=None, blocker=None):
    return Verdict(a, b, 2.0, kind, warn, "kind", None, follower, blocker)


def stats(env):
    return {k.replace("ladder_shadow_", ""): v
            for k, v in env.get_error_statistics().items()
            if k.startswith("ladder_shadow_")}


# ------------------------------------------------------------------ 1. off
def test_off_counts_nothing():
    env = make_env(False)
    check("shadow_flag_off", env.ladder_shadow, False)
    check("no_groups", stats(env)["groups"], 0)


# ------------------------------------------------------------------ 2. head-on
def test_head_on_is_a_2_cycle():
    env = make_env(True)
    env._ladder_shadow_step([V("f0", "f1", "head_on")])
    s = stats(env)
    check("one_group", s["groups"], 1)
    check("two_fleet_group", s["groups_2"], 1)
    check("one_2_cycle", s["cycles_2"], 1)
    check("no_multi_cycle", s["cycles_3plus"], 0)


# ------------------------------------------------------------------ 3. chain
def test_chain_has_no_cycle():
    env = make_env(True)
    env._ladder_shadow_step([V("f0", "f1", "following", follower="f0"),
                             V("f1", "f2", "blocked", blocker="f2")])
    s = stats(env)
    check("one_group_of_three", (s["groups"], s["groups_3plus"], s["max_group"]), (1, 1, 3))
    check("no_cycle_in_a_chain", s["cycles_seen"], 0)


# ------------------------------------------------------------------ 4. ring of three
def test_ring_of_three_is_a_multi_cycle():
    env = make_env(True)
    env._ladder_shadow_step([V("f0", "f1", "following", follower="f0"),
                             V("f1", "f2", "following", follower="f1"),
                             V("f2", "f0", "blocked", blocker="f0")])
    s = stats(env)
    check("one_multi_cycle", s["cycles_3plus"], 1)
    check("cycle_size_three", s["max_cycle"], 3)


# ------------------------------------------------------------------ 5. ignored
def test_unwarned_and_sequential_ignored():
    env = make_env(True)
    env._ladder_shadow_step([V("f0", "f1", "head_on", warn=False),
                             V("f2", "f3", "sequential")])
    s = stats(env)
    check("nothing_counted", (s["groups"], s["cycles_seen"]), (0, 0))


# ------------------------------------------------------------------ 6. persistence
def test_persistence():
    env = make_env(True)
    ring = [V("f0", "f1", "head_on")]
    env._ladder_shadow_step(ring)                 # one check
    env._ladder_shadow_step([])                   # gone
    check("cleared_in_grace", stats(env)["cycles_cleared_in_grace"], 1)

    for _ in range(3):
        env._ladder_shadow_step([V("f2", "f3", "head_on")])
    env._ladder_shadow_step([])
    s = stats(env)
    check("needed_a_retreat", s["cycles_needing_retreat"], 1)
    check("longest_cycle_three_checks", s["max_cycle_persist"], 3)

    env._ladder_shadow_step([V("f4", "f5", "head_on")])
    env._ladder_shadow_step([V("f4", "f5", "head_on")])
    check("still_alive_at_the_end_counted", stats(env)["cycles_needing_retreat"], 2)
    check("cycles_seen_total", stats(env)["cycles_seen"], 3)


# ------------------------------------------------------------------ 7. the real classifier
def test_blocker_recorded_by_classifier():
    env = make_env(True)
    a, b = env.nodes[0], env.nodes[1]
    pc = env.conflicts
    ob = pc.occupied(b)
    route_into_b = [set(ob)]                       # a's next cell is b's cell
    kind, meet, fol = pc.classify(route_into_b, [], pc.occupied(a), ob, False, True)
    check("classifier_says_blocked", kind, "blocked")
    check("meet_is_on_the_blocker", meet in ob, True)


if __name__ == "__main__":
    for fn in list(globals().values()):
        if callable(fn) and getattr(fn, "__name__", "").startswith("test_"):
            print(f"\n--- {fn.__name__} ---")
            fn()
    print("\n" + ("ALL PASS" if not FAIL else f"FAILURES: {FAIL}"))