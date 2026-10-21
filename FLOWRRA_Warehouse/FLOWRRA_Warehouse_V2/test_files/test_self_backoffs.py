"""
test_self_backoffs.py -- step 3a measurement: self-initiated back-offs.

A back-off is SELF-initiated when a fleet inside a conflict, moving on its OWN
policy (no RULES order, no ladder retrace, no hold), ends the step further from
its goal. As the reflex forms -- the Learner copying RULES, the rung costs
pricing escalation -- these should rise while RULES-ordered retreats fall.

WHAT THIS CHECKS
  1. A conflicted fleet on its own policy that moves away from its goal counts.
  2. One that moves closer does not.
  3. A fleet under a RULES order or a retrace order is not watched.
  4. A goal change between steps is not a back-off.
  5. A fleet outside every conflict is not watched.

Run:  python test_self_backoffs.py      (ends with ALL PASS)
"""
from config_warehouse import CONFIG
from core_warehouse import FLOWRRA
from test_smoke_integration import build_instance

FAIL = []


def check(name, got, want):
    ok = got == want
    print(f"{'PASS' if ok else 'FAIL'}  {name}: got {got!r} want {want!r}")
    if not ok:
        FAIL.append(name)


def make_env():
    saved = dict(CONFIG["conflict"].get("ladder", {}) or {})
    CONFIG["conflict"]["ladder"] = {"enabled": True, "shadow": False, "retrace": True,
                                    "grace_checks": 1, "trail_length": 8,
                                    "aging_reset_steps": 3, "l3_stuck_steps": 30}
    try:
        G, gp, missions, gdm, pool = build_instance(24)
        env = FLOWRRA(G, gp, missions, mode="training", goal_distance_maps=gdm,
                      shared_pool_mode=True, goal_pool=pool)
    finally:
        CONFIG["conflict"]["ladder"] = saved
    env._rule_actions = {}
    return env


def movers(env, n):
    return [x for x in env.nodes if x.id not in env.immobile_nodes][:n]


def set_hops(node, h):
    node.get_graph_distance_to_goal = lambda: h


def test_counts_only_own_policy_moves_away():
    env = make_env()
    a, b, c, d, e = movers(env, 5)
    for x in (a, b, c, d, e):
        set_hops(x, 10.0)
    env.loop.warning_nodes = {a.id, b.id, c.id, d.id}
    env.loop.deadlocked_nodes = set()
    env._rule_actions = {c.id: 0}                        # RULES decided c
    env._retrace[d.id] = {"target": None, "others": set(), "since": 0,
                          "budget": 99, "arrived": False}  # d is retreating
    env._ladder_watch_backoffs()
    check("watched_own_policy_conflicted_only", sorted(env._backoff_watch),
          sorted([a.id, b.id]))
    set_hops(a, 11.0)                                    # a stepped away
    set_hops(b, 9.0)                                     # b moved closer
    env._rule_actions = {}
    env._ladder_watch_backoffs()
    check("one_self_backoff", env.ladder_self_backoffs, 1)


def test_goal_change_is_not_a_backoff():
    env = make_env()
    (a,) = movers(env, 1)
    set_hops(a, 5.0)
    env.loop.warning_nodes = {a.id}
    env.loop.deadlocked_nodes = set()
    env._ladder_watch_backoffs()
    a.current_goal_id = "a_new_goal"
    set_hops(a, 20.0)
    env._ladder_watch_backoffs()
    check("new_goal_not_counted", env.ladder_self_backoffs, 0)


def test_stats_reported():
    env = make_env()
    est = env.get_error_statistics()
    check("in_the_csv", ("ladder_self_backoffs" in est, "ladder_backoff_watched" in est),
          (True, True))


if __name__ == "__main__":
    for fn in list(globals().values()):
        if callable(fn) and getattr(fn, "__name__", "").startswith("test_"):
            print(f"\n--- {fn.__name__} ---")
            fn()
    print("\n" + ("ALL PASS" if not FAIL else f"FAILURES: {FAIL}"))