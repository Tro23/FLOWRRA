"""
test_ladder_authority.py -- THE CONFLICT LADDER, step 2a-i: one authority.

WHAT 2a-i CHANGES (config conflict.ladder.enabled; off = exactly the old code)
  * The learned recovery head is OFF. Its values drifted with or without the
    teacher (pooling_run1: -0.69 by ep 60; teacher_run1: +16.7 on costs-only
    rewards). L3 is a fixed rule instead: a warned pair lasting l3_stuck_steps
    with neither fleet closer to its goal gets separated -- and only that pair,
    past the preemptive filters, which exist for fleets still mid-pass.
  * A recovery hold no longer silences RULES (the `and not _held` lock-out).
  * Aging resets per conflict: after aging_reset_steps steps outside every
    warned pair, a fleet's waiting count returns to 0.

WHAT THIS CHECKS
  1. Off: the learned head is still asked, and a needless ask is still charged.
  2. On: the learned head is never asked; nothing at risk means nothing happens.
  3. The stuck detector: fires after l3_stuck_steps without progress, not before,
     never when a fleet got closer, and restarts its window after firing.
  4. L3 separates ONLY the stuck fleets, and goes past the held-pair filter.
  5. Aging resets after the calm streak, and never while the fleet is in conflict.
The lock-out itself sits inside env.step(); the episode counter
ladder_rules_while_held shows it working in the smoke run.

Run:  python test_ladder_authority.py      (ends with ALL PASS)
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


class AlwaysAsks:
    """A learned recovery head that always asks for a spatial collapse."""
    def __init__(self):
        self.asked = 0

    def choose_recovery(self):
        self.asked += 1
        return 1


def make_env(ladder_on: bool, stuck_steps: int = 30):
    saved = dict(CONFIG["conflict"].get("ladder", {}) or {})
    CONFIG["conflict"]["ladder"] = {"enabled": ladder_on, "aging_reset_steps": 3,
                                    "l3_stuck_steps": stuck_steps}
    try:
        G, grid_pos, missions, gdm, goal_pool = build_instance(24)
        env = FLOWRRA(G, grid_pos, missions, mode="training",
                      goal_distance_maps=gdm, shared_pool_mode=True,
                      goal_pool=goal_pool)
    finally:
        CONFIG["conflict"]["ladder"] = saved
    return env


def two_moving(env):
    ids = [n.id for n in env.nodes if n.id not in env.immobile_nodes]
    return ids[0], ids[1]


# ------------------------------------------------------------------ 1. off
def test_off_still_asks_the_learned_head():
    env = make_env(False)
    head = AlwaysAsks()
    env.gnn = head
    env._step_warning, env._step_deadlocked = set(), set()
    before = env.recovery_wasted
    env._policy_recovery_step()
    check("learned_head_asked", head.asked, 1)
    check("needless_ask_charged_as_before", env.recovery_wasted - before, 1)


# ------------------------------------------------------------------ 2. on
def test_on_never_asks_the_learned_head():
    env = make_env(True)
    head = AlwaysAsks()
    env.gnn = head
    env._step_warning, env._step_deadlocked = set(), set()
    env.loop.warning_pairs = []
    before_w, before_i = env.recovery_wasted, env.recovery_invocations
    env._policy_recovery_step()
    check("learned_head_not_asked", head.asked, 0)
    check("no_needless_ask", env.recovery_wasted - before_w, 0)
    check("no_invocation", env.recovery_invocations - before_i, 0)
    check("mode_recorded_as_none", env.last_recovery_mode, 0)


# ------------------------------------------------------------------ 3. stuck detector
def test_stuck_detector():
    env = make_env(True, stuck_steps=5)
    a, b = two_moving(env)
    env.loop.warning_pairs = [(a, b, 2.0)]
    env.step_count = 100
    check("first_sight_starts_window", env._ladder_stuck(), set())
    env.step_count = 104
    check("not_before_the_window", env._ladder_stuck(), set())
    env.step_count = 105
    check("fires_after_window_without_progress", env._ladder_stuck(), {a, b})
    env.step_count = 106
    check("window_restarts_after_firing", env._ladder_stuck(), set())

    # progress: fleet a gets closer during the window
    env2 = make_env(True, stuck_steps=5)
    a2, b2 = two_moving(env2)
    env2.loop.warning_pairs = [(a2, b2, 2.0)]
    env2.step_count = 100
    env2._ladder_stuck()
    node_a = next(n for n in env2.nodes if n.id == a2)
    h0 = node_a.get_graph_distance_to_goal()
    node_a.get_graph_distance_to_goal = lambda: h0 - 1
    env2.step_count = 105
    check("progress_means_not_stuck", env2._ladder_stuck(), set())

    # a pair that stops being warned is forgotten
    env.loop.warning_pairs = []
    env._ladder_stuck()
    check("cleared_pair_forgotten", len(env._ladder_pairs), 0)


# ------------------------------------------------------------------ 4. L3 acts on the stuck pair only
def test_l3_separates_only_the_stuck_pair():
    env = make_env(True)
    ids = [n.id for n in env.nodes if n.id not in env.immobile_nodes][:4]
    a, b, c, d = ids
    seen = {}

    def capture(**kw):
        seen["dead"] = set(kw["deadlocked_nodes"])
        return {}

    env.recovery.collapse_and_reinitialize = capture
    env._drop_handled_pairs = lambda dead: set()       # the held-pair filter drops all
    env.preempt_skip_held_pairs = True
    env._step_deadlocked = set()
    env._step_warning = {a, b, c, d}

    env._ladder_l3_set = {a, b}
    env._run_recovery(forced=False)
    check("only_the_stuck_pair_separated", seen.get("dead"), {a, b})
    check("l3_set_consumed", env._ladder_l3_set, None)

    seen.clear()
    env._step_warning = {a, b, c, d}
    out = env._run_recovery(forced=False)
    # Without L3 the preemptive filters still apply -- here the convoy filter,
    # which runs first, already skips it. With L3 above, BOTH filters were
    # passed: the collapse ran on exactly the stuck pair.
    check("without_l3_the_filters_still_apply",
          out.get("reinit_from") in ("skipped_convoy", "skipped_held_pair"), True)
    check("nothing_separated_without_l3", seen.get("dead"), None)


# ------------------------------------------------------------------ 5. per-conflict aging
def test_aging_resets_per_conflict():
    env = make_env(True)
    rules = env.rules
    check("rules_present", rules is not None, True)
    check("reset_steps_read", rules.aging_reset_steps, 3)
    a, b = two_moving(env)
    node_ids = [n.id for n in env.nodes]
    idle = [0] * len(node_ids)

    rules.waited[a] = 5
    env.loop.warning_nodes = {a}            # a stays in conflict
    env.loop.deadlocked_nodes = set()
    for _ in range(4):
        rules.plan(idle, node_ids)
    check("no_reset_while_in_conflict", rules.waited.get(a, 0) >= 5, True)

    env.loop.warning_nodes = set()          # a is calm from here on
    before = rules.aging_resets
    for _ in range(3):
        rules.plan(idle, node_ids)
    check("reset_after_calm_streak", rules.aging_resets - before >= 1, True)
    check("waiting_count_back_near_zero", rules.waited.get(a, 0) <= 1, True)

    off = make_env(False)
    check("off_keeps_delivery_only_rule", off.rules.aging_reset_steps, 0)


if __name__ == "__main__":
    for fn in list(globals().values()):
        if callable(fn) and getattr(fn, "__name__", "").startswith("test_"):
            print(f"\n--- {fn.__name__} ---")
            fn()
    print("\n" + ("ALL PASS" if not FAIL else f"FAILURES: {FAIL}"))