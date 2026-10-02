"""
test_measurement_fixes.py -- CONFLICT_DESIGN.md step 1, the measurement fixes.

  1. RISK STEPS ARE COUNTED EVEN WHEN THE HEAD DECLINES. risk_steps_acted /
     risk_steps read 100% by construction because a declined risk step returned
     before it was counted.
  2. THE PREVENTION WINDOW resolves each event exactly as documented: judged at
     ages 1..k (never age 0), "collided" if any watched fleet collided in that
     window, "clear" otherwise, "pending" if the episode ends first.
  3. THE RANDOMISED TEST's bookkeeping (drive_shortest_path.py --recovery coin):
     every opportunity opens exactly one event, in the arm the coin chose.
  4. EFFICIENCY: each dock order's cycle is dock -> goal -> nearest dock, checked
     against networkx directly, and the ideal is fleets x steps x speed / cycle.

Plus the counters the pre-registered criteria are stated in: repeat offences
must equal the "REPEAT OFFENCE" lines actually printed, and holds must be
consistent with the hold bounds. And the driver itself: deterministic per seed,
and it refuses a config key that does not exist.

Real episodes on the smoke-test warehouse, driven by the shortest-path driver
(no network, no learning), so every assertion is about the environment.
"""

import contextlib
import io
import math
import re
import sys
import types

import numpy as np
import networkx as nx

from config_warehouse import CONFIG
import drive_shortest_path as drv

FAIL = []


def check(name, got, want):
    ok = got == want
    print(f"{'PASS' if ok else 'FAIL'}  {name}: got {got!r} want {want!r}")
    if not ok:
        FAIL.append(name)


def args(**kw):
    a = types.SimpleNamespace(synthetic=True, agents=24, steps=60, seeds="0",
                              recovery="never", coin_p=0.5, order_window="final",
                              verbose=False, map="synthetic", maps_dir="", scens_dir="")
    for k, v in kw.items():
        setattr(a, k, v)
    return a


def build_env(stream: bool, seed: int = 0, agents: int = 24):
    """A real FLOWRRA on the smoke warehouse, driven by the shortest-path driver."""
    drv.apply_config({"stream.enabled": stream})
    from core_warehouse import FLOWRRA
    with contextlib.redirect_stdout(io.StringIO()):
        G, grid, miss, kw = drv.synthetic_instance(args(agents=agents), seed)
        env = FLOWRRA(G, grid, miss, **kw)
    return env, G


def drive(env, steps, recovery="never", seed=0, capture=False):
    d = drv.ShortestPathDriver(env, recovery, 0.5, seed)
    env.gnn = d
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        for _ in range(steps):
            env.step(episode_step=1, total_episodes=1)
    return d, buf.getvalue()


# ============================================================== fix 1
def test_risk_steps_counted_when_the_head_declines():
    env, _ = build_env(stream=False)
    d, _ = drive(env, 60, recovery="never")
    est = env.get_error_statistics()
    print(f"      risk steps {est['risk_steps']}  acted {est['risk_steps_acted']}  "
          f"warning-only {est['warning_steps']}")
    # The old code counted a risk step only when the head acted, so with a
    # head that never acts it reported 0. The episode is full of warnings.
    check("declined_risk_steps_are_counted", est["risk_steps"] > 0, True)
    check("warning_steps_counted_too", est["warning_steps"] > 0, True)
    check("nothing_acted", est["risk_steps_acted"], 0)
    check("rate_is_zero_not_100pct", est["intervention_rate"], 0.0)


def test_acted_counts_exactly_the_invocations():
    env, _ = build_env(stream=False)
    d, _ = drive(env, 60, recovery="always")
    est = env.get_error_statistics()
    print(f"      opportunities {d.opportunities}  invoked {d.invoked}  "
          f"risk {est['risk_steps']}  acted {est['risk_steps_acted']}")
    check("acted_equals_driver_invocations", est["risk_steps_acted"], d.invoked)
    check("acted_never_exceeds_risk_steps",
          est["risk_steps"] >= est["risk_steps_acted"], True)
    check("warning_steps_are_the_opportunities", est["warning_steps"], d.opportunities)


# ============================================================== fix 2
def test_window_semantics_by_hand():
    env, _ = build_env(stream=False)
    k = env.prevention_window
    check("window_from_config", k, CONFIG["recovery_policy"]["prevention_window"])

    def at(step, collided=()):
        env.step_count = step
        env._step_deadlocked = set(collided)
        env._watch_resolve()

    env._watch_events, env._watch_log = [], []
    env._watch = {"preempt": {"opened": 0, "collided": 0, "clear": 0,
                              "fleets_watched": 0, "fleets_hit": 0}}
    env.step_count = 10
    env._watch_open("preempt", {"a", "b"})     # a collides at age 3
    env._watch_open("preempt", {"c"})          # never collides
    env._watch_open("preempt", set())          # empty: must not open
    at(10, collided={"a", "c"})                # age 0: must NOT count
    for t in range(11, 10 + k):
        at(t, collided={"a"} if t == 13 else ())
    s = env._watch["preempt"]
    check("nothing_resolves_before_age_k", (s["collided"], s["clear"]), (0, 0))
    at(10 + k)
    check("collided_event_resolved", s["collided"], 1)
    check("clear_event_resolved", s["clear"], 1)
    check("empty_set_not_opened", s["opened"], 2)
    check("fleets_hit_counted", s["fleets_hit"], 1)
    check("fleets_watched_counted", s["fleets_watched"], 3)
    # "c" was in the collision set at age 0 only. Had age 0 been judged, its
    # event would have resolved "collided"; it resolved "clear" instead.
    check("age0_collision_ignored",
          [r[3] for r in env._watch_log if r[2] == 1], [0])

    env.step_count = 50
    env._watch_open("preempt", {"z"})
    at(51)
    st = env._watch_stats()
    check("unfinished_event_pending", st["preempt_pending"], 1)
    check("prevention_rate", st["preempt_prevention_rate"], 0.5)
    check("log_rows", [(r[0], r[2], r[3]) for r in env._watch_log],
          [("preempt", 2, 1), ("preempt", 1, 0)])


def test_every_preemptive_recovery_opens_one_event():
    env, _ = build_env(stream=False)
    d, _ = drive(env, 60, recovery="always")
    est = env.get_error_statistics()
    print(f"      preemptive recoveries {est['recovery_preemptive']}  watched "
          f"{est['preempt_watched']}  clear {est['preempt_clear_k']}  collided "
          f"{est['preempt_collided_k']}  pending {est['preempt_pending']}")
    check("preemptive_recoveries_happened", est["recovery_preemptive"] > 0, True)
    check("one_event_per_preemptive_recovery",
          est["preempt_watched"], est["recovery_preemptive"])
    check("events_accounted_for",
          est["preempt_clear_k"] + est["preempt_collided_k"] + est["preempt_pending"],
          est["preempt_watched"])


# ============================================================== fix 3
def test_coin_bookkeeping():
    a = args(recovery="coin", steps=60)
    row, events = drv.run_one(a, "today", {}, 0)
    print(f"      opportunities {row['opportunities']}  invoked {row['invoked']}  "
          f"events resolved {len(events)}")
    opened = row["opp_invoke_opened"] + row["opp_decline_opened"]
    check("one_event_per_opportunity", opened, row["opportunities"])
    check("invoke_events_are_the_invocations", row["opp_invoke_opened"], row["invoked"])
    check("both_arms_populated",
          row["opp_invoke_opened"] > 0 and row["opp_decline_opened"] > 0, True)
    check("resolved_plus_pending",
          len(events) + row["opp_invoke_pending"] + row["opp_decline_pending"], opened)
    check("preemptive_recoveries_only_when_invoked",
          row["recovery_preemptive"] <= row["invoked"], True)


# ============================================================== fix 4
def test_cycle_matches_networkx():
    env, G = build_env(stream=True)
    docks = list(env.exit_nodes)
    s = env._sstats
    bad = 0
    for dock in docks[:6]:
        before_sum, before_n = s["cycle_sum"], s["cycle_n"]
        g = env._issue_order(dock)
        if g is None:
            continue
        to = nx.shortest_path_length(G, dock, g)
        back = min(nx.shortest_path_length(G, g, e) for e in docks)
        if (s["cycle_sum"] - before_sum, s["cycle_n"] - before_n) != (to + back, 1):
            bad += 1
            print(f"      dock {dock} goal {g}: recorded {s['cycle_sum'] - before_sum}, "
                  f"networkx {to} + {back}")
    check("cycle_is_dock_goal_nearest_dock", bad, 0)


def test_efficiency_formula():
    env, _ = build_env(stream=True)
    check("nan_before_any_dock_order",
          math.isnan(env._stream_efficiency()["stream_efficiency"]), True)
    env._sstats.update(cycle_sum=594, cycle_n=10, deliveries=146)
    env.step_count = 800
    env._stream_fleets = 60
    e = env._stream_efficiency()
    ideal = 60 * 800 * CONFIG["warehouse"]["base_speed"] / 59.4
    print(f"      ideal {e['stream_ideal_deliveries']:.1f}  efficiency {e['stream_efficiency']:.3f}")
    check("ideal_matches_stream_design", round(e["stream_ideal_deliveries"], 6), round(ideal, 6))
    check("efficiency", round(e["stream_efficiency"], 6), round(146 / ideal, 6))
    check("in_stream_stats", "stream_efficiency" in env._stream_stats(), True)


def test_efficiency_in_a_real_stream_episode():
    env, _ = build_env(stream=True)
    drive(env, 60)
    st = env.get_error_statistics()
    print(f"      deliveries {st['stream_deliveries']}  cycle {st['stream_ideal_cycle_hops']:.1f} hops"
          f"  ideal {st['stream_ideal_deliveries']:.1f}  efficiency {st['stream_efficiency']:.2f}")
    check("dock_orders_recorded", env._sstats["cycle_n"] > 0, True)
    check("efficiency_finite", bool(np.isfinite(st["stream_efficiency"])), True)


# ============================================================== criteria counters
def test_repeat_offences_equal_the_printed_lines():
    env, _ = build_env(stream=False)
    d, out = drive(env, 60, recovery="always", capture=True)
    printed = [int(m) for m in re.findall(r"REPEAT OFFENCE #(\d+)", out)]
    est = env.get_error_statistics()
    print(f"      printed {len(printed)}  counted {est['conflict_repeat_offences']}  "
          f"worst {est['conflict_max_pair_repeat']}")
    check("repeats_exercised", len(printed) > 0, True)
    check("count_equals_printed", est["conflict_repeat_offences"], len(printed))
    check("worst_equals_printed", est["conflict_max_pair_repeat"], max(printed or [0]))


def test_holds_are_consistent():
    env, _ = build_env(stream=False)
    drive(env, 60, recovery="always")
    est = env.get_error_statistics()
    h, cap, steps = est["conflict_holds"], est["conflict_holds_at_cap"], est["conflict_hold_steps"]
    lo = h * env.recovery.base_yield_steps
    hi = h * env.recovery.max_yield_steps
    print(f"      holds {h}  at cap {cap}  hold steps {steps} (bounds {lo}..{hi})")
    check("holds_exercised", h > 0, True)
    check("at_cap_not_more_than_holds", cap <= h, True)
    check("hold_steps_within_bounds", lo <= steps <= hi, True)


# ============================================================== proposed vs executed
def test_proposed_versus_executed():
    """
    Among fleets at risk, what the network PROPOSED against what was DONE. The
    shortest-path driver almost never proposes waiting, so: without the rules,
    proposed and executed agree except where recovery holds a fleet; with them,
    the rules account for the extra waits -- and every divergence is attributed.
    """
    RULES = {"conflict.path_warnings": True, "conflict.directional_braking": True,
             "conflict.corridor_entry": True, "conflict.priority": True,
             "conflict.node_aligned_moves": True, "conflict.yield_to_stopped": True}
    got = {}
    for label, sets in (("off", {}), ("rules", RULES)):
        drv.apply_config(dict(sets, **{"stream.enabled": True}))
        from core_warehouse import FLOWRRA
        with contextlib.redirect_stdout(io.StringIO()):
            G, grid, miss, kw = drv.synthetic_instance(args(agents=24), 0)
            env = FLOWRRA(G, grid, miss, **kw)
        drive(env, 60, recovery="never")
        st = env.get_error_statistics()
        got[label] = st
        print(f"      {label:<6} at-risk decisions {st['choice_n']:>4}   wait: proposed "
              f"{st['choice_policy_wait_share']:.0%}  done {st['choice_exec_wait_share']:.0%}   "
              f"overridden by rule {st['choice_over_rule']}, hold {st['choice_over_hold']}, "
              f"other {st['choice_over_other']}")
    for label, st in got.items():
        n = st["choice_n"]
        check(f"{label}_proposed_classes_sum", sum(st[f"choice_policy_{k}"] for k in
              ("wait", "toward", "away", "side")), n)
        check(f"{label}_executed_classes_sum", sum(st[f"choice_exec_{k}"] for k in
              ("wait", "toward", "away", "side")), n)
    check("exercised", got["rules"]["choice_n"] > 0, True)
    check("no_rule_overrides_without_rules", got["off"]["choice_over_rule"], 0)
    check("rules_impose_waits_the_policy_did_not_choose",
          got["rules"]["choice_exec_wait_share"] > got["rules"]["choice_policy_wait_share"], True)
    check("rules_are_credited", got["rules"]["choice_over_rule"] > 0, True)


# ============================================================== the driver
def test_driver_is_deterministic():
    a = args(recovery="coin", steps=40)
    r1, e1 = drv.run_one(a, "x", {}, 3)
    r2, e2 = drv.run_one(a, "x", {}, 3)
    r1.pop("ms_per_step"); r2.pop("ms_per_step")
    check("same_seed_same_row", r1 == r2, True)
    check("same_seed_same_events", e1 == e2, True)


def test_unknown_config_key_is_refused():
    try:
        drv.apply_config({"recovery_policy.not_a_key": 1})
        check("unknown_key_refused", False, True)
    except SystemExit:
        check("unknown_key_refused", True, True)
    drv.apply_config({})


def test_config_restored_in_place():
    section = CONFIG["recovery_policy"]
    drv.apply_config({"recovery_policy.prevention_window": 9})
    check("override_applied", CONFIG["recovery_policy"]["prevention_window"], 9)
    check("section_object_kept", CONFIG["recovery_policy"] is section, True)
    drv.apply_config({})
    check("restored", CONFIG["recovery_policy"]["prevention_window"],
          drv._PRISTINE["recovery_policy"]["prevention_window"])


if __name__ == "__main__":
    for fn in list(globals().values()):
        if callable(fn) and getattr(fn, "__name__", "").startswith("test_"):
            print(f"\n--- {fn.__name__} ---")
            fn()
    drv.apply_config({})
    print("\n" + ("ALL PASS" if not FAIL else f"FAILURES: {FAIL}"))
    sys.exit(1 if FAIL else 0)
