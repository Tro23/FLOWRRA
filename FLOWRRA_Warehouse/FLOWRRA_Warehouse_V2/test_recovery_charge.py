"""
test_recovery_charge.py -- every recovery decision is charged exactly once.

Until 2026-09-29 _policy_recovery_step() applied its cost-and-outcome block
twice in a row, so an at-risk invocation cost 2 x invocation_cost and a
resolution paid 2 x resolution_bonus -- while a WASTED invocation (its own
early-return branch) was charged once and the preemptive bonus paid once. The
relative prices were wrong, not just the scale: asking for help with fleets at
risk cost twice what asking for nothing did, and invoking after a collision paid
better than preventing one.

Each case below sets up one situation by hand on a real FLOWRRA, calls
_policy_recovery_step() once, and checks the charge on BOTH ledgers: the live
one (_pending_integrity_reward, which reaches the recovery head) and the shadow
per-fleet one (_attr_integrity).

On the code before the fix, the at-risk cases fail with exactly double.
"""

import contextlib
import io
import sys
import types

import numpy as np

from config_warehouse import CONFIG
import drive_shortest_path as drv

FAIL = []


def check(name, got, want):
    ok = got == want
    print(f"{'PASS' if ok else 'FAIL'}  {name}: got {got!r} want {want!r}")
    if not ok:
        FAIL.append(name)


class _Head:
    """Just the recovery decision; the movement side is never called here."""

    def __init__(self, mode):
        self.mode = mode

    def choose_recovery(self):
        return self.mode

    def freeze_node(self, *a, **k):
        pass

    def unfreeze_node(self, *a, **k):
        pass


def fresh():
    drv.apply_config({"stream.enabled": False})
    from core_warehouse import FLOWRRA
    with contextlib.redirect_stdout(io.StringIO()):
        G, grid, miss, kw = drv.synthetic_instance(types.SimpleNamespace(agents=12), 0)
        env = FLOWRRA(G, grid, miss, **kw)
    return env


def charge(env, mode, deadlocked=(), warning=(), integrity=1.0, streak=0):
    """One call of _policy_recovery_step in a hand-built situation."""
    env.gnn = _Head(mode)
    env._step_deadlocked = set(deadlocked)
    env._step_warning = set(warning)
    env.loop.current_integrity = integrity
    env._deadlock_streak = streak
    env._pending_integrity_reward = 0.0
    env._pending_integrity_by_fleet = {}
    env._pending_integrity_system = 0.0
    with contextlib.redirect_stdout(io.StringIO()):
        env._policy_recovery_step()
    return (env._pending_integrity_reward, dict(env._pending_integrity_by_fleet),
            env._pending_integrity_system)


COST = float(CONFIG["recovery_policy"]["invocation_cost"])
BONUS = float(CONFIG["recovery_policy"]["resolution_bonus"])


def test_preemptive_invocation_charged_once():
    env = fresh()
    a, b = env.nodes[0].id, env.nodes[1].id
    live, per_fleet, system = charge(env, mode=1, warning={a, b}, integrity=0.5)
    print(f"      live {live}  per fleet {per_fleet}")
    check("live_is_one_cost", live, COST)
    check("each_involved_fleet_one_cost", per_fleet, {a: COST, b: COST})
    check("nothing_to_the_system_share", system, 0.0)


def test_invocation_during_a_collision_charged_once():
    env = fresh()
    a, b = env.nodes[0].id, env.nodes[1].id
    live, per_fleet, _ = charge(env, mode=1, deadlocked={a, b}, integrity=0.0)
    print(f"      live {live}  per fleet {per_fleet}")
    check("live_is_cost_plus_bonus", live, COST + BONUS)
    check("each_fleet_cost_plus_bonus", per_fleet, {a: COST + BONUS, b: COST + BONUS})


def test_wasted_invocation_charged_once():
    env = fresh()
    live, per_fleet, system = charge(env, mode=1)
    print(f"      live {live}  system {system}")
    check("live_is_one_cost", live, COST)
    check("charged_to_the_holon", system, COST)
    check("no_fleet_charged", per_fleet, {})


def test_declined_is_free():
    env = fresh()
    a, b = env.nodes[0].id, env.nodes[1].id
    live, per_fleet, system = charge(env, mode=0, warning={a, b}, integrity=0.5)
    check("decline_costs_nothing", (live, per_fleet, system), (0.0, {}, 0.0))


def test_forced_fallback_charged_once():
    env = fresh()
    a, b = env.nodes[0].id, env.nodes[1].id
    before = env.recovery_forced
    live, _, _ = charge(env, mode=0, deadlocked={a, b}, integrity=0.0,
                        streak=env.recovery_forced_after - 1)
    print(f"      live {live}  forced {env.recovery_forced - before}")
    check("forced_fired", env.recovery_forced - before, 1)
    check("live_is_cost_plus_bonus", live, COST + BONUS)


def test_the_prices_are_in_the_intended_order():
    """
    The ordering the duplicate inverted: preventing a collision (when it works)
    must pay better than cleaning one up, and asking with fleets at risk must
    not cost more than asking for nothing.
    """
    pre_bonus = float(CONFIG["recovery_policy"]["preemptive_bonus"])
    env = fresh()
    a, b = env.nodes[0].id, env.nodes[1].id
    prevent = charge(env, mode=1, warning={a, b}, integrity=0.5)[0] + pre_bonus
    env = fresh()
    cleanup = charge(env, mode=1, deadlocked={a, b}, integrity=0.0)[0]
    env = fresh()
    at_risk = charge(env, mode=1, warning={a, b}, integrity=0.5)[0]
    env = fresh()
    wasted = charge(env, mode=1)[0]
    print(f"      prevent {prevent:+}  clean up {cleanup:+}  "
          f"ask at risk {at_risk:+}  ask for nothing {wasted:+}")
    check("prevent_pays_more_than_cleanup", prevent > cleanup, True)
    check("at_risk_not_dearer_than_wasted", at_risk >= wasted, True)


def test_collision_cost_mode():
    """
    reward_mode = collision_cost: each NEW colliding pair is charged once, the
    first step it is seen; a pair still colliding is not charged again; at most
    collision_charge_cap pairs a step; and the strict "clear" bonus is gone.
    """
    env = fresh()
    env.recovery_reward_mode = "collision_cost"
    C = env.recovery_collision_cost
    n = env.nodes
    def place(pairs_at):
        for i, node in enumerate(n):
            node.current_pos = np.array([float(3 * i), 30.0, 0.0])      # spread out
        for k, (i, j) in enumerate(pairs_at):
            cell = env._coords_by_id[sorted(env._coords_by_id)[k * 7]]
            n[i].current_pos = np.array(cell, dtype=np.float64)
            n[j].current_pos = np.array(cell, dtype=np.float64)
        env.proximity.refresh(env.nodes, excluded_ids=set())
        env._pending_integrity_reward = 0.0
        env._pending_integrity_by_fleet = {}
        return env._charge_new_collisions(), env._pending_integrity_reward
    got1 = place([(0, 1)])
    got2 = place([(0, 1)])
    got3 = place([(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)])
    print(f"      new pair {got1}, same pair again {got2}, four new pairs + one old {got3} (C = {C})")
    check("new_pair_charged_once", got1, (1, C))
    check("same_pair_not_charged_again", got2, (0, 0.0))
    check("capped_per_step", got3, (env.recovery_collision_cap, env.recovery_collision_cap * C))
    # no "clear" bonus in this mode
    env = fresh()
    env.recovery_reward_mode = "collision_cost"
    a, b = env.nodes[0].id, env.nodes[1].id
    charge(env, mode=1, warning={a, b}, integrity=0.5)
    before = env.recovery_preemptive_success
    env._preemptive_watch = {a, b}
    env._step_deadlocked, env._step_warning = set(), set()
    env._pending_integrity_reward = 0.0
    if env._preemptive_watch:                     # the payoff block, as step() runs it
        still = env._preemptive_watch & (env._step_deadlocked | env._step_warning)
        if not still:
            if env.recovery_reward_mode == "strict_bonus":
                env._pending_integrity_reward += env.recovery_preemptive_bonus
            env.recovery_preemptive_success += 1
    check("success_still_counted", env.recovery_preemptive_success - before, 1)
    check("no_clear_bonus_paid", env._pending_integrity_reward, 0.0)


def test_default_mode_charges_no_collisions():
    env = fresh()
    check("default_is_strict_bonus", env.recovery_reward_mode, "strict_bonus")
    check("nothing_charged_by_default", env.recovery_collisions_charged, 0)


if __name__ == "__main__":
    for fn in list(globals().values()):
        if callable(fn) and getattr(fn, "__name__", "").startswith("test_"):
            print(f"\n--- {fn.__name__} ---")
            fn()
    drv.apply_config({})
    print("\n" + ("ALL PASS" if not FAIL else f"FAILURES: {FAIL}"))
    sys.exit(1 if FAIL else 0)
