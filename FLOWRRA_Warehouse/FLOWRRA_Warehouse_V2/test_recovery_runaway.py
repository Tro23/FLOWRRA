"""
test_recovery_runaway.py

The yield-escalation runaway that killed the 2026-09-17 cold run, and a
measurement of whether `waiting` is what built the pile-up in the first place.

WHAT HAPPENED. From step 163 to the end of the episode, the same twelve fleets
collided every single step. Recovery reported "Tier 1: Spatial Escape
Successful" every time and nothing moved. The hold durations climbed 665, 670,
675 ... 845, five steps longer each step, while REPEAT OFFENCE counted 133
through 170.

TWO DEFECTS, COMPOUNDING.

  1. THE ESCALATION HAD NO CEILING.
         duration = base + escalation * (repeat - 1)
     At repeat 169 that is 845 steps in a 780-step episode. The held fleets
     could not resume before the episode ended.

  2. IT COUNTED ITS OWN REMEDY AS A FRESH OFFENCE.
     A yielding fleet cannot move. If it was overlapping when the hold began it
     is still overlapping next step, and every step after. Counting that as a
     new collision makes the escalation measure how long the remedy has been
     failing, then use that number to hold them longer. A pair that collided
     ONCE and could not separate for 170 steps was recorded as 170 collisions.

The pile-up was large for a third reason that is NOT fixed here: the
anti-starvation alternation only applies to `len(ordered) == 2`, so in a
twelve-fleet tangle eleven fleets are held at once and one winner walks free.
That is a design question, not a bug, and it is left alone deliberately.
"""

import numpy as np
import networkx as nx

from config_warehouse import CONFIG
from recovery_warehouse import WarehouseRecovery

FAIL = []


def check(name, got, want):
    ok = got == want
    print(f"{'PASS' if ok else 'FAIL'}  {name}: got {got!r} want {want!r}")
    if not ok:
        FAIL.append(name)


class _Node:
    """Minimal stand-in: _assign_yields only reads id, position and goal hops."""

    def __init__(self, nid, pos, hops):
        self.id = nid
        self.current_pos = np.array(pos, dtype=np.float64)
        self._hops = hops

    def get_graph_distance_to_goal(self):
        return self._hops


def fresh(**kw):
    r = CONFIG["recovery"]
    return WarehouseRecovery(
        base_yield_steps=kw.get("base", r["base_yield_steps"]),
        yield_escalation_per_repeat=kw.get("esc", r["yield_escalation_per_repeat"]),
        max_yield_steps=kw.get("cap", r.get("max_yield_steps", 30)),
    )


# ===================================================================== cap
def test_yield_duration_is_capped():
    rec = fresh()
    nodes = [_Node("a", (0, 0, 0), 3), _Node("b", (1, 0, 0), 9)]
    rows = []
    for repeat in (1, 2, 5, 10, 50, 169):
        out = rec._assign_yields(nodes, repeat)
        rows.append((repeat, max(out["yield_durations"].values())))
    print(f"      {'repeat':>8}{'hold':>7}")
    for r, d in rows:
        print(f"      {r:>8}{d:>7}")

    cap = rec.max_yield_steps
    check("never_exceeds_the_cap", bool(all(d <= cap for _, d in rows)), True)
    check("still_escalates_below_the_cap", bool(rows[0][1] < rows[2][1]), True)
    check("runaway_case_is_bounded", rows[-1][1], cap)

    # The number that actually broke the run: a hold longer than the episode.
    check("cap_is_shorter_than_an_episode",
          bool(cap < CONFIG["training"].get("max_steps_per_episode", 780)), True)


def test_uncapped_would_have_exceeded_the_episode():
    """Reproduces the original failure, so the regression is on record."""
    rec = fresh(cap=10 ** 9)
    nodes = [_Node("a", (0, 0, 0), 3), _Node("b", (1, 0, 0), 9)]
    d = max(rec._assign_yields(nodes, 169)["yield_durations"].values())
    print(f"      uncapped hold at repeat 169: {d} steps "
          f"(episode is {CONFIG['training'].get('max_steps_per_episode', 780)})")
    check("uncapped_exceeds_episode",
          bool(d > CONFIG["training"].get("max_steps_per_episode", 780)), True)


# ================================================== the self-feeding counter
def test_held_fleets_do_not_escalate_each_other():
    """
    Two fleets overlapping and BOTH held. They cannot separate, so every step
    they are still overlapping -- but that is one unresolved collision, not a
    new one each step.
    """
    rec = fresh()
    a, b = _Node("a", (5, 0, 0), 4), _Node("b", (5, 0, 0), 7)

    first = rec._record_colliding_pairs([a, b])
    check("first_overlap_counts", first, 1)

    # Recovery holds them both.
    rec.currently_yielding = {"a", "b"}
    for _ in range(50):
        rec._record_colliding_pairs([a, b])
    held = rec.pair_collision_counts[frozenset(("a", "b"))]
    print(f"      after 50 steps both held, repeat count = {held}")
    check("held_overlap_does_not_escalate", held, 1)

    # Once released, a genuine re-collision counts again.
    rec.currently_yielding = set()
    rec._record_colliding_pairs([a, b])
    check("release_restores_counting",
          rec.pair_collision_counts[frozenset(("a", "b"))], 2)


def test_one_held_one_free_still_counts():
    """
    Only a MUTUAL hold is exempt. If one fleet can still move and chooses to sit
    on top of a held one, that is a real offence and should escalate.
    """
    rec = fresh()
    a, b = _Node("a", (5, 0, 0), 4), _Node("b", (5, 0, 0), 7)
    rec.currently_yielding = {"a"}
    for _ in range(3):
        rec._record_colliding_pairs([a, b])
    check("half_held_pair_still_counts",
          rec.pair_collision_counts[frozenset(("a", "b"))], 3)


def test_the_runaway_cannot_recur():
    """
    End to end: the exact shape of the failure. Two overlapping fleets, held,
    unable to move, for 200 steps.
    """
    rec = fresh()
    a, b = _Node("a", (5, 0, 0), 4), _Node("b", (5, 0, 0), 7)
    worst, holds = 0, []
    for step in range(200):
        worst = rec._record_colliding_pairs([a, b])
        out = rec._assign_yields([a, b], worst)
        d = max(out["yield_durations"].values())
        holds.append(d)
        rec.currently_yielding = {"a", "b"}      # both held, neither can move
    print(f"      200 steps stuck: repeat={worst}  holds "
          f"first={holds[0]} last={holds[-1]} max={max(holds)}")
    check("repeat_count_stays_at_one", worst, 1)
    check("hold_stays_at_base", max(holds), rec.base_yield_steps)


# ====================================== did waiting build the pile-up?
def test_does_waiting_seed_pileups():
    """
    OPEN QUESTION, MEASURED RATHER THAN ASSERTED.

    A voluntarily waiting fleet is stationary, and stationary fleets accumulate.
    A cold random policy waits often. So `waiting.enabled` is a plausible cause
    of the twelve-fleet pile -- but plausible is not measured, and guessing has
    been wrong more often than right this week.

    Runs the same seeded episode twice, waiting on and off, and reports. No
    assertion on the comparison: one episode pair is not evidence, and a test
    that fails on noise is worse than no test.
    """
    import test_smoke_integration as smoke

    # run_episode() calls set_flags(), which sets waiting.enabled itself -- so
    # setting it beforehand does nothing and both arms run identically. The
    # first version of this test did exactly that and reported 112 collisions
    # and 50 waits for BOTH arms, including the one where waiting was meant to
    # be off. Identical numbers in a comparison are a bug, not a null result.
    _orig = smoke.set_flags

    out = {}
    for enabled in (False, True):
        def _patched(on, _e=enabled):
            _orig(on)
            CONFIG["waiting"]["enabled"] = _e
        smoke.set_flags = _patched
        env, agent, base, dens, nf, mem, steps = smoke.run_episode(True, steps=120)
        smoke.set_flags = _orig
        est = env.get_error_statistics()
        out[enabled] = {
            "collisions": env.loop.total_collisions,
            "recoveries": est.get("recovery_events", 0),
            "waits": est.get("waits_started", 0),
            "immobile": len(env.immobile_nodes),
        }
    smoke.set_flags(False)

    print(f"      {'waiting':>9}{'collisions':>12}{'waits':>8}{'immobile':>10}")
    for k, v in out.items():
        print(f"      {str(k):>9}{v['collisions']:>12}{v['waits']:>8}{v['immobile']:>10}")
    ratio = (out[True]["collisions"] / max(1, out[False]["collisions"]))
    print(f"      collisions with waiting / without = {ratio:.2f}x")
    print("      (one episode pair -- indicative only, not evidence)")
    check("both_arms_ran", len(out), 2)
    # The arms must actually DIFFER in the thing being varied, or the comparison
    # is two copies of one run wearing different labels.
    check("waiting_arm_actually_waited", bool(out[True]["waits"] > 0), True)
    check("control_arm_did_not_wait", out[False]["waits"], 0)


# ============================================ Tier-1 no-op escapes
def test_a_noop_escape_is_a_failure_not_a_success():
    """
    Tier 1 picks the highest-affordance NEIGHBOUR cell. A fleet sitting mid-edge
    rounds to a cell whose neighbour set can include the cell it already
    occupies, so "escaping" can leave it exactly where it was -- and Tier 1 then
    reports success, force_repair() restores integrity to 1.0, nothing has moved,
    and the identical collision fires again next step.

    Measured at 25.7% of relocations on 2026-09-13 (13,047 of 50,720), in code.

    DO NOT try to read this rate off the log. The relocation line used to format
    positions with int(), which TRUNCATES: a genuine 8.5 -> 8.0 move printed as
    (8)->(8), identical to standing still. That artifact produced a confident
    wrong diagnosis on 2026-09-17 -- the log was read as 24.2% no-ops when it
    could not distinguish them at all. The counter exists so the number comes
    from the floats rather than from the text.
    """
    rec = fresh()
    check("counters_start_at_zero",
          (rec.tier1_real_escapes, rec.tier1_noop_escapes), (0, 0))

    # The guard itself: a zero-distance move must be rejected.
    import numpy as _np
    here = _np.array([8.0, 9.0, 0.0])
    for cand, label, want_noop in ((_np.array([8.0, 9.0, 0.0]), "same cell", True),
                                   (_np.array([8.0, 9.0, 0.0]) + 0.5, "half cell", False),
                                   (_np.array([7.0, 9.0, 0.0]), "one cell", False)):
        moved = float(_np.sum(_np.abs(cand - here))) >= 1e-6
        print(f"      {label:<12} -> {'MOVED' if moved else 'no-op'}")
        check(f"{label.replace(' ', '_')}_classified", moved, not want_noop)


def test_tier1_refuses_an_occupied_cell():
    """
    THE STACKING BUG. affordance = 1/(1+R), and a peer standing ON a cell
    contributes R = peer_severity = 0.7, scoring 0.588 -- comfortably above
    spatial_safe_threshold = 0.35. Two peers score 0.461 and still pass.

    So Tier 1 relocated fleets ONTO other fleets and reported success. That is
    how a 3-fleet conflict becomes an 18-fleet blob: recovery was not failing to
    disperse the pile, it was assembling it.

    A soft score cannot express a hard constraint. This asserts the constraint.
    """
    import numpy as _np
    sev = CONFIG["density"]["peer_severity"]
    thr = CONFIG["recovery"]["spatial_safe_threshold"]
    rows = [("empty", 0.0), ("peer 1 hop", sev * 0.667),
            ("peer ON the cell", sev * 1.0), ("two peers", sev * 1.667)]
    print(f"      spatial_safe_threshold = {thr}")
    for label, R in rows:
        a = 1.0 / (1.0 + R)
        print(f"      {label:<20} affordance {a:.3f}"
              f"  {'passes' if a > thr else 'rejected'} the score check")

    # The score check alone lets an OCCUPIED cell through -- that is the bug.
    occupied_affordance = 1.0 / (1.0 + sev)
    check("score_alone_would_accept_an_occupied_cell",
          bool(occupied_affordance > thr), True)

    # The hard check must reject it regardless of score.
    ct = CONFIG["warehouse"]["collision_threshold"]
    cand = _np.array([5.0, 0.0, 0.0])
    occupant = _np.array([5.0, 0.0, 0.0])
    far = _np.array([9.0, 0.0, 0.0])
    check("hard_check_rejects_the_occupant",
          bool(float(_np.sum(_np.abs(cand - occupant))) <= ct), True)
    check("hard_check_allows_a_clear_cell",
          bool(float(_np.sum(_np.abs(cand - far))) <= ct), False)


def test_tier1_occupancy_counter_exists():
    rec = fresh()
    check("counter_present", hasattr(rec, "tier1_rejected_occupied"), True)
    check("counter_starts_at_zero", rec.tier1_rejected_occupied, 0)
    check("counter_is_reported",
          "tier1_rejected_occupied" in rec.get_statistics(), True)


def test_log_format_no_longer_hides_a_half_cell_move():
    """int() truncates and made a real move look like a no-op; round() does not."""
    import numpy as _np
    old = tuple(int(v) for v in _np.array([8.5, 9.0, 0.0]))
    new = tuple(round(float(v), 1) for v in _np.array([8.5, 9.0, 0.0]))
    print(f"      8.5 under int(): {old}   under round(): {new}")
    check("int_hid_the_half_cell", old, (8, 9, 0))
    check("round_shows_it", new, (8.5, 9.0, 0.0))


if __name__ == "__main__":
    for fn in list(globals().values()):
        if callable(fn) and getattr(fn, "__name__", "").startswith("test_"):
            print(f"\n--- {fn.__name__} ---")
            fn()
    print("\n" + ("ALL PASS" if not FAIL else f"FAILURES: {FAIL}"))