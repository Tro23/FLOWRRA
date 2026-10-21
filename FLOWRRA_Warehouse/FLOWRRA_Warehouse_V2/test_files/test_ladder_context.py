"""
test_ladder_context.py -- step 3b: fleets can SEE the ladder.

WHAT 3b ADDS (config conflict.ladder.context; needs ladder.enabled)
  Seven values appended LAST to every fleet's base state:
      in a cycle (0/1) | time in conflict (steps / T, capped at 1)
      | group's rung (one-hot L0..L3; all zero outside any conflict)
      | cheaper side (my retreat / (mine + theirs); <0.5 = mine is cheaper)
  Computed at the start of a step for the choice, and again after the move --
  from the same after-move verdicts the reward uses -- for the stored next
  state, so a next state never carries the previous step's context. The state
  width grows by 7, so a run with it on starts cold.

WHAT THIS CHECKS
  1. Width: +7 with it on, unchanged off; state_layout names the new block.
  2. Cycle and time in conflict; the after-move preview does not advance time.
  3. Rung: L0 in conflict, L1 retrace, L2 corridor back-out, L3 safety net.
  4. Cheaper side from the trails.
  5. The values land at the end of the state vector, in order.

Run:  python test_ladder_context.py      (ends with ALL PASS)
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


def make_env(context=True):
    saved = dict(CONFIG["conflict"].get("ladder", {}) or {})
    CONFIG["conflict"]["ladder"] = {"enabled": True, "shadow": False, "retrace": True,
                                    "context": context, "grace_checks": 1, "trail_length": 8,
                                    "aging_reset_steps": 3, "l3_stuck_steps": 30}
    try:
        G, gp, missions, gdm, pool = build_instance(24)
        env = FLOWRRA(G, gp, missions, mode="training", goal_distance_maps=gdm,
                      shared_pool_mode=True, goal_pool=pool)
    finally:
        CONFIG["conflict"]["ladder"] = saved
    env.ROUTES = {}
    env._route_cells = lambda node: set(env.ROUTES.get(node.id, set()))
    env.loop.deadlocked_nodes = set()
    return env


def movers(env, n):
    out = []
    for x in env.nodes:
        if x.id not in env.immobile_nodes and len(env.rules._anchors(x)) == 1:
            out.append(x)
        if len(out) == n:
            return out


def V(a, b, kind):
    return Verdict(a, b, 2.0, kind, True, "kind", None, None, None)


def test_width():
    off = make_env(context=False)
    w0 = len(off.nodes[0].get_state_vector(off.nodes))
    on = make_env(context=True)
    w1 = len(on.nodes[0].get_state_vector(on.nodes))
    check("seven_more_values", w1 - w0, 7)
    lay = on.nodes[0].state_layout()
    check("layout_names_the_block", lay.get("ladder_context"), (w1 - 7, w1))


def test_cycle_and_time():
    env = make_env()
    a, b, c = movers(env, 3)
    vs = [V(a.id, b.id, "head_on")]
    env._ladder_context_update(vs, advance=True)
    check("pair_in_a_cycle", (a.lc_in_cycle, b.lc_in_cycle, c.lc_in_cycle), (1.0, 1.0, 0.0))
    check("rung_L0_in_conflict", (a.lc_rung, c.lc_rung), (0, -1))
    check("one_step_in_conflict", round(a.lc_conflict_frac, 4), round(1 / 30, 4))
    env._ladder_context_update(vs, advance=True)
    env._ladder_context_update(vs, advance=False)       # after-move preview
    check("preview_shows_next_step", round(a.lc_conflict_frac, 4), round(3 / 30, 4))
    check("preview_does_not_advance", env._conflict_steps[a.id], 2)
    env._ladder_context_update([], advance=True)
    check("resets_when_out_of_conflict", (a.lc_conflict_frac, a.lc_in_cycle), (0.0, 0.0))


def test_rungs():
    env = make_env()
    a, b, c, d = movers(env, 4)
    env._retrace[a.id] = {"target": None, "others": {b.id}, "since": 0,
                          "budget": 99, "arrived": False}
    env.rules.orders[c.id] = {"target": None, "dmap": {}, "winners": {d.id}}
    env._ladder_context_update([], advance=True)
    check("retrace_is_L1_both_sides", (a.lc_rung, b.lc_rung), (1, 1))
    check("corridor_is_L2_both_sides", (c.lc_rung, d.lc_rung), (2, 2))
    env._l3_recent[a.id] = env.step_count
    env._ladder_context_update([], advance=True)
    check("safety_net_is_L3", a.lc_rung, 3)


def test_cheaper_side():
    env = make_env()
    a, b = movers(env, 2)
    a_cell, b_cell = env.rules._anchors(a)[0], env.rules._anchors(b)[0]
    occupied = {x for n in env.nodes for x in env.rules._anchors(n)}
    prev = next(w for w in env.G.neighbors(a_cell) if w not in occupied)
    env._trail[a.id] = [prev, a_cell]                    # a can back up one hop
    env._trail[b.id] = [b_cell]                          # b cannot
    env.ROUTES = {a.id: {b_cell}, b.id: {a_cell}}
    env._ladder_context_update([V(a.id, b.id, "head_on")], advance=True)
    check("a_is_the_cheaper_side", (a.lc_cheaper, b.lc_cheaper), (0.0, 1.0))
    tail = list(a.get_state_vector(env.nodes)[-7:])
    check("values_at_the_end_in_order",
          [round(float(x), 4) for x in tail],
          [1.0, round(1 / 30, 4), 1.0, 0.0, 0.0, 0.0, 0.0])


if __name__ == "__main__":
    for fn in list(globals().values()):
        if callable(fn) and getattr(fn, "__name__", "").startswith("test_"):
            print(f"\n--- {fn.__name__} ---")
            fn()
    print("\n" + ("ALL PASS" if not FAIL else f"FAILURES: {FAIL}"))