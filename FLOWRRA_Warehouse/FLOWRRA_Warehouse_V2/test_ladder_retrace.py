"""
test_ladder_retrace.py -- THE CONFLICT LADDER, step 2a-iii: retrace (2a-iv folded in).

WHAT 2a-iii ADDS (config conflict.ladder.retrace; needs ladder.enabled)
  shadow_run1 found 79 wait-for cycles in 10 episodes: 97% two fleets facing
  each other, 59% cleared within one check by themselves, 41% needed someone to
  back up. Nothing did: RULES' yield-to-stopped guard deliberately steps aside
  from loops, and the longest-lived cycles (29 and 31 steps) sat in exactly the
  episodes where it did. Now:
    * every fleet keeps a TRAIL, the last cells it actually stood on;
    * a cycle seen at more than grace_checks consecutive checks gets ONE member
      backed up its own trail -- the cheapest retreat (fewest hops to a free cell
      off every other member's route); ties go to the fleet further from its goal;
    * one hop per step, then hold -- until its route no longer meets the others'
      (COMMITMENT: release on "they have passed", not on "the cycle dissolved",
      which backing off causes at once);
    * fleets RULES is already backing out of a corridor are left to RULES;
    * 3-fleet cycles: the same rule, the cheapest member backs up.
  Orders ride in RULES' action dict: they execute last, reach held fleets, and
  are recorded as RULES decisions -- teaching labels for the Learner (step 3).

WHAT THIS CHECKS
  1. Off: no orders.
  2. Retreat cost: 1 hop when the last cell is clear; further when it lies on the
     other's route; impossible when another fleet stands on the way back.
  3. Grace: a cycle seen once gets no order; seen twice, the cheaper side backs up.
  4. The ordered move heads for the retreat cell; on arrival the fleet holds.
  5. Commitment: held while routes still meet; released once they do not.
  6. A budget ends an order that cannot finish.
  7. A fleet in a RULES corridor order is never given a retrace order.
  8. A 3-fleet cycle gets exactly one order.

Run:  python test_ladder_retrace.py      (ends with ALL PASS)
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


def make_env(retrace=True):
    saved = dict(CONFIG["conflict"].get("ladder", {}) or {})
    CONFIG["conflict"]["ladder"] = {"enabled": True, "shadow": False, "retrace": retrace,
                                    "grace_checks": 1, "trail_length": 8,
                                    "aging_reset_steps": 3, "l3_stuck_steps": 30}
    try:
        G, grid_pos, missions, gdm, goal_pool = build_instance(24)
        env = FLOWRRA(G, grid_pos, missions, mode="training",
                      goal_distance_maps=gdm, shared_pool_mode=True,
                      goal_pool=goal_pool)
    finally:
        CONFIG["conflict"]["ladder"] = saved
    return env


def setup(env, n=2):
    """n mobile fleets standing on single cells, each with a one-cell trail behind
    it on a free neighbouring cell. Routes are controlled through env.ROUTES."""
    occupied = set()
    for node in env.nodes:
        occupied |= set(env.rules._anchors(node))
    picked = []
    for node in env.nodes:
        if node.id in env.immobile_nodes:
            continue
        an = env.rules._anchors(node)
        if len(an) != 1:
            continue
        free = [w for w in env.G.neighbors(an[0]) if w not in occupied]
        if not free:
            continue
        env._trail[node.id] = [free[0], an[0]]
        occupied.add(free[0])
        picked.append((node, an[0], free[0]))
        if len(picked) == n:
            break
    env.ROUTES = {}
    env._route_cells = lambda node: set(env.ROUTES.get(node.id, set()))
    return picked


# ------------------------------------------------------------------ 1. off
def test_off():
    env = make_env(retrace=False)
    check("retrace_flag_off", env.ladder_retrace, False)


# ------------------------------------------------------------------ 2. retreat cost
def test_retreat_cost():
    env = make_env()
    (a, a_cur, a_prev), (b, b_cur, b_prev) = setup(env)
    env.ROUTES = {b.id: {a_cur}}
    check("one_hop_when_clear", env._retreat_cost(a, {b.id}), (1.0, a_prev))

    env.ROUTES = {b.id: {a_cur, a_prev}}          # my last cell is on b's route
    prev2 = next((w for w in env.G.neighbors(a_prev)
                  if w not in (a_cur, a_prev) and w not in env.ROUTES[b.id]), None)
    env._trail[a.id] = [prev2, a_prev, a_cur]
    check("walks_further_back_when_on_their_route",
          env._retreat_cost(a, {b.id}), (2.0, prev2))

    env._trail[a.id] = [b_cur, a_cur]             # b now stands on my way back
    check("blocked_trail_is_impossible", env._retreat_cost(a, {b.id})[0], float("inf"))
    env._trail[a.id] = []
    check("no_trail_is_impossible", env._retreat_cost(a, {b.id})[0], float("inf"))


# ------------------------------------------------------------------ 3-5. grace, move, commitment
def test_grace_move_and_commitment():
    env = make_env()
    (a, a_cur, a_prev), (b, b_cur, b_prev) = setup(env)
    env._trail[b.id] = [b_cur]                    # b has nowhere to go back to
    env.ROUTES = {a.id: {b_cur}, b.id: {a_cur}}
    c = frozenset((a.id, b.id))

    env._shadow_cycles = {c: 1}
    env._ladder_plan_retreats({c})
    check("seen_once_no_order", len(env._retrace), 0)

    env._shadow_cycles = {c: 2}
    env._ladder_plan_retreats({c})
    check("seen_twice_cheaper_side_ordered", sorted(env._retrace), [a.id])
    check("order_targets_the_trail_cell", env._retrace[a.id]["target"], a_prev)
    check("one_order_counted", env.ladder_retrace_stats["orders"], 1)

    env._pair_kind = {c: "head_on"}               # still facing each other
    act = env._ladder_retrace_actions()[a.id]
    check("move_heads_for_the_retreat_cell", env.rules._target_cell(a, act), a_prev)

    env.rules._anchors = (lambda real: (lambda node: [a_prev] if node.id == a.id
                                        else real(node)))(env.rules._anchors)
    check("holds_on_arrival", env._ladder_retrace_actions()[a.id], 0)
    check("still_ordered_while_head_on", a.id in env._retrace, True)

    env._pair_kind = {c: "contested"}             # racing for the same cell
    env._ladder_retrace_actions()
    check("still_ordered_while_racing_for_a_cell", a.id in env._retrace, True)

    env._pair_kind = {c: "following"}             # b passed; now a harmless convoy
    env._ladder_retrace_actions()
    check("released_when_only_a_convoy_remains", a.id in env._retrace, False)
    check("release_counted", env.ladder_retrace_stats["released"], 1)


# ------------------------------------------------------------------ 6. budget
def test_budget_is_in_steps():
    env = make_env()
    check("two_steps_per_hop_at_half_speed", env.ladder_steps_per_hop, 2)
    check("one_hop_budget_30_steps", env._retrace_budget(1), 30)
    check("three_hop_budget_38_steps", env._retrace_budget(3), 38)


def test_budget_ends_an_order():
    env = make_env()
    (a, a_cur, a_prev), (b, b_cur, b_prev) = setup(env)
    env.ROUTES = {a.id: {b_cur}, b.id: {a_cur}}
    env._retrace[a.id] = {"target": a_prev, "others": {b.id}, "since": 0,
                          "budget": 5, "arrived": False}
    env.step_count = 100
    env._pair_kind = {frozenset((a.id, b.id)): "blocked"}
    env._ladder_retrace_actions()
    st = env.ladder_retrace_stats
    check("timed_out", (a.id in env._retrace, st["timeouts"]), (False, 1))
    check("why_never_arrived", st["timeout_not_arrived"], 1)
    check("why_last_relationship_blocked", st["timeout_last_blocked"], 1)


# ------------------------------------------------------------------ 7. one authority
def test_corridor_orders_left_to_rules():
    env = make_env()
    (a, a_cur, a_prev), (b, b_cur, b_prev) = setup(env)
    env.ROUTES = {a.id: {b_cur}, b.id: {a_cur}}
    c = frozenset((a.id, b.id))
    env.rules.orders[b.id] = {"target": None}
    env._shadow_cycles = {c: 2}
    env._ladder_plan_retreats({c})
    check("no_retrace_on_a_rules_corridor_pair", len(env._retrace), 0)
    check("skip_counted", env.ladder_retrace_stats["skipped_corridor"], 1)


# ------------------------------------------------------------------ 8. three fleets
def test_three_fleet_cycle_one_order():
    env = make_env()
    picked = setup(env, n=3)
    (a, a_cur, _), (b, b_cur, _), (c3, c_cur, _) = picked
    env.ROUTES = {a.id: {b_cur}, b.id: {c_cur}, c3.id: {a_cur}}
    cyc = frozenset((a.id, b.id, c3.id))
    env._shadow_cycles = {cyc: 2}
    env._ladder_plan_retreats({cyc})
    check("exactly_one_member_backs_up", len(env._retrace), 1)
    check("multi_counted", env.ladder_retrace_stats["multi"], 1)


if __name__ == "__main__":
    for fn in list(globals().values()):
        if callable(fn) and getattr(fn, "__name__", "").startswith("test_"):
            print(f"\n--- {fn.__name__} ---")
            fn()
    print("\n" + ("ALL PASS" if not FAIL else f"FAILURES: {FAIL}"))