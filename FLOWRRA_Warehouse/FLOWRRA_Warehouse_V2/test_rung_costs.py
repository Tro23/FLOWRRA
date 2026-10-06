"""
test_rung_costs.py -- step 2c: Safety pays by rung, weighted by path.

WHAT 2c ADDS (config training.rung_costs; off = counted only, rewards unchanged)
  Every escalation of the conflict ladder is charged ONCE, when it happens, to
  the Safety reward of the fleets it involved:
      L1  a retrace order          c1 per hop retreated      (-2.4 / hop)
      L2  a RULES corridor back-out  c1 per hop to the bay + c2  (-6 extra)
      L3  the safety net fires     c3                        (-50, a collision)
  Who pays: the conflict group at weight 1.0; a fleet whose ROUTE heads into the
  group's cells at heading_in_weight (0.25); everyone else nothing. Path, not
  distance: a bystander parked nearby pays nothing. A fleet that backs off on its
  own, inside the grace check, pays nothing at all.

  Calibration: one hop takes two steps at half speed and the worst warning step
  costs -1.2, so c1 = -2.4 -- giving way costs what lingering in the jam would.

Also here: the "collisions while retreating" diagnostic -- does reversing ever
cause contact?

WHAT THIS CHECKS
  1. Off: charges are counted, nothing reaches a reward.
  2. Path weights: group 1.0, heading-in 0.25, bystander 0.
  3. L1: a retrace order charges its cycle c1 x hops.
  4. L2: a new corridor order charges retreater and winners c1 x hops + c2, once.
  5. L3: the fixed rule firing charges the stuck pair c3.
  6. The per-step vector lines up with the fleets and empties after use.
  7. A fatal collision during a retreat is counted.

Run:  python test_rung_costs.py      (ends with ALL PASS)
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


def close(a, b):
    return abs(a - b) < 1e-9


def make_env(rung_on=True):
    saved_l = dict(CONFIG["conflict"].get("ladder", {}) or {})
    saved_r = dict(CONFIG["training"].get("rung_costs", {}) or {})
    CONFIG["conflict"]["ladder"] = {"enabled": True, "shadow": False, "retrace": True,
                                    "grace_checks": 1, "trail_length": 8,
                                    "aging_reset_steps": 3, "l3_stuck_steps": 30}
    CONFIG["training"]["rung_costs"] = {"enabled": rung_on, "c1_per_hop": -2.4,
                                        "c2_pullover": -6.0, "c3_l3": -50.0,
                                        "heading_in_weight": 0.25}
    try:
        G, grid_pos, missions, gdm, goal_pool = build_instance(24)
        env = FLOWRRA(G, grid_pos, missions, mode="training",
                      goal_distance_maps=gdm, shared_pool_mode=True,
                      goal_pool=goal_pool)
    finally:
        CONFIG["conflict"]["ladder"] = saved_l
        CONFIG["training"]["rung_costs"] = saved_r
    env.ROUTES = {}
    env._route_cells = lambda node: set(env.ROUTES.get(node.id, set()))
    return env


def three(env):
    """Three mobile fleets on single cells: a, b (a conflict) and c (another)."""
    out = []
    for n in env.nodes:
        if n.id in env.immobile_nodes:
            continue
        an = env.rules._anchors(n)
        if len(an) == 1:
            out.append((n, an[0]))
        if len(out) == 3:
            return out


# ------------------------------------------------------------------ 1. off
def test_off_counts_but_never_charges():
    env = make_env(rung_on=False)
    (a, _), (b, _), _ = three(env)
    env._charge_rung("l3", {a.id, b.id}, -50.0)
    check("counted", env.rung_stats["l3_events"], 1)
    check("would_have_charged", env.rung_stats["l3_charged"], -100.0)
    check("nothing_pending_for_rewards", env._rung_pending, {})


# ------------------------------------------------------------------ 2. path weights
def test_path_weights():
    env = make_env()
    (a, a_cell), (b, b_cell), (c, c_cell) = three(env)
    env.ROUTES = {c.id: {a_cell}}                 # c is driving into the jam
    env._charge_rung("l1", {a.id, b.id}, -2.4)
    p = env._rung_pending
    check("group_member_pays_full", close(p[a.id], -2.4) and close(p[b.id], -2.4), True)
    check("heading_in_pays_a_quarter", close(p[c.id], -0.6), True)
    check("bystanders_pay_nothing", len(p), 3)

    env2 = make_env()
    (a2, a2c), (b2, _), (c2, _) = three(env2)
    env2.ROUTES = {}                               # c is near but heading elsewhere
    env2._charge_rung("l1", {a2.id, b2.id}, -2.4)
    check("near_but_not_heading_in_pays_nothing", c2.id in env2._rung_pending, False)


# ------------------------------------------------------------------ 3. L1
def test_l1_charged_when_a_retrace_order_is_made():
    env = make_env()
    (a, a_cell), (b, b_cell), _ = three(env)
    prev = next(w for w in env.G.neighbors(a_cell)
                if w not in {x for n in env.nodes for x in env.rules._anchors(n)})
    env._trail[a.id] = [prev, a_cell]
    env._trail[b.id] = [b_cell]
    env.ROUTES = {a.id: {b_cell}, b.id: {a_cell}}
    cyc = frozenset((a.id, b.id))
    env._shadow_cycles = {cyc: 2}
    env._ladder_plan_retreats({cyc})
    check("order_made", a.id in env._retrace, True)
    check("l1_event", env.rung_stats["l1_events"], 1)
    check("both_members_pay_one_hop", close(env._rung_pending[a.id], -2.4)
          and close(env._rung_pending[b.id], -2.4), True)


# ------------------------------------------------------------------ 4. L2
def test_l2_charged_once_per_new_corridor_order():
    env = make_env()
    (a, a_cell), (b, _), _ = three(env)
    env.rules.orders[a.id] = {"target": None, "dmap": {a_cell: 3}, "winners": {b.id}}
    env._charge_corridor_orders()
    want = -2.4 * 3 - 6.0
    check("retreater_pays_hops_plus_pullover", close(env._rung_pending[a.id], want), True)
    check("winner_shares_it", close(env._rung_pending[b.id], want), True)
    env._charge_corridor_orders()                  # same order, next step
    check("charged_once_not_every_step", env.rung_stats["l2_events"], 1)


# ------------------------------------------------------------------ 5. L3
def test_l3_charged_when_the_fixed_rule_fires():
    env = make_env()
    (a, _), (b, _), _ = three(env)
    env._ladder_stuck = lambda: {a.id, b.id}
    env._step_warning, env._step_deadlocked = {a.id, b.id}, set()
    env._run_recovery = lambda forced: {}
    env._policy_recovery_step()
    check("l3_event", env.rung_stats["l3_events"], 1)
    check("stuck_pair_pays_a_collision", close(env._rung_pending[a.id], -50.0)
          and close(env._rung_pending[b.id], -50.0), True)


# ------------------------------------------------------------------ 6. the reward vector
def test_vector_lines_up_and_empties():
    env = make_env()
    (a, _), _, _ = three(env)
    env._rung_pending = {a.id: -2.4}
    v = env._rung_vector()
    idx = [n.id for n in env.nodes].index(a.id)
    check("in_node_order", (len(v), close(v[idx], -2.4), close(float(v.sum()), -2.4)),
          (len(env.nodes), True, True))
    check("emptied_after_use", env._rung_pending, {})


# ------------------------------------------------------------------ 7. collision diagnostic
def test_collision_while_retreating_counted():
    env = make_env()
    (a, a_cell), (b, _), _ = three(env)
    env._retrace[a.id] = {"target": a_cell, "others": {b.id}, "since": 0,
                          "budget": 99, "arrived": True}
    env._pair_kind = {frozenset((a.id, b.id)): "head_on"}
    env.loop.deadlocked_nodes = {a.id}
    env._ladder_retrace_actions()
    check("counted", env.ladder_retrace_stats["collisions_while_retreating"], 1)


if __name__ == "__main__":
    for fn in list(globals().values()):
        if callable(fn) and getattr(fn, "__name__", "").startswith("test_"):
            print(f"\n--- {fn.__name__} ---")
            fn()
    print("\n" + ("ALL PASS" if not FAIL else f"FAILURES: {FAIL}"))