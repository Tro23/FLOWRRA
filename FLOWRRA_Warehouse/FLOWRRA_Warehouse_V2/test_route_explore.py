"""
test_route_explore.py -- explore everywhere, value the policy that runs
(build log entry 34, Fix 3; switch conflict.route_explore).

THE FINDING (dry run A, 13 October). With route by default, only the route's
move is ever executed in the clear -- the route overrode even a fleet that had
chosen to explore. The honest target corrects that move's value; the other moves
there keep stale values, and as soon as learning started the network's own
choice in the clear disagreed with the route on 35-55% of moves (the frozen
champion: 7-23%).

THE FIX. A fleet whose move this step is an exploring one is not overridden by
the route: the network explores in the clear too, at its own epsilon, and those
moves get corrected by real experience. The honest target values such a fleet
at the ROUTE's move -- the policy that runs there whenever it does not explore.
Training only: deploy_eval (and trace_fleet through it) force it off.

WHAT THIS CHECKS
  1. The agent reports exactly which fleets explored (all at epsilon 1, none at 0).
  Inside real FLOWRRA steps (24 fleets, the stream, route + guard + honest target
  on; exploration raised to 0.3 so it happens often):
  2. Switch on: no exploring fleet in the clear is overridden by the route; every
     non-exploring one is; some explored moves really ran (a detour).
  3. The honest labels: an explored fleet in the clear is valued at the route's
     move; the rest as in test_honest_target.
  4. The counter route_explore_kept matches; the guard's invariant holds.
  5. Switch off: exploring fleets in the clear are routed, exactly as before.
  6. deploy_eval's safety setup forces the switch off in every mode.

Run:  python test_route_explore.py      (ends with ALL PASS; a few minutes)
"""

import random

import numpy as np
import torch

from agent_warehouse import GNNAgent

FAIL = []


def check(name, got, want):
    ok = got == want
    print(f"{'PASS' if ok else 'FAIL'}  {name}: got {got!r} want {want!r}")
    if not ok:
        FAIL.append(name)


# ------------------------------------------------------------------ 1. the agent's mask
def test_agent_reports_exploration():
    a = GNNAgent(node_feature_dim=8, edge_feature_dim=0, action_size=7,
                 hidden_dim=16, num_layers=1, n_heads=2,
                 reward_heads=["safety", "delivery", "efficiency"], head_weights=[6.0, 4.0, 1.0],
                 batch_size=4, dropout=0.0, seed=0)
    rng = np.random.default_rng(0)
    nf = rng.normal(size=(5, 8)).astype(np.float32)
    adj = np.ones((5, 5), dtype=np.float32)
    vm = np.ones((5, 7), dtype=bool)
    a.choose_actions(nf, adj, 1, 1, valid_action_masks=vm, eps_min=1.0, eps_peak=1.0)
    check("all_explore_at_eps_1", a.last_explore_mask.tolist(), [True] * 5)
    a.choose_actions(nf, adj, 1, 1, valid_action_masks=vm, eps_min=0.0, eps_peak=0.0)
    check("none_explore_at_eps_0", a.last_explore_mask.tolist(), [False] * 5)


# ------------------------------------------------------------------ 2-5. real steps
def env_part():
    import core_warehouse
    from config_warehouse import CONFIG
    from node_warehouse import precompute_goal_distances
    from test_smoke_integration import build_instance, set_flags

    BOOK = {"enabled": True, "orders": 60, "release_steps": 150}
    saved = {k: dict(CONFIG.get(k, {}) or {}) for k in
             ("stream", "obstacles", "safety", "conflict", "reward_decomposition",
              "step", "training")}

    def configure(explore_on):
        CONFIG["stream"] = dict(saved["stream"], enabled=True, order_book=dict(BOOK))
        CONFIG["stream"].pop("order_floor_window", None)
        CONFIG["obstacles"] = dict(saved["obstacles"], enabled=False)
        sf = CONFIG.setdefault("safety", {})
        for k in ("intent_guard", "bubbles_by_kind", "hops_everywhere"):
            sf[k] = True
        CONFIG.setdefault("conflict", {})["route_by_default"] = True
        CONFIG["conflict"]["route_explore"] = explore_on
        CONFIG.setdefault("step", {})["simultaneous"] = True
        CONFIG.setdefault("training", {})["honest_target"] = True
        # Run 7's labels (network-driven -1); SARSA is tested in test_sarsa.py.
        CONFIG["training"]["sarsa"] = False
        # The router (entry 42) adds a learner label to every push; tested in
        # test_learner_router.py. Off here, so this keeps testing its own fix.
        CONFIG["training"]["learner"] = dict(CONFIG["training"].get("learner", {}) or {}, enabled=False)

    def make_env(seed=7, n_fleets=24):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        core_warehouse._OBSTACLE_EPISODES = 0
        core_warehouse._WAVE_EPISODES = 0
        set_flags(True)
        G, grid_pos, missions, _gdm, goal_pool = build_instance(n_fleets)
        rng = np.random.default_rng(3)
        nodes = sorted(G.nodes())
        bank = sorted(set(goal_pool) | {nodes[int(i)] for i in rng.integers(0, len(nodes), 60)})
        gdm = precompute_goal_distances(G, [{"goal_node": g} for g in bank])
        env = core_warehouse.FLOWRRA(G, grid_pos, missions, mode="training",
                                     goal_distance_maps=gdm, shared_pool_mode=True,
                                     goal_pool=goal_pool, order_bank=bank, order_seed=0)
        base_dim = len(env.nodes[0].get_state_vector(env.nodes))
        rd = CONFIG["reward_decomposition"]
        env.gnn = GNNAgent(
            node_feature_dim=base_dim + env.density.output_dim, edge_feature_dim=0,
            action_size=CONFIG["gnn"]["action_size"], hidden_dim=CONFIG["gnn"]["hidden_dim"],
            num_layers=CONFIG["gnn"]["num_layers"], n_heads=CONFIG["gnn"]["num_heads"],
            reward_heads=rd["heads"], head_weights=rd["weights"],
            dropout=CONFIG["gnn"]["dropout"], lr=CONFIG["gnn"]["learning_rate"],
            gamma=CONFIG["training"]["gamma"],
            buffer_capacity=CONFIG["training"]["buffer_capacity"],
            batch_size=CONFIG["training"]["batch_size"],
            stability_coef=CONFIG["gnn"]["stability_coef"],
            encoder_mode=CONFIG["gnn"]["encoder"], base_feature_dim=base_dim,
            density_diamond_mask=env.density._diamond_mask,
            double_dqn=True, per_fleet_terminal=True)
        # Explore often, so the case under test happens many times.
        env.gnn.epsilon_gaussian = lambda *a, **k: 0.3
        return env

    def run(env, steps=80):
        pushes = []
        _orig = env.gnn.memory.push

        def _rec(**kw):
            pushes.append(None if kw.get("next_policy_action") is None
                          else np.array(kw["next_policy_action"]).copy())
            return _orig(**kw)
        env.gnn.memory.push = _rec
        snaps = []
        for _ in range(steps):
            env.step(episode_step=1, total_episodes=1)
            pend = env._honest_pending
            snaps.append({
                "ids": list(env._honest_ids),
                "raw": np.array(env._honest_raw).copy(),
                "explore": np.array(env.gnn.last_explore_mask, dtype=bool).copy(),
                "eligible": np.array(env._route_eligible, dtype=bool).copy(),
                "routed": np.array(env._route_mask, dtype=bool).copy(),
                "moves": np.array(env._route_moves).copy(),
                "executed": (np.array(pend[0]["actions"]).copy() if pend is not None else None),
                "next_ids": (list(pend[1]) if pend is not None else None),
            })
            if env.is_episode_over():
                break
        env.flush_pending_transition()
        return snaps, pushes

    try:
        # ---------------- ON
        configure(True)
        env = make_env()
        snaps, pushes = run(env)
        overridden = sum(int((s["eligible"] & s["explore"] & s["routed"]).sum()) for s in snaps)
        missed = sum(int((s["eligible"] & ~s["explore"] & ~s["routed"]).sum()) for s in snaps)
        kept = sum(int((s["eligible"] & s["explore"]).sum()) for s in snaps)
        detours = sum(int(((s["eligible"] & s["explore"])
                           & (s["executed"] == s["raw"]) & (s["raw"] != s["moves"])).sum())
                      for s in snaps if s["executed"] is not None)
        check("on_exploring_in_clear_not_overridden", overridden, 0)
        check("on_non_exploring_in_clear_routed", missed, 0)
        check("on_exploration_in_clear_happened", kept > 0, True)
        check("on_some_detours_really_ran", detours > 0, True)
        est = env.get_error_statistics()
        check("on_counter_matches", est["route_explore_kept"], kept)
        check("on_invariant_holds", est["guard_violations"], 0)

        bad, n_route_label = 0, 0
        for t in range(len(snaps) - 1):
            nxt, now = snaps[t]["next_ids"], snaps[t + 1]
            if nxt is None or now["executed"] is None:
                continue
            pos = {fid: j for j, fid in enumerate(now["ids"])}
            npa = pushes[t]
            for k, fid in enumerate(nxt):
                j = pos.get(fid)
                if j is None:
                    want = -1
                elif now["eligible"][j] and not now["routed"][j]:
                    want = int(now["moves"][j])
                    n_route_label += 1
                else:
                    ex, rw = int(now["executed"][j]), int(now["raw"][j])
                    want = -1 if (not now["routed"][j] and ex == rw) else ex
                if int(npa[k]) != want:
                    bad += 1
        check("on_every_label_follows_the_rule", bad, 0)
        check("on_explored_in_clear_valued_at_route_move", n_route_label > 0, True)
        print(f"      {len(snaps)} steps | explored in the clear {kept} | real detours {detours} "
              f"| labelled with the route's move {n_route_label}")

        # ---------------- OFF
        configure(False)
        env2 = make_env()
        snaps2, _ = run(env2, steps=40)
        kept2 = sum(int((s["eligible"] & s["explore"] & ~s["routed"]).sum()) for s in snaps2)
        check("off_exploring_in_clear_routed_as_before", kept2, 0)
        check("off_counter_zero", env2.get_error_statistics()["route_explore_kept"], 0)

        # ---------------- deployment forces it off
        import deploy_eval
        for mode in ("config", "on", "off"):
            CONFIG["conflict"]["route_explore"] = True
            deploy_eval.apply_safety_mode(mode)
            check(f"deploy_forces_off({mode})", CONFIG["conflict"]["route_explore"], False)
    finally:
        for k, v in saved.items():
            CONFIG[k] = v


if __name__ == "__main__":
    test_agent_reports_exploration()
    env_part()
    print("\n" + ("ALL PASS" if not FAIL else f"FAILURES: {FAIL}"))
