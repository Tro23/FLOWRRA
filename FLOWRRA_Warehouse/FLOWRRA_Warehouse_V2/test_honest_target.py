"""
test_honest_target.py -- value the policy actually run (build log entry 34).

THE PROBLEM (entry 31). With route by default the route drove ~85% of moves.
The learning target valued every next state at the network's BEST move, but in
those states the network's other moves never happen, so no experience corrects
their values; the max picks the most drifted one and the values climb. Run 6's
delivery head reached 2.13 against a ceiling of 1.42.

THE FIX (switch training.honest_target). Each step's transition is held one
step. When the next step has run, every fleet in the transition's next state
gets next_policy_action:
  * -1 where the network drove it (not routed, and its own choice executed):
    learn() bootstraps from the network's best move, as always;
  * otherwise the move actually executed there (the route's, a rule's, a hold's,
    the guard's -- a stop is 0): learn() bootstraps from THAT move.

WHAT THIS CHECKS
  Part 1, the learning step (small hand-built agent):
    1. Switch off: a transition stored without the field keeps the 14-field
       layout, and padding is unchanged.
    2. The field pads with -1, and a batch mixing old and new transitions reads
       the old ones as -1 ("the network chose").
    3. All -1 gives targets, loss and weights IDENTICAL to no field at all --
       plain DQN and Double DQN.
    4. A given move is valued at THAT move (a route move, a stop = 0), exactly
       r + gamma * Q_target(s', given), for every head; -1 fleets are unchanged.
  Part 2, inside real FLOWRRA steps (24 fleets, the order stream, route and
  guard on):
    5. Exactly one transition per step, in order, none lost (the end-of-episode
       flush included).
    6. Every next_policy_action follows the rule above, matched by fleet id,
       including across roster changes (fleets leaving and entering).
    7. Route-driven fleets are valued at the move made; network-driven ones at
       -1; at least one overruled-to-wait (guard or hold) is valued at 0.
    8. The learning step runs on these transitions and uses the moves.
    9. Switch off inside FLOWRRA: pushed at once, 14 fields, no hold.

Run:  python test_honest_target.py      (ends with ALL PASS; a few minutes)
"""

import copy
import random

import numpy as np
import torch

from agent_warehouse import DEVICE, GNNAgent, GraphReplayBuffer

FAIL = []

A = 7          # 6 moves + idle
N = 3
HEADS = ["safety", "delivery", "efficiency"]
WEIGHTS = [6.0, 4.0, 1.0]


def check(name, got, want):
    ok = got == want
    print(f"{'PASS' if ok else 'FAIL'}  {name}: got {got!r} want {want!r}")
    if not ok:
        FAIL.append(name)


# ------------------------------------------------------------------ helpers
def transition(rng, n=N, npa=None):
    kw = dict(
        node_features=rng.normal(size=(n, 8)).astype(np.float32),
        adj_matrix=np.ones((n, n), dtype=np.float32),
        actions=rng.integers(0, A, size=n).astype(np.int64),
        rewards=rng.normal(size=(n, len(HEADS))).astype(np.float32),
        next_node_features=rng.normal(size=(n, 8)).astype(np.float32),
        next_adj_matrix=np.ones((n, n), dtype=np.float32),
        done=False,
        integrity=1.0,
        active_mask=np.ones(n, dtype=np.float32),
        recovery_action=0,
        next_valid_mask=np.ones((n, A), dtype=bool),
        next_active_mask=np.ones(n, dtype=np.float32),
    )
    if npa is not None:
        kw["next_policy_action"] = np.array(npa, dtype=np.int64)
    return kw


def small_agent(double_dqn=False):
    return GNNAgent(
        node_feature_dim=8, edge_feature_dim=0, action_size=A,
        hidden_dim=16, num_layers=1, n_heads=2,
        reward_heads=HEADS, head_weights=WEIGHTS,
        double_dqn=double_dqn, per_fleet_terminal=True,
        batch_size=4, dropout=0.0, lr=3e-4, seed=0, gamma=0.99,
    )


def learn_once(agent, seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    loss = agent.learn()
    return loss, agent._last_targets.clone()


# ------------------------------------------------------------------ 1-2. storage
def test_storage_and_padding():
    rng = np.random.default_rng(0)
    buf = GraphReplayBuffer(10)
    buf.push(**transition(rng))
    old = buf.buffer[0]
    check("switch_off_keeps_14_fields", len(old), 14)
    check("switch_off_pads_to_14", len(GraphReplayBuffer._pad_transition(old, 5)), 14)

    buf.push(**transition(rng, npa=[-1, 3, 0]))
    new = buf.buffer[1]
    check("with_field_15_fields", len(new), 15)
    p = GraphReplayBuffer._pad_transition(new, 5)
    check("field_pads_with_minus_one", p[14].tolist(), [-1, 3, 0, -1, -1])
    check("no_padding_keeps_field", GraphReplayBuffer._pad_transition(new, N)[14].tolist(), [-1, 3, 0])

    random.seed(0)
    b = buf.sample(2)
    check("sample_returns_17", len(b), 17)
    rows = sorted(b[16].tolist())
    check("mixed_batch_old_reads_minus_one", rows, sorted([[-1, -1, -1], [-1, 3, 0]]))

    buf2 = GraphReplayBuffer(10)
    buf2.push(**transition(rng))
    buf2.push(**transition(rng))
    random.seed(0)
    check("all_old_batch_field_is_none", buf2.sample(2)[16] is None, True)


# ------------------------------------------------------------------ 3. identical when -1
def test_minus_one_is_identical():
    for dd in (False, True):
        rng = np.random.default_rng(1)
        trs = [transition(rng) for _ in range(4)]
        a1 = small_agent(dd)
        a2 = copy.deepcopy(a1)
        for t in trs:
            a1.memory.push(**t)
            a2.memory.push(**dict(t, next_policy_action=np.full(N, -1, dtype=np.int64)))
        l1, t1 = learn_once(a1)
        l2, t2 = learn_once(a2)
        check(f"all_minus_one_same_targets(double={dd})", bool(torch.equal(t1, t2)), True)
        check(f"all_minus_one_same_loss(double={dd})", l1 == l2, True)
        same_w = all(torch.equal(p, q) for p, q in zip(a1.policy_net.parameters(),
                                                       a2.policy_net.parameters()))
        check(f"all_minus_one_same_weights(double={dd})", same_w, True)


# ------------------------------------------------------------------ 4. given move
def test_given_move_is_used():
    for dd in (False, True):
        rng = np.random.default_rng(2)
        base = transition(rng)
        # Four copies of ONE transition, so every batch row is the same and the
        # expected target can be computed for it directly.
        given = np.array([-1, 3, 0], dtype=np.int64)   # fleet 1: a route move; fleet 2: a stop
        a_ref = small_agent(dd)
        a_new = copy.deepcopy(a_ref)
        for _ in range(4):
            a_ref.memory.push(**base)
            a_new.memory.push(**dict(base, next_policy_action=given))
        _, t_ref = learn_once(a_ref)
        # Expected for the given fleets, from the TARGET net before learning.
        # On the agent's device (the GPU when there is one), like learn() itself.
        with torch.no_grad():
            nf = torch.from_numpy(base["next_node_features"])[None].to(DEVICE)
            na = torch.from_numpy(base["next_adj_matrix"])[None].to(DEVICE)
            mask = torch.ones(1, N, device=DEVICE)
            _, _, nq = a_new.target_net(nf, na, node_mask=mask)       # [1, N, K, A]
        r = torch.from_numpy(base["rewards"]).to(DEVICE)
        _, t_new = learn_once(a_new)
        g = a_new.gamma
        exp1 = r[1] + g * nq[0, 1, :, 3]
        exp2 = r[2] + g * nq[0, 2, :, 0]
        ok1 = bool(torch.allclose(t_new[0, 1], exp1, atol=1e-5))
        ok2 = bool(torch.allclose(t_new[0, 2], exp2, atol=1e-5))
        check(f"route_move_valued_at_that_move(double={dd})", ok1, True)
        check(f"stop_valued_at_wait(double={dd})", ok2, True)
        check(f"network_fleet_unchanged(double={dd})",
              bool(torch.equal(t_new[:, 0], t_ref[:, 0])), True)
        check(f"honest_share_reported(double={dd})", round(a_new.last_honest_share, 4),
              round(2 / 3, 4))


# ------------------------------------------------------------------ 5-9. real steps
def env_part():
    import core_warehouse
    from config_warehouse import CONFIG
    from node_warehouse import precompute_goal_distances
    from test_smoke_integration import build_instance, set_flags

    BOOK = {"enabled": True, "orders": 60, "release_steps": 150}
    saved = {k: dict(CONFIG.get(k, {}) or {}) for k in
             ("stream", "obstacles", "safety", "conflict", "reward_decomposition",
              "step", "training")}

    def configure(honest):
        CONFIG["stream"] = dict(saved["stream"], enabled=True, order_book=dict(BOOK))
        CONFIG["stream"].pop("order_floor_window", None)
        CONFIG["obstacles"] = dict(saved["obstacles"], enabled=False)
        sf = CONFIG.setdefault("safety", {})
        for k in ("intent_guard", "bubbles_by_kind", "hops_everywhere"):
            sf[k] = True
        CONFIG.setdefault("conflict", {})["route_by_default"] = True
        # Fix 1 on its own: route_explore (Fix 3) changes which fleets the route
        # drives, and is tested in test_route_explore.py.
        CONFIG["conflict"]["route_explore"] = False
        CONFIG.setdefault("step", {})["simultaneous"] = True
        CONFIG.setdefault("training", {})["honest_target"] = honest

    def make_env(seed=7, n_fleets=24):
        from agent_warehouse import GNNAgent as _G
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
        env.gnn = _G(
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
        return env

    try:
        # ---------------- ON
        configure(True)
        env = make_env()
        pushes = []
        _orig_push = env.gnn.memory.push

        def _rec_push(**kw):
            pushes.append({"actions": np.array(kw["actions"]).copy(),
                           "npa": (None if kw.get("next_policy_action") is None
                                   else np.array(kw["next_policy_action"]).copy()),
                           "nam": np.array(kw["next_active_mask"]).copy()})
            return _orig_push(**kw)
        env.gnn.memory.push = _rec_push

        snaps = []          # per step: ids, raw, routed, executed, next_ids
        steps = 0
        for _ in range(140):
            env.step(episode_step=1, total_episodes=1)
            steps += 1
            pend = env._honest_pending
            snaps.append({
                "ids": list(env._honest_ids),
                "raw": np.array(env._honest_raw).copy(),
                "routed": (np.array(env._route_mask, dtype=bool).copy()
                           if env._route_mask is not None
                           else np.zeros(len(env._honest_ids), dtype=bool)),
                "executed": (np.array(pend[0]["actions"]).copy() if pend is not None
                             else pushes[-1]["actions"]),
                "next_ids": (list(pend[1]) if pend is not None else None),
            })
            if env.is_episode_over():
                break
        flushed = env.flush_pending_transition()

        check("one_transition_per_step", len(pushes), steps)
        check("final_flush_pushed_one", flushed, 1)
        check("every_push_carries_the_field", all(p["npa"] is not None for p in pushes), True)
        in_order = all(np.array_equal(pushes[t]["actions"], snaps[t]["executed"])
                       for t in range(steps))
        check("transitions_in_step_order", in_order, True)

        bad, n_exec_routed, n_net, n_stop, roster_changes = 0, 0, 0, 0, 0
        for t in range(steps - 1):
            nxt, now = snaps[t]["next_ids"], snaps[t + 1]
            if nxt is None:
                continue
            if set(nxt) != set(now["ids"]):
                roster_changes += 1
            pos = {fid: j for j, fid in enumerate(now["ids"])}
            npa = pushes[t]["npa"]
            for k, fid in enumerate(nxt):
                j = pos.get(fid)
                if j is None:
                    want = -1
                else:
                    ex, rw, ro = int(now["executed"][j]), int(now["raw"][j]), bool(now["routed"][j])
                    want = -1 if (not ro and ex == rw) else ex
                    if pushes[t]["nam"][k] > 0:
                        if ro and want >= 0:
                            n_exec_routed += 1
                        if want == -1:
                            n_net += 1
                        if (not ro) and want == 0 and rw != 0:
                            n_stop += 1
                if int(npa[k]) != want:
                    bad += 1
        check("every_next_move_follows_the_rule", bad, 0)
        check("route_driven_valued_as_made", n_exec_routed > 0, True)
        check("network_driven_valued_at_its_best", n_net > 0, True)
        check("overruled_to_wait_valued_at_0", n_stop > 0, True)
        check("roster_changes_seen_and_matched", roster_changes > 0, True)
        print(f"      {steps} steps | route-driven valued as made {n_exec_routed} | "
              f"network-driven {n_net} | overruled to wait {n_stop} | roster changes {roster_changes}")

        est = env.get_error_statistics()
        check("stats_count_pushes", est["honest_pushed"], steps)
        print(f"      honest_exec_share {est['honest_exec_share']:.2f}")

        # The learning step on these transitions.
        loss = env.gnn.learn()
        check("learn_runs_on_held_transitions", np.isfinite(loss), True)
        check("learn_uses_the_moves", env.gnn.last_honest_share > 0.0, True)

        # ---------------- OFF
        configure(False)
        env2 = make_env()
        n2 = 0
        for _ in range(30):
            env2.step(episode_step=1, total_episodes=1)
            n2 += 1
        check("off_pushed_at_once", len(env2.gnn.memory), n2)
        check("off_no_hold", env2._honest_pending is None, True)
        check("off_14_fields", {len(e) for e in env2.gnn.memory.buffer}, {14})
        check("off_flush_is_noop", env2.flush_pending_transition(), 0)
    finally:
        for k, v in saved.items():
            CONFIG[k] = v


if __name__ == "__main__":
    test_storage_and_padding()
    test_minus_one_is_identical()
    test_given_move_is_used()
    env_part()
    print("\n" + ("ALL PASS" if not FAIL else f"FAILURES: {FAIL}"))
