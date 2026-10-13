"""
test_sarsa.py -- SARSA everywhere: every next move is the move made
(build log entry 38; switch training.sarsa).

THE PROBLEM. After the honest target (entry 34) one max was left. Where the
NETWORK drove a fleet in the next state, the learning target still used its
best move by today's values (-1 -> argmax in learn()). That move may never run
there -- the guard would stop it, a rule would override it -- so its value is
never corrected, and run 7's values kept climbing (entry 35).

THE FIX. With training.sarsa on, the core writes the move the network actually
made (its exploring moves included) for those fleets too. Every active
next-state fleet then carries a real move, learn() never needs its argmax for
them, and the target is r + gamma * Q_target(s', move made): SARSA. Nothing
else changes: route-, rule-, hold- and guard-driven fleets keep the move made,
and an explorer in the clear keeps the route's move (Fix 3).

WHAT THIS CHECKS
  Part 1, the learning step (small hand-built agent):
    1. Every fleet given: the target is exactly r + gamma * Q_target(s', given)
       for every fleet and head (plain and Double DQN).
    2. No max is left: with every fleet given, scrambling the ONLINE network
       (which picks the argmax in Double DQN) changes no target. The control:
       with -1 it does change them.
  Part 2, inside real FLOWRRA steps (24 fleets, the stream, route + guard +
  route-explore + honest target on; exploration raised to 0.3 so every case
  occurs often). The same seed is run with SARSA off and on; nothing learns
  during the steps, so both runs make exactly the same moves:
    3. The moves are identical in both runs (the switch changes only labels).
    4. Where run 7's rule gave a move, SARSA gives the same move.
    5. Where run 7's rule gave -1 for a fleet still on the floor (the network
       drove it), SARSA gives the move it made -- exploring moves included.
       A fleet that left the floor reads -1 in both (its bootstrap is masked).
    6. No active next-state fleet reads -1 with SARSA on (flushes aside).
    7. Counters: honest_sarsa = the network-driven fleets labelled;
       honest_labelled_share = 1.0; honest_exec_share the same in both runs.
    8. learn() on these transitions sees every active fleet given (share 1.0).
  Part 3:
    9. SARSA on with the honest target off: a warning, and nothing changes
       (pushed at once, 14 fields, no hold).

Run:  python test_sarsa.py      (ends with ALL PASS; a few minutes)
"""

import copy
import random

import numpy as np
import torch

from agent_warehouse import DEVICE, GNNAgent
from test_honest_target import N, learn_once, small_agent, transition

FAIL = []


def check(name, got, want):
    ok = got == want
    print(f"{'PASS' if ok else 'FAIL'}  {name}: got {got!r} want {want!r}")
    if not ok:
        FAIL.append(name)


# ------------------------------------------------------------------ 1. exact target
def test_every_fleet_given_is_exact():
    for dd in (False, True):
        rng = np.random.default_rng(11)
        base = transition(rng)
        given = np.array([2, 3, 0], dtype=np.int64)    # a move, another move, a stop
        a = small_agent(dd)
        for _ in range(4):
            a.memory.push(**dict(base, next_policy_action=given))
        with torch.no_grad():
            nf = torch.from_numpy(base["next_node_features"])[None].to(DEVICE)
            na = torch.from_numpy(base["next_adj_matrix"])[None].to(DEVICE)
            mask = torch.ones(1, N, device=DEVICE)
            _, _, nq = a.target_net(nf, na, node_mask=mask)          # [1, N, K, A]
        r = torch.from_numpy(base["rewards"]).to(DEVICE)
        _, t = learn_once(a)
        ok = all(bool(torch.allclose(t[0, i], r[i] + a.gamma * nq[0, i, :, int(given[i])],
                                     atol=1e-5)) for i in range(N))
        check(f"target_is_r_plus_gamma_q_of_move_made(double={dd})", ok, True)
        check(f"every_fleet_given_share_1(double={dd})", round(a.last_honest_share, 6), 1.0)


# ------------------------------------------------------------------ 2. no max left
def _scramble(net, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    with torch.no_grad():
        for p in net.parameters():
            p.copy_((torch.randn(p.shape, generator=g) * 0.5).to(p.device))


def _targets_before_and_after_scramble(npa):
    rng = np.random.default_rng(12)
    base = transition(rng)
    a1 = small_agent(double_dqn=True)
    for _ in range(4):
        a1.memory.push(**dict(base, next_policy_action=np.array(npa, dtype=np.int64)))
    a2 = copy.deepcopy(a1)
    _scramble(a2.policy_net, seed=99)          # the ONLINE net only; the target net is untouched
    _, t1 = learn_once(a1)
    _, t2 = learn_once(a2)
    return bool(torch.allclose(t1, t2, atol=1e-6))


def test_no_max_left():
    check("all_given_online_net_cannot_change_targets",
          _targets_before_and_after_scramble([2, 3, 0]), True)
    check("control_minus_one_online_net_changes_targets",
          _targets_before_and_after_scramble([-1, -1, -1]), False)


# ------------------------------------------------------------------ 3-9. real steps
def env_part():
    import core_warehouse
    from config_warehouse import CONFIG
    from node_warehouse import precompute_goal_distances
    from test_smoke_integration import build_instance, set_flags

    BOOK = {"enabled": True, "orders": 60, "release_steps": 150}
    saved = {k: dict(CONFIG.get(k, {}) or {}) for k in
             ("stream", "obstacles", "safety", "conflict", "reward_decomposition",
              "step", "training")}

    def configure(sarsa, honest=True):
        CONFIG["stream"] = dict(saved["stream"], enabled=True, order_book=dict(BOOK))
        CONFIG["stream"].pop("order_floor_window", None)
        CONFIG["obstacles"] = dict(saved["obstacles"], enabled=False)
        sf = CONFIG.setdefault("safety", {})
        for k in ("intent_guard", "bubbles_by_kind", "hops_everywhere"):
            sf[k] = True
        CONFIG.setdefault("conflict", {})["route_by_default"] = True
        CONFIG["conflict"]["route_explore"] = True
        CONFIG.setdefault("step", {})["simultaneous"] = True
        CONFIG.setdefault("training", {})["honest_target"] = honest
        CONFIG["training"]["sarsa"] = sarsa
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
        # Explore often, so every case under test happens many times.
        env.gnn.epsilon_gaussian = lambda *a, **k: 0.3
        return env

    def run(env, steps=80):
        """Step without learning; record every push (marking those made by a
        flush) and, per step, what the label rule needs."""
        pushes = []
        state = {"flushing": False}
        _orig_push = env.gnn.memory.push
        _orig_flush = env.flush_pending_transition

        def _rec_push(**kw):
            pushes.append({"actions": np.array(kw["actions"]).copy(),
                           "npa": (None if kw.get("next_policy_action") is None
                                   else np.array(kw["next_policy_action"]).copy()),
                           "nam": np.array(kw["next_active_mask"]).copy(),
                           "flushed": state["flushing"]})
            return _orig_push(**kw)

        def _rec_flush():
            state["flushing"] = True
            try:
                return _orig_flush()
            finally:
                state["flushing"] = False
        env.gnn.memory.push = _rec_push
        env.flush_pending_transition = _rec_flush

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
                # this step's own transition, held: its actions are the moves made
                "executed": (np.array(pend[0]["actions"]).copy() if pend is not None else None),
                "next_ids": (list(pend[1]) if pend is not None else None),
            })
            if env.is_episode_over():
                break
        return snaps, pushes

    try:
        # ---------------- the same seed, SARSA off and on
        configure(sarsa=False)
        env_off = make_env()
        snaps_off, pushes_off = run(env_off)
        est_off = env_off.get_error_statistics()

        configure(sarsa=True)
        env_on = make_env()
        snaps_on, pushes_on = run(env_on)
        est_on = env_on.get_error_statistics()

        same_moves = (len(pushes_off) == len(pushes_on) and all(
            np.array_equal(p["actions"], q["actions"]) for p, q in zip(pushes_off, pushes_on)))
        check("same_seed_same_moves_off_and_on", same_moves, True)

        # The move each fleet made at step t+1: step t+1's own (held) transition.
        bad_kept, bad_net, bad_left, active_minus_one = 0, 0, 0, 0
        n_kept, n_net, n_net_explore, n_left = 0, 0, 0, 0
        for t in range(len(pushes_on) - 1):
            if pushes_on[t]["flushed"] or pushes_off[t]["flushed"]:
                continue
            nxt, now = snaps_on[t]["next_ids"], snaps_on[t + 1]
            if nxt is None or now["executed"] is None:
                continue
            executed = now["executed"]
            pos = {fid: j for j, fid in enumerate(now["ids"])}
            o, s, nam = pushes_off[t]["npa"], pushes_on[t]["npa"], pushes_on[t]["nam"]
            for k, fid in enumerate(nxt):
                j = pos.get(fid)
                if int(o[k]) >= 0:
                    n_kept += 1
                    bad_kept += int(int(s[k]) != int(o[k]))
                elif j is None:
                    n_left += 1
                    bad_left += int(int(s[k]) != -1)
                else:
                    n_net += 1
                    bad_net += int(int(s[k]) != int(executed[j]))
                    if now["explore"][j] and not now["eligible"][j]:
                        n_net_explore += 1
                if nam[k] > 0 and int(s[k]) < 0:      # an active fleet left to the argmax
                    active_minus_one += 1
        check("run7_labels_kept_where_given", bad_kept, 0)
        check("network_driven_labelled_with_move_made", bad_net, 0)
        check("fleet_that_left_reads_minus_one", bad_left, 0)
        check("no_active_fleet_reads_minus_one", active_minus_one, 0)
        check("network_driven_cases_seen", n_net > 0, True)
        check("network_exploring_moves_labelled_too", n_net_explore > 0, True)
        print(f"      {len(pushes_on)} pushes | kept from run 7's rule {n_kept} | "
              f"network-driven now labelled {n_net} (of them exploring {n_net_explore}) | "
              f"left the floor {n_left}")

        check("counter_off_is_zero", est_off["honest_sarsa"], 0)
        check("counter_on_counts_network_driven", est_on["honest_sarsa"] > 0, True)
        check("labelled_share_on_is_1", round(est_on["honest_labelled_share"], 6), 1.0)
        check("exec_share_same_off_and_on",
              round(est_on["honest_exec_share"], 9), round(est_off["honest_exec_share"], 9))
        check("sarsa_on_reported", (est_off["honest_sarsa_on"], est_on["honest_sarsa_on"]), (0, 1))

        # learn() BEFORE the final flush: every stored transition is complete.
        loss = env_on.gnn.learn()
        check("learn_runs", bool(np.isfinite(loss)), True)
        check("learn_sees_every_active_fleet_given", round(env_on.gnn.last_honest_share, 6), 1.0)
        check("final_flush_pushes_one", env_on.flush_pending_transition(), 1)

        # ---------------- SARSA on, honest target off: nothing changes
        configure(sarsa=True, honest=False)
        env3 = make_env()
        n3 = 0
        for _ in range(20):
            env3.step(episode_step=1, total_episodes=1)
            n3 += 1
        check("without_honest_pushed_at_once", len(env3.gnn.memory), n3)
        check("without_honest_14_fields", {len(e) for e in env3.gnn.memory.buffer}, {14})
        check("without_honest_reported_off", env3.get_error_statistics()["honest_sarsa_on"], 0)
    finally:
        for k, v in saved.items():
            CONFIG[k] = v


if __name__ == "__main__":
    test_every_fleet_given_is_exact()
    test_no_max_left()
    env_part()
    print("\n" + ("ALL PASS" if not FAIL else f"FAILURES: {FAIL}"))
