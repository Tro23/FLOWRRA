"""
test_self_skip.py -- the self path (gnn.self_skip, build log entry 43).

THE IDEA. Message passing mixes each fleet's summary with its neighbours'. The
diagnostic (diag_clear.py) showed a fleet's own route direction fading as
neighbours are mixed in: the Learner reads the route at 97-100% with no
neighbour, 67% with 8 or more. The self path gives each Q head -- and the
Learner -- a second input beside the trunk's summary: the fleet's OWN encoding
from before message passing, through a bias-free linear map added to the head's
first layer. The maps start at exactly zero.

WHAT THIS CHECKS
  Part 1, small hand-built agents:
    1. Building the self path draws nothing from the global random generator,
       and every other layer is exactly as without it.
    2. At zero it is the same network: identical Q-values, per-head values and
       Learner logits on the same input.
    3. A warm start from a checkpoint WITHOUT the self path (ladder5) loads
       every other weight and BOTH optimizers' states exactly; the self path
       starts at zero; checkpoint_has_self_skip tells the two apart.
    4. The first update is the same as without it (the main path's weights end
       identical), and the self path itself learns (its weights leave zero),
       the Q heads' maps and the Learner's alike.
    5. A checkpoint WITH the self path round-trips: weights and its optimizer.
  Part 2, real FLOWRRA steps (24 fleets, the stream, guard, SARSA, router):
    6. With the conv encoder of the real config, 40 steps with learning run,
       every loss finite, and the self path moves off zero.
    7. deploy_eval's builder builds a self-path checkpoint WITH it and a plain
       one without.

Run:  python test_self_skip.py      (ends with ALL PASS; about a minute)
"""

import os
import random
import tempfile

import numpy as np
import torch

from agent_warehouse import DEVICE, GNNAgent

FAIL = []
A = 7
N = 4
HEADS = ["safety", "delivery", "efficiency"]


def check(name, got, want):
    ok = got == want
    print(f"{'PASS' if ok else 'FAIL'}  {name}: got {got!r} want {want!r}")
    if not ok:
        FAIL.append(name)


def small(skip, learner=True, lr=1e-2, seed=0):
    kw = dict(node_feature_dim=8, edge_feature_dim=0, action_size=A,
              hidden_dim=16, num_layers=1, n_heads=2,
              reward_heads=HEADS, head_weights=[6.0, 4.0, 1.0],
              batch_size=4, dropout=0.0, lr=lr, seed=seed, gamma=0.99,
              self_skip=skip)
    if learner:
        kw.update(learner_beta=0.0, learner_weight=1.0, learner_nudge="prob",
                  learner_beta_mode="gap", learner_beta_scale=0.0, learner_gap_min=1)
    return GNNAgent(**kw)


def transition(rng, n=N, label=None):
    kw = dict(
        node_features=rng.normal(size=(n, 8)).astype(np.float32),
        adj_matrix=np.ones((n, n), dtype=np.float32),
        actions=rng.integers(0, A, size=n).astype(np.int64),
        rewards=rng.normal(scale=0.1, size=(n, len(HEADS))).astype(np.float32),
        next_node_features=rng.normal(size=(n, 8)).astype(np.float32),
        next_adj_matrix=np.ones((n, n), dtype=np.float32),
        done=False, integrity=1.0, active_mask=np.ones(n, dtype=np.float32),
        recovery_action=0,
        next_valid_mask=np.ones((n, A), dtype=bool),
        next_active_mask=np.ones(n, dtype=np.float32),
        valid_mask=np.ones((n, A), dtype=bool),
        teacher_mask=np.zeros(n, dtype=np.float32),
    )
    if label is not None:
        kw["learner_label"] = np.asarray(label, dtype=np.int64)
    return kw


def plain_params(agent):
    """Every parameter except the self path's, in module order."""
    sk = {id(p) for p in agent._skip_params()}
    return [p for p in agent.policy_net.parameters() if id(p) not in sk]


def skip_params(agent):
    return agent._skip_params()


# ------------------------------------------------------------------ 1. isolated init
def test_isolated_init():
    torch.manual_seed(5)
    a = small(skip=False, seed=None)
    s_off = torch.get_rng_state().clone()
    torch.manual_seed(5)
    b = small(skip=True, seed=None)
    s_on = torch.get_rng_state().clone()
    check("global_generator_untouched_by_the_self_path", bool(torch.equal(s_off, s_on)), True)
    same = all(torch.equal(x, y) for x, y in zip(plain_params(a), plain_params(b)))
    check("every_other_layer_identical", same, True)
    check("same_parameter_count_outside_the_self_path",
          len(plain_params(a)), len(list(a.policy_net.parameters())))
    check("self_path_starts_at_zero",
          all(bool((p == 0).all()) for p in skip_params(b)), True)
    check("self_path_covers_heads_and_learner",
          (b.policy_net.self_skip is not None and len(b.policy_net.self_skip) == len(HEADS)
           and b.policy_net.learner_skip is not None), True)


# ------------------------------------------------------------------ 2. same network at zero
def test_same_network_at_zero():
    a, b = small(skip=False), small(skip=True)
    a.policy_net.eval(); b.policy_net.eval()
    g = torch.Generator().manual_seed(3)
    x = torch.randn(2, N, 8, generator=g).to(DEVICE)
    adj = torch.ones(2, N, N).to(DEVICE)
    with torch.no_grad():
        qa, _, pa = a.policy_net(x, adj)
        la = a.policy_net.last_learner_logits.clone()
        qb, _, pb = b.policy_net(x, adj)
        lb = b.policy_net.last_learner_logits.clone()
    check("q_values_identical", bool(torch.equal(qa, qb)), True)
    check("per_head_values_identical", bool(torch.equal(pa, pb)), True)
    check("learner_logits_identical", bool(torch.equal(la, lb)), True)


# ------------------------------------------------------------------ 3. warm start
def test_warm_start_from_a_checkpoint_without_the_self_path():
    rng = np.random.default_rng(6)
    src = small(skip=False)
    label = [1 << 3, -1, 1 << 5, -1]
    for _ in range(6):
        src.memory.push(**transition(rng, label=label))
    random.seed(1); torch.manual_seed(1)
    src.learn()
    with tempfile.TemporaryDirectory() as d:
        p0 = os.path.join(d, "plain.pth")
        src.save(p0)
        check("checkpoint_has_no_self_path", GNNAgent.checkpoint_has_self_skip(p0), False)
        dst = small(skip=True, seed=3)
        dst.load(p0)
        same = all(torch.equal(x, y) for x, y in zip(plain_params(src), plain_params(dst)))
        check("every_other_weight_loaded_exactly", same, True)
        for nm in ("optimizer", "learner_optimizer"):
            so = getattr(src, nm).state_dict()["state"]
            do = getattr(dst, nm).state_dict()["state"]
            ok = (so.keys() == do.keys() and all(
                torch.equal(so[k]["exp_avg"], do[k]["exp_avg"])
                and torch.equal(so[k]["exp_avg_sq"], do[k]["exp_avg_sq"]) for k in so))
            check(f"{nm}_state_loaded_exactly", ok, True)
        check("self_path_zero_after_warm_start",
              all(bool((p == 0).all()) for p in skip_params(dst)), True)
        src.policy_net.eval(); dst.policy_net.eval()
        x = torch.randn(1, N, 8).to(DEVICE)
        adj = torch.ones(1, N, N).to(DEVICE)
        with torch.no_grad():
            check("warm_started_network_is_the_same",
                  bool(torch.equal(src.policy_net(x, adj)[0], dst.policy_net(x, adj)[0])), True)


# ------------------------------------------------------------------ 4. first update
def test_first_update_and_it_learns():
    rng_a, rng_b = np.random.default_rng(1), np.random.default_rng(1)
    off, on = small(skip=False), small(skip=True)
    label = [1 << 3, -1, 1 << 5, -1]
    for _ in range(6):
        off.memory.push(**transition(rng_a, label=label))
        on.memory.push(**transition(rng_b, label=label))
    random.seed(10); torch.manual_seed(10)
    off.learn()
    random.seed(10); torch.manual_seed(10)
    on.learn()
    same = all(torch.allclose(x, y, rtol=0.0, atol=1e-7)
               for x, y in zip(plain_params(off), plain_params(on)))
    check("first_update_main_path_identical", same, True)
    moved_q = any(bool((p != 0).any()) for p in on.policy_net.self_skip.parameters())
    moved_l = bool((on.policy_net.learner_skip.weight != 0).any())
    check("q_heads_self_path_learns", moved_q, True)
    check("learner_self_path_learns", moved_l, True)
    for i in range(4):
        random.seed(20 + i); torch.manual_seed(20 + i)
        on.learn()
    check("losses_finite_after_more_updates",
          all(np.isfinite(v) for v in on.last_head_losses.values()), True)


# ------------------------------------------------------------------ 5. round trip
def test_round_trip():
    rng = np.random.default_rng(4)
    a = small(skip=True)
    for _ in range(6):
        a.memory.push(**transition(rng, label=[1 << 2, -1, 1 << 4, -1]))
    for i in range(3):
        random.seed(i); torch.manual_seed(i)
        a.learn()
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "skip.pth")
        a.save(p)
        check("checkpoint_has_self_path", GNNAgent.checkpoint_has_self_skip(p), True)
        b = small(skip=True, seed=9)
        b.load(p)
        same = all(torch.equal(x, y) for x, y in zip(a.policy_net.parameters(),
                                                     b.policy_net.parameters()))
        check("self_path_checkpoint_round_trip", same, True)
        sa, sb = a.skip_optimizer.state_dict()["state"], b.skip_optimizer.state_dict()["state"]
        check("skip_optimizer_round_trip",
              sa.keys() == sb.keys() and all(torch.equal(sa[k]["exp_avg"], sb[k]["exp_avg"])
                                             for k in sa), True)
        c = small(skip=False, seed=9)
        try:
            c.load(p)
            loud = False
        except RuntimeError:
            loud = True
        check("self_path_checkpoint_into_plain_agent_fails_loudly", loud, True)


# ------------------------------------------------------------------ 6-7. real steps
def env_part():
    import core_warehouse
    from config_warehouse import CONFIG
    from node_warehouse import precompute_goal_distances
    from test_smoke_integration import build_instance, set_flags

    BOOK = {"enabled": True, "orders": 60, "release_steps": 150}
    saved = {k: dict(CONFIG.get(k, {}) or {}) for k in
             ("stream", "obstacles", "safety", "conflict", "step", "training", "gnn")}
    LRN = dict(CONFIG["training"].get("learner", {}) or {})

    def configure():
        CONFIG["stream"] = dict(saved["stream"], enabled=True, order_book=dict(BOOK))
        CONFIG["stream"].pop("order_floor_window", None)
        CONFIG["obstacles"] = dict(saved["obstacles"], enabled=False)
        sf = CONFIG.setdefault("safety", {})
        for k in ("intent_guard", "bubbles_by_kind", "hops_everywhere"):
            sf[k] = True
        CONFIG.setdefault("conflict", {})["route_by_default"] = False
        CONFIG.setdefault("step", {})["simultaneous"] = True
        CONFIG.setdefault("training", {})["honest_target"] = True
        CONFIG["training"]["sarsa"] = True
        CONFIG["training"]["learner"] = dict(LRN, enabled=True, mode="router")
        CONFIG["gnn"] = dict(saved["gnn"], next_state_adjacency="choice",
                             next_state_edge_features=True)

    def make_env(skip, seed=7, n_fleets=24):
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
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
        _edim = int(CONFIG["gnn"].get("edge_feature_dim", 0))
        env.gnn = GNNAgent(
            node_feature_dim=base_dim + env.density.output_dim, edge_feature_dim=_edim,
            action_size=CONFIG["gnn"]["action_size"], hidden_dim=CONFIG["gnn"]["hidden_dim"],
            num_layers=CONFIG["gnn"]["num_layers"], n_heads=CONFIG["gnn"]["num_heads"],
            reward_heads=rd["heads"], head_weights=rd["weights"],
            dropout=CONFIG["gnn"]["dropout"], lr=CONFIG["gnn"]["learning_rate"],
            gamma=CONFIG["training"]["gamma"],
            buffer_capacity=CONFIG["training"]["buffer_capacity"], batch_size=16,
            stability_coef=CONFIG["gnn"]["stability_coef"],
            encoder_mode=CONFIG["gnn"]["encoder"], base_feature_dim=base_dim,
            density_diamond_mask=env.density._diamond_mask,
            double_dqn=True, per_fleet_terminal=True,
            learner_beta=0.0, learner_weight=1.0, learner_nudge="prob",
            learner_beta_mode="gap", learner_beta_scale=1.0, learner_gap_min=50,
            self_skip=skip)
        env.gnn.epsilon_gaussian = lambda *a, **k: 0.1
        return env, (G, grid_pos, missions, goal_pool)

    try:
        configure()
        env, inst = make_env(skip=True)
        finite, learned = True, 0
        for _ in range(40):
            env.step(episode_step=1, total_episodes=1)
            if len(env.gnn.memory) >= env.gnn.batch_size:
                loss = env.gnn.learn()
                learned += 1
                vals = [float(loss)] + [float(v) for v in env.gnn.last_head_losses.values()]
                finite = finite and all(np.isfinite(vals))
            if env.is_episode_over():
                break
        check("real_steps_learned", learned > 10, True)
        check("real_steps_losses_finite", finite, True)
        check("real_self_path_moved", any(bool((p != 0).any()) for p in env.gnn._skip_params()), True)

        from benchmark_flowrra import build_agent
        G, grid_pos, missions, goal_pool = inst
        env2, _ = make_env(skip=False)
        with tempfile.TemporaryDirectory() as d:
            p_on, p_off = os.path.join(d, "on.pth"), os.path.join(d, "off.pth")
            env.gnn.save(p_on)
            env2.gnn.save(p_off)
            a_on = build_agent(G, grid_pos, missions, goal_pool, p_on)
            a_off = build_agent(G, grid_pos, missions, goal_pool, p_off)
            check("deploy_self_path_arm_has_it", a_on.policy_net.self_skip is not None, True)
            check("deploy_plain_arm_has_none", a_off.policy_net.self_skip is None, True)
    finally:
        for k, v in saved.items():
            CONFIG[k] = v


if __name__ == "__main__":
    test_isolated_init()
    test_same_network_at_zero()
    test_warm_start_from_a_checkpoint_without_the_self_path()
    test_first_update_and_it_learns()
    test_round_trip()
    env_part()
    print("\n" + ("ALL PASS" if not FAIL else f"FAILURES: {FAIL}"))
