"""
test_learner_router.py -- the Learner as the router (build log entry 42;
training.learner, mode "router").

THE IDEA. In the clear the hard-coded route drives, so the network's own moves
there never run and their values are never corrected (it disagrees with the
route on ~40% of clear moves). The Learner head learns the route's move for
every fleet IN THE CLEAR -- computed, executed or not -- and nudges only those
fleets: score = Q + beta x P_Learner, beta = scale x the heads' median gap.
The runner hands the clear over: route ON with the Learner in shadow (scale
0) until it agrees with the route, then route OFF and the scale falls to 0.

WHAT THIS CHECKS
  Part 1, the agent (small hand-built agents):
    1. The Learner's initialisation draws from its own random stream: building
       it leaves the global generator, and every other layer, exactly as
       without it.
    2. Learning with the router's labels moves ONLY the Learner: the trunk and
       the Q heads end bit-identical to an agent that has no Learner at all
       (separate optimizer, separate clipping, detached trunk).
    3. The labels teach it: taught exactly where a label is set; it learns them.
       A label is a bit mask of the route's equally-short moves: a tie is
       learned as equally right (soft target) and agreement counts either.
    4. A warm start from a checkpoint WITHOUT a Learner (ladder4) loads: every
       other weight and the main optimizer's state exactly; the Learner fresh.
       checkpoint_has_learner tells the two apart.
    5. Scale 0 and a uniform Learner change no choice; a certain Learner moves
       only fleets inside the nudge region.
    6. The schedule (router_plan / router_scale): 80 and 12 episodes.
  Part 2, inside real FLOWRRA steps (24 fleets, the stream, guard, SARSA):
    7. Route ON: the router's clear set is exactly the route's own eligibility,
       and the route's move is one of each label's moves, every step; the
       nudge region is the clear set; every push carries that step's labels.
    8. Route OFF: the route drives nobody, the labels are still computed, and
       the nudge region is still the clear set.
    9. Shadow is silent: router on (scale 0) against the Learner switched off,
       same seed. (a) The same steps give identical moves. (b) Learning on the
       same memory moves the trunk and Q heads no further from the Learner-free
       agent than a copy of that agent moves from itself (the GPU's own
       non-repeatability is the yardstick); only the Learner learned.
   10. deploy_eval's builder: a Learner checkpoint loads WITH its Learner (at
       the schedule's end scale), a Learner-free one without.

Run:  python test_learner_router.py      (ends with ALL PASS; a few minutes)
"""

import copy
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


def small(learner, lr=1e-2, seed=0):
    kw = dict(node_feature_dim=8, edge_feature_dim=0, action_size=A,
              hidden_dim=16, num_layers=1, n_heads=2,
              reward_heads=HEADS, head_weights=[6.0, 4.0, 1.0],
              batch_size=4, dropout=0.0, lr=lr, seed=seed, gamma=0.99)
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


def non_learner_params(agent):
    lh = agent.policy_net.learner_head
    ids = set() if lh is None else {id(p) for p in lh.parameters()}
    return [p for p in agent.policy_net.parameters() if id(p) not in ids]


# ------------------------------------------------------------------ 1. isolated init
def test_isolated_init():
    torch.manual_seed(5)
    a = small(learner=False, seed=None)
    s_off = torch.get_rng_state().clone()
    torch.manual_seed(5)
    b = small(learner=True, seed=None)
    s_on = torch.get_rng_state().clone()
    check("global_generator_untouched_by_the_learner", bool(torch.equal(s_off, s_on)), True)
    same = all(torch.equal(x, y) for x, y in zip(non_learner_params(a), non_learner_params(b)))
    check("every_other_layer_identical", same, True)
    c = small(learner=True, seed=11)
    same_l = all(torch.equal(x, y) for x, y in zip(b.policy_net.learner_head.parameters(),
                                                     c.policy_net.learner_head.parameters()))
    check("learner_init_is_its_own_fixed_stream", same_l, True)


# ------------------------------------------------------------------ 2. only the Learner moves
def test_only_the_learner_moves():
    rng_a, rng_b = np.random.default_rng(1), np.random.default_rng(1)
    off, on = small(learner=False), small(learner=True)
    label = [1 << 3, -1, 1 << 5, -1]        # bit masks of the route's moves; -1: not in the clear
    for _ in range(6):
        t_a, t_b = transition(rng_a), transition(rng_b, label=label)
        off.memory.push(**t_a)
        on.memory.push(**t_b)
    for i in range(5):
        random.seed(10 + i); torch.manual_seed(10 + i)
        off.learn()
        random.seed(10 + i); torch.manual_seed(10 + i)
        on.learn()
    same = all(torch.allclose(x, y, rtol=0.0, atol=1e-7)
               for x, y in zip(non_learner_params(off), non_learner_params(on)))
    check("trunk_and_q_heads_identical_to_no_learner", same, True)
    fresh = small(learner=True)
    moved = any(not torch.equal(x, y) for x, y in zip(on.policy_net.learner_head.parameters(),
                                                       fresh.policy_net.learner_head.parameters()))
    check("the_learner_itself_learned", moved, True)
    check("sample_carries_router_labels", on.memory.sample(2)[17] is not None, True)
    check("no_label_no_router_field", off.memory.sample(2)[17] is None, True)


# ------------------------------------------------------------------ 3. the labels teach it
def test_labels_teach_it():
    rng = np.random.default_rng(3)
    a = small(learner=True)
    for _ in range(8):
        a.memory.push(**transition(rng, label=[1 << 5, -1, 1 << 5, -1]))
    random.seed(4); torch.manual_seed(4)
    a.learn()
    check("taught_only_where_a_label_is_set", a.last_learner_n, 8.0)   # 2 of 4 fleets x batch 4
    first = a.last_learner_loss
    for _ in range(150):
        a.learn()
    print(f"      learner loss {first:.3f} -> {a.last_learner_loss:.3f} | agreement {a.last_learner_agree:.2f}")
    check("loss_falls", a.last_learner_loss < first, True)
    check("learns_the_route_move", a.last_learner_agree, 1.0)

    # A TIE: moves 2 and 4 equally short. The soft label teaches both as
    # equally right, so the Learner stays neutral between them (the heads
    # choose) and counts as agreeing on either.
    rng = np.random.default_rng(13)
    b = small(learner=True)
    tie = (1 << 2) | (1 << 4)
    for _ in range(8):
        b.memory.push(**transition(rng, label=[tie, -1, tie, -1]))
    random.seed(5); torch.manual_seed(5)
    for _ in range(200):
        b.learn()
    random.seed(6)
    bt = b.memory.sample(4)                 # the states it was taught on
    with torch.no_grad():
        b.policy_net.eval()
        b.policy_net(bt[0], bt[1], node_mask=bt[14])
        pr = torch.softmax(b.policy_net.last_learner_logits, dim=2)   # [4, N, A]
        p = pr[:, [0, 2], :].reshape(-1, A).mean(dim=0).cpu().numpy()  # the labelled fleets
    print(f"      tie 2|4: p2 {p[2]:.2f} p4 {p[4]:.2f} others {1 - p[2] - p[4]:.2f}")
    check("tie_learned_as_equally_right", bool(abs(p[2] - p[4]) < 0.15 and p[2] + p[4] > 0.8), True)
    check("tie_agreement_counts_either", b.last_learner_agree, 1.0)


# ------------------------------------------------------------------ 4. warm start
def test_warm_start_from_a_checkpoint_without_learner():
    rng = np.random.default_rng(6)
    src = small(learner=False)
    for _ in range(6):
        src.memory.push(**transition(rng))
    random.seed(1); torch.manual_seed(1)
    src.learn()
    with tempfile.TemporaryDirectory() as d:
        p0 = os.path.join(d, "no_learner.pth")
        src.save(p0)
        check("checkpoint_has_no_learner", GNNAgent.checkpoint_has_learner(p0), False)
        dst = small(learner=True, seed=3)
        dst.load(p0)
        same = all(torch.equal(x, y) for x, y in zip(non_learner_params(src), non_learner_params(dst)))
        check("every_other_weight_loaded_exactly", same, True)
        so, do = src.optimizer.state_dict()["state"], dst.optimizer.state_dict()["state"]
        same_opt = (so.keys() == do.keys() and all(
            torch.equal(so[k]["exp_avg"], do[k]["exp_avg"]) for k in so))
        check("main_optimizer_state_loaded_exactly", same_opt, True)
        fresh = small(learner=True)
        same_l = all(torch.equal(x, y) for x, y in zip(dst.policy_net.learner_head.parameters(),
                                                         fresh.policy_net.learner_head.parameters()))
        check("learner_starts_fresh", same_l, True)
        p1 = os.path.join(d, "with_learner.pth")
        dst.save(p1)
        check("checkpoint_has_learner", GNNAgent.checkpoint_has_learner(p1), True)
        back = small(learner=True, seed=9)
        back.load(p1)
        same_b = all(torch.equal(x, y) for x, y in zip(back.policy_net.parameters(),
                                                         dst.policy_net.parameters()))
        check("learner_checkpoint_round_trip", same_b, True)


# ------------------------------------------------------------------ 5. the nudge
def _greedy(a, feats, adj, valid, region):
    return a.choose_actions(node_features=feats, adj_matrix=adj, episode_number=0,
                            total_episodes=1, eps_min=0.0, eps_peak=0.0,
                            valid_action_masks=valid, conflict_mask=region)


def test_the_nudge():
    rng = np.random.default_rng(7)
    n = 6
    feats = rng.normal(size=(n, 8)).astype(np.float32)
    adj = np.ones((n, n), dtype=np.float32)
    valid = np.ones((n, A), dtype=bool)
    base = _greedy(small(learner=False), feats, adj, valid, None)
    k = (int(base[0]) + 1) % A
    region = np.array([True, True, True, False, False, False])

    def certain(scale, uniform=False):
        a = small(learner=True)
        a._gap_window.extend([1e6] * 10)            # a huge typical gap: a certain Learner wins
        a.learner_beta_scale = scale
        last = a.policy_net.learner_head[-1]
        with torch.no_grad():
            last.weight.zero_()
            last.bias.zero_()
            if not uniform:
                last.bias[k] = 50.0
        return a

    check("scale_0_changes_nothing",
          _greedy(certain(0.0), feats, adj, valid, region).tolist(), base.tolist())
    check("uniform_learner_changes_nothing",
          _greedy(certain(3.0, uniform=True), feats, adj, valid, region).tolist(), base.tolist())
    out = _greedy(certain(3.0), feats, adj, valid, region)
    check("certain_learner_moves_the_region", [int(x) for x in out[region]], [k] * 3)
    check("outside_the_region_untouched", out[~region].tolist(), base[~region].tolist())
    a = certain(0.0)
    _greedy(a, feats, adj, valid, region)
    check("shadow_still_collects_gaps", len(a.learner_q_gaps) > 0, True)
    check("shadow_changes_no_choice", a.learner_changed, 0)


# ------------------------------------------------------------------ 6. the schedule
def test_schedule():
    from main_runner_warehouse import router_plan, router_scale
    lrn = {"router": {"switch_agreement": 0.98, "warmup_min": 3, "warmup_frac": 0.0625,
                      "tail_frac": 0.1875, "scale_start": 3.0, "scale_end": 0.0}}
    p80, p12 = router_plan(lrn, 80), router_plan(lrn, 12)
    check("plan_80", (p80["warmup_min"], p80["fade_end"]), (5, 65))
    check("plan_12", (p12["warmup_min"], p12["fade_end"]), (3, 10))
    check("before_the_switch_scale_0", router_scale(5, 6, p80), 0.0)
    check("never_switched_scale_0", router_scale(40, None, p80), 0.0)
    check("at_the_switch_full_scale", router_scale(6, 6, p80), 3.0)
    check("halfway", round(router_scale(6 + 59 // 2, 6, p80), 3), round(3.0 * (1 - 29 / 59), 3))
    check("at_fade_end_0", router_scale(65, 6, p80), 0.0)
    check("tail_0", router_scale(80, 6, p80), 0.0)


# ------------------------------------------------------------------ 7-10. real steps
def env_part():
    import core_warehouse
    from config_warehouse import CONFIG
    from node_warehouse import precompute_goal_distances
    from test_smoke_integration import build_instance, set_flags

    BOOK = {"enabled": True, "orders": 60, "release_steps": 150}
    saved = {k: dict(CONFIG.get(k, {}) or {}) for k in
             ("stream", "obstacles", "safety", "conflict", "reward_decomposition",
              "step", "training")}
    LRN = dict(CONFIG["training"].get("learner", {}) or {})

    def configure(router, route_on=True):
        CONFIG["stream"] = dict(saved["stream"], enabled=True, order_book=dict(BOOK))
        CONFIG["stream"].pop("order_floor_window", None)
        CONFIG["obstacles"] = dict(saved["obstacles"], enabled=False)
        sf = CONFIG.setdefault("safety", {})
        for k in ("intent_guard", "bubbles_by_kind", "hops_everywhere"):
            sf[k] = True
        CONFIG.setdefault("conflict", {})["route_by_default"] = route_on
        CONFIG["conflict"]["route_explore"] = True
        CONFIG.setdefault("step", {})["simultaneous"] = True
        CONFIG.setdefault("training", {})["honest_target"] = True
        CONFIG["training"]["sarsa"] = True
        CONFIG["training"]["learner"] = dict(LRN, enabled=router, mode="router")

    def make_env(learner_head, seed=7, n_fleets=24, scale=0.0):
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
        # Edge width as the runner and deploy_eval's builder set it, so the
        # saved checkpoints in part 10 load into deploy's agents.
        _edim = int(CONFIG["gnn"].get("edge_feature_dim", 0))
        _lad = (CONFIG.get("conflict", {}) or {}).get("ladder", {}) or {}
        if _lad.get("enabled") and _lad.get("pair_relations"):
            _edim += 6
        kw = dict(
            node_feature_dim=base_dim + env.density.output_dim, edge_feature_dim=_edim,
            action_size=CONFIG["gnn"]["action_size"], hidden_dim=CONFIG["gnn"]["hidden_dim"],
            num_layers=CONFIG["gnn"]["num_layers"], n_heads=CONFIG["gnn"]["num_heads"],
            reward_heads=rd["heads"], head_weights=rd["weights"],
            dropout=CONFIG["gnn"]["dropout"], lr=CONFIG["gnn"]["learning_rate"],
            gamma=CONFIG["training"]["gamma"],
            buffer_capacity=CONFIG["training"]["buffer_capacity"],
            batch_size=16,
            stability_coef=CONFIG["gnn"]["stability_coef"],
            encoder_mode=CONFIG["gnn"]["encoder"], base_feature_dim=base_dim,
            density_diamond_mask=env.density._diamond_mask,
            double_dqn=True, per_fleet_terminal=True)
        if learner_head:
            kw.update(learner_beta=0.0, learner_weight=1.0, learner_nudge="prob",
                      learner_beta_mode="gap", learner_beta_scale=scale, learner_gap_min=50)
        env.gnn = GNNAgent(**kw)
        env.gnn.epsilon_gaussian = lambda *a, **k: 0.1
        return env

    def run(env, steps, learn=False):
        regions, snaps, pushes, moves = [], [], [], []
        _orig_choose = env.gnn.choose_actions

        def _rec_choose(**kw):
            cm = kw.get("conflict_mask")
            regions.append(None if cm is None else np.array(cm, dtype=bool).copy())
            return _orig_choose(**kw)
        env.gnn.choose_actions = _rec_choose
        _orig_push = env.gnn.memory.push

        def _rec_push(**kw):
            pushes.append(None if kw.get("learner_label") is None
                          else np.array(kw["learner_label"]).copy())
            return _orig_push(**kw)
        env.gnn.memory.push = _rec_push
        for _ in range(steps):
            env.step(episode_step=1, total_episodes=1)
            snaps.append({
                "labels": (None if env._router_labels is None else env._router_labels.copy()),
                "eligible": (None if env._route_eligible is None
                             else np.array(env._route_eligible, dtype=bool).copy()),
                "rmoves": (None if env._route_moves is None else np.array(env._route_moves).copy()),
                "route_mask": env._route_mask,
            })
            pend = env._honest_pending
            moves.append(None if pend is None else np.array(pend[0]["actions"]).copy())
            if learn and len(env.gnn.memory) >= env.gnn.batch_size:
                env.gnn.learn()
            if env.is_episode_over():
                break
        return regions, snaps, pushes, moves

    try:
        # ---------------- 7. route ON: the router's clear set IS the route's
        configure(router=True, route_on=True)
        env = make_env(learner_head=True)
        regions, snaps, pushes, _ = run(env, 60)
        bad, n_clear, bad_region, n_tied = 0, 0, 0, 0
        for s, r in zip(snaps, regions):
            if s["labels"] is None or s["eligible"] is None:
                continue
            # In the clear exactly where the route is eligible, and the
            # route's own move is one of the label's equally-short moves.
            bad += int(((s["labels"] >= 0) != s["eligible"]).sum())
            el = s["eligible"]
            bad += int((((s["labels"][el] >> s["rmoves"][el]) & 1) != 1).sum())
            n_clear += int(el.sum())
            n_tied += int(sum(bin(int(x)).count("1") > 1 for x in s["labels"][el]))
            region = np.zeros(len(s["labels"]), dtype=bool) if r is None else r
            bad_region += int((region != (s["labels"] >= 0)).sum())
        check("route_on_labels_are_the_routes_moves", bad, 0)
        check("route_on_clear_fleets_seen", n_clear > 0, True)
        check("nudge_region_is_the_clear_set", bad_region, 0)
        pb = sum(int((p != s["labels"]).sum()) for p, s in zip(pushes, snaps)
                 if p is not None and s["labels"] is not None and len(p) == len(s["labels"]))
        check("every_push_carries_its_steps_labels", pb, 0)
        check("every_push_has_labels", all(p is not None for p in pushes), True)
        est = env.get_error_statistics()
        check("stats_router_on", (est["router_on"], est["router_route_on"]), (1, 1))
        check("stats_agreement_in_range", 0.0 <= est["router_learner_agree"] <= 1.0, True)
        print(f"      route on: {n_clear} clear fleet-steps ({n_tied} with equally short moves) "
              f"| tie share {est['router_tie_share'] * 100:.0f}% | Learner agrees "
              f"{est['router_learner_agree'] * 100:.0f}% (untrained) | heads differ "
              f"{est['router_q_differs'] * 100:.0f}%")

        # ---------------- 8. route OFF: labels still computed, the route drives nobody
        configure(router=True, route_on=False)
        env = make_env(learner_head=True)
        regions, snaps, pushes, _ = run(env, 40)
        check("route_off_route_drives_nobody", all(s["route_mask"] is None for s in snaps), True)
        n_lab = sum(int((s["labels"] >= 0).sum()) for s in snaps if s["labels"] is not None)
        check("route_off_labels_still_computed", n_lab > 0, True)
        bad_region = sum(int(((np.zeros(len(s["labels"]), dtype=bool) if r is None else r)
                              != (s["labels"] >= 0)).sum())
                         for s, r in zip(snaps, regions) if s["labels"] is not None)
        check("route_off_nudge_region_is_the_clear_set", bad_region, 0)
        check("stats_route_off", env.get_error_statistics()["router_route_on"], 0)

        # ---------------- 9. shadow is silent
        # Measured in two halves, so the GPU's own non-repeatability cannot be
        # mistaken for the Learner (Rohit's machine, 14 October: the Learner-free
        # run does not repeat itself bit for bit once it learns -- atomic adds
        # in a backward pass -- and over 90 learning steps one flipped near-tie
        # sends two otherwise identical runs down different days).
        #   (a) CHOICES: the same steps with no learning -- forward passes only,
        #       which repeat exactly -- must give identical moves.
        #   (b) UPDATES: then both learn on their (identical) memories, with the
        #       same draws; the trunk and Q heads must stay as close to the
        #       Learner-free agent as a copy of that agent stays to itself.
        def same_moves(ma, mb):
            return len(ma) == len(mb) and all(
                (x is None and y is None) or (x is not None and y is not None and np.array_equal(x, y))
                for x, y in zip(ma, mb))

        def max_diff(ga, gb):
            return max(float((x - y).abs().max().item())
                       for x, y in zip(non_learner_params(ga), non_learner_params(gb)))

        configure(router=False, route_on=True)
        env_a = make_env(learner_head=False)
        _, _, _, moves_a = run(env_a, 60, learn=False)
        configure(router=True, route_on=True)
        env_b = make_env(learner_head=True, scale=0.0)
        _, _, _, moves_b = run(env_b, 60, learn=False)
        check("shadow_same_moves_as_no_learner", same_moves(moves_a, moves_b), True)
        check("shadow_nudge_changed_nothing", env_b.gnn.learner_changed, 0)

        ga, gb = env_a.gnn, env_b.gnn
        for g in (ga, gb):                                             # drop the recorders
            for owner, name in ((g.memory, "push"), (g, "choose_actions")):
                if name in vars(owner):
                    delattr(owner, name)
        ga2 = copy.deepcopy(ga)                                        # the control
        for i in range(20):
            for g in (ga, ga2, gb):
                random.seed(100 + i); torch.manual_seed(100 + i)
                g.learn()
        ctrl, test = max_diff(ga, ga2), max_diff(ga, gb)
        print(f"      after 20 updates on the same memory: max weight difference "
              f"control {ctrl:.2e} | Learner in shadow {test:.2e}")
        check("shadow_updates_as_close_as_the_control", test <= max(1e-7, 10.0 * ctrl), True)
        check("shadow_learner_learned", gb.last_learner_n > 0, True)
        env_a.gnn, env_b.gnn = ga, gb

        # ---------------- 10. deploy builds each arm by its checkpoint
        from benchmark_flowrra import build_agent
        from test_smoke_integration import build_instance as _bi
        G, grid_pos, missions, _gdm, goal_pool = _bi(24)
        with tempfile.TemporaryDirectory() as d:
            p_on, p_off = os.path.join(d, "on.pth"), os.path.join(d, "off.pth")
            env_b.gnn.save(p_on)
            env_a.gnn.save(p_off)
            a_on = build_agent(G, grid_pos, missions, goal_pool, p_on)
            a_off = build_agent(G, grid_pos, missions, goal_pool, p_off)
            check("deploy_learner_arm_has_its_learner", a_on.policy_net.learner_head is not None, True)
            check("deploy_plain_arm_has_none", a_off.policy_net.learner_head is None, True)
            check("deploy_learner_at_end_scale",
                  a_on.learner_beta_scale,
                  float((LRN.get("router", {}) or {}).get("scale_end", 0.0)))
    finally:
        for k, v in saved.items():
            CONFIG[k] = v


if __name__ == "__main__":
    test_isolated_init()
    test_only_the_learner_moves()
    test_labels_teach_it()
    test_warm_start_from_a_checkpoint_without_learner()
    test_the_nudge()
    test_schedule()
    env_part()
    print("\n" + ("ALL PASS" if not FAIL else f"FAILURES: {FAIL}"))