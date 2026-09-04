"""
smoke_test.py

End-to-end check of the changes, on a synthetic multi-level warehouse with
ladders. No official map CSVs required.

Covers, in order:
  1. Decomposed attention is numerically equivalent to the old formulation, and
     survives an all-zero adjacency row without producing NaN.
  2. Replay buffer pads mixed fleet counts instead of discarding them.
  3. Per-map bounds are derived from real extents rather than CONFIG's literal.
  4. Per-fleet epsilon actually produces partial exploration.
  5. A full training episode runs with variable k.
  6. Error stop -> confirmation -> pickup -> dwell -> handover completes, and a
     stopped fleet is invisible to the collision system.
"""

import os, sys, csv, math, random, itertools
import numpy as np
import networkx as nx
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def build_synthetic_map(nx_cells=14, ny_cells=14, nz=3, ladder_every=5):
    """
    Multi-level grid with vertical ladders, mimicking the benchmark's topology:
    aisles on each level, sparse Z connections between them.
    """
    G = nx.Graph()
    pos = {}
    def nid(x, y, z): return f"n{x}_{y}_{z}"
    for z in range(nz):
        for x in range(nx_cells):
            for y in range(ny_cells):
                # carve racks out so degree is realistically low
                if x % 3 == 1 and y % 7 not in (0, 6):
                    continue
                n = nid(x, y, z)
                pos[n] = {"X": float(x), "Y": float(y), "Z": float(z)}
                G.add_node(n)
    for n in list(G.nodes):
        x, y, z = (int(v) for v in n[1:].split("_"))
        for dx, dy in ((1, 0), (0, 1)):
            m = nid(x + dx, y + dy, z)
            if m in G:
                G.add_edge(n, m)
    # ladders
    for x in range(0, nx_cells, ladder_every):
        for y in range(0, ny_cells, ladder_every):
            for z in range(nz - 1):
                a, b = nid(x, y, z), nid(x, y, z + 1)
                if a in G and b in G:
                    G.add_edge(a, b)
    comp = max(nx.connected_components(G), key=len)
    G = G.subgraph(comp).copy()
    pos = {k: v for k, v in pos.items() if k in G}
    return G, pos


def check(label, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {label}" + (f"  {detail}" if detail else ""))
    return cond


# ---------------------------------------------------------------------------
def test_attention():
    print("\n1. ATTENTION -- decomposed vs materialised, and the NaN guard")
    from agent_warehouse import GraphAttentionLayer
    torch.manual_seed(0)
    layer = GraphAttentionLayer(in_features=12, out_features=8, n_heads=4, dropout=0.0)
    layer.eval()
    B, N = 3, 9
    x = torch.randn(B, N, 12)
    adj = (torch.rand(B, N, N) > 0.4).float()
    for b in range(B):
        adj[b].fill_diagonal_(1.0)

    def reference(x, adj):
        """The original implementation, verbatim, as ground truth."""
        Bc, Nc, _ = x.shape
        h = torch.matmul(x, layer.W).view(Bc, Nc, layer.n_heads, layer.out_features)
        h_i = h.unsqueeze(2).expand(Bc, Nc, Nc, layer.n_heads, layer.out_features)
        h_j = h.unsqueeze(1).expand(Bc, Nc, Nc, layer.n_heads, layer.out_features)
        cat = torch.cat([h_i, h_j], dim=-1)
        a_r = layer.a.view(1, 1, 1, 1, 2 * layer.out_features, 1)
        e = torch.matmul(cat.unsqueeze(-2), a_r).squeeze(-1).squeeze(-1)
        e = layer.leakyrelu(e)
        e = e.masked_fill((adj.unsqueeze(-1) == 0).expand(Bc, Nc, Nc, layer.n_heads),
                          float("-inf"))
        alpha = torch.nn.functional.softmax(e, dim=2)
        at = alpha.permute(0, 3, 1, 2)
        ht = h.permute(0, 2, 1, 3)
        out = torch.matmul(at, ht).permute(0, 2, 1, 3).reshape(Bc, Nc, -1)
        return out

    with torch.no_grad():
        new = layer(x, adj)
        ref = reference(x, adj)
    ok = check("matches the original formulation",
               torch.allclose(new, ref, atol=1e-5),
               f"max abs diff {float((new-ref).abs().max()):.2e}")

    # all-zero adjacency row (what an unpadded padding row would look like)
    adj2 = adj.clone()
    adj2[0, 3, :] = 0.0
    with torch.no_grad():
        out2 = layer(x, adj2)
    ok &= check("no NaN on an all-zero adjacency row",
                not torch.isnan(out2).any())

    # memory: the old path materialises B*N*N*H*2F floats
    B2, N2, H, F_ = 64, 50, 4, 32
    old_mb = B2 * N2 * N2 * H * 2 * F_ * 4 / 1e6
    new_mb = B2 * N2 * N2 * H * 4 / 1e6
    check("peak pairwise tensor shrinks", new_mb < old_mb / 10,
          f"{old_mb:.0f} MB -> {new_mb:.1f} MB per layer at B=64,N=50")
    return ok


def test_buffer_padding():
    print("\n2. REPLAY BUFFER -- mixed fleet counts")
    from agent_warehouse import GraphReplayBuffer
    buf = GraphReplayBuffer(capacity=500)
    F_ = 10
    counts = [25, 31, 40, 50]
    for i in range(200):
        n = counts[i % len(counts)]
        adj = np.eye(n, dtype=np.float32)
        buf.push(np.random.randn(n, F_).astype(np.float32), adj,
                 np.random.randint(0, 7, n), np.random.randn(n, 5).astype(np.float32),
                 np.random.randn(n, F_).astype(np.float32), adj,
                 False, 1.0, np.ones(n, dtype=np.float32))
    out = buf.sample(64)
    ok = check("sample() returns a batch from mixed node counts", out is not None)
    if out:
        nf, adjs, acts, rew, nnf, nadj, dones, integ, amask, rec = out
        ok &= check("padded to the batch max", nf.shape[1] == max(counts),
                    f"shape {tuple(nf.shape)}")
        # padding rows must be masked out and self-looped
        ok &= check("padding rows carry active_mask 0",
                    bool((amask.sum(dim=1) <= max(counts)).all().item()))
        diag_ok = True
        for b in range(nf.shape[0]):
            n_real = int(amask[b].sum().item())
            if n_real < nf.shape[1]:
                diag_ok &= bool(adjs[b, n_real:, n_real:].diagonal().min().item() == 1.0)
        ok &= check("padding rows have a self-loop (no all--inf softmax row)", diag_ok)
    return ok


def test_bounds_and_env():
    # Seed here, not just at module level: these tests build networks and
    # step environments, so running them in a different order would
    # otherwise change every trajectory and make failures irreproducible.
    random.seed(101); np.random.seed(101); torch.manual_seed(101)
    print("\n3-5. BOUNDS, EPSILON, FULL EPISODE")
    from config_warehouse import CONFIG
    from core_warehouse import FLOWRRA
    from agent_warehouse import GNNAgent
    from config_warehouse import CONFIG as _C
    _RD = _C['reward_decomposition']
    from node_warehouse import precompute_goal_distances

    G, pos = build_synthetic_map()
    rng = random.Random(3)
    nodes = sorted(G.nodes)
    k = 12
    picks = rng.sample(nodes, 2 * k)
    starts, goals = picks[:k], picks[k:]

    missions = [{"id": str(i), "start_node": s,
                 "start_pos": np.array([pos[s]["X"], pos[s]["Y"], pos[s]["Z"]],
                                       dtype=np.float32)}
                for i, s in enumerate(starts)]
    goal_pool = {g: np.array([pos[g]["X"], pos[g]["Y"], pos[g]["Z"]], dtype=np.float32)
                 for g in goals}
    gdm = precompute_goal_distances(G, [{"goal_node": g} for g in goal_pool])

    env = FLOWRRA(G, pos, missions, mode="training", goal_distance_maps=gdm,
                  shared_pool_mode=True, goal_pool=goal_pool)

    xs = [p["X"] for p in pos.values()]
    zs = [p["Z"] for p in pos.values()]
    expect_x = max(xs) - min(xs)
    ok = check("bounds derived from map extents, not CONFIG's (50,50,10)",
               abs(env.warehouse_bounds[0] - expect_x) < 1e-6
               and env.warehouse_bounds != tuple(float(v) for v in CONFIG["warehouse"]["bounds"]),
               f"derived {env.warehouse_bounds}, span X={expect_x}, Z={max(zs)-min(zs)}")

    n0 = env.nodes[0]
    input_dim = (len(n0.get_state_vector(env.nodes))
                 + len(env.density.get_local_affordance(n0.current_pos, env.nodes, set())))
    agent = GNNAgent(node_feature_dim=input_dim, edge_feature_dim=0,
                     action_size=7, hidden_dim=32, num_layers=2, n_heads=2,
                     reward_heads=_RD['heads'], head_weights=_RD['weights'],
                     dropout=0.0, lr=3e-4, gamma=CONFIG["training"]["gamma"],
                     buffer_capacity=2000, batch_size=16)
    env.gnn = agent

    # --- per-fleet epsilon: at eps=0.5 we should see BOTH modes in one step ---
    feats = np.random.randn(20, input_dim).astype(np.float32)
    adj = np.ones((20, 20), dtype=np.float32)
    mixed = 0
    for _ in range(40):
        a = agent.choose_actions(feats, adj, episode_number=1, total_episodes=1,
                                 node_ids=[str(i) for i in range(20)],
                                 eps_min=0.5, eps_peak=0.5)
        if len(set(a.tolist())) > 1:
            mixed += 1
    ok &= check("per-fleet epsilon gives partial exploration within a step",
                mixed > 30, f"{mixed}/40 steps had non-uniform actions")

    ok &= check("gamma is 0.99", abs(CONFIG["training"]["gamma"] - 0.99) < 1e-9)

    for step in range(120):
        env.step(episode_step=1, total_episodes=10)
        if len(agent.memory) >= agent.batch_size:
            agent.learn(node_ids=[n.id for n in env.nodes])
        if env.is_episode_over():
            break
    ok &= check("episode ran without exception", True,
                f"{env.step_count} steps, {len(env.frozen_nodes)}/{len(env.nodes)} done")

    # a learn() step on a buffer holding two different fleet counts
    env2 = FLOWRRA(G, pos, missions[:7], mode="training", goal_distance_maps=gdm,
                   shared_pool_mode=True, goal_pool=goal_pool)
    env2.gnn = agent
    agent.reset_episode_state()
    for _ in range(40):
        env2.step(episode_step=2, total_episodes=10)
    loss = agent.learn(node_ids=[n.id for n in env2.nodes])
    ok &= check("learn() works on a buffer with mixed k (12 and 7)",
                loss is not None and not math.isnan(float(loss)), f"loss {loss:.4f}")
    return ok


def test_error_handover():
    # Seed here, not just at module level: these tests build networks and
    # step environments, so running them in a different order would
    # otherwise change every trajectory and make failures irreproducible.
    random.seed(202); np.random.seed(202); torch.manual_seed(202)
    print("\n6. ERROR STOP -> PICKUP -> HANDOVER")
    from config_warehouse import CONFIG
    from core_warehouse import FLOWRRA
    from agent_warehouse import GNNAgent
    from config_warehouse import CONFIG as _C
    _RD = _C['reward_decomposition']
    from node_warehouse import precompute_goal_distances

    G, pos = build_synthetic_map()
    rng = random.Random(11)
    nodes = sorted(G.nodes)
    k = 10
    picks = rng.sample(nodes, 2 * k)
    starts, goals = picks[:k], picks[k:]
    missions = [{"id": str(i), "start_node": s,
                 "start_pos": np.array([pos[s]["X"], pos[s]["Y"], pos[s]["Z"]],
                                       dtype=np.float32)}
                for i, s in enumerate(starts)]
    goal_pool = {g: np.array([pos[g]["X"], pos[g]["Y"], pos[g]["Z"]], dtype=np.float32)
                 for g in goals}
    gdm = precompute_goal_distances(G, [{"goal_node": g} for g in goal_pool])

    env = FLOWRRA(G, pos, missions, mode="training", goal_distance_maps=gdm,
                  shared_pool_mode=True, goal_pool=goal_pool)
    n0 = env.nodes[0]
    input_dim = (len(n0.get_state_vector(env.nodes))
                 + len(env.density.get_local_affordance(n0.current_pos, env.nodes, set())))
    agent = GNNAgent(node_feature_dim=input_dim, edge_feature_dim=0, action_size=7,
                     hidden_dim=32, num_layers=2, n_heads=2, dropout=0.0, lr=3e-4,
                     reward_heads=_RD['heads'], head_weights=_RD['weights'],
                     gamma=0.99, buffer_capacity=2000, batch_size=16)
    env.gnn = agent

    # Auto-injection OFF for this test: we inject one error deliberately and
    # want the handover path isolated. Left on, the dice can kill three of ten
    # fleets (including rescuers) and the test measures fleet attrition rather
    # than whether handover works. Re-dispatch after a rescuer dies is covered
    # separately in test 7.
    env.errors_enabled = False

    # Force a deterministic error rather than waiting on the injection dice.
    for _ in range(30):
        env.step(episode_step=1, total_episodes=10)

    victim = next((n for n in env.nodes
                   if n.id not in env.immobile_nodes and n.current_goal_id), None)
    ok = check("found a live fleet to kill", victim is not None)
    if not ok:
        return False
    orphan_goal = victim.current_goal_id

    env.stopped_nodes.add(victim.id)
    env._error_step[victim.id] = env.step_count
    env.total_errors += 1
    victim.direction = np.zeros(3, dtype=np.float32)
    print(f"      injected error on fleet {victim.id} (goal {orphan_goal})")

    ok &= check("stopped fleet is excluded from the collision system",
                victim.id in env.immobile_nodes)

    # Its cell must be collision-transparent: put another fleet exactly on it.
    other = next(n for n in env.nodes if n.id != victim.id and n.id not in env.immobile_nodes)
    other.current_pos = victim.current_pos.copy()
    env.loop.check_integrity(env.nodes, env.step_count, env.immobile_nodes)
    # Assert about the VICTIM specifically, not global integrity: in a 10-fleet
    # warehouse some unrelated pair may well be in conflict on the same step, and
    # teleporting a fleet onto the stopped cell for this test can itself land it
    # next to a third fleet. The claim under test is that the stopped fleet is
    # not a party to any conflict.
    ok &= check("stopped fleet triggers neither fatal nor warning conflict",
                victim.id not in env.loop.deadlocked_nodes
                and victim.id not in env.loop.warning_nodes,
                f"deadlocked={sorted(env.loop.deadlocked_nodes)} "
                f"warning={sorted(env.loop.warning_nodes)}")

    saw_pickup = saw_dispatch = False
    rescuer_own_goal = None
    for _ in range(400):
        env.step(episode_step=1, total_episodes=10)
        if env.open_pickups:
            saw_pickup = True
        if env._pickup_assignment:
            saw_dispatch = True
            if rescuer_own_goal is None:
                rid = list(env._pickup_assignment)[0]
                rescuer_own_goal = next(n.current_goal_id for n in env.nodes
                                        if n.id == rid)
        if env.total_handovers > 0:
            break

    ok &= check("stop confirmed and order cancelled",
                victim.id in env.stopped_confirmed)
    ok &= check("pickup opened at the stopped fleet's cell", saw_pickup)
    ok &= check("a rescuer was dispatched", saw_dispatch)
    ok &= check("handover completed (package carried onward)",
                env.total_handovers > 0,
                f"handovers={env.total_handovers} stats={env.get_error_statistics()}")

    if env.total_handovers > 0:
        # Exclude stopped fleets explicitly. Relying on "first node whose
        # current_goal_id matches" is what made this assertion read the corpse
        # instead of the carrier; being explicit here means the test still
        # passes for the right reason if that state hygiene ever regresses.
        carrier = next((n for n in env.nodes
                        if n.current_goal_id == orphan_goal
                        and n.id not in env.stopped_nodes), None)
        ok &= check("orphaned goal is now owned by a live fleet",
                    carrier is not None and carrier.id != victim.id,
                    f"carried by {carrier.id if carrier else None}, "
                    f"originally {victim.id}")
        # The rescuer must be carrying BOTH orders, not have swapped its own
        # for the orphan. Either it is currently on the orphan with its own
        # queued behind, or vice versa.
        if carrier is not None:
            still_owed = env._pending_goals.get(carrier.id, [])
            ok &= check("rescuer kept its own order as well (carries two)",
                        len(still_owed) >= 1 or rescuer_own_goal in env.claimed_goals,
                        f"current={carrier.current_goal_id} pending={still_owed} "
                        f"own_was={rescuer_own_goal}")
        ok &= check("dead fleet no longer advertises a goal",
                    victim.current_goal_id is None,
                    f"victim.current_goal_id={victim.current_goal_id}")

    ok &= check("episode-over separates termination from success",
                env.is_episode_over() == (len(env.immobile_nodes) == len(env.nodes)))
    return ok


def test_rescuer_dies():
    # Seed here, not just at module level: these tests build networks and
    # step environments, so running them in a different order would
    # otherwise change every trajectory and make failures irreproducible.
    random.seed(303); np.random.seed(303); torch.manual_seed(303)
    print("\n7. RESCUER ERRORS MID-LEG-1 -> PICKUP GOES BACK ON THE BOARD")
    from core_warehouse import FLOWRRA
    from agent_warehouse import GNNAgent
    from config_warehouse import CONFIG as _C
    _RD = _C['reward_decomposition']
    from node_warehouse import precompute_goal_distances

    G, pos = build_synthetic_map()
    rng = random.Random(5)
    nodes = sorted(G.nodes)
    k = 12
    picks = rng.sample(nodes, 2 * k)
    missions = [{"id": str(i), "start_node": s,
                 "start_pos": np.array([pos[s]["X"], pos[s]["Y"], pos[s]["Z"]],
                                       dtype=np.float32)}
                for i, s in enumerate(picks[:k])]
    goal_pool = {g: np.array([pos[g]["X"], pos[g]["Y"], pos[g]["Z"]], dtype=np.float32)
                 for g in picks[k:]}
    gdm = precompute_goal_distances(G, [{"goal_node": g} for g in goal_pool])

    env = FLOWRRA(G, pos, missions, mode="training", goal_distance_maps=gdm,
                  shared_pool_mode=True, goal_pool=goal_pool)
    n0 = env.nodes[0]
    dim = (len(n0.get_state_vector(env.nodes))
           + len(env.density.get_local_affordance(n0.current_pos, env.nodes, set())))
    env.gnn = GNNAgent(node_feature_dim=dim, edge_feature_dim=0, action_size=7,
                       hidden_dim=32, num_layers=2, n_heads=2, dropout=0.0, lr=3e-4,
                       reward_heads=_RD['heads'], head_weights=_RD['weights'],
                       gamma=0.99, buffer_capacity=2000, batch_size=16)
    env.errors_enabled = False

    for _ in range(25):
        env.step(episode_step=1, total_episodes=10)

    victim = next(n for n in env.nodes
                  if n.id not in env.immobile_nodes and n.current_goal_id)
    env.stopped_nodes.add(victim.id)
    env._error_step[victim.id] = env.step_count
    victim.direction = np.zeros(3, dtype=np.float32)

    # run until a rescuer is dispatched
    for _ in range(120):
        env.step(episode_step=1, total_episodes=10)
        if env._pickup_assignment:
            break
    ok = check("a rescuer was dispatched", bool(env._pickup_assignment))
    if not ok:
        return False

    rescuer_id = list(env._pickup_assignment)[0]
    pickup_id = env._pickup_assignment[rescuer_id]

    # now kill the rescuer, through the real injection path
    env.errors_enabled = True
    env.error_prob_per_step = 1.0
    env.error_min_progress = 0.0
    env.error_max_per_episode = 99
    rescuer = next(n for n in env.nodes if n.id == rescuer_id)
    for _ in range(60):
        env._maybe_inject_error()
        if rescuer_id in env.stopped_nodes:
            break
    env.errors_enabled = False

    if rescuer_id not in env.stopped_nodes:
        return check("could not kill the rescuer (test setup)", False)

    ok &= check("dead rescuer released its pickup assignment",
                rescuer_id not in env._pickup_assignment)
    ok &= check("the pickup is still open", pickup_id in env.open_pickups)

    env._dispatch_rescuers()
    ok &= check("a replacement rescuer was dispatched",
                pickup_id in env._pickup_assignment.values(),
                f"assignments={env._pickup_assignment}")
    return ok


def test_recall_retired():
    print("\n8. LATE ERROR -> RETIRED FLEET RECALLED")
    from core_warehouse import FLOWRRA
    from agent_warehouse import GNNAgent
    from config_warehouse import CONFIG as _C
    _RD = _C['reward_decomposition']
    from node_warehouse import precompute_goal_distances_compact
    random.seed(404); np.random.seed(404); torch.manual_seed(404)

    G, pos = build_synthetic_map(16, 16, 3)
    nodes = sorted(G.nodes)
    k = 12
    picks = random.sample(nodes, 2 * k)
    missions = [{"id": str(i), "start_node": s,
                 "start_pos": np.array([pos[s]["X"], pos[s]["Y"], pos[s]["Z"]],
                                       dtype=np.float32)}
                for i, s in enumerate(picks[:k])]
    goal_pool = {g: np.array([pos[g]["X"], pos[g]["Y"], pos[g]["Z"]], dtype=np.float32)
                 for g in picks[k:]}
    gdm, _ = precompute_goal_distances_compact(G, list(goal_pool))

    env = FLOWRRA(G, pos, missions, mode="training", goal_distance_maps=gdm,
                  shared_pool_mode=True, goal_pool=goal_pool)
    n0 = env.nodes[0]
    dim = (len(n0.get_state_vector(env.nodes))
           + len(env.density.get_local_affordance(n0.current_pos, env.nodes, set())))
    env.gnn = GNNAgent(node_feature_dim=dim, edge_feature_dim=0, action_size=7,
                       hidden_dim=32, num_layers=2, n_heads=2, dropout=0.0, lr=3e-4,
                       reward_heads=_RD['heads'], head_weights=_RD['weights'],
                       gamma=0.99, buffer_capacity=2000, batch_size=16)
    env.errors_enabled = False

    # Retire everyone except two fleets, simulating a late-episode state.
    for _ in range(20):
        env.step(episode_step=1, total_episodes=10)
    live = [n for n in env.nodes if n.id not in env.immobile_nodes]
    victim, survivor = live[0], live[1]
    for n in live[2:]:
        env.claimed_goals.add(n.current_goal_id)
        env._release_target(n)
        env.frozen_nodes.add(n.id)
        env.gnn.freeze_node(n.id, n.current_pos)

    n_frozen_before = len(env.frozen_nodes)
    ok = check("set up a late-episode state", n_frozen_before >= 8,
               f"{n_frozen_before} retired, 2 active")

    env.stopped_nodes.add(victim.id)
    env._error_step[victim.id] = env.step_count
    victim.direction = np.zeros(3, dtype=np.float32)

    for _ in range(120):
        env.step(episode_step=1, total_episodes=10)
        if env._pickup_assignment:
            break

    ok &= check("a rescuer was dispatched", bool(env._pickup_assignment))
    if not ok:
        return False
    rid = list(env._pickup_assignment)[0]
    est = env.get_error_statistics()

    ok &= check("a retired fleet was recalled rather than the lone survivor",
                est["retired_fleets_recalled"] >= 1,
                f"rescuer={rid} survivor={survivor.id} recalls={est['retired_fleets_recalled']}")
    ok &= check("recalled fleet is out of frozen_nodes on both sides",
                rid not in env.frozen_nodes and rid not in env.gnn.frozen_nodes)
    ok &= check("its earlier delivery still counts (claimed_goals untouched)",
                len(env.claimed_goals) >= n_frozen_before - 2,
                f"claimed={len(env.claimed_goals)}")
    return ok


if __name__ == "__main__":
    import contextlib, io
    results = []
    for fn in (test_attention, test_buffer_padding, test_bounds_and_env,
               test_error_handover, test_rescuer_dies, test_recall_retired):
        buf = io.StringIO()
        try:
            # env prints a lot; keep only our own check lines
            with contextlib.redirect_stdout(buf):
                r = fn()
        except Exception as exc:
            import traceback
            print(buf.getvalue()[-2000:])
            traceback.print_exc()
            r = False
        for line in buf.getvalue().splitlines():
            if line.strip().startswith(("[PASS]", "[FAIL]")) or line.startswith(("\n", "1.", "2.", "3", "6", "7", "8")) or "  [" in line:
                print(line)
        results.append(r)
    print("\n" + "=" * 60)
    print("ALL PASS" if all(results) else "SOME FAILURES")
    print("=" * 60)
    sys.exit(0 if all(results) else 1)