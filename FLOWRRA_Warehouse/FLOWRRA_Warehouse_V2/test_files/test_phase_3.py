"""
test_phase3.py -- edge-aware attention carrying path conflict.

THE PROBLEM. Before this, the only thing the GAT knew about a pair of fleets was
that they were neighbours. Not how far apart, not in which direction, and not
whether their planned routes crossed. Every pairwise fact had to be inferred
from two node vectors that do not contain each other's positions at all.

Path conflict is the motivating case: "do our routes want the same cell, and how
soon" is a property of a PAIR. Encoding it twice, once in each node, is a worse
representation of the same fact -- and the fleet that most needs it is the one
deciding whether to hold.

IT IS ALSO THE RELEASE SIGNAL WAITING HAS BEEN MISSING. "Blocked" means our
projections intersect; "clearing" means they no longer do -- which is visible
BEFORE the blocker has physically moved away, where a nearest-peer distance
cannot tell you anything until it already has.

THREE CHANNELS, riding inside the adjacency tensor as [N, N, 1 + E]:
    0  mask        binary, exactly what the layer has always used
    1  closeness   1 - d/interaction_radius on GRAPH distance
    2  conflict    max over shared cells of w_i * w_j, discounted by how soon
    3  head-on     each fleet's next cell is the other's current one

Head-on earns a channel of its own because a SWAP is invisible to a cell-overlap
test: neither fleet ever occupies the other's target at the same t. It is also
the collision that matters most, since neither can yield by continuing.

edge_feature_dim = 0 must remain a working identity path, so the whole mechanism
stays ablatable after the retrain instead of being an act of faith.
"""

import numpy as np
import networkx as nx
import torch

from config_warehouse import CONFIG
from agent_warehouse import DEVICE, GraphAttentionLayer, GraphReplayBuffer

FAIL = []


def check(name, got, want):
    ok = got == want
    print(f"{'PASS' if ok else 'FAIL'}  {name}: got {got!r} want {want!r}")
    if not ok:
        FAIL.append(name)


# ------------------------------------------------------------------ world
def corridor_env(n_fleets, starts, goals, width=16):
    """A single straight aisle, so conflicts are easy to construct by hand."""
    from core_warehouse import FLOWRRA
    from node_warehouse import precompute_goal_distances

    G = nx.Graph()
    pos = {}
    for x in range(width):
        nid = f"n_{x}"
        G.add_node(nid)
        pos[nid] = {"X": float(x), "Y": 0.0, "Z": 0.0}
    for x in range(width - 1):
        G.add_edge(f"n_{x}", f"n_{x+1}")
    grid = {(int(v["X"]), 0, 0): k for k, v in pos.items()}

    missions, gp = [], {}
    for i, (s, g) in enumerate(zip(starts, goals)):
        sn, gn = f"n_{s}", f"n_{g}"
        missions.append({
            "id": f"f{i}", "start_node": sn, "goal_node": gn,
            "start_pos": np.array([float(s), 0.0, 0.0], dtype=np.float32),
            "goal_pos": np.array([float(g), 0.0, 0.0], dtype=np.float32)})
        gp[gn] = np.array([float(g), 0.0, 0.0], dtype=np.float32)
    gdm = precompute_goal_distances(G, [{"goal_node": g} for g in gp])
    env = FLOWRRA(G, grid, missions, mode="init", goal_distance_maps=gdm,
                  shared_pool_mode=True, goal_pool=gp)

    # FORCE THE GOALS BACK.
    #
    # Under shared_pool_mode the orchestrator runs an optimal 1:1 Hungarian
    # assignment at construction, so the goals in `missions` are a suggestion,
    # not an instruction. It swapped them here -- f0 at x=5 is nearer x=0 than
    # x=12 -- which silently turned a CONVERGING fixture into a diverging one
    # and made the conflict test assert the opposite of what it meant to.
    #
    # Reassigning after construction is the only way to control intent in a
    # test. The path memo keys on current_goal_id, so it invalidates itself.
    for i, (node, g) in enumerate(zip(env.nodes, goals)):
        gn = f"n_{g}"
        node.goal_pos = np.array([float(g), 0.0, 0.0], dtype=np.float32)
        node.goal_distance_map = dict(nx.single_source_shortest_path_length(G, gn))
        node.current_goal_id = gn
    return env


def edges_for(env):
    env.proximity.refresh(env.nodes, excluded_ids=env.immobile_nodes)
    adj = (env._build_adjacency_graph() if env.adjacency_metric == "graph"
           else env._build_adjacency())
    return env._build_edge_features(adj)


def with_edges(fn, dim=3, metric="graph"):
    old_d = CONFIG["gnn"]["edge_feature_dim"]
    old_m = CONFIG["proximity"]["adjacency_metric"]
    CONFIG["gnn"]["edge_feature_dim"] = dim
    CONFIG["proximity"]["adjacency_metric"] = metric
    try:
        return fn()
    finally:
        CONFIG["gnn"]["edge_feature_dim"] = old_d
        CONFIG["proximity"]["adjacency_metric"] = old_m


# ================================================================== identity
def test_zero_dim_is_a_true_identity_path():
    """The ablation arm. It must behave EXACTLY as before, not approximately."""
    torch.manual_seed(0)
    lay = GraphAttentionLayer(8, 8, n_heads=2, dropout=0.0, edge_feature_dim=0)
    check("no_edge_projection", lay.edge_proj, None)

    x = torch.randn(2, 5, 8)
    adj = torch.ones(2, 5, 5)
    lay.eval()
    a = lay(x, adj)
    b = lay(x, adj, edge_attr=torch.randn(2, 5, 5, 3))
    check("edge_attr_ignored_when_dim_zero",
          bool(torch.equal(a, b)), True)

    env = with_edges(lambda: corridor_env(2, [2, 9], [12, 0]), dim=0)
    adj0 = env._build_edge_features(env._build_adjacency())
    check("adjacency_stays_single_channel", adj0.shape[-1], 1)


# ================================================================== features
def test_conflict_fires_for_converging_routes():
    """Two fleets heading INTO each other along one aisle."""
    def go():
        env = corridor_env(2, [5, 9], [12, 0])   # f0 goes right, f1 goes left
        return edges_for(env)
    e = with_edges(go)
    close, conflict, head_on = e[0, 1, 1], e[0, 1, 2], e[0, 1, 3]
    print(f"      converging: close={close:.3f} conflict={conflict:.3f} "
          f"head_on={head_on:.0f}")
    check("neighbours_at_all", bool(e[0, 1, 0] > 0), True)
    check("conflict_detected", bool(conflict > 0), True)
    check("symmetric", bool(np.allclose(e[..., 1:], e[..., 1:].transpose(1, 0, 2))), True)


def test_no_conflict_for_diverging_routes():
    """Same separation, opposite intent: they are moving APART."""
    def go():
        env = corridor_env(2, [5, 9], [0, 15])   # f0 goes left, f1 goes right
        return edges_for(env)
    e = with_edges(go)
    print(f"      diverging: close={e[0,1,1]:.3f} conflict={e[0,1,2]:.3f}")
    check("still_neighbours", bool(e[0, 1, 0] > 0), True)
    check("no_conflict_when_diverging", bool(e[0, 1, 2] == 0.0), True)


def test_head_on_fires_on_a_swap():
    """
    Adjacent fleets whose next cells are each other's current one. A swap is
    invisible to cell-overlap -- neither occupies the other's target at the same
    t -- which is why it needs its own channel.
    """
    def go():
        env = corridor_env(2, [7, 8], [12, 0])   # f0 -> 8, f1 -> 7
        return edges_for(env)
    e = with_edges(go)
    print(f"      swap: close={e[0,1,1]:.3f} conflict={e[0,1,2]:.3f} "
          f"head_on={e[0,1,3]:.0f}")
    check("head_on_detected", float(e[0, 1, 3]), 1.0)
    check("head_on_symmetric", float(e[1, 0, 3]), 1.0)


def test_closeness_decreases_with_graph_distance():
    def go():
        env = corridor_env(3, [5, 7, 12], [15, 14, 13])
        return edges_for(env)
    e = with_edges(go)
    near, far = float(e[0, 1, 1]), float(e[0, 2, 1])
    print(f"      closeness: 2 cells apart {near:.3f}, 7 apart {far:.3f}")
    check("closer_scores_higher", bool(near > far), True)
    check("bounded", bool(0.0 <= far and near <= 1.0), True)


# ================================================================== plumbing
def test_padding_preserves_channels():
    """
    Edge features ride inside the adjacency, so _pad_transition has to carry the
    channel axis -- and a padded row must still self-attend on the BINARY
    channel or softmax sees an all--inf row and returns NaN.
    """
    adj = np.zeros((3, 3, 4), dtype=np.float32)
    adj[:, :, 0] = 1.0
    adj[0, 1, 2] = 0.7
    # 11 elements since the next-state validity mask was added: the bootstrap
    # must not max over structurally impossible actions.
    nvm = np.zeros((3, 7), dtype=bool)
    nvm[:, 0] = True
    exp = (np.zeros((3, 8), dtype=np.float32), adj,
           np.zeros(3, dtype=np.int64), np.zeros((3, 2), dtype=np.float32),
           np.zeros((3, 8), dtype=np.float32), adj.copy(),
           False, 1.0, np.ones(3, dtype=np.float32), 0, nvm)
    out = GraphReplayBuffer._pad_transition(exp, 5)
    padded = out[1]
    check("padded_shape", padded.shape, (5, 5, 4))
    check("original_block_intact", round(float(padded[0, 1, 2]), 5), 0.7)
    check("phantom_rows_self_attend", float(padded[4, 4, 0]), 1.0)
    check("phantom_edge_features_zero", float(padded[4, 4, 2]), 0.0)


def test_attention_actually_uses_the_edge_features():
    """A channel the network ignores is not a feature."""
    torch.manual_seed(1)
    lay = GraphAttentionLayer(8, 8, n_heads=2, dropout=0.0, edge_feature_dim=3).eval()
    with torch.no_grad():
        lay.edge_proj.weight.normal_(0, 1.0)   # undo the small init for the test
    x = torch.randn(1, 4, 8)
    adj = torch.ones(1, 4, 4)
    zero = torch.zeros(1, 4, 4, 3)
    conflict = zero.clone()
    conflict[0, 0, 1, 1] = 1.0
    a = lay(x, adj, zero)
    b = lay(x, adj, conflict)
    d = float((a - b).abs().max().detach())
    print(f"      max output change from one conflict edge: {d:.4f}")
    check("edge_features_change_the_output", bool(d > 1e-4), True)


def test_gradients_reach_the_edge_projection():
    torch.manual_seed(2)
    lay = GraphAttentionLayer(8, 8, n_heads=2, dropout=0.0, edge_feature_dim=3)
    out = lay(torch.randn(2, 4, 8), torch.ones(2, 4, 4), torch.randn(2, 4, 4, 3))
    out.pow(2).mean().backward()
    g = lay.edge_proj.weight.grad
    check("edge_proj_has_gradient", g is not None and bool(torch.isfinite(g).all()), True)
    check("edge_proj_gradient_nonzero", bool(g.abs().sum() > 0), True)


def test_network_accepts_both_adjacency_shapes():
    from agent_warehouse import GNNAgent
    rd = CONFIG["reward_decomposition"]
    for edim, adj in ((0, torch.ones(1, 6, 6)),
                      (3, torch.rand(1, 6, 6, 4))):
        if edim:
            adj[..., 0] = 1.0
        ag = GNNAgent(node_feature_dim=40, edge_feature_dim=edim, action_size=7,
                      hidden_dim=32, num_layers=2, n_heads=2,
                      reward_heads=rd["heads"], head_weights=rd["weights"])
        # .to(DEVICE): GNNAgent puts both networks on cuda when it is
        # available, so CPU inputs raise "Expected all tensors to be on the same
        # device". Every real caller goes through choose_actions() or learn(),
        # which move their tensors -- only a test that pokes policy_net directly
        # has to do it by hand.
        with torch.no_grad():
            q, _, _ = ag.policy_net(torch.randn(1, 6, 40).to(DEVICE),
                                    adj.to(DEVICE))
        check(f"forward_ok_edim{edim}",
              (tuple(q.shape), bool(torch.isfinite(q).all())), ((1, 6, 7), True))


def test_arrival_order_is_antisymmetric_and_enables_a_one_step_hold():
    """
    The scenario: a corridor where two fleets want the SAME cell. If they arrive
    together, that is a collision. If one arrives a step later, it is a clean
    pass -- and the later fleet is the one that should hold.

    Conflict alone cannot express that: it is symmetric, so both fleets read the
    same number and have no basis to behave differently. Arrival order is the
    only channel that flips.
    """
    def simultaneous():
        # Both 2 cells from cell 7, so both want it at the same step.
        env = corridor_env(2, [5, 9], [9, 5])
        return edges_for(env)

    def staggered():
        # f0 is 2 cells from 7, f1 is 3 -- one step apart.
        env = corridor_env(2, [5, 10], [10, 5])
        return edges_for(env)

    e_sim = with_edges(simultaneous, dim=4)
    e_stg = with_edges(staggered, dim=4)

    c_sim, o_sim = float(e_sim[0, 1, 2]), float(e_sim[0, 1, 4])
    c_stg, o_stg = float(e_stg[0, 1, 2]), float(e_stg[0, 1, 4])
    print(f"      simultaneous: conflict={c_sim:.3f} order={o_sim:+.0f}")
    print(f"      staggered   : conflict={c_stg:.3f} order={o_stg:+.0f}")

    check("simultaneous_conflicts_more", bool(c_sim > c_stg), True)
    check("order_antisymmetric", float(e_stg[1, 0, 4]), -o_stg)
    check("earlier_fleet_marked_first", bool(o_stg != 0.0), True)

    # Everything except order stays symmetric.
    for k, name in ((1, "closeness"), (2, "conflict"), (3, "head_on")):
        check(f"{name}_symmetric",
              bool(np.allclose(e_stg[..., k], e_stg[..., k].T)), True)
    check("order_is_the_only_antisymmetric_channel",
          bool(np.allclose(e_stg[..., 4], -e_stg[..., 4].T)), True)


def junction_env(d_horizontal, d_vertical, jx=5, width=11, height=6):
    """
    A T-junction, so the two approach distances are INDEPENDENT.

        (5,5)
          |
          |            vertical branch
          |
    0-----J-----10     horizontal corridor, J = (jx, 0)

    Fleet A starts d_horizontal cells left of J heading right through it.
    Fleet B starts d_vertical cells up the branch heading down through it and
    then left. Both contest cell J, at t = d_horizontal and t = d_vertical.

    A straight corridor cannot do this: there, changing the separation also
    changes its PARITY, so the fleets alternate between meeting AT a cell and
    swapping BETWEEN cells -- two different collision types. That is what made
    the first version of this test non-monotonic, and it was the fixture's
    fault, not the feature's.
    """
    from core_warehouse import FLOWRRA
    from node_warehouse import precompute_goal_distances

    G = nx.Graph()
    pos = {}
    for x in range(width):
        G.add_node(f"h_{x}")
        pos[f"h_{x}"] = (x, 0)
    for x in range(width - 1):
        G.add_edge(f"h_{x}", f"h_{x+1}")
    for y in range(1, height):
        G.add_node(f"v_{y}")
        pos[f"v_{y}"] = (jx, y)
    G.add_edge(f"h_{jx}", "v_1")
    for y in range(1, height - 1):
        G.add_edge(f"v_{y}", f"v_{y+1}")

    grid = {(x, y, 0): nid for nid, (x, y) in pos.items()}
    a_start, b_start = f"h_{jx - d_horizontal}", f"v_{d_vertical}"
    a_goal, b_goal = f"h_{width - 1}", "h_0"

    missions, gp = [], {}
    for fid, sn, gn in (("f0", a_start, a_goal), ("f1", b_start, b_goal)):
        sc, gc = pos[sn], pos[gn]
        missions.append({
            "id": fid, "start_node": sn, "goal_node": gn,
            "start_pos": np.array([sc[0], sc[1], 0.0], dtype=np.float32),
            "goal_pos": np.array([gc[0], gc[1], 0.0], dtype=np.float32)})
        gp[gn] = np.array([gc[0], gc[1], 0.0], dtype=np.float32)

    gdm = precompute_goal_distances(G, [{"goal_node": g} for g in gp])
    env = FLOWRRA(G, grid, missions, mode="init", goal_distance_maps=gdm,
                  shared_pool_mode=True, goal_pool=gp)
    for node, (sn, gn) in zip(env.nodes, ((a_start, a_goal), (b_start, b_goal))):
        gc = pos[gn]
        node.current_pos = np.array([pos[sn][0], pos[sn][1], 0.0], dtype=np.float64)
        node.goal_pos = np.array([gc[0], gc[1], 0.0], dtype=np.float32)
        node.goal_distance_map = dict(nx.single_source_shortest_path_length(G, gn))
        node.current_goal_id = gn
    return env


def test_a_one_step_offset_genuinely_deconflicts():
    """
    The reason the temporal term was sharpened. The first version discounted by
    1/(1 + max(t) + |dt|), so one step of separation only cut the score by a
    third -- not enough for a fleet to learn that inserting a single step of
    delay RESOLVES a crossing rather than merely softening it. overlap =
    1/(1+|dt|) halves per step.
    """
    def at(gap):
        e = edges_for(junction_env(d_horizontal=2, d_vertical=2 + gap))
        return float(e[0, 1, 2]), float(e[0, 1, 4])

    rows = [with_edges(lambda g=g: at(g), dim=4) for g in (0, 1, 2)]
    scores = [r[0] for r in rows]
    orders = [r[1] for r in rows]
    print(f"      both 2 cells from the junction, then B delayed by 1 and 2:")
    for g, (c, o) in zip((0, 1, 2), rows):
        print(f"        gap {g}: conflict={c:.3f} order={o:+.0f}")
    check("conflict_falls_with_separation",
          scores == sorted(scores, reverse=True), True)
    check("one_step_at_least_halves_it",
          bool(scores[1] <= 0.55 * scores[0] + 1e-9), True)
    check("simultaneous_has_no_order", orders[0], 0.0)
    check("delayed_peer_makes_me_first", bool(orders[1] > 0), True)


if __name__ == "__main__":
    for fn in list(globals().values()):
        if callable(fn) and getattr(fn, "__name__", "").startswith("test_"):
            print(f"\n--- {fn.__name__} ---")
            fn()
    print("\n" + ("ALL PASS" if not FAIL else f"FAILURES: {FAIL}"))