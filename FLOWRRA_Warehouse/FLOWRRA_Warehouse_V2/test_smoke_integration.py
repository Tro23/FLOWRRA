"""
test_smoke_integration.py

Runs REAL FLOWRRA episodes on a synthetic warehouse, with every Phase-1 and
Phase-2 feature switched ON, and checks that information actually flows.

WHY THIS EXISTS. Every unit test so far exercises one component against a stub
or a hand-built fixture. None of them runs the pipeline end to end, and several
code paths have therefore NEVER EXECUTED in a real episode -- the waiting branch
most of all, since `waiting.enabled` has defaulted to False since the day it was
written.

The lesson that prompted it: the lesion hook was written on 2026-09-13 and first
executed on 2026-09-14, when it raised AttributeError immediately. It had
survived a full session and a hundred assertions because it sat behind
`if self._lesion_names` and every run used the empty control arm. A code path
gated behind a config flag is a code path that is not being tested.

WHAT IS CHECKED
  * both encoder paths run and produce finite Q-values
  * every state block stays finite across a whole episode
  * the new counters MOVE -- a feature that never fires is not "passing"
  * the diagnostics that should be ~0 are ~0 (projection and kernel fallbacks,
    ray_origin_recovered)
  * flags OFF reproduces the old widths exactly
"""

import os
import sys

# DETERMINISM. Seeding random, numpy and torch is the load-bearing part, and it
# is done per-episode in run_episode().
#
# A single CPU thread on top of that, because torch's conv reductions are
# summed in whatever order the thread pool happens to produce. The differences
# are ~1e-7, but an argmax over Q-values does not care how small a difference
# is -- it just picks the other action, and the trajectory diverges from there.
#
# CORRECTION ON THE RECORD: an earlier version of this file re-exec'd with
# PYTHONHASHSEED=0, on the theory that sets of string fleet ids were iterating
# in hash order and changing recovery order. That was WRONG. Fingerprinting
# every step under PYTHONHASHSEED 0, 1 and 2 gives byte-identical trajectories.
# The flakiness stopped when torch seeding was added, one edit earlier, and the
# hash seed got the credit.
torch_threads_pinned = False
try:
    import torch as _torch
    _torch.set_num_threads(1)
    torch_threads_pinned = True
except Exception:
    pass

import numpy as np
import networkx as nx

from config_warehouse import CONFIG

FAIL = []
WARN = []


def check(name, got, want):
    ok = got == want
    print(f"{'PASS' if ok else 'FAIL'}  {name}: got {got!r} want {want!r}")
    if not ok:
        FAIL.append(name)


def note(name, msg):
    print(f"NOTE  {name}: {msg}")
    WARN.append(name)


# --------------------------------------------------------------- warehouse
def build_warehouse(width=24, aisles=6):
    """Aisles at even y, cross-aisles at both ends and the middle."""
    G = nx.Graph()
    pos = {}
    ys = list(range(0, aisles * 2, 2))
    for y in ys:
        for x in range(width):
            nid = f"n_{x}_{y}"
            G.add_node(nid)
            pos[nid] = {"X": float(x), "Y": float(y), "Z": 0.0}
        for x in range(width - 1):
            G.add_edge(f"n_{x}_{y}", f"n_{x+1}_{y}")
    for x in (0, width // 2, width - 1):
        for y in range(ys[0], ys[-1] + 1):
            nid = f"n_{x}_{y}"
            if nid not in G:
                G.add_node(nid)
                pos[nid] = {"X": float(x), "Y": float(y), "Z": 0.0}
        for y in range(ys[0], ys[-1]):
            G.add_edge(f"n_{x}_{y}", f"n_{x}_{y+1}")

    grid_pos = {(int(v["X"]), int(v["Y"]), int(v["Z"])): k for k, v in pos.items()}
    return G, pos, grid_pos


def build_instance(n_fleets=24, seed=0):
    from node_warehouse import precompute_goal_distances
    G, pos, grid_pos = build_warehouse()
    rng = np.random.default_rng(seed)
    nodes = sorted(G.nodes())

    missions = []
    goal_pool = {}
    for i in range(n_fleets):
        start = nodes[int(rng.integers(0, len(nodes)))]
        goal = nodes[int(rng.integers(0, len(nodes)))]
        missions.append({
            "id": f"f{i}",
            "start_node": start,
            "goal_node": goal,
            "start_pos": np.array([pos[start]["X"], pos[start]["Y"],
                                   pos[start]["Z"]], dtype=np.float32),
            "goal_pos": np.array([pos[goal]["X"], pos[goal]["Y"],
                                  pos[goal]["Z"]], dtype=np.float32),
        })
        goal_pool[goal] = np.array([pos[goal]["X"], pos[goal]["Y"],
                                    pos[goal]["Z"]], dtype=np.float32)

    gdm = precompute_goal_distances(G, [{"goal_node": g} for g in goal_pool])
    return G, grid_pos, missions, gdm, goal_pool


# The configured slow-channel setting, captured once so "flags on" can restore it.
_CONFIGURED_SLOW = CONFIG["density"].get("slow_channel", False)
# Likewise the holon-integrity feature (one extra base feature, 83 -> 84).
_CONFIGURED_HOLON = CONFIG.get("perception", {}).get("holon_integrity", False)


def set_flags(on: bool):
    """All Phase-1 and Phase-2 flags together. They are meant to move as a set."""
    # The slow channel is a THIRD density input channel, so it cannot exist in
    # the old single-number 'affordance' mode -- density_warehouse refuses that
    # combination. "Flags off" (old widths) therefore forces it off. "Flags on"
    # restores whatever the config says, so the suite exercises the real
    # configuration instead of a hardcoded one. Before this, the harness never
    # touched slow_channel, and with a config that enables it the flags-off
    # test crashed on construction.
    CONFIG["density"]["slow_channel"] = _CONFIGURED_SLOW if on else False
    # "Flags off" means the OLD widths, so the holon feature is off too.
    CONFIG.setdefault("perception", {})["holon_integrity"] = _CONFIGURED_HOLON if on else False
    CONFIG["proximity"]["adjacency_metric"] = "graph" if on else "manhattan"
    CONFIG["density"]["ray_transform"] = "smooth" if on else "clip25"
    CONFIG["density"]["output_mode"] = "channels" if on else "affordance"
    CONFIG["gnn"]["encoder"] = "conv" if on else "flat"
    CONFIG["waiting"]["enabled"] = bool(on)
    CONFIG["perception"]["idle_mode"] = "full"


def run_episode(on: bool, steps=60, n_fleets=24, obstacles=None):
    import torch
    from core_warehouse import FLOWRRA
    from agent_warehouse import GNNAgent

    # SEEDED. Without this the agent's epsilon-greedy exploration differs every
    # run, so trajectories, collisions and therefore every counter differ too --
    # and a test whose assertions depend on the weather is not a test. Found the
    # hard way: no_kernel_fallbacks passed and failed alternately on identical
    # code before this was added.
    import random as _random
    _random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)

    set_flags(on)
    G, grid_pos, missions, gdm, goal_pool = build_instance(n_fleets)
    env = FLOWRRA(G, grid_pos, missions, mode="training",
                  goal_distance_maps=gdm, shared_pool_mode=True,
                  goal_pool=goal_pool)
    if obstacles:
        # Never place one under a fleet that is already standing there.
        #
        # The hard veto in get_valid_action_mask() stops a fleet MOVING onto an
        # obstacle; nothing can stop one that was already on the cell when the
        # obstacle appeared. That is exactly why ObstacleField.reset() takes an
        # `avoid` set -- and why injecting cells directly, as this test does,
        # has to do the same job by hand.
        occupied = {tuple(np.round(n.current_pos).astype(int)) for n in env.nodes}
        env.static_obstacles = {c for c in obstacles if c not in occupied}

    n0 = env.nodes[0]
    base_dim = len(n0.get_state_vector(env.nodes))
    dens_dim = env.density.output_dim
    rd = CONFIG["reward_decomposition"]
    agent = GNNAgent(
        node_feature_dim=base_dim + dens_dim, edge_feature_dim=0,
        action_size=CONFIG["gnn"]["action_size"],
        hidden_dim=CONFIG["gnn"]["hidden_dim"],
        num_layers=CONFIG["gnn"]["num_layers"],
        n_heads=CONFIG["gnn"]["num_heads"],
        reward_heads=rd["heads"], head_weights=rd["weights"],
        dropout=CONFIG["gnn"]["dropout"], lr=CONFIG["gnn"]["learning_rate"],
        gamma=CONFIG["training"]["gamma"],
        buffer_capacity=CONFIG["training"]["buffer_capacity"],
        batch_size=CONFIG["training"]["batch_size"],
        stability_coef=CONFIG["gnn"]["stability_coef"],
        encoder_mode=CONFIG["gnn"]["encoder"],
        base_feature_dim=base_dim,
        density_diamond_mask=env.density._diamond_mask,
    )
    env.gnn = agent

    nonfinite = 0
    for t in range(steps):
        env.step(episode_step=1, total_episodes=1)
        for n in env.nodes:
            v = n.get_state_vector(env.nodes)
            if not np.all(np.isfinite(v)):
                nonfinite += 1
        if env.is_episode_over():
            break

    est = env.get_error_statistics()
    return env, agent, base_dim, dens_dim, nonfinite, len(env.gnn.memory), t + 1


# ==================================================================== tests
def test_flags_off_reproduces_old_widths():
    env, agent, base, dens, nf, mem, steps = run_episode(False, steps=12)
    check("off_density_dim", dens, 231)
    check("off_total_width", base + dens, 314)
    check("off_encoder", CONFIG["gnn"]["encoder"], "flat")
    check("off_no_nonfinite_states", nf, 0)
    check("off_buffer_filled", mem > 0, True)


def test_flags_on_runs_end_to_end():
    env, agent, base, dens, nf, mem, steps = run_episode(True, steps=60)
    print(f"      ran {steps} steps, buffer={mem}, "
          f"active={len(env.get_active_nodes())}/{len(env.nodes)}")
    # Exact, not loosened: 231 diamond cells per density channel, 83 base
    # features. Two channels (mask, repulsion) -> 462 / 545; the slow channel
    # adds a third -> 693 / 776. The channel count comes from the config, so
    # this holds for whichever configuration the suite is run against.
    _ch = 3 if CONFIG["density"].get("slow_channel", False) else 2
    check("on_density_dim", dens, _ch * 231)
    _base = 83 + (1 if CONFIG.get("perception", {}).get("holon_integrity", False) else 0)
    check("on_total_width", base + dens, _base + _ch * 231)
    check("on_no_nonfinite_states", nf, 0)
    check("on_buffer_filled", mem > 0, True)

    # Q-values carry -inf at STRUCTURALLY INVALID actions, by design: they are
    # masked before the argmax so an impossible move can never be selected. On a
    # degree-~2.27 graph most of the 7 actions are invalid, so ~56% of entries
    # being -inf is correct. What must be finite is the VALID entries.
    q = agent.last_q_values
    check("q_values_exist", q is not None, True)
    if q is not None:
        finite = np.isfinite(q)
        check("valid_actions_have_finite_q", bool(finite.any()), True)
        check("every_fleet_has_at_least_one_valid_action",
              bool(finite.any(axis=1).all()), True)
        print(f"      Q {q.shape}: {int((~finite).sum())} of {q.size} masked "
              f"(-inf at structurally invalid actions)")

    est = env.get_error_statistics()
    print(f"      waits_started={est['waits_started']} "
          f"mutual={est['mutual_waits']} capped={est['waits_capped']} "
          f"waiting_now={est['waiting_now']}")
    print(f"      proj_fallbacks={est['projection_fallbacks']} "
          f"kernel_fallbacks={est['kernel_manhattan_fallbacks']} "
          f"ray_origin_recovered={est['ray_origin_recovered']}")

    # Diagnostics that should be ~0. Non-zero means a fallback is carrying the
    # behaviour instead of backstopping it.
    # projection_fallbacks counts FLEET-STEPS whose intended path could not be
    # computed -- a fleet standing on its goal with no descent left, or one whose
    # goal was just reassigned. A handful is expected; a large number means the
    # BFS descent is failing and dead reckoning is carrying the behaviour.
    if est["projection_fallbacks"] > 0.02 * len(env.nodes) * steps:
        check("projection_fallbacks_are_rare", est["projection_fallbacks"], 0)
    else:
        print(f"      projection fallbacks {est['projection_fallbacks']} "
              f"of ~{len(env.nodes) * steps} fleet-steps -- rare, as expected")
    # A RATE, not zero.
    #
    # Same correction as projection_fallbacks above, learned the same way. This
    # assertion demanded exactly 0 and fired intermittently -- about one run in
    # three -- and I explained it wrongly twice: first as PYTHONHASHSEED (step
    # fingerprints under seeds 0, 1 and 2 are byte-identical, so no), then as
    # torch thread count (pinning to one thread made it MORE frequent, so no).
    #
    # What the counter means: a stamp source whose cell is not a graph node, so
    # the graph kernel could not place it. The known cause -- dead reckoning
    # projecting off the map edge -- is fixed. A residue of a few hundred in a
    # 1,440-fleet-step episode is one bad cell stamped by every observer for the
    # ~8 steps collapse memory survives, which is rare and bounded rather than
    # systematic. The cells are now recorded, so the next occurrence arrives
    # with its own evidence instead of another hypothesis.
    kf = est["kernel_manhattan_fallbacks"]
    if kf:
        print(f"      kernel fallbacks {kf} from cells "
              f"{est.get('kernel_fallback_cells')}")
    check("no_kernel_fallbacks", kf, 0)
    print(f"      off-grid memory splats refused: "
          f"{est.get('memory_offgrid_rejected', 0)}")
    check("no_ray_origin_recovery", est["ray_origin_recovered"], 0)

    # A feature that never fires is not passing.
    if est["waits_started"] == 0:
        note("waiting_never_fired",
             "no fleet ever held still next to a mobile peer in this run -- "
             "the branch is untested, not proven. Raise fleet count or steps.")


def test_new_state_blocks_carry_real_values():
    env, agent, base, dens, nf, mem, steps = run_episode(True, steps=40)
    layout = env.nodes[0].state_layout()
    vecs = np.stack([n.get_state_vector(env.nodes) for n in env.nodes])

    for block in ("ray_hit_waiting", "ray_hit_permanent", "ray_hit_unknown",
                  "gibbs_state", "situation_features"):
        lo, hi = layout[block]
        sl = vecs[:, lo:hi]
        check(f"{block}_finite", bool(np.all(np.isfinite(sl))), True)

    lo, hi = layout["gibbs_state"]
    ent = vecs[:, lo]
    tt = vecs[:, lo + 1]
    print(f"      entropy  min={ent.min():.3f} max={ent.max():.3f} "
          f"mean={ent.mean():.3f}")
    print(f"      T        min={tt.min():.3f} max={tt.max():.3f}")
    check("entropy_in_range", bool((ent >= 0).all() and (ent <= 1).all()), True)
    check("T_in_range", bool((tt > 0).all() and (tt <= 1).all()), True)
    if ent.std() < 1e-6:
        note("entropy_constant",
             f"entropy is the same for every fleet ({ent[0]:.3f}) -- it only "
             "varies under contention, so this may just mean the map is empty")

    lo, hi = layout["ray_hit_permanent"]
    perm = vecs[:, lo:hi]
    print(f"      rays hitting a permanent peer: {int(perm.sum())}")


def test_obstacle_vetoes_inside_a_real_episode():
    """The hard veto has to survive contact with the action loop, not just a
    unit test on a bare FleetNode."""
    env, agent, base, dens, nf, mem, steps = run_episode(
        True, steps=20, obstacles={(5, 0, 0), (12, 2, 0)})
    check("obstacles_registered", len(env.static_obstacles) > 0, True)
    check("obstacles_reported",
          env.get_error_statistics()["static_obstacles"],
          len(env.static_obstacles))

    occupied = {tuple(np.round(n.current_pos).astype(int)) for n in env.nodes}
    trespass = occupied & env.static_obstacles
    check("no_fleet_ever_stands_on_an_obstacle", sorted(trespass), [])


def test_adjacency_graph_mode_is_sparser():
    """Graph distance must link FEWER pairs than Manhattan -- never more."""
    from core_warehouse import FLOWRRA
    G, grid_pos, missions, gdm, goal_pool = build_instance(24)

    set_flags(False)
    e1 = FLOWRRA(G, grid_pos, missions, mode="init", goal_distance_maps=gdm,
                 shared_pool_mode=True, goal_pool=goal_pool)
    e1.proximity.refresh(e1.nodes, excluded_ids=e1.immobile_nodes)
    a_man = e1._build_adjacency()

    set_flags(True)
    e2 = FLOWRRA(G, grid_pos, missions, mode="init", goal_distance_maps=gdm,
                 shared_pool_mode=True, goal_pool=goal_pool)
    e2.proximity.refresh(e2.nodes, excluded_ids=e2.immobile_nodes)
    a_gph = e2._build_adjacency_graph()

    print(f"      edges: manhattan={int(a_man.sum())} graph={int(a_gph.sum())}")
    check("graph_adjacency_is_a_subset",
          bool(((a_gph > 0) & (a_man == 0)).sum() == 0), True)
    check("graph_adjacency_is_sparser",
          bool(a_gph.sum() <= a_man.sum()), True)


if __name__ == "__main__":
    for fn in list(globals().values()):
        if callable(fn) and getattr(fn, "__name__", "").startswith("test_"):
            print(f"\n--- {fn.__name__} ---")
            fn()
    set_flags(False)
    print()
    if WARN:
        print(f"NOTES (not failures): {WARN}")
    print("ALL PASS" if not FAIL else f"FAILURES: {FAIL}")


def trace_episode(steps=40, n_fleets=24):
    """Per-step fingerprint of the whole fleet state, for diffing two runs."""
    import hashlib
    import random as _random
    import torch
    from core_warehouse import FLOWRRA
    from agent_warehouse import GNNAgent
    _random.seed(7); np.random.seed(7); torch.manual_seed(7)
    set_flags(True)
    G, grid_pos, missions, gdm, goal_pool = build_instance(n_fleets)
    env = FLOWRRA(G, grid_pos, missions, mode="training",
                  goal_distance_maps=gdm, shared_pool_mode=True,
                  goal_pool=goal_pool)
    n0 = env.nodes[0]
    base = len(n0.get_state_vector(env.nodes))
    rd = CONFIG["reward_decomposition"]
    env.gnn = GNNAgent(
        node_feature_dim=base + env.density.output_dim, edge_feature_dim=0,
        action_size=7, hidden_dim=CONFIG["gnn"]["hidden_dim"],
        num_layers=CONFIG["gnn"]["num_layers"], n_heads=CONFIG["gnn"]["num_heads"],
        reward_heads=rd["heads"], head_weights=rd["weights"],
        encoder_mode="conv", base_feature_dim=base,
        density_diamond_mask=env.density._diamond_mask)
    out = []
    for t in range(steps):
        env.step(episode_step=1, total_episodes=1)
        pos = "".join(f"{n.id}:{n.current_pos.round(3).tolist()}" for n in env.nodes)
        out.append(hashlib.md5(pos.encode()).hexdigest()[:8])
        if env.is_episode_over():
            break
    return out