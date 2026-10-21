"""
test_guard_integration.py -- the intent guard, bubbles by kind and route by
default inside REAL FLOWRRA steps (build log entry 29).

WHAT IT CHECKS, on test_smoke_integration's synthetic warehouse (24 fleets in a
small floor, so they meet often), with the order stream and its fixed book:
  1. THE INVARIANT: the env's own live check (guard_violations) stays 0 -- no
     pair the guard checked ended a step under min_gap hops, or drove through
     the other -- over 300 steps;
  2. the guard actually engaged (guard_pairs_short > 0): the test is not passing
     on an empty floor;
  3. route by default decided some moves and the network the rest (on this
     crowded floor the network decides most: the share is printed, and read on
     the real maps from deploy_eval --safety on);
  4. the same seed gives the same episode, step for step (positions and guard
     counters), so the guard did not bring in any hidden randomness;
  5. with every new switch OFF, nothing new happens: no guard counters, no
     routed moves.
It also prints collisions (from movement they should not happen with the guard;
re-entries and recovery moves can still place fleets close).

Run:  python test_guard_integration.py      (ends with ALL PASS; a few minutes)
"""

import random

import numpy as np

from config_warehouse import CONFIG
import core_warehouse
from node_warehouse import precompute_goal_distances
from test_smoke_integration import build_instance, set_flags

FAIL = []
BOOK = {"enabled": True, "orders": 60, "release_steps": 150}


def check(name, got, want):
    ok = got == want
    print(f"{'PASS' if ok else 'FAIL'}  {name}: got {got!r} want {want!r}")
    if not ok:
        FAIL.append(name)


def configure(safety_on: bool):
    sf = CONFIG.setdefault("safety", {})
    for k in ("intent_guard", "bubbles_by_kind", "hops_everywhere"):
        sf[k] = safety_on
    sf.setdefault("min_gap", 1.0)
    CONFIG.setdefault("conflict", {})["route_by_default"] = safety_on
    CONFIG.setdefault("reward_decomposition", {})["presence_charge"] = not safety_on
    CONFIG.setdefault("step", {})["simultaneous"] = True


def run(steps=300, n_fleets=24, seed=7):
    import torch
    from agent_warehouse import GNNAgent
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
        density_diamond_mask=env.density._diamond_mask)
    trail = []
    for t in range(steps):
        env.step(episode_step=1, total_episodes=1)
        trail.append((tuple(tuple(np.round(n.current_pos, 6)) for n in env.nodes),
                      env.guard_pairs_short, env.guard_stops))
        if env.is_episode_over():
            break
    return env, trail


def main():
    saved = {k: (dict(CONFIG.get(k, {}) or {})) for k in
             ("stream", "obstacles", "safety", "conflict", "reward_decomposition", "step")}
    try:
        CONFIG["stream"] = dict(saved["stream"], enabled=True, order_book=dict(BOOK))
        CONFIG["stream"].pop("order_floor_window", None)
        CONFIG["obstacles"] = dict(saved["obstacles"], enabled=False)

        configure(True)
        env, trail = run()
        est = env.get_error_statistics()
        print(f"      guard ON: {len(trail)} steps | guard stepped in on {est['guard_steps_active']} steps "
              f"| pairs too close {est['guard_pairs_short']} | slowed {est['guard_fleets_slowed']} "
              f"| stopped {est['guard_stops']} | fallback {est['guard_fallback_stops']} "
              f"| route decided {est['route_share'] * 100:.0f}% | collisions {env.loop.total_collisions}")
        check("invariant_never_broken", est["guard_violations"], 0)
        check("guard_engaged", est["guard_pairs_short"] > 0, True)
        # On this floor (24 fleets packed in, so most are in a warned pair most
        # of the time) the network decides most moves: the first run gave route
        # 28%. So this asks only that BOTH decide some; the share on the real
        # maps is read from deploy_eval --safety on, not asserted here.
        check("route_decided_some_moves", est["route_share"] > 0.0, True)
        check("network_still_decides_some", est["route_share"] < 1.0, True)

        env2, trail2 = run()
        check("same_seed_same_episode", trail2 == trail, True)

        configure(False)
        env3, _ = run(steps=60)
        est3 = env3.get_error_statistics()
        check("switches_off_no_guard", (est3["guard_on"], est3["guard_pairs_checked"]), (0, 0))
        check("switches_off_no_route", (est3["route_on"], env3.route_routed), (0, 0))
    finally:
        for k, v in saved.items():
            CONFIG[k] = v


if __name__ == "__main__":
    main()
    print("\n" + ("ALL PASS" if not FAIL else f"FAILURES: {FAIL}"))