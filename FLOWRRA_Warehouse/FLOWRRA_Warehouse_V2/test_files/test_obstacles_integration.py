"""
test_obstacles_integration.py -- "cleanup" obstacles inside REAL FLOWRRA steps.

test_obstacles_cleanup.py checks the obstacle module alone. This one runs the
whole orchestrator on test_smoke_integration's synthetic warehouse with
obstacles on, and checks the hooks in core_warehouse.py:
  * the core tells the obstacle module where every fleet stands, every step;
  * no fleet ever stands on a human or on debris, and no human on a fleet;
  * every fleet sees the same obstacle cells the core holds;
  * the obstacle counters reach get_error_statistics() (as obstacle_*);
  * the obstacle stream is seeded: two identical runs give identical cells.

Run:  python test_obstacles_integration.py      (ends with ALL PASS; ~1 minute)
"""

import random

import numpy as np

from config_warehouse import CONFIG
import core_warehouse
from test_smoke_integration import build_instance, set_flags

FAIL = []

FAST = {"enabled": True, "mode": "cleanup", "seed": 0, "debris_interval": 15,
        "max_debris": 2, "response_delay": 0, "cleanup_steps": 3,
        "human_move_period": 1, "human_patience": 2, "max_debris_life": 400}


def check(name, got, want):
    ok = got == want
    print(f"{'PASS' if ok else 'FAIL'}  {name}: got {got!r} want {want!r}")
    if not ok:
        FAIL.append(name)


def run(steps=150, n_fleets=24):
    import torch
    from agent_warehouse import GNNAgent
    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    core_warehouse._OBSTACLE_EPISODES = 0          # same obstacle stream each call
    core_warehouse._WAVE_EPISODES = 0              # and the same failure waves
    set_flags(True)
    G, grid_pos, missions, gdm, goal_pool = build_instance(n_fleets)
    env = core_warehouse.FLOWRRA(G, grid_pos, missions, mode="training",
                                 goal_distance_maps=gdm, shared_pool_mode=True,
                                 goal_pool=goal_pool)
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

    fleet_on_obstacle = human_on_fleet = mismatched_view = 0
    history = []
    before = {}
    for t in range(steps):
        prev_obst = {grid_pos[c] for c in env.static_obstacles if c in grid_pos}
        before = {n.id: (n.current_pos.copy(), set(env.rules._anchors(n)) if env.rules else set())
                  for n in env.nodes}
        env.step(episode_step=1, total_episodes=1)
        fleets = env._fleet_cell_ids()
        obst_ids = {grid_pos[c] for c in env.static_obstacles if c in grid_pos}
        clash = fleets & obst_ids
        fleet_on_obstacle += len(clash)
        human_on_fleet += sum(1 for h in env.obstacle_field.c_humans if h["cell"] in fleets)
        for cell in sorted(clash):
            # Who is on it, and which side arrived this step: the diagnosis.
            for n in env.nodes:
                an = set(env.rules._anchors(n)) if env.rules else set()
                if cell in an:
                    p0, a0 = before.get(n.id, (None, set()))
                    kind = ("human" if any(h["cell"] == cell for h in env.obstacle_field.c_humans)
                            else "debris")
                    print(f"      CLASH step {t}: fleet {n.id} on {cell} ({kind}); "
                          f"pos {p0} -> {n.current_pos}, anchors {sorted(a0)} -> {sorted(an)}; "
                          f"obstacle there before the step: {cell in prev_obst}; "
                          f"last action {getattr(n, 'last_action', '?')}; "
                          f"immobile {n.id in env.immobile_nodes}")
        mismatched_view += sum(1 for n in env.nodes if n.static_obstacles != env.static_obstacles)
        history.append(tuple(sorted(env.static_obstacles)))
        if env.is_episode_over():
            break
    return env, fleet_on_obstacle, human_on_fleet, mismatched_view, history


def test_obstacles_in_real_steps():
    saved = dict(CONFIG.get("obstacles", {}) or {})
    CONFIG["obstacles"] = dict(FAST)
    try:
        env, f_on_o, h_on_f, mism, hist1 = run()
        _, _, _, _, hist2 = run()
        if hist1 != hist2:
            k = next(i for i, (a, b) in enumerate(zip(hist1, hist2)) if a != b)
            print(f"      the two runs first differ at step {k}")
    finally:
        CONFIG["obstacles"] = saved
    est = env.get_error_statistics()
    s = env.obstacle_field.statistics()
    print(f"      debris spawned {s['debris_spawned']}, cleaned {s['debris_cleaned']}, "
          f"expired {s['debris_expired']}; humans in {s['humans_entered']}, "
          f"out {s['humans_exited']}; waits {s['human_wait_steps']}, "
          f"reroutes {s['human_reroutes']}, near misses {s['near_miss_steps']}")
    check("obstacles_were_used", s["debris_spawned"] > 0 and s["humans_entered"] > 0, True)
    check("some_debris_cleaned", s["debris_cleaned"] > 0, True)
    check("no_fleet_ever_on_an_obstacle", f_on_o, 0)
    check("no_human_ever_on_a_fleet", h_on_f, 0)
    check("every_fleet_sees_the_core_obstacles", mism, 0)
    check("counters_reach_the_statistics", "obstacle_debris_spawned" in est, True)
    check("obstacle_stream_is_seeded", hist1 == hist2, True)


if __name__ == "__main__":
    test_obstacles_in_real_steps()
    print("\n" + ("ALL PASS" if not FAIL else f"FAILURES: {FAIL}"))