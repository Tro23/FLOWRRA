"""
Centralized Configuration for FLOWRRA Loop-GNN Swarm (V2 Mechanics).
"""

# import numpy as np

CONFIG = {
    "spatial": {
        "dimensions": 2,
        "world_bounds": (10, 10),
        "toroidal": True,
    },
    "loop": {
        "ideal_distance": 0.6,  # Ideal distance between connected nodes
        "stiffness": 0.5,  # Spring force holding the loop
        "break_threshold": 1.5,  # Distance at which the loop breaks
    },
    # Static obstacles: (x, y, radius)
    "obstacles": [
        (3.0, 3.0, 0.8),
        (7.0, 7.0, 0.8),
        (5.0, 2.0, 0.5),
        (2.0, 8.0, 0.6),
        (8.0, 3.0, 0.7),
        (4.0, 6.0, 0.5),
    ],
    # Moving obstacles: (x, y, radius, vx, vy)
    "moving_obstacles": [
        (6.0, 5.0, 0.4, 0.5, 0.3),  # Slow moving obstacle
        (2.0, 4.0, 0.3, -0.3, 0.4),
    ],
    "exploration": {
        "map_resolution": 0.2,
        "sensor_range": 1.5,
    },
    "rewards": {
        "r_flow": 0.2,  # Movement reward
        "r_explore": 1.5,  # New cell discovery
        "r_collision": 10.0,  # Obstacle collision penalty
        "r_loop_integrity": 2.0,  # Loop maintenance reward
        "r_collapse_penalty": 5.0,  # Heavy penalty for broken loop
        "r_idle": 0.05,  # Idle penalty
    },
    "wfc": {
        "history_length": 70,  # How many steps to remember
        "tail_length": 15,  # Size of the "Comet Tail" to smooth over
        "collapse_threshold": 0.6,  # Coherence below this triggers collapse
        "tau": 3,  # Consecutive failing steps before trigger
    },
    "node": {
        "num_nodes": 12,
        "move_speed": 0.05,
        "fov_angle": 360,
    },
    "repulsion": {
        "local_grid_size": (7, 7),
        "global_grid_shape": (50, 50),
    },
    "gnn": {
        "hidden_dim": 128,
        "lr": 0.0003,
        "gamma": 0.95,
        "num_layers": 3,
        "n_heads": 4,
    },
}
