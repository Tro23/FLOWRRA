"""
Centralized Configuration for FLOWRRA Loop-GNN Swarm (V2 Mechanics).
"""

# import numpy as np

CONFIG = {
    "spatial": {
        "dimensions": 2,
        "world_bounds": (7, 7),
        "toroidal": True,
    },
    "loop": {
        "ideal_distance": 0.7,  # Ideal distance between connected nodes
        "stiffness": 0.2,  # Spring force holding the loop
        "break_threshold": 20.0,  # Distance at which the loop breaks
    },
    # Static obstacles: (x, y, radius)
    "obstacles": [
        # (3.0, 3.0, 0.8),
        (6.5, 6.0, 0.8),
        (6.5, 2.0, 0.5),
        (2.0, 8.0, 0.6),
        (7.5, 3.0, 0.7),
        # (4.0, 6.0, 0.5),
    ],
    # Moving obstacles: (x, y, radius, vx, vy)
    "moving_obstacles": [
        # (6.0, 5.0, 0.4, 0.5, 0.3),  # Slow moving obstacle
        (2.0, 4.0, 0.3, -0.3, 0.4),
    ],
    "exploration": {
        "map_resolution": 1.0,
        "sensor_range": 1.5,
    },
    "rewards": {
        "r_flow": 1.5,  # Movement reward
        "r_explore": 30.0,  # New cell discovery
        "r_collision": 10.0,  # Obstacle collision penalty
        "r_loop_integrity": 2.0,  # Loop maintenance reward
        "r_collapse_penalty": 3.0,  # Heavy penalty for broken loop
        "r_idle": 2.9,  # Idle penalty
    },
    "wfc": {
        "history_length": 150,  # How many steps to remember
        "tail_length": 50,  # Size of the "Comet Tail" to smooth over
        "collapse_threshold": 0.4,  # Coherence below this triggers collapse
        "tau": 8,  # Consecutive failing steps before trigger
    },
    "node": {
        "num_nodes": 10,
        "move_speed": 0.25,
        "fov_angle": 360,
    },
    "repulsion": {
        "local_grid_size": (3, 3),
        "global_grid_shape": (50, 50),
        "decay_lambda": 0.99999,
        "blur_data": 0.0,
    },
    "gnn": {
        "hidden_dim": 128,
        "lr": 0.0003,
        "gamma": 0.95,
        "num_layers": 3,
        "n_heads": 4,
    },
}
