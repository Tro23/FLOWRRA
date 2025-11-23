"""
config.py - Stability Patch
"""

CONFIG = {
    "spatial": {
        "dimensions": 2,
        "world_bounds": (1.0, 1.0),
        "toroidal": True,
    },
    "loop": {
        "ideal_distance": 0.11,
        "stiffness": 0.50,
        # FIX 1: Relax the breaking point.
        # Nodes need room to stretch around the 0.07 radius obstacles.
        "break_threshold": 0.35,
    },
    "obstacles": [
        (0.929, 0.857, 0.074),
        (0.929, 0.286, 0.071),
        (0.286, 0.543, 0.076),
        (0.071, 0.429, 0.070),
    ],
    "moving_obstacles": [
        (0.286, 0.571, 0.043, -0.0043, 0.0057),
    ],
    "exploration": {
        "map_resolution": 0.02,
        "sensor_range": 0.20,
    },
    "rewards": {
        "r_flow": 2.0,  # Reduced slightly
        "r_explore": 15.0,  # REDUCED: Stop the Kamikaze behavior (was 40.0)
        "r_collision": 45.0,  # High penalty
        "r_loop_integrity": 10.0,  # INCREASED: Value the formation more
        "r_collapse_penalty": 20.0,  # TRIPLED: Breaking the loop must hurt
        "r_idle": 0.2,
        "r_reconnection": 15.0,  # High reward for healing
    },
    "wfc": {
        "history_length": 200,
        "tail_length": 30,
        "collapse_threshold": 0.4,
        "tau": 10,  # INCREASED: Give them more time to recover before reset
    },
    "node": {
        "num_nodes": 10,
        "move_speed": 0.015,
        "fov_angle": 360,
    },
    "repulsion": {
        "local_grid_size": (5, 5),
        "global_grid_shape": (80, 80),
        "decay_lambda": 0.995,
        "blur_data": 0.1,
        "beta": 1.5,  # Strength multiplier
    },
    "gnn": {
        "hidden_dim": 128,
        "lr": 0.0002,  # SLOWER LEARNING: Prevent erratic gradient jumps
        "gamma": 0.98,  # Look further into the future
        "num_layers": 3,
        "n_heads": 4,
    },
}
