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
        "ideal_distance": 0.08,
        "stiffness": 0.45,
        # FIX 1: Relax the breaking point.
        # Nodes need room to stretch around the 0.07 radius obstacles.
        "break_threshold": 0.45,
    },
    "obstacles": [
        (0.629, 0.557, 0.020),
        (0.529, 0.286, 0.020),
        (0.286, 0.543, 0.020),
        (0.071, 0.429, 0.020),
        (0.029, 0.208, 0.020),
        (0.786, 0.304, 0.020),
        (0.471, 0.502, 0.020),
        (0.829, 0.457, 0.020),
        (0.779, 0.506, 0.020),
        (0.236, 0.103, 0.020),
        (0.771, 0.709, 0.020),
        (0.329, 0.478, 0.020),
        (0.406, 0.494, 0.020),
        (0.571, 0.122, 0.020),
    ],
    "moving_obstacles": [
        (0.286, 0.571, 0.020, -0.0043, 0.0057),
        (0.599, 0.341, 0.020, -0.0033, 0.0035),
        (0.386, 0.571, 0.020, -0.0056, 0.0066),
        (0.786, 0.443, 0.020, -0.0056, 0.0066),
        (0.076, 0.461, 0.020, -0.0083, 0.0047),
        (0.689, 0.781, 0.020, -0.0022, 0.0015),
        (0.496, 0.641, 0.020, -0.0046, 0.0077),
        (0.506, 0.773, 0.020, -0.0016, 0.0066),
    ],
    "exploration": {
        "map_resolution": 0.004,
        "sensor_range": 0.25,
    },
    "rewards": {
        "r_flow": 2.0,  # Reduced slightly
        "r_explore": 10.0,  # REDUCED: Stop the Kamikaze behavior (was 40.0)
        "r_collision": 40.0,  # High penalty
        "r_loop_integrity": 10.0,  # INCREASED: Value the formation more
        "r_collapse_penalty": 20.0,  # TRIPLED: Breaking the loop must hurt
        "r_idle": 1.0,
        "r_reconnection": 10.0,  # High reward for healing
    },
    "wfc": {
        "history_length": 150,
        "tail_length": 20,
        "collapse_threshold": 0.4,
        "tau": 10,  # INCREASED: Give them more time to recover before reset
    },
    "node": {
        "num_nodes": 20,
        "move_speed": 0.017,
        "fov_angle": 360,
    },
    "repulsion": {
        "local_grid_size": (11, 11),
        "global_grid_shape": (250, 250),
        "decay_lambda": 0.995,
        "blur_data": 0.1,
        "beta": 2.0,  # Strength multiplier
    },
    "gnn": {
        "hidden_dim": 128,
        "lr": 0.0002,  # SLOWER LEARNING: Prevent erratic gradient jumps
        "gamma": 0.98,  # Look further into the future
        "num_layers": 3,
        "n_heads": 4,
    },
}
