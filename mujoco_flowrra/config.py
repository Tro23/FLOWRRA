"""
config.py

Global Configuration for FLOWRRA MuJoCo Environment.
"""

CONFIG = {
    "node": {
        "num_nodes": 10,           # Size of the swarm
    },
    "spatial": {
        "dimensions": 3,           # MuJoCo is strictly 3D
        "world_bounds": [-10, 10]  # Used for generic spatial constraints
    },
    "exploration": {
        "sensor_range": 4.0,       # How far drones can see peers/obstacles
        "map_resolution": 0.5
    },
    "repulsion": {
        "local_grid_size": (5, 5, 5), # 3D local vision tensor for the GNN
        "global_grid_shape": (40, 40, 40)
    },
    "loop": {
        "ideal_distance": 1.5,     # Target distance between nodes in the topology
        "break_threshold": 3.0,    # Distance at which the connection physically snaps
        "stiffness": 1.0           # Legacy topological stiffness
    },
    "wfc": {
        "history_length": 200,     # How many frames of relative shape to remember
        "tail_length": 3,
        "collapse_threshold": 0.6,
        "tau": 5                   # Consecutive unstable frames before WFC triggers
    },
    "gnn": {
        "hidden_dim": 128,         # Neural network capacity
        "lr": 0.0003,
        "gamma": 0.95,
        "stability_coef": 0.5
    },
    "rewards": {
        "r_flow": 10.0,            # Reward multiplier for actual physical distance moved
        "r_collision": 25.0,       # Heavy penalty for hitting a wall (velocity drops to 0)
        "r_loop_integrity": 5.0,   # Reward for keeping the topological ring intact
        "r_idle": 1.0,             # Penalty for hovering in place
        "r_collapse_penalty": 15.0 # Penalty for triggering a Wave Function Collapse
    }
}