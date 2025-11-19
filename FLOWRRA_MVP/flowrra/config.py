"""
Centralized Configuration for FLOWRRA Exploration MVP.
"""

CONFIG = {
    'spatial': {
        'dimensions': 2,       # Keep 2D for MVP Exploration Demo
        'world_bounds': (1.0, 1.0),
        'toroidal': False,     # False for exploration (we want walls)
    },
    'exploration': {
        'map_resolution': 0.02, # High res for accurate % coverage
        'sensor_range': 0.15,
        'frontier_sample_rate': 5, # Recalculate frontiers every 5 steps (Optimization)
    },
    'rewards': {
        'r_flow': 0.5,         # Reward for moving with the flow
        'r_explore': 2.0,      # Big reward for discovering new tiles
        'r_collision': 5.0,    # Penalty for hitting obstacles/peers
        'r_idle': 0.1          # Penalty for staying still
    },
    'node': {
        'num_nodes': 10,
        'move_speed': 0.02,
        'fov_angle': 360,      # 360 vision for exploration
    },
    'repulsion': {
        'local_grid_size': (5, 5),
        'global_grid_shape': (50, 50),
    },
    'gnn': {
        'hidden_dim': 128,
        'lr': 0.001,
        'gamma': 0.99
    }
}