"""
config.py

Centralized hyperparameter configuration for FLOWRRA-GNN.
Supports both 2D and 3D modes with flexible grid sizing.
"""

# =============================================================================
# SPATIAL CONFIGURATION
# =============================================================================
SPATIAL_CONFIG = {
    'dimensions': 3,  # 2 for 2D, 3 for 3D
    'world_bounds': (1.0, 1.0, 1.0),  # Normalized space [0,1]^n
    'toroidal_wrap': True,  # Wrap around edges (pacman style)
}

# =============================================================================
# REPULSION FIELD CONFIGURATION
# =============================================================================
REPULSION_CONFIG = {
    # Local grid (what each node computes)
    'local_grid_size': (5, 5, 5),  # Voxels around each node (5x5x5 for 3D)
    'local_grid_resolution': 0.05,  # Spatial resolution of local grid
    
    # Global grid (visualization only)
    'global_grid_shape': (60, 60, 60),  # Full world discretization
    
    # Repulsion field parameters
    'eta': 0.5,  # Learning rate for splatting
    'gamma_f': 0.9,  # Decay factor for comet-tail
    'k_f': 5,  # Number of future projection steps
    'sigma_f': 0.05,  # Gaussian kernel width
    'decay_lambda': 0.01,  # Per-step field decay
    'blur_delta': 0.2,  # Diffusion mix factor
    'beta': 0.7,  # Repulsion strength multiplier
}

# =============================================================================
# NODE CONFIGURATION
# =============================================================================
NODE_CONFIG = {
    'num_nodes': 15,  # Number of agents in swarm
    'sensor_range': 0.15,  # Detection radius
    'move_speed': 0.015,  # Movement speed per step
    'rotation_speed': 2.0,  # Angular rotation speed (degrees for 2D)
    
    # 3D orientation parameters
    'azimuth_steps': 24,  # Horizontal angle discretization
    'elevation_steps': 12,  # Vertical angle discretization (3D only)
}

# =============================================================================
# GNN ARCHITECTURE
# =============================================================================
GNN_CONFIG = {
    'architecture': 'GAT',  # 'GAT' (Graph Attention) or 'GraphSAGE'
    'hidden_dim': 128,  # Hidden layer dimension
    'num_layers': 3,  # Number of graph conv layers
    'num_heads': 4,  # Attention heads (for GAT)
    'dropout': 0.1,  # Dropout rate
    'aggregation': 'mean',  # 'mean', 'max', or 'sum'
    
    # Input/output sizes (auto-computed but can override)
    'node_feature_dim': None,  # Auto: pos(3) + vel(3) + local_grid(125) + detections
    'edge_feature_dim': 8,  # distance(1) + bearing(3) + rel_vel(3) + type(1)
    'action_space_size': None,  # Auto: 6 directions + 4 angle actions for 3D
}

# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================
TRAINING_CONFIG = {
    # Learning parameters
    'learning_rate': 0.0003,
    'gamma': 0.95,  # Discount factor
    'batch_size': 32,
    'buffer_capacity': 15000,
    
    # Exploration strategy
    'epsilon_strategy': 'gaussian',  # 'gaussian', 'exponential', 'linear'
    'epsilon_min': 0.05,
    'epsilon_peak': 0.8,
    'epsilon_decay': 0.9995,
    
    # Training schedule
    'total_training_steps': 30000,
    'episode_steps': 1000,
    'warm_up_steps': 200,
    'target_update_freq': 100,  # Steps between target network updates
    
    # Reward shaping
    'coherence_weight': 0.5,
    'exploration_weight': 1.5,
    'movement_weight': 57.0,
    'collision_penalty': -2.0,
    'collapse_penalty': -3.0,
}

# =============================================================================
# ENVIRONMENT B (OBSTACLES)
# =============================================================================
ENVIRONMENT_B_CONFIG = {
    'grid_size': 60,  # Discrete grid for obstacles
    'num_fixed_obstacles': 15,
    'num_moving_obstacles': 7,
    'obstacle_move_prob': 0.8,  # Probability obstacle moves each step
}

# =============================================================================
# WAVE FUNCTION COLLAPSE
# =============================================================================
WFC_CONFIG = {
    'history_length': 200,
    'tail_length': 15,
    'collapse_threshold': 0.88,
    'tau': 2,  # Consecutive unstable steps to trigger collapse
}

# =============================================================================
# VISUALIZATION & LOGGING
# =============================================================================
VISUALIZATION_CONFIG = {
    'visual_dir': 'flowrra_gnn_visuals',
    'log_file': 'flowrra_gnn_training.csv',
    'model_save_path': 'flowrra_gnn_agent.pth',
    
    # Visualization frequency
    'visualize_every_n_steps': 500,
    'log_every_n_steps': 50,
    
    # Deployment
    'deployment_steps': 50,
    'deployment_episodes': 3,
    'deployment_fps': 10,  # For GIF generation
}

# =============================================================================
# DIMENSION-SPECIFIC PRESETS
# =============================================================================

def get_2d_config():
    """Returns configuration optimized for 2D experiments."""
    config = {
        'spatial': SPATIAL_CONFIG.copy(),
        'repulsion': REPULSION_CONFIG.copy(),
        'node': NODE_CONFIG.copy(),
        'gnn': GNN_CONFIG.copy(),
        'training': TRAINING_CONFIG.copy(),
        'env_b': ENVIRONMENT_B_CONFIG.copy(),
        'wfc': WFC_CONFIG.copy(),
        'viz': VISUALIZATION_CONFIG.copy(),
    }
    
    # 2D-specific overrides
    config['spatial']['dimensions'] = 2
    config['spatial']['world_bounds'] = (1.0, 1.0)
    config['repulsion']['local_grid_size'] = (5, 5)
    config['repulsion']['global_grid_shape'] = (60, 60)
    config['node']['elevation_steps'] = 1  # No elevation in 2D
    
    return config

def get_3d_config():
    """Returns configuration optimized for 3D experiments."""
    config = {
        'spatial': SPATIAL_CONFIG.copy(),
        'repulsion': REPULSION_CONFIG.copy(),
        'node': NODE_CONFIG.copy(),
        'gnn': GNN_CONFIG.copy(),
        'training': TRAINING_CONFIG.copy(),
        'env_b': ENVIRONMENT_B_CONFIG.copy(),
        'wfc': WFC_CONFIG.copy(),
        'viz': VISUALIZATION_CONFIG.copy(),
    }
    
    # Already defaults to 3D, but being explicit
    config['spatial']['dimensions'] = 3
    config['spatial']['world_bounds'] = (1.0, 1.0, 1.0)
    config['repulsion']['local_grid_size'] = (5, 5, 5)
    config['repulsion']['global_grid_shape'] = (60, 60, 60)
    
    return config

def get_fast_prototype_config():
    """Lightweight config for rapid prototyping."""
    config = get_2d_config()
    
    config['node']['num_nodes'] = 8
    config['repulsion']['local_grid_size'] = (3, 3)
    config['repulsion']['global_grid_shape'] = (30, 30)
    config['training']['total_training_steps'] = 5000
    config['training']['episode_steps'] = 200
    config['env_b']['num_fixed_obstacles'] = 8
    config['env_b']['num_moving_obstacles'] = 3
    
    return config

def get_large_scale_config():
    """Config for testing scalability (100+ nodes)."""
    config = get_3d_config()
    
    config['node']['num_nodes'] = 100
    config['repulsion']['local_grid_size'] = (7, 7, 7)
    config['gnn']['hidden_dim'] = 256
    config['gnn']['num_layers'] = 4
    config['training']['buffer_capacity'] = 30000
    config['training']['total_training_steps'] = 50000
    
    return config

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def validate_config(config):
    """Validates configuration consistency."""
    dims = config['spatial']['dimensions']
    
    assert dims in [2, 3], "Dimensions must be 2 or 3"
    assert len(config['spatial']['world_bounds']) == dims, \
        f"world_bounds must have {dims} elements"
    assert len(config['repulsion']['local_grid_size']) == dims, \
        f"local_grid_size must have {dims} elements"
    assert len(config['repulsion']['global_grid_shape']) == dims, \
        f"global_grid_shape must have {dims} elements"
    
    if dims == 2:
        assert config['node']['elevation_steps'] == 1, \
            "2D mode should have elevation_steps=1"
    
    print(f"✓ Configuration validated for {dims}D mode")
    return True

def print_config_summary(config):
    """Pretty-prints configuration summary."""
    dims = config['spatial']['dimensions']
    num_nodes = config['node']['num_nodes']
    local_grid = config['repulsion']['local_grid_size']
    
    print("=" * 60)
    print(f"FLOWRRA-GNN Configuration Summary")
    print("=" * 60)
    print(f"Dimensions: {dims}D")
    print(f"Nodes: {num_nodes}")
    print(f"Local Grid: {local_grid}")
    print(f"GNN Architecture: {config['gnn']['architecture']}")
    print(f"Hidden Dim: {config['gnn']['hidden_dim']}")
    print(f"Training Steps: {config['training']['total_training_steps']}")
    print("=" * 60)