"""
config_federation.py

Configuration for Federated FLOWRRA System.

Adds federation-specific parameters while maintaining
compatibility with existing holon/node configs.
"""

CONFIG = {
    # ==================== FEDERATION LAYER ====================
    "federation": {
        "num_holons": 4,  # Must be perfect square (4, 9, 16, etc.)
        "world_bounds": (1.0, 1.0),  # Global space size
        "breach_threshold": 0.02,  # Distance that triggers breach alert
        "coordination_mode": "positional",  # 'positional' or 'topological' (future)
        "enable_dynamic_splitting": False,  # Phase 2 feature
    },
    # ==================== HOLON CONFIG ====================
    "holon": {
        "mode": "training",  # 'training' or 'deployment'
        "independent_training": True,  # Each holon trains separately
        "share_experience": False,  # Phase 2: federated learning
    },
    # ==================== NODE CONFIG ====================
    "node": {
        "total_nodes": 40,  # Total across all holons
        "num_nodes_per_holon": None,  # Auto-calculated: total_nodes / num_holons
        "move_speed": 0.015,
        "sensor_range": 0.15,
    },
    # ==================== SPATIAL CONFIG ====================
    "spatial": {
        "dimensions": 2,  # 2D or 3D
        "world_bounds": (1.0, 1.0),  # Should match federation.world_bounds
    },
    # ==================== GNN CONFIG ====================
    "gnn": {
        "hidden_dim": 128,
        "num_layers": 3,
        "n_heads": 4,
        "lr": 0.0003,
        "gamma": 0.95,
        "dropout": 0.1,
    },
    # ==================== LOOP STRUCTURE ====================
    "loop": {
        "ideal_distance": 0.1,  # Smaller for denser packing
        "stiffness": 0.5,
        "break_threshold": 0.35,
    },
    # ==================== REWARDS ====================
    "rewards": {
        "r_flow": 1.0,
        "r_collision": 25.0,
        "r_idle": 0.5,
        "r_loop_integrity": 5.0,
        "r_collapse_penalty": 10.0,
        "r_explore": 15.0,
        "r_reconnection": 0.5,
        # NEW: Boundary breach penalty
        "r_boundary_breach": 3.0,  # Per breach
    },
    # ==================== REPULSION FIELD ====================
    "repulsion": {
        "local_grid_size": (5, 5, 5),
        "global_grid_shape": (60, 60, 60),
        "eta": 0.5,
        "gamma_f": 0.9,
        "k_f": 5,
        "sigma_f": 0.05,
        "decay_lambda": 0.01,
        "blur_delta": 0.2,
        "beta": 0.7,
    },
    # ==================== EXPLORATION ====================
    "exploration": {
        "map_resolution": 0.02,
        "sensor_range": 0.15,
    },
    # ==================== WFC RECOVERY ====================
    "wfc": {
        "history_length": 200,
        "tail_length": 15,
        "collapse_threshold": 0.6,
        "tau": 3,
    },
    # ==================== OBSTACLES ====================
    "obstacles": [
        # (x, y, radius) - Will be scaled by world_bounds
        # Example: Central obstacle
        # (0.5, 0.5, 0.1),
    ],
    "moving_obstacles": [
        # (x, y, radius, vx, vy)
        # Example: Horizontal mover
        # (0.3, 0.5, 0.08, 0.001, 0.0),
    ],
    # ==================== TRAINING ====================
    "training": {
        "num_episodes": 2000,  # Reduced for faster federated training
        "steps_per_episode": 400,
        "target_update_frequency": 100,
        "save_frequency": 200,
        "metrics_save_frequency": 50,
    },
    # ==================== VISUALIZATION ====================
    "visualization": {
        "show_partitions": True,  # Draw holon boundaries
        "show_breach_alerts": True,  # Highlight breached nodes
        "partition_color_scheme": "rainbow",  # Color holons differently
    },
}


def validate_config(cfg: dict):
    """Validate configuration and auto-calculate derived values."""

    # Calculate nodes per holon
    total_nodes = cfg["node"]["total_nodes"]
    num_holons = cfg["federation"]["num_holons"]

    if total_nodes % num_holons != 0:
        # Adjust total nodes to be evenly divisible
        adjusted = (total_nodes // num_holons) * num_holons
        print(
            f"[Config] WARNING: Adjusted total_nodes from {total_nodes} to {adjusted} (evenly divisible by {num_holons})"
        )
        cfg["node"]["total_nodes"] = adjusted

    cfg["node"]["num_nodes_per_holon"] = cfg["node"]["total_nodes"] // num_holons

    # Ensure world_bounds match
    if cfg["spatial"]["world_bounds"] != cfg["federation"]["world_bounds"]:
        print(
            f"[Config] WARNING: Syncing spatial.world_bounds to federation.world_bounds"
        )
        cfg["spatial"]["world_bounds"] = cfg["federation"]["world_bounds"]

    # Validate num_holons is perfect square
    import math

    sqrt_holons = int(math.sqrt(num_holons))
    if sqrt_holons**2 != num_holons:
        raise ValueError(f"num_holons must be perfect square, got {num_holons}")

    print(f"\n[Config] Validated:")
    print(f"  Total nodes: {cfg['node']['total_nodes']}")
    print(f"  Nodes per holon: {cfg['node']['num_nodes_per_holon']}")
    print(f"  Holons: {num_holons} ({sqrt_holons}x{sqrt_holons} grid)")
    print()

    return cfg


# Auto-validate on import
CONFIG = validate_config(CONFIG)
