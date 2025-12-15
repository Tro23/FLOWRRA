"""
config.py - COMPLETE VERSION

Configuration for Federated FLOWRRA System with Phase 2 support.

Supports both:
- Phase 1: Standard GNN (agent.py)
- Phase 2: Recurrent GNN (r_gnn_agent.py)
"""

# Import numpy for validation
import numpy as np

CONFIG = {
    # ==================== FEDERATION LAYER ====================
    "federation": {
        "num_holons": 4,  # Must be perfect square (4, 9, 16, etc.)
        "world_bounds": (1.0, 1.0),  # Global space size
        "breach_threshold": 0.03,  # Distance that triggers breach alert (increased for stability)
        "coordination_mode": "positional",  # 'positional' or 'topological' (future)
        "enable_dynamic_splitting": False,  # Phase 2 feature
    },
    # ==================== HOLON CONFIG ====================
    "holon": {
        "mode": "training",  # 'training' or 'deployment'
        "independent_training": True,  # Each holon trains separately
        "share_experience": False,  # Phase 3: federated learning
        "use_r_gnn": False,  # Set True to enable Phase 2 R-GNN
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
    # ==================== GNN CONFIG (Phase 1) ====================
    "gnn": {
        "hidden_dim": 128,
        "num_layers": 3,
        "n_heads": 4,
        "lr": 0.0003,
        "gamma": 0.95,
        "dropout": 0.1,
        "buffer_capacity": 15000,
    },
    # ==================== R-GNN CONFIG (Phase 2) ====================
    "r_gnn": {
        "hidden_dim": 128,
        "num_gat_layers": 3,
        "n_heads": 4,
        "lstm_layers": 1,  # Number of LSTM layers
        "sequence_length": 4,  # Temporal sequence length
        "lr": 0.0003,
        "gamma": 0.95,
        "dropout": 0.1,
        "buffer_capacity": 15000,
    },
    # ==================== LOOP STRUCTURE ====================
    "loop": {
        "ideal_distance": 0.12,  # Adjusted for federated spacing
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
        "local_grid_size": (5, 5),  # 2D only for now
        "global_grid_shape": (60, 60),  # 2D only
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
        # (x, y, radius) - Scaled by world_bounds
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
        "num_episodes": 1000,  # Reduced for faster federated training
        "steps_per_episode": 400,
        "target_update_frequency": 100,
        "save_frequency": 100,
        "metrics_save_frequency": 20,
    },
    # ==================== VISUALIZATION ====================
    "visualization": {
        "show_partitions": True,  # Draw holon boundaries
        "show_breach_alerts": True,  # Highlight breached nodes
        "partition_color_scheme": "rainbow",  # Color holons differently
        "render_frequency": 50,  # Render every N steps (0 = disabled)
    },
}


def validate_config(cfg: dict) -> dict:
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

    # Validate repulsion grid dimensions match spatial dimensions
    dims = cfg["spatial"]["dimensions"]
    if dims == 2:
        # Ensure 2D grids
        if len(cfg["repulsion"]["local_grid_size"]) != 2:
            cfg["repulsion"]["local_grid_size"] = cfg["repulsion"]["local_grid_size"][
                :2
            ]
        if len(cfg["repulsion"]["global_grid_shape"]) != 2:
            cfg["repulsion"]["global_grid_shape"] = cfg["repulsion"][
                "global_grid_shape"
            ][:2]

    # Validate loop parameters for federated setup
    nodes_per_holon = cfg["node"]["num_nodes_per_holon"]
    ideal_dist = cfg["loop"]["ideal_distance"]

    # Calculate expected equilibrium radius
    equilibrium_radius = (nodes_per_holon * ideal_dist) / (2 * np.pi)

    # Check if it fits in holon bounds
    holon_width = cfg["federation"]["world_bounds"][0] / sqrt_holons
    max_radius = holon_width / 2 * 0.8  # 80% of holon half-width

    if equilibrium_radius > max_radius:
        # Adjust ideal_distance to fit
        adjusted_ideal_dist = (max_radius * 2 * np.pi) / nodes_per_holon
        print(
            f"[Config] WARNING: ideal_distance {ideal_dist:.3f} too large for holon size"
        )
        print(f"[Config] Adjusted to {adjusted_ideal_dist:.3f} to fit equilibrium ring")
        cfg["loop"]["ideal_distance"] = adjusted_ideal_dist

    print(f"\n[Config] Validated:")
    print(f"  Total nodes: {cfg['node']['total_nodes']}")
    print(f"  Nodes per holon: {cfg['node']['num_nodes_per_holon']}")
    print(f"  Holons: {num_holons} ({sqrt_holons}x{sqrt_holons} grid)")
    print(
        f"  Agent type: {'R-GNN (Phase 2)' if cfg['holon']['use_r_gnn'] else 'GNN (Phase 1)'}"
    )
    print()

    return cfg


# Auto-validate on import
CONFIG = validate_config(CONFIG)
