"""
config.py - DIMENSION-SAFE VERSION

Critical Fix: Ensures repulsion grids match spatial dimensions
"""

CONFIG = {
    # ==================== FEDERATION LAYER ====================
    "federation": {
        "num_holons": 4,
        "world_bounds": (1.0, 1.0),  # 2D only for now
        "breach_threshold": 0.1,
        "coordination_mode": "positional",
        "enable_dynamic_splitting": False,
    },
    # ==================== HOLON CONFIG ====================
    "holon": {
        "mode": "training",
        "independent_training": True,
        "share_experience": False,
        "use_r_gnn": False,
    },
    # ==================== NODE CONFIG ====================
    "node": {
        "total_nodes": 40,
        "num_nodes_per_holon": None,  # Auto-calculated
        "move_speed": 0.015,
        "sensor_range": 0.15,
    },
    # ==================== SPATIAL CONFIG ====================
    "spatial": {
        "dimensions": 2,  # CRITICAL: Must be 2 for federated mode
        "world_bounds": (1.0, 1.0),
    },
    # ==================== GNN CONFIG ====================
    "gnn": {
        "hidden_dim": 128,
        "num_layers": 3,
        "n_heads": 4,
        "lr": 0.0003,
        "gamma": 0.95,
        "dropout": 0.1,
        "buffer_capacity": 15000,
    },
    # ==================== LOOP STRUCTURE ====================
    "loop": {
        "ideal_distance": 0.10,
        "stiffness": 0.5,
        "break_threshold": 0.22,
    },
    # ==================== REWARDS ====================
    "rewards": {
        "r_flow": 1.0,
        "r_collision": 25.0,
        "r_idle": 0.5,
        "r_loop_integrity": 5.0,
        "r_collapse_penalty": 10.0,
        "r_explore": 5.0,
        "r_reconnection": 0.5,
        "r_boundary_breach": 3.0,
    },
    # ==================== REPULSION FIELD ====================
    "repulsion": {
        # CRITICAL FIX: These will be auto-adjusted based on dimensions
        "local_grid_size": (5, 5),  # Start as 2D, validated below
        "global_grid_shape": (60, 60),  # Start as 2D, validated below
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
    "moving_obstacles": [],
    # ==================== TRAINING ====================
    "training": {
        "num_episodes": 50,
        "steps_per_episode": 100,
        "target_update_frequency": 100,
        "save_frequency": 100,
        "metrics_save_frequency": 20,
    },
    # ==================== VISUALIZATION ====================
    "visualization": {
        "show_partitions": True,
        "show_breach_alerts": True,
        "partition_color_scheme": "rainbow",
        "render_frequency": 50,
        "save_history": True,  # NEW: Save for deployment visualization
    },
}


def validate_config(cfg: dict) -> dict:
    """Validate configuration and ensure dimension consistency."""
    import math

    import numpy as np

    # Get dimensions
    dims = cfg["spatial"]["dimensions"]

    # CRITICAL FIX: Ensure repulsion grids match dimensions
    if dims == 2:
        # Force 2D grids
        cfg["repulsion"]["local_grid_size"] = (5, 5)
        cfg["repulsion"]["global_grid_shape"] = (60, 60)
        print(f"[Config] Forced 2D repulsion grids")
    elif dims == 3:
        cfg["repulsion"]["local_grid_size"] = (5, 5, 5)
        cfg["repulsion"]["global_grid_shape"] = (60, 60, 60)
        print(f"[Config] Using 3D repulsion grids")

    # Calculate nodes per holon
    total_nodes = cfg["node"]["total_nodes"]
    num_holons = cfg["federation"]["num_holons"]

    if total_nodes % num_holons != 0:
        adjusted = (total_nodes // num_holons) * num_holons
        print(
            f"[Config] WARNING: Adjusted total_nodes from {total_nodes} to {adjusted}"
        )
        cfg["node"]["total_nodes"] = adjusted

    cfg["node"]["num_nodes_per_holon"] = cfg["node"]["total_nodes"] // num_holons

    # Ensure world_bounds match
    if cfg["spatial"]["world_bounds"] != cfg["federation"]["world_bounds"]:
        print(f"[Config] Syncing spatial.world_bounds to federation.world_bounds")
        cfg["spatial"]["world_bounds"] = cfg["federation"]["world_bounds"]

    # Validate num_holons is perfect square
    sqrt_holons = int(math.sqrt(num_holons))
    if sqrt_holons**2 != num_holons:
        raise ValueError(f"num_holons must be perfect square, got {num_holons}")

    # Validate loop parameters
    nodes_per_holon = cfg["node"]["num_nodes_per_holon"]
    ideal_dist = cfg["loop"]["ideal_distance"]
    equilibrium_radius = (nodes_per_holon * ideal_dist) / (2 * np.pi)

    holon_width = cfg["federation"]["world_bounds"][0] / sqrt_holons
    max_radius = holon_width / 2 * 0.8

    if equilibrium_radius > max_radius:
        adjusted_ideal_dist = (max_radius * 2 * np.pi) / nodes_per_holon
        print(
            f"[Config] WARNING: Adjusted ideal_distance from {ideal_dist:.3f} to {adjusted_ideal_dist:.3f}"
        )
        cfg["loop"]["ideal_distance"] = adjusted_ideal_dist

    print(f"\n[Config] Validated:")
    print(f"  Dimensions: {dims}D")
    print(f"  Repulsion grids: {cfg['repulsion']['local_grid_size']}")
    print(f"  Total nodes: {cfg['node']['total_nodes']}")
    print(f"  Nodes per holon: {cfg['node']['num_nodes_per_holon']}")
    print(f"  Holons: {num_holons} ({sqrt_holons}x{sqrt_holons} grid)")
    print()

    return cfg


# Auto-validate on import
CONFIG = validate_config(CONFIG)
