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
        # NEW: Strategic Freezing Parameters
        "enable_strategic_freezing": True,  # Enable frozen node feature
        "freeze_at_coverage": 0.55,  # Freeze when coverage hits 85%
        "freeze_edge_nodes": True,  # Prioritize freezing edge nodes
        "min_active_nodes": 4,  # Never freeze below this many active nodes
        "freeze_stability_steps": 50,  # Node must be stable for N steps to freeze
    },
    # ==================== NODE CONFIG ====================
    "node": {
        "total_nodes": 32,
        "num_nodes_per_holon": None,  # Auto-calculated
        "move_speed": 0.010,
        "sensor_range": 0.25,
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
        "lr": 0.0001,
        "gamma": 0.98,
        "dropout": 0.1,
        "buffer_capacity": 15000,
        "stability_coef": 0.55,  # NEW: Weight for stability loss
    },
    # ==================== LOOP STRUCTURE ====================
    "loop": {
        "ideal_distance": 0.12,
        "stiffness": 0.45,
        "break_threshold": 0.32,
    },
    # ==================== PATROL LOGIC (NEW) ====================
    "patrol": {
        "enabled": True,
        "waypoint_threshold": 0.15,  # Distance to consider waypoint reached
        "stick_prob": 0.05,  # Probability to change waypoint randomly
        "bias_force": 0.002,  # Strength of the patrol pull (weak enough to not break loop)
    },
    # ==================== REWARDS ====================
    "rewards": {
        "r_flow": 5.0,
        "r_collision": 30.0,
        "r_idle": 2.0,
        "r_loop_integrity": 10.0,
        "r_collapse_penalty": 25.0,
        "r_explore": 12.0,
        "r_reconnection_spatial": 40.0,  # HIGH reward for spatial (forward) recovery
        "r_reconnection_temporal": 10.0,  # LOW reward for temporal (backward) recovery
        "r_boundary_breach": 10.0,
        "r_frozen_node_bonus": 50.0,  # Big bonus when node successfully freezes
        "r_frozen_utility": 1.0,  # Small reward when active nodes use frozen nodes as landmarks
        "r_reconnection": 5.0,
    },
    # ==================== REPULSION FIELD ====================
    "repulsion": {
        # CRITICAL FIX: These will be auto-adjusted based on dimensions
        "local_grid_size": (5, 5),  # Start as 2D, validated below
        "global_grid_shape": (80, 80),  # Start as 2D, validated below
        "eta": 0.5,
        "gamma_f": 0.9,
        "k_f": 5,
        "sigma_f": 0.05,
        "decay_lambda": 0.9,
        "blur_delta": 0.1,
        "beta": 0.3,
    },
    # ==================== EXPLORATION ====================
    "exploration": {
        "map_resolution": 0.01,
        "sensor_range": 0.20,
    },
    # ==================== WFC RECOVERY ====================
    "wfc": {
        "history_length": 150,
        "tail_length": 8,
        "collapse_threshold": 0.55,
        "tau": 2,
        # NEW: Spatial collapse tuning
        "spatial_search_radius_mult": 1.2,  # Multiplier for local_extent
        "spatial_samples": 32,  # Number of candidate positions
        "spatial_accept_threshold": 0.60,  # Lower = more lenient acceptance
        "spatial_improvement_min": 0.988,  # Minimum improvement ratio
    },
    # ==================== OBSTACLES ====================
    "obstacles": [
        (0.229, 0.257, 0.008),
        (0.329, 0.186, 0.008),
        (0.286, 0.143, 0.008),
        (0.071, 0.429, 0.008),
        (0.029, 0.208, 0.008),
        (0.029, 0.657, 0.008),
        (0.129, 0.786, 0.008),
        (0.486, 0.843, 0.008),
        (0.071, 0.882, 0.008),
        (0.629, 0.257, 0.008),
        (0.829, 0.486, 0.008),
        (0.986, 0.143, 0.008),
        (0.571, 0.429, 0.008),
        (0.729, 0.208, 0.008),
        (0.629, 0.557, 0.008),
        (0.749, 0.786, 0.008),
        (0.986, 0.643, 0.008),
        (0.571, 0.829, 0.008),
        (0.729, 0.908, 0.008),
    ],
    "moving_obstacles": [],
    # ==================== TRAINING ====================
    "training": {
        "num_episodes": 100,
        "steps_per_episode": 500,
        "target_update_frequency": 100,
        "save_frequency": 100,
        "metrics_save_frequency": 20,
    },
    # ==================== VISUALIZATION ====================
    "visualization": {
        "show_partitions": True,
        "show_breach_alerts": True,
        "show_frozen_nodes": True,  # NEW: Highlight frozen nodes
        "frozen_node_color": "gold",  # NEW: Color for frozen nodes
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
    if cfg["holon"]["enable_strategic_freezing"]:
        print(f"  Strategic Freezing: ENABLED")
        print(
            f"    Freeze at coverage: {cfg['holon']['freeze_at_coverage'] * 100:.0f}%"
        )
        print(f"    Min active nodes: {cfg['holon']['min_active_nodes']}")
    print()

    return cfg


# Auto-validate on import
CONFIG = validate_config(CONFIG)
