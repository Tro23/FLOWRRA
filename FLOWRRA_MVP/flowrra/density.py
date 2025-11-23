"""
density.py

N-dimensional repulsion field with local grid computation.
Enhanced with collision event splatting for retrocausal learning.

Key additions:
- splat_collision_event: Adds repulsion at failure sites
- Forward projection along velocity (comet-tail)
- Severity-weighted splatting
"""

from typing import Any, Dict, List, Tuple

import numpy as np


class DensityFunctionEstimatorND:
    """
    Manages N-dimensional repulsive density field with local computation.

    Each node computes its own local grid rather than maintaining
    a global field, making this highly scalable.

    Enhanced with collision learning: when collisions or WFC triggers occur,
    repulsion is splatted to teach avoidance.
    """

    def __init__(
        self,
        dimensions: int = 3,
        local_grid_size: Tuple[int, ...] = (5, 5, 5),
        global_grid_shape: Tuple[int, ...] = (60, 60, 60),
        eta: float = 0.5,
        gamma_f: float = 0.9,
        k_f: int = 5,
        sigma_f: float = 0.05,
        decay_lambda: float = 0.01,
        blur_delta: float = 0.2,
        beta: float = 0.7,
    ):
        """
        Args:
            dimensions: 2 or 3
            local_grid_size: Size of local grid per node (e.g., (5,5,5))
            global_grid_shape: Full world grid for visualization only
            eta: Learning rate for repulsion splatting
            gamma_f: Comet-tail decay factor (forward projection)
            k_f: Number of future projection steps
            sigma_f: Gaussian kernel width
            decay_lambda: Global field decay rate
            blur_delta: Diffusion blend factor
            beta: Repulsion strength multiplier
        """
        self.dimensions = dimensions
        self.local_grid_size = local_grid_size
        self.global_grid_shape = global_grid_shape

        # Validate dimensions
        assert dimensions in [2, 3], "Only 2D and 3D supported"
        assert len(local_grid_size) == dimensions
        assert len(global_grid_shape) == dimensions

        # Repulsion parameters
        self.eta = eta
        self.gamma_f = gamma_f
        self.k_f = k_f
        self.sigma_f = sigma_f
        self.decay_lambda = decay_lambda
        self.blur_delta = blur_delta
        self.beta = beta

        # Global grid (for visualization + collision learning)
        self.repulsion_field = np.zeros(global_grid_shape)

        # Precompute Gaussian kernel for efficiency
        self._precompute_kernel()

        # Track collision splatting events
        self.total_collision_splats = 0
        self.total_wfc_splats = 0

    def check_collision(
        self,
        node_pos: np.ndarray,
        obstacles: List[Tuple[float, float, float]],
        other_nodes: List[Any],
        self_id: int,
        threshold: float = 0.15,
    ) -> bool:
        """
        Checks if the node is colliding with any static obstacle or other node.

        Args:
            node_pos: Position of the node.
            obstacles: List of (x, y, radius) tuples.
            other_nodes: List of Node objects.
            self_id: ID of the node checking (to avoid self-collision).
            threshold: Collision radius for node-node.
        """
        # 1. Check Static Obstacles
        for obs in obstacles:
            ox, oy, rad = obs
            # Simple Euclidean distance (ignoring torus wrap for obstacles for safety)
            dist = np.linalg.norm(node_pos - np.array([ox, oy]))
            if dist < (rad + threshold):  # Hit obstacle radius + node body
                return True

        # 2. Check Peer Collisions
        for other in other_nodes:
            if other.id == self_id:
                continue
            dist = np.linalg.norm(node_pos - other.pos)
            if dist < threshold:
                return True

        return False

    def _precompute_kernel(self):
        """Precomputes Gaussian kernel for splatting."""
        # Create coordinate grids
        ranges = [np.arange(size) - size // 2 for size in self.local_grid_size]

        if self.dimensions == 2:
            Y, X = np.meshgrid(ranges[1], ranges[0], indexing="ij")
            distances_sq = X**2 + Y**2
        else:  # 3D
            Z, Y, X = np.meshgrid(ranges[2], ranges[1], ranges[0], indexing="ij")
            distances_sq = X**2 + Y**2 + Z**2

        # Gaussian falloff
        self.kernel_template = np.exp(
            -distances_sq / (2 * (self.sigma_f * max(self.local_grid_size)) ** 2)
        )

    def reset(self):
        """Resets the global repulsion field."""
        self.repulsion_field = np.zeros(self.global_grid_shape)
        self.total_collision_splats = 0
        self.total_wfc_splats = 0

    def splat_collision_event(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        severity: float,
        node_id: int = None,
        is_wfc_event: bool = False,
    ):
        """
        CRITICAL METHOD: Splat repulsion at collision/failure sites.

        This implements retrocausal learning from the design doc:
        - When collision occurs: splat repulsion to teach "avoid here"
        - When WFC triggers: splat along failed path to teach "avoid this configuration"

        Uses forward projection along velocity (comet-tail) to anticipate movement.

        Args:
            position: Where the failure occurred (normalized [0,1])
            velocity: Direction of failed movement
            severity: How bad the failure was (0-1)
            node_id: Which node failed (for logging)
            is_wfc_event: True if this is a WFC collapse, False if collision
        """
        # Track event type
        if is_wfc_event:
            self.total_wfc_splats += 1
        else:
            self.total_collision_splats += 1

        # Normalize velocity for projection
        vel_mag = np.linalg.norm(velocity)
        if vel_mag < 1e-6:
            # If no velocity, splat in all directions (omnidirectional repulsion)
            velocity = np.zeros_like(position)
        else:
            velocity = velocity / vel_mag  # Unit direction

        # Project forward along velocity (comet-tail effect)
        for k in range(self.k_f):
            # Future position along trajectory
            future_pos = position + velocity * k * 0.02  # Small step size

            # Toroidal wrapping
            future_pos = np.mod(future_pos, 1.0)

            # Convert to global grid coordinates
            grid_pos = (future_pos * np.array(self.global_grid_shape)).astype(int)
            grid_pos = np.clip(grid_pos, 0, np.array(self.global_grid_shape) - 1)

            # Temporal decay: earlier projections stronger
            temporal_weight = self.gamma_f**k

            # Total weight
            weight = self.eta * severity * temporal_weight

            # Splat Gaussian kernel around this point
            self._splat_kernel_at_grid_pos(grid_pos, weight)

    def _splat_kernel_at_grid_pos(self, grid_pos: np.ndarray, weight: float):
        """
        Splat a Gaussian kernel at the given grid position.

        Args:
            grid_pos: Integer grid coordinates
            weight: Amplitude of the splat
        """
        # Kernel size (use local_grid_size as kernel footprint)
        kernel_half = np.array(self.local_grid_size) // 2

        # Bounds for splatting
        if self.dimensions == 2:
            x_min = max(0, grid_pos[0] - kernel_half[0])
            x_max = min(self.global_grid_shape[0], grid_pos[0] + kernel_half[0] + 1)
            y_min = max(0, grid_pos[1] - kernel_half[1])
            y_max = min(self.global_grid_shape[1], grid_pos[1] + kernel_half[1] + 1)

            # Extract kernel region
            kx_start = kernel_half[0] - (grid_pos[0] - x_min)
            kx_end = kx_start + (x_max - x_min)
            ky_start = kernel_half[1] - (grid_pos[1] - y_min)
            ky_end = ky_start + (y_max - y_min)

            # Splat
            self.repulsion_field[x_min:x_max, y_min:y_max] += (
                weight * self.kernel_template[kx_start:kx_end, ky_start:ky_end]
            )

        else:  # 3D
            x_min = max(0, grid_pos[0] - kernel_half[0])
            x_max = min(self.global_grid_shape[0], grid_pos[0] + kernel_half[0] + 1)
            y_min = max(0, grid_pos[1] - kernel_half[1])
            y_max = min(self.global_grid_shape[1], grid_pos[1] + kernel_half[1] + 1)
            z_min = max(0, grid_pos[2] - kernel_half[2])
            z_max = min(self.global_grid_shape[2], grid_pos[2] + kernel_half[2] + 1)

            # Extract kernel region
            kx_start = kernel_half[0] - (grid_pos[0] - x_min)
            kx_end = kx_start + (x_max - x_min)
            ky_start = kernel_half[1] - (grid_pos[1] - y_min)
            ky_end = ky_start + (y_max - y_min)
            kz_start = kernel_half[2] - (grid_pos[2] - z_min)
            kz_end = kz_start + (z_max - z_min)

            # Splat
            self.repulsion_field[x_min:x_max, y_min:y_max, z_min:z_max] += (
                weight
                * self.kernel_template[
                    kx_start:kx_end, ky_start:ky_end, kz_start:kz_end
                ]
            )

    def get_repulsion_potential_for_node(
        self, node_pos: np.ndarray, repulsion_sources: List[Dict[str, Any]]
    ) -> np.ndarray:
        """
        Calculates LOCAL repulsion grid for a single node.

        This is the key scalability feature - each node independently
        computes its local field based only on detected sources.

        Args:
            node_pos: Node's current position (N-dimensional)
            repulsion_sources: List of detections (nodes + obstacles)
                Each dict has: 'pos', 'velocity', 'distance', 'type'

        Returns:
            np.ndarray: Local grid of repulsion values
        """
        local_grid = np.zeros(self.local_grid_size)

        # Compute local grid bounds in world coordinates
        local_extent = 1.0 / max(self.global_grid_shape) * max(self.local_grid_size)

        for source in repulsion_sources:
            source_pos = source["pos"]
            source_vel = source["velocity"]

            # Project forward in time (comet-tail effect)
            for k in range(self.k_f):
                future_pos = source_pos + source_vel * k

                # Convert to local grid coordinates
                relative_pos = future_pos - node_pos

                # Toroidal wrapping
                relative_pos = np.mod(relative_pos + 0.5, 1.0) - 0.5

                # Map to local grid indices
                local_indices = self._world_to_local_grid(relative_pos, local_extent)

                if local_indices is None:
                    continue  # Outside local grid

                # Temporal decay factor
                temporal_weight = self.gamma_f**k

                # Spatial kernel
                spatial_weight = self._get_kernel_weight(local_indices)

                # Splat onto local grid
                if self._is_valid_index(local_indices):
                    local_grid[tuple(local_indices)] += (
                        self.eta * temporal_weight * spatial_weight
                    )

        return self.beta * local_grid

    def _world_to_local_grid(
        self, relative_pos: np.ndarray, local_extent: float
    ) -> np.ndarray:
        """
        Converts relative world position to local grid indices.

        Returns None if outside local grid bounds.
        """
        # Normalize to grid coordinates
        grid_coords = (relative_pos / local_extent) * np.array(self.local_grid_size)

        # Center the grid
        grid_coords += np.array(self.local_grid_size) / 2

        # Convert to integer indices
        indices = np.floor(grid_coords).astype(int)

        # Check bounds
        if np.any(indices < 0) or np.any(indices >= np.array(self.local_grid_size)):
            return None

        return indices

    def _is_valid_index(self, indices: np.ndarray) -> bool:
        """Checks if indices are within local grid bounds."""
        if indices is None:
            return False
        return np.all(indices >= 0) and np.all(indices < np.array(self.local_grid_size))

    def _get_kernel_weight(self, indices: np.ndarray) -> float:
        """Gets Gaussian kernel weight at given indices."""
        if not self._is_valid_index(indices):
            return 0.0
        return self.kernel_template[tuple(indices)]

    def update_from_sensor_data(
        self, all_nodes: List[Any], all_obstacle_states: List[Dict[str, Any]]
    ):
        """
        Updates the GLOBAL repulsion field for visualization.

        This aggregates all local fields onto a single global grid.
        In a fully distributed system, this step would be unnecessary.
        """
        # Decay existing field (gradual forgetting)
        self.repulsion_field *= 1.0 - self.decay_lambda

        for node in all_nodes:
            # Get detections for this node
            all_detections = node.sense_nodes(all_nodes) + node.sense_obstacles(
                all_obstacle_states
            )

            # Compute local field
            local_grid = self.get_repulsion_potential_for_node(
                node_pos=node.pos, repulsion_sources=all_detections
            )

            # Splat local grid onto global grid
            self._splat_local_to_global(node.pos, local_grid)

    def _splat_local_to_global(self, node_pos: np.ndarray, local_grid: np.ndarray):
        """
        Splats a local grid onto the global visualization grid.
        """
        # Convert node position to global grid coordinates
        global_center = (node_pos * np.array(self.global_grid_shape)).astype(int)

        # Local grid offset ranges
        local_ranges = [np.arange(size) - size // 2 for size in self.local_grid_size]

        # Iterate over local grid
        if self.dimensions == 2:
            for i, dx in enumerate(local_ranges[0]):
                for j, dy in enumerate(local_ranges[1]):
                    gx = (global_center[0] + dx) % self.global_grid_shape[0]
                    gy = (global_center[1] + dy) % self.global_grid_shape[1]
                    self.repulsion_field[gx, gy] += local_grid[i, j]

        else:  # 3D
            for i, dx in enumerate(local_ranges[0]):
                for j, dy in enumerate(local_ranges[1]):
                    for k, dz in enumerate(local_ranges[2]):
                        gx = (global_center[0] + dx) % self.global_grid_shape[0]
                        gy = (global_center[1] + dy) % self.global_grid_shape[1]
                        gz = (global_center[2] + dz) % self.global_grid_shape[2]
                        self.repulsion_field[gx, gy, gz] += local_grid[i, j, k]

    def get_full_repulsion_grid(self) -> np.ndarray:
        """Returns the global repulsion field for visualization."""
        return self.repulsion_field

    def apply_diffusion(self):
        """
        Applies diffusion (blurring) to smooth the global field.
        Optional step for visualization quality.
        """
        if self.dimensions == 2:
            from scipy.ndimage import gaussian_filter

            blurred = gaussian_filter(self.repulsion_field, sigma=1.0)
        else:  # 3D
            from scipy.ndimage import gaussian_filter

            blurred = gaussian_filter(self.repulsion_field, sigma=1.0)

        self.repulsion_field = (
            1 - self.blur_delta
        ) * self.repulsion_field + self.blur_delta * blurred

    def get_statistics(self) -> Dict[str, Any]:
        """Get density field statistics."""
        return {
            "total_collision_splats": self.total_collision_splats,
            "total_wfc_splats": self.total_wfc_splats,
            "repulsion_field_mean": float(np.mean(self.repulsion_field)),
            "repulsion_field_max": float(np.max(self.repulsion_field)),
            "repulsion_field_nonzero_fraction": float(
                np.sum(self.repulsion_field > 0) / self.repulsion_field.size
            ),
        }
