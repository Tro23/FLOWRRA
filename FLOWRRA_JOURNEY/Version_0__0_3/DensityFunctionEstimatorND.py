"""
DensityFunctionEstimatorND.py

N-dimensional repulsion field with local grid computation.
Supports both 2D and 3D with configurable local grid sizes.
"""
import numpy as np
from typing import List, Dict, Any, Tuple

class DensityFunctionEstimatorND:
    """
    Manages N-dimensional repulsive density field with local computation.
    
    Each node computes its own local grid rather than maintaining
    a global field, making this highly scalable.
    """
    
    def __init__(self,
                 dimensions: int = 3,
                 local_grid_size: Tuple[int, ...] = (5, 5, 5),
                 global_grid_shape: Tuple[int, ...] = (60, 60, 60),
                 eta: float = 0.5,
                 gamma_f: float = 0.9,
                 k_f: int = 5,
                 sigma_f: float = 0.05,
                 decay_lambda: float = 0.01,
                 blur_delta: float = 0.2,
                 beta: float = 0.7):
        """
        Args:
            dimensions: 2 or 3
            local_grid_size: Size of local grid per node (e.g., (5,5,5))
            global_grid_shape: Full world grid for visualization only
            eta: Learning rate for repulsion splatting
            gamma_f: Comet-tail decay factor
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
        
        # Global grid (for visualization only)
        self.repulsion_field = np.zeros(global_grid_shape)
        
        # Precompute Gaussian kernel for efficiency
        self._precompute_kernel()
    
    def _precompute_kernel(self):
        """Precomputes Gaussian kernel for splatting."""
        # Create coordinate grids
        ranges = [np.arange(size) - size // 2 for size in self.local_grid_size]
        
        if self.dimensions == 2:
            Y, X = np.meshgrid(ranges[1], ranges[0], indexing='ij')
            distances_sq = X**2 + Y**2
        else:  # 3D
            Z, Y, X = np.meshgrid(ranges[2], ranges[1], ranges[0], indexing='ij')
            distances_sq = X**2 + Y**2 + Z**2
        
        # Gaussian falloff
        self.kernel_template = np.exp(-distances_sq / (2 * (self.sigma_f * max(self.local_grid_size))**2))
    
    def reset(self):
        """Resets the global repulsion field."""
        self.repulsion_field = np.zeros(self.global_grid_shape)
    
    def get_repulsion_potential_for_node(self,
                                        node_pos: np.ndarray,
                                        repulsion_sources: List[Dict[str, Any]]) -> np.ndarray:
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
            source_pos = source['pos']
            source_vel = source['velocity']
            
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
                temporal_weight = self.gamma_f ** k
                
                # Spatial kernel
                spatial_weight = self._get_kernel_weight(local_indices)
                
                # Splat onto local grid
                if self._is_valid_index(local_indices):
                    local_grid[tuple(local_indices)] += \
                        self.eta * temporal_weight * spatial_weight
        
        return self.beta * local_grid
    
    def _world_to_local_grid(self, relative_pos: np.ndarray, 
                            local_extent: float) -> np.ndarray:
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
    
    def update_from_sensor_data(self,
                               all_nodes: List[Any],
                               all_obstacle_states: List[Dict[str, Any]]):
        """
        Updates the GLOBAL repulsion field for visualization.
        
        This aggregates all local fields onto a single global grid.
        In a fully distributed system, this step would be unnecessary.
        """
        # Decay existing field
        self.repulsion_field *= (1.0 - self.decay_lambda)
        
        for node in all_nodes:
            # Get detections for this node
            all_detections = node.sense_nodes(all_nodes) + \
                           node.sense_obstacles(all_obstacle_states)
            
            # Compute local field
            local_grid = self.get_repulsion_potential_for_node(
                node_pos=node.pos,
                repulsion_sources=all_detections
            )
            
            # Splat local grid onto global grid
            self._splat_local_to_global(node.pos, local_grid)
    
    def _splat_local_to_global(self, node_pos: np.ndarray, 
                               local_grid: np.ndarray):
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
        
        self.repulsion_field = (1 - self.blur_delta) * self.repulsion_field + \
                              self.blur_delta * blurred