"""
density.py - DOMAIN-AGNOSTIC VERSION

Clean N-dimensional repulsion/affordance field estimator.

Key Changes:
1. Removed domain-specific assumptions
2. Configurable field semantics (repulsion vs attraction)
3. Pluggable source types (obstacles, goals, other agents)
4. Generic splatting interface
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# =============================================================================
# SOURCE TYPE DEFINITIONS
# =============================================================================

class FieldSource:
    """
    Generic source for density fields.
    
    Can represent:
    - Repulsive sources (obstacles, other agents)
    - Attractive sources (goals, charging stations)
    - Neutral sources (landmarks)
    """
    
    def __init__(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        influence_type: str = 'repulsive',  # 'repulsive', 'attractive', 'neutral'
        influence_strength: float = 1.0,
        influence_radius: float = 0.1,
        metadata: Optional[Dict] = None
    ):
        self.position = position.copy()
        self.velocity = velocity.copy()
        self.influence_type = influence_type
        self.influence_strength = influence_strength
        self.influence_radius = influence_radius
        self.metadata = metadata or {}
    
    @classmethod
    def from_detection(cls, detection: Dict[str, Any], influence_type: str = 'repulsive'):
        """Create source from sensor detection."""
        return cls(
            position=detection['pos'],
            velocity=detection['velocity'],
            influence_type=influence_type,
            influence_strength=1.0,
            metadata={'detection': detection}
        )


# =============================================================================
# DOMAIN-AGNOSTIC DENSITY FIELD
# =============================================================================

class DensityFunctionEstimatorND:
    """
    N-dimensional density field with configurable semantics.
    
    Supports:
    - Repulsion fields (avoid obstacles)
    - Attraction fields (seek goals)
    - Mixed fields (navigate around obstacles toward goals)
    
    No domain assumptions - just potential fields.
    """
    
    def __init__(
        self,
        dimensions: int = 3,
        local_grid_size: Tuple[int, ...] = (5, 5, 5),
        global_grid_shape: Tuple[int, ...] = (60, 60, 60),
        field_mode: str = 'affordance',  # 'affordance', 'repulsion', 'attraction'
        eta: float = 0.5,
        gamma_f: float = 0.9,
        k_f: int = 5,
        sigma_f: float = 0.05,
        decay_lambda: float = 0.01,
        blur_delta: float = 0.2,
        beta: float = 0.7
    ):
        """
        Args:
            dimensions: 2 or 3
            local_grid_size: Local grid per agent
            global_grid_shape: Global visualization grid
            field_mode: 'affordance' (high=good), 'repulsion' (high=bad), 'attraction' (high=good)
            eta: Splatting strength
            gamma_f: Temporal decay for comet-tail
            k_f: Projection steps
            sigma_f: Gaussian kernel width
            decay_lambda: Global field decay
            blur_delta: Diffusion strength
            beta: Field flip strength (for affordance mode)
        """
        self.dimensions = dimensions
        self.local_grid_size = local_grid_size
        self.global_grid_shape = global_grid_shape
        self.field_mode = field_mode
        
        # Validate
        assert dimensions in [2, 3]
        assert len(local_grid_size) == dimensions
        assert len(global_grid_shape) == dimensions
        assert field_mode in ['affordance', 'repulsion', 'attraction']
        
        # Parameters
        self.eta = eta
        self.gamma_f = gamma_f
        self.k_f = k_f
        self.sigma_f = sigma_f
        self.decay_lambda = decay_lambda
        self.blur_delta = blur_delta
        self.beta = beta
        
        # Global grid (for visualization)
        self.field = np.zeros(global_grid_shape)
        
        # Local extent calculation
        self.local_extent = (1.0 / max(global_grid_shape)) * max(local_grid_size)
        
        # Precompute kernel
        self._precompute_kernel()
        
        # Statistics
        self.total_splats = 0
        self.total_collision_splats = 0
        self.total_wfc_splats = 0
    
    def _precompute_kernel(self):
        """Precompute Gaussian kernel for efficiency."""
        ranges = [np.arange(size) - size // 2 for size in self.local_grid_size]
        
        if self.dimensions == 2:
            Y, X = np.meshgrid(ranges[1], ranges[0], indexing='ij')
            distances_sq = X**2 + Y**2
        else:  # 3D
            Z, Y, X = np.meshgrid(ranges[2], ranges[1], ranges[0], indexing='ij')
            distances_sq = X**2 + Y**2 + Z**2
        
        # Gaussian kernel
        self.kernel_template = np.exp(
            -distances_sq / (2 * (self.sigma_f * max(self.local_grid_size)) ** 2)
        )
    
    def reset(self):
        """Reset field to zero."""
        self.field = np.zeros(self.global_grid_shape)
        self.total_splats = 0
        self.total_collision_splats = 0
        self.total_wfc_splats = 0
    
    # =========================================================================
    # CORE: LOCAL AFFORDANCE COMPUTATION
    # =========================================================================
    
    def get_local_field_for_agent(
        self,
        agent_pos: np.ndarray,
        sources: List[FieldSource],
        agent_velocity: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Compute local field for a single agent.
        
        This is the KEY scalability feature - O(detected_sources) not O(all_agents).
        
        Args:
            agent_pos: Agent position
            sources: Detected sources (obstacles, other agents, goals)
            agent_velocity: Agent's current velocity (for frame-of-reference)
        
        Returns:
            Local grid of field values
        """
        local_grid = np.zeros(self.local_grid_size)
        
        for source in sources:
            # Project source forward in time (comet-tail)
            for k in range(self.k_f):
                future_pos = source.position + source.velocity * k
                
                # Relative position in agent's frame
                relative_pos = future_pos - agent_pos
                
                # Toroidal wrapping
                relative_pos = np.mod(relative_pos + 0.5, 1.0) - 0.5
                
                # Map to local grid
                indices = self._world_to_local_grid(relative_pos)
                if indices is None:
                    continue
                
                # Temporal decay
                temporal_weight = self.gamma_f ** k
                
                # Spatial kernel
                spatial_weight = self._get_kernel_weight(indices)
                
                # Influence sign (repulsive vs attractive)
                if source.influence_type == 'repulsive':
                    sign = +1.0
                elif source.influence_type == 'attractive':
                    sign = -1.0
                else:  # neutral
                    sign = 0.0
                
                # Splat
                if self._is_valid_index(indices):
                    influence = (
                        sign *
                        self.eta *
                        source.influence_strength *
                        temporal_weight *
                        spatial_weight
                    )
                    local_grid[tuple(indices)] += influence
        
        # Apply field semantics
        return self._apply_field_semantics(local_grid)
    
    def get_affordance_potential_for_node(
        self,
        node_pos: np.ndarray,
        repulsion_sources: List[Dict[str, Any]]
    ) -> np.ndarray:
        """
        BACKWARD COMPATIBILITY METHOD.
        
        Converts old-style detections to FieldSource objects.
        """
        sources = [
            FieldSource.from_detection(det, influence_type='repulsive')
            for det in repulsion_sources
        ]
        
        return self.get_local_field_for_agent(node_pos, sources)
    
    def _apply_field_semantics(self, raw_field: np.ndarray) -> np.ndarray:
        """
        Apply field mode semantics.
        
        - affordance: High values = good to move there (flip repulsion)
        - repulsion: High values = bad to move there
        - attraction: High values = good to move there (already correct)
        """
        if self.field_mode == 'affordance':
            # Flip: open space = 1.0, obstacles = lower
            normalized = np.clip(raw_field, 0.0, 1.0)
            return 1.0 - (self.beta * normalized)
        
        elif self.field_mode == 'repulsion':
            # Keep as-is: high values = avoid
            return raw_field
        
        elif self.field_mode == 'attraction':
            # Negate: high values = seek
            return -raw_field
        
        return raw_field
    
    # =========================================================================
    # SPLATTING: Learning from Events
    # =========================================================================
    
    def splat_event(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        influence_type: str = 'repulsive',
        severity: float = 1.0,
        metadata: Optional[Dict] = None
    ):
        """
        Generic event splatting.
        
        Use cases:
        - Collision: splat repulsion to avoid in future
        - Goal reached: splat attraction to remember
        - WFC trigger: splat repulsion along failed path
        
        Args:
            position: Where event occurred
            velocity: Direction of event (for comet-tail)
            influence_type: 'repulsive' or 'attractive'
            severity: How strongly to splat (0-1)
            metadata: Event metadata (for tracking)
        """
        self.total_splats += 1
        
        if metadata and metadata.get('is_collision'):
            self.total_collision_splats += 1
        if metadata and metadata.get('is_wfc'):
            self.total_wfc_splats += 1
        
        # Normalize velocity
        vel_mag = np.linalg.norm(velocity)
        if vel_mag < 1e-6:
            velocity = np.zeros_like(position)
        else:
            velocity = velocity / vel_mag
        
        # Project forward along trajectory
        sign = +1.0 if influence_type == 'repulsive' else -1.0
        
        for k in range(self.k_f):
            future_pos = position + velocity * k * 0.02
            future_pos = np.mod(future_pos, 1.0)
            
            # Convert to global grid
            grid_pos = (future_pos * np.array(self.global_grid_shape)).astype(int)
            grid_pos = np.clip(grid_pos, 0, np.array(self.global_grid_shape) - 1)
            
            # Temporal weight
            temporal_weight = self.gamma_f ** k
            
            # Total weight
            weight = sign * self.eta * severity * temporal_weight
            
            # Splat kernel
            self._splat_kernel_at_grid_pos(grid_pos, weight)
    
    def splat_collision_event(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        severity: float,
        node_id: int = None,
        is_wfc_event: bool = False
    ):
        """
        BACKWARD COMPATIBILITY METHOD.
        
        Converts old-style collision splatting to new generic method.
        """
        metadata = {
            'node_id': node_id,
            'is_collision': not is_wfc_event,
            'is_wfc': is_wfc_event
        }
        
        self.splat_event(
            position=position,
            velocity=velocity,
            influence_type='repulsive',
            severity=severity,
            metadata=metadata
        )
    
    def _splat_kernel_at_grid_pos(self, grid_pos: np.ndarray, weight: float):
        """Splat Gaussian kernel at grid position."""
        kernel_half = np.array(self.local_grid_size) // 2
        
        if self.dimensions == 2:
            x_min = max(0, grid_pos[0] - kernel_half[0])
            x_max = min(self.global_grid_shape[0], grid_pos[0] + kernel_half[0] + 1)
            y_min = max(0, grid_pos[1] - kernel_half[1])
            y_max = min(self.global_grid_shape[1], grid_pos[1] + kernel_half[1] + 1)
            
            kx_start = kernel_half[0] - (grid_pos[0] - x_min)
            kx_end = kx_start + (x_max - x_min)
            ky_start = kernel_half[1] - (grid_pos[1] - y_min)
            ky_end = ky_start + (y_max - y_min)
            
            self.field[x_min:x_max, y_min:y_max] += (
                weight * self.kernel_template[kx_start:kx_end, ky_start:ky_end]
            )
        
        else:  # 3D
            x_min = max(0, grid_pos[0] - kernel_half[0])
            x_max = min(self.global_grid_shape[0], grid_pos[0] + kernel_half[0] + 1)
            y_min = max(0, grid_pos[1] - kernel_half[1])
            y_max = min(self.global_grid_shape[1], grid_pos[1] + kernel_half[1] + 1)
            z_min = max(0, grid_pos[2] - kernel_half[2])
            z_max = min(self.global_grid_shape[2], grid_pos[2] + kernel_half[2] + 1)
            
            kx_start = kernel_half[0] - (grid_pos[0] - x_min)
            kx_end = kx_start + (x_max - x_min)
            ky_start = kernel_half[1] - (grid_pos[1] - y_min)
            ky_end = ky_start + (y_max - y_min)
            kz_start = kernel_half[2] - (grid_pos[2] - z_min)
            kz_end = kz_start + (z_max - z_min)
            
            self.field[x_min:x_max, y_min:y_max, z_min:z_max] += (
                weight * self.kernel_template[
                    kx_start:kx_end, ky_start:ky_end, kz_start:kz_end
                ]
            )
    
    # =========================================================================
    # GLOBAL FIELD UPDATES (For visualization)
    # =========================================================================
    
    def update_from_sensor_data(
        self,
        all_nodes: List[Any],
        all_obstacle_states: List[Dict[str, Any]]
    ):
        """
        Update global field from all agents.
        
        This is for visualization - agents use local fields.
        """
        # Decay existing field
        self.field *= (1.0 - self.decay_lambda)
        
        for node in all_nodes:
            # Get detections
            node_detections = node.sense_nodes(all_nodes)
            obstacle_detections = node.sense_obstacles(all_obstacle_states)
            
            # Convert to sources
            sources = []
            for det in node_detections + obstacle_detections:
                sources.append(FieldSource.from_detection(det))
            
            # Compute local field
            local_grid = self.get_local_field_for_agent(node.pos, sources)
            
            # Splat to global
            self._splat_local_to_global(node.pos, local_grid)
    
    def _splat_local_to_global(self, agent_pos: np.ndarray, local_grid: np.ndarray):
        """Splat local grid onto global field."""
        global_center = (agent_pos * np.array(self.global_grid_shape)).astype(int)
        
        local_ranges = [np.arange(size) - size // 2 for size in self.local_grid_size]
        
        if self.dimensions == 2:
            for i, dx in enumerate(local_ranges[0]):
                for j, dy in enumerate(local_ranges[1]):
                    gx = (global_center[0] + dx) % self.global_grid_shape[0]
                    gy = (global_center[1] + dy) % self.global_grid_shape[1]
                    self.field[gx, gy] += local_grid[i, j]
        else:  # 3D
            for i, dx in enumerate(local_ranges[0]):
                for j, dy in enumerate(local_ranges[1]):
                    for k, dz in enumerate(local_ranges[2]):
                        gx = (global_center[0] + dx) % self.global_grid_shape[0]
                        gy = (global_center[1] + dy) % self.global_grid_shape[1]
                        gz = (global_center[2] + dz) % self.global_grid_shape[2]
                        self.field[gx, gy, gz] += local_grid[i, j, k]
    
    # =========================================================================
    # UTILITIES
    # =========================================================================
    
    def _world_to_local_grid(self, relative_pos: np.ndarray) -> Optional[np.ndarray]:
        """Convert relative position to local grid indices."""
        grid_coords = (relative_pos / self.local_extent) * np.array(self.local_grid_size)
        grid_coords += np.array(self.local_grid_size) / 2
        
        indices = np.floor(grid_coords).astype(int)
        
        if np.any(indices < 0) or np.any(indices >= np.array(self.local_grid_size)):
            return None
        
        return indices
    
    def _is_valid_index(self, indices: Optional[np.ndarray]) -> bool:
        """Check if indices are valid."""
        if indices is None:
            return False
        return np.all(indices >= 0) and np.all(indices < np.array(self.local_grid_size))
    
    def _get_kernel_weight(self, indices: np.ndarray) -> float:
        """Get kernel weight at indices."""
        if not self._is_valid_index(indices):
            return 0.0
        return float(self.kernel_template[tuple(indices)])
    
    def get_full_repulsion_grid(self) -> np.ndarray:
        """Get global field for visualization."""
        return self.field
    
    def apply_diffusion(self):
        """Apply Gaussian blur for smoothing."""
        from scipy.ndimage import gaussian_filter
        
        blurred = gaussian_filter(self.field, sigma=1.0)
        self.field = (1 - self.blur_delta) * self.field + self.blur_delta * blurred
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get field statistics."""
        return {
            'total_splats': self.total_splats,
            'total_collision_splats': self.total_collision_splats,
            'total_wfc_splats': self.total_wfc_splats,
            'field_mode': self.field_mode,
            'field_mean': float(np.mean(self.field)),
            'field_max': float(np.max(self.field)),
            'field_nonzero_fraction': float(np.sum(self.field > 0) / self.field.size)
        }
