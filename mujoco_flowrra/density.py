"""
density.py

Continuous Gridless Density Function Estimator for FLOWRRA.
Uses Gaussian Mixture Models (GMM) to evaluate affordance on-the-fly.
Infinitely scalable, zero memory overhead for empty space.
"""

from typing import Any, Dict, List, Tuple
import numpy as np

class DensityFunctionEstimatorND:
    def __init__(
        self,
        dimensions: int = 3,
        local_grid_size: Tuple[int, ...] = (5, 5, 5),
        local_extent: float = 2.0, # How many meters the local grid covers
        sigma: float = 0.5,        # Width of the repulsion Gaussian
        beta: float = 0.8,         # Max repulsion strength (0 to 1)
        tail_length: int = 3,      # How many steps ahead to project velocity
        tail_decay: float = 0.6    # How fast the velocity comet-tail fades
    ):
        self.dimensions = dimensions
        self.local_grid_size = local_grid_size
        self.local_extent = local_extent
        self.sigma = sigma
        self.beta = beta
        self.tail_length = tail_length
        self.tail_decay = tail_decay

        # Retrocausal Memory: List of dicts {'pos': array, 'weight': float}
        self.wfc_memory_splats: List[Dict[str, Any]] = []
        
        # Pre-calculate the local grid offsets so we don't do it every frame
        self._local_offsets = self._generate_local_grid_offsets()

    def _generate_local_grid_offsets(self) -> np.ndarray:
        """Creates the relative [X, Y, Z] coordinates for the local sensor grid."""
        ranges = [np.linspace(-self.local_extent/2, self.local_extent/2, size) 
                  for size in self.local_grid_size]
        
        if self.dimensions == 2:
            X, Y = np.meshgrid(*ranges, indexing="ij")
            offsets = np.stack([X.ravel(), Y.ravel()], axis=-1)
        else:
            X, Y, Z = np.meshgrid(*ranges, indexing="ij")
            offsets = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1)
            
        return offsets # Shape: [N_points, dimensions]

    def _evaluate_gaussian(self, query_points: np.ndarray, center: np.ndarray, weight: float) -> np.ndarray:
        """
        Evaluates a Gaussian centered at 'center' for an array of query_points.
        query_points shape: [N_points, dimensions]
        """
        diff = query_points - center
        # Squared Euclidean distance
        dist_sq = np.sum(diff**2, axis=-1)
        # Standard Gaussian formula
        return weight * np.exp(-dist_sq / (2 * self.sigma**2))

    def get_affordance_potential_for_node(self, node_pos: np.ndarray, repulsion_sources: List[Dict[str, Any]]) -> np.ndarray:
        """
        Calculates the GNN's local vision tensor by sampling the continuous GMM math function.
        Returns a flat array of shape [local_grid_size product].
        """
        # 1. Generate the exact world coordinates the drone is looking at
        query_points = node_pos + self._local_offsets 
        
        # 2. Start with an empty repulsion field (0.0 everywhere)
        total_repulsion = np.zeros(query_points.shape[0])

        # 3. Add Live Physics Gaussians (Obstacles & Peers)
        for source in repulsion_sources:
            src_pos = source["pos"]
            src_vel = source.get("velocity", np.zeros(self.dimensions))
            
            # The Comet-Tail: Project Gaussians forward along the velocity vector
            for k in range(self.tail_length):
                future_pos = src_pos + (src_vel * k * 0.1) # 0.1s lookahead per step
                tail_weight = self.beta * (self.tail_decay ** k)
                
                total_repulsion += self._evaluate_gaussian(query_points, future_pos, tail_weight)

        # 4. Add Retrocausal WFC Memory Gaussians (Past Crashes)
        # Decay them slightly over time so the swarm eventually tries exploring there again
        surviving_splats = []
        for splat in self.wfc_memory_splats:
            total_repulsion += self._evaluate_gaussian(query_points, splat["pos"], splat["weight"])
            
            # Decay the memory
            splat["weight"] *= 0.99 
            if splat["weight"] > 0.05: # Forget it if it's too weak
                surviving_splats.append(splat)
        self.wfc_memory_splats = surviving_splats

        # 5. The Affordance Flip!
        # Clip repulsion to 1.0, then subtract from 1.0 to get Affordance
        normalized_repulsion = np.clip(total_repulsion, 0.0, 1.0)
        affordance = 1.0 - normalized_repulsion
        
        # Return reshaped to the exact tensor shape the GNN expects
        return affordance.reshape(self.local_grid_size)

    def splat_collision_event(self, position: np.ndarray, velocity: np.ndarray, severity: float, node_id: int = None, is_wfc_event: bool = False):
        """
        Instead of modifying a grid, we just append a mathematical Gaussian to the universe.
        """
        # Add the main impact site
        self.wfc_memory_splats.append({
            "pos": position.copy(),
            "weight": severity * self.beta
        })
        
        # If it was a fast crash, splat a Gaussian slightly backward along the trajectory 
        # to teach the drones to brake *before* they hit the danger zone
        vel_mag = np.linalg.norm(velocity)
        if vel_mag > 0.1:
            brake_pos = position - (velocity * 0.5)
            self.wfc_memory_splats.append({
                "pos": brake_pos,
                "weight": severity * self.beta * 0.5
            })

    def get_statistics(self) -> Dict[str, Any]:
        return {
            "active_memory_splats": len(self.wfc_memory_splats),
            "dimensions": self.dimensions
        }