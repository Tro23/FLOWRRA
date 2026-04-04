"""
density.py

Continuous Gridless Density Function Estimator for FLOWRRA.
Uses Gaussian Mixture Models (GMM) to evaluate affordance on-the-fly.
Now equipped with a Spatial Hash Grid for O(1) neighboring lookups.
"""

from collections import defaultdict
from typing import Any, Dict, List, Tuple

import numpy as np


class DensityFunctionEstimatorND:
    def __init__(
        self,
        dimensions: int = 3,
        local_grid_size: Tuple[int, ...] = (5, 5, 5),
        local_extent: float = 2.0,
        sigma: float = 0.5,
        beta: float = 0.8,
        tail_length: int = 3,
        tail_decay: float = 0.6,
        hash_cell_size: float = 4.0,  # <--- NEW: Size of the spatial hash cells
    ):
        self.dimensions = dimensions
        self.local_grid_size = local_grid_size
        self.local_extent = local_extent
        self.sigma = sigma
        self.beta = beta
        self.tail_length = tail_length
        self.tail_decay = tail_decay
        self.hash_cell_size = hash_cell_size

        # Retrocausal Memory: List of dicts {'pos': array, 'weight': float}
        self.wfc_memory_splats: List[Dict[str, Any]] = []

        self._local_offsets = self._generate_local_grid_offsets()

    def _generate_local_grid_offsets(self) -> np.ndarray:
        ranges = [
            np.linspace(-self.local_extent / 2, self.local_extent / 2, size)
            for size in self.local_grid_size
        ]

        if self.dimensions == 2:
            X, Y = np.meshgrid(*ranges, indexing="ij")
            offsets = np.stack([X.ravel(), Y.ravel()], axis=-1)
        else:
            X, Y, Z = np.meshgrid(*ranges, indexing="ij")
            offsets = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1)

        return offsets

    # =======================================================
    # SPATIAL HASH LOGIC
    # =======================================================
    def _get_hash_key(self, pos: np.ndarray) -> Tuple[int, ...]:
        """Converts a 3D position into a discrete grid coordinate tuple."""
        return tuple(np.floor(pos / self.hash_cell_size).astype(int))

    def _build_spatial_hash(
        self, sources: List[Dict[str, Any]]
    ) -> Dict[Tuple[int, ...], List[Dict[str, Any]]]:
        """Bins all sources into a dictionary based on their spatial grid location."""
        spatial_hash = defaultdict(list)
        for source in sources:
            key = self._get_hash_key(source["pos"])
            spatial_hash[key].append(source)
        return spatial_hash

    # =======================================================
    # THE MATH (THE READER)
    # =======================================================
    def _evaluate_gaussian(
        self, query_points: np.ndarray, center: np.ndarray, weight: float
    ) -> np.ndarray:
        """Evaluates a Gaussian centered at 'center' for an array of query_points."""
        diff = query_points - center
        dist_sq = np.sum(diff**2, axis=-1)
        return weight * np.exp(-dist_sq / (2 * self.sigma**2))

    def get_affordance_potential_for_node(
        self, node_pos: np.ndarray, repulsion_sources: List[Dict[str, Any]]
    ) -> np.ndarray:
        query_points = node_pos + self._local_offsets
        total_repulsion = np.zeros(query_points.shape[0])

        # 1. Build the spatial hash for this frame
        spatial_hash = self._build_spatial_hash(repulsion_sources)
        node_key = self._get_hash_key(node_pos)

        # 2. Only check the node's cell and immediate neighbors (3x3x3 grid)
        relevant_sources = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    neighbor_key = (
                        node_key[0] + dx,
                        node_key[1] + dy,
                        node_key[2] + dz,
                    )
                    relevant_sources.extend(spatial_hash.get(neighbor_key, []))

        # 3. Add Live Physics Gaussians (Only from relevant sources!)
        for source in relevant_sources:
            src_pos = source["pos"]
            src_vel = source.get("velocity", np.zeros(self.dimensions))

            for k in range(self.tail_length):
                future_pos = src_pos + (src_vel * k * 0.1)
                tail_weight = self.beta * (self.tail_decay**k)
                total_repulsion += self._evaluate_gaussian(
                    query_points, future_pos, tail_weight
                )

        # 4. Add Retrocausal WFC Memory Gaussians (Past Crashes)
        surviving_splats = []
        for splat in self.wfc_memory_splats:
            # Spatial Hash optimization for memories: only evaluate if it's close!
            if np.linalg.norm(splat["pos"] - node_pos) < (self.hash_cell_size * 1.5):
                total_repulsion += self._evaluate_gaussian(
                    query_points, splat["pos"], splat["weight"]
                )

            splat["weight"] *= 0.99
            if splat["weight"] > 0.05:
                surviving_splats.append(splat)
        self.wfc_memory_splats = surviving_splats

        # 5. The Affordance Flip
        normalized_repulsion = np.clip(total_repulsion, 0.0, 1.0)
        affordance = 1.0 - normalized_repulsion

        return affordance.reshape(self.local_grid_size)

    # =======================================================
    # THE MEMORY (THE WRITER)
    # =======================================================
    def splat_collision_event(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        severity: float,
        node_id: int = None,
        is_wfc_event: bool = False,
    ):
        """
        Instead of modifying a grid, we just append a mathematical Gaussian to the universe.
        """
        # Add the main impact site
        self.wfc_memory_splats.append(
            {"pos": position.copy(), "weight": severity * self.beta}
        )

        # If it was a fast crash, splat a Gaussian slightly backward along the trajectory
        vel_mag = np.linalg.norm(velocity)
        if vel_mag > 0.1:
            brake_pos = position - (velocity * 0.5)
            self.wfc_memory_splats.append(
                {"pos": brake_pos, "weight": severity * self.beta * 0.5}
            )

    def get_statistics(self) -> Dict[str, Any]:
        return {
            "active_memory_splats": len(self.wfc_memory_splats),
            "dimensions": self.dimensions,
        }
