"""
federation/quadtree.py

Dynamic Quadtree Spatial Partitioner for Holon Management.

Responsibilities:
- Partition 2D space into M cells based on node density
- Assign spatial bounds to each holon
- Support lazy splitting/merging (future enhancement)
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np


@dataclass
class SpatialPartition:
    """
    Represents a spatial region assigned to a holon.
    """
    id: int
    bounds_x: Tuple[float, float]  # (x_min, x_max)
    bounds_y: Tuple[float, float]  # (y_min, y_max)
    center: np.ndarray

    def contains(self, pos: np.ndarray) -> bool:
        """Check if position is within this partition."""
        x, y = pos[0], pos[1]
        return (self.bounds_x[0] <= x < self.bounds_x[1] and
                self.bounds_y[0] <= y < self.bounds_y[1])

    def get_neighbors(self, all_partitions: List['SpatialPartition']) -> List[int]:
        """Get IDs of adjacent partitions."""
        neighbors = []

        for other in all_partitions:
            if other.id == self.id:
                continue

            # Check if shares edge (not just corner)
            shares_x = (abs(self.bounds_x[1] - other.bounds_x[0]) < 1e-6 or
                       abs(self.bounds_x[0] - other.bounds_x[1]) < 1e-6)
            shares_y = (abs(self.bounds_y[1] - other.bounds_y[0]) < 1e-6 or
                       abs(self.bounds_y[0] - other.bounds_y[1]) < 1e-6)

            overlaps_x = not (self.bounds_x[1] <= other.bounds_x[0] or
                            self.bounds_x[0] >= other.bounds_x[1])
            overlaps_y = not (self.bounds_y[1] <= other.bounds_y[0] or
                            self.bounds_y[0] >= other.bounds_y[1])

            if (shares_x and overlaps_y) or (shares_y and overlaps_x):
                neighbors.append(other.id)

        return neighbors

    def distance_to_boundary(self, pos: np.ndarray) -> Tuple[float, str]:
        """
        Calculate distance to nearest boundary and which edge.

        Returns:
            (distance, edge_name) where edge_name is 'north', 'south', 'east', 'west'
        """
        x, y = pos[0], pos[1]

        distances = {
            'west': x - self.bounds_x[0],
            'east': self.bounds_x[1] - x,
            'south': y - self.bounds_y[0],
            'north': self.bounds_y[1] - y
        }

        edge = min(distances, key=distances.get)
        return distances[edge], edge


class QuadtreePartitioner:
    """
    Manages spatial partitioning of the global space.

    Phase 1: Static grid (num_holons must be perfect square)
    Phase 2: Dynamic splitting based on node density
    """

    def __init__(self, num_holons: int, world_bounds: Tuple[float, float]):
        """
        Args:
            num_holons: Number of holons (must be 4, 9, 16, etc.)
            world_bounds: (width, height) of global space
        """
        self.num_holons = num_holons
        self.world_bounds = world_bounds

        # Validate perfect square
        self.grid_size = int(np.sqrt(num_holons))
        assert self.grid_size ** 2 == num_holons, \
            f"num_holons must be perfect square, got {num_holons}"

        self.partitions: List[SpatialPartition] = []
        self._initialize_grid()

    def _initialize_grid(self):
        """Create initial grid partitions."""
        cell_width = self.world_bounds[0] / self.grid_size
        cell_height = self.world_bounds[1] / self.grid_size

        partition_id = 0
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                x_min = col * cell_width
                x_max = (col + 1) * cell_width
                y_min = row * cell_height
                y_max = (row + 1) * cell_height

                center = np.array([(x_min + x_max) / 2, (y_min + y_max) / 2])

                partition = SpatialPartition(
                    id=partition_id,
                    bounds_x=(x_min, x_max),
                    bounds_y=(y_min, y_max),
                    center=center
                )

                self.partitions.append(partition)
                partition_id += 1

        print(f"[Quadtree] Initialized {self.num_holons} partitions in {self.grid_size}x{self.grid_size} grid")

    def get_partition_for_position(self, pos: np.ndarray) -> Optional[SpatialPartition]:
        """Find which partition contains this position."""
        for partition in self.partitions:
            if partition.contains(pos):
                return partition

        # Handle edge case: position exactly on boundary
        # Assign to nearest partition center
        distances = [np.linalg.norm(pos - p.center) for p in self.partitions]
        nearest_idx = np.argmin(distances)
        return self.partitions[nearest_idx]

    def get_partition_by_id(self, partition_id: int) -> Optional[SpatialPartition]:
        """Get partition by ID."""
        for partition in self.partitions:
            if partition.id == partition_id:
                return partition
        return None

    def get_all_partitions(self) -> List[SpatialPartition]:
        """Get all partitions."""
        return self.partitions

    def visualize_grid(self) -> str:
        """Generate ASCII visualization of partition grid."""
        lines = ["\n[Quadtree Grid Layout]"]
        lines.append(f"Grid size: {self.grid_size}x{self.grid_size}")
        lines.append(f"World bounds: {self.world_bounds}")
        lines.append("")

        for row in range(self.grid_size):
            row_partitions = []
            for col in range(self.grid_size):
                partition_id = row * self.grid_size + col
                row_partitions.append(f"[H{partition_id}]")
            lines.append(" ".join(row_partitions))

        return "\n".join(lines)
