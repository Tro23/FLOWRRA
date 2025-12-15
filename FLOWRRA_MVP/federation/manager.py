"""
federation/manager.py

The Federation Manager - "Consciousness Above"

Responsibilities:
1. Spatial partitioning (via Quadtree)
2. Breach detection (nodes crossing holon boundaries)
3. Constraint alerts (send to affected holons)
4. Metrics aggregation (for visualization)

Does NOT:
- Control nodes directly
- Apply forces
- Train neural networks
- Make decisions for holons
"""

from typing import Any, Dict, List, Tuple

import numpy as np

from .quadtree import QuadtreePartitioner, SpatialPartition


class BreachAlert:
    """
    Alert sent to holon when nodes violate boundaries.
    Phase 1: Positional (global coordinates)
    """

    def __init__(
        self,
        node_id: int,
        global_pos: np.ndarray,
        assigned_bounds: Dict[str, Tuple[float, float]],
        violation_vector: np.ndarray,
        severity: float,
        boundary_edge: str,
    ):
        self.node_id = node_id
        self.global_pos = global_pos.copy()
        self.assigned_bounds = assigned_bounds
        self.violation_vector = violation_vector.copy()
        self.severity = severity  # 0-1 scale
        self.boundary_edge = boundary_edge  # 'north', 'south', 'east', 'west'

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for holon consumption."""
        return {
            "type": "positional",
            "node_id": self.node_id,
            "global_pos": self.global_pos,
            "holon_bounds": self.assigned_bounds,
            "violation_vector": self.violation_vector,
            "severity": self.severity,
            "boundary_edge": self.boundary_edge,
            "suggested_correction": -self.violation_vector * 1.5,  # Push back
        }


class FederationManager:
    """
    The Watcher - monitors boundaries, sends constraint reminders.
    Maintains minimal global state.
    """

    def __init__(
        self,
        num_holons: int,
        world_bounds: Tuple[float, float] = (1.0, 1.0),
        breach_threshold: float = 0.02,  # How far out of bounds triggers alert
        coordination_mode: str = "positional",
    ):
        """
        Args:
            num_holons: Number of holons (must be perfect square)
            world_bounds: Global space dimensions
            breach_threshold: Distance outside bounds that triggers alert
            coordination_mode: 'positional' or 'topological' (future)
        """
        self.num_holons = num_holons
        self.world_bounds = world_bounds
        self.breach_threshold = breach_threshold
        self.coordination_mode = coordination_mode

        # Spatial partitioner
        self.quadtree = QuadtreePartitioner(num_holons, world_bounds)

        # Metrics tracking
        self.step_count = 0
        self.total_breaches = 0
        self.breach_history: List[Dict[str, Any]] = []

        print(f"\n{'=' * 60}")
        print(f"[Federation] Initialized with {num_holons} holons")
        print(f"[Federation] World bounds: {world_bounds}")
        print(f"[Federation] Breach threshold: {breach_threshold}")
        print(self.quadtree.visualize_grid())
        print(f"{'=' * 60}\n")

    def detect_breaches(
        self, holon_states: Dict[int, Dict[str, Any]]
    ) -> Dict[int, List[BreachAlert]]:
        """
        Detect nodes that have crossed holon boundaries.

        Args:
            holon_states: Dict mapping holon_id -> {
                'nodes': List of node objects,
                'partition_id': int
            }

        Returns:
            Dict mapping holon_id -> List[BreachAlert]
        """
        breach_alerts: Dict[int, List[BreachAlert]] = {
            holon_id: [] for holon_id in holon_states.keys()
        }

        # Check each holon's nodes
        for holon_id, state in holon_states.items():
            nodes = state["nodes"]
            assigned_partition = self.quadtree.get_partition_by_id(
                state["partition_id"]
            )

            if assigned_partition is None:
                continue

            for node in nodes:
                # Check if node is within assigned bounds
                if not assigned_partition.contains(node.pos):
                    # BREACH DETECTED
                    self.total_breaches += 1

                    # Calculate violation details
                    dist_to_boundary, edge = assigned_partition.distance_to_boundary(
                        node.pos
                    )

                    # Violation vector (how far out of bounds)
                    actual_partition = self.quadtree.get_partition_for_position(
                        node.pos
                    )

                    if (
                        actual_partition
                        and actual_partition.id != assigned_partition.id
                    ):
                        # Node is in wrong partition
                        violation_vector = node.pos - assigned_partition.center
                    else:
                        # Node is just outside boundary
                        violation_vector = self._calculate_violation_vector(
                            node.pos, assigned_partition, edge
                        )

                    # Calculate severity (0-1 scale)
                    severity = min(
                        1.0, np.linalg.norm(violation_vector) / self.breach_threshold
                    )

                    # Create alert
                    alert = BreachAlert(
                        node_id=node.id,
                        global_pos=node.pos,
                        assigned_bounds={
                            "x": assigned_partition.bounds_x,
                            "y": assigned_partition.bounds_y,
                        },
                        violation_vector=violation_vector,
                        severity=severity,
                        boundary_edge=edge,
                    )

                    breach_alerts[holon_id].append(alert)

                    # Log breach
                    self.breach_history.append(
                        {
                            "step": self.step_count,
                            "holon_id": holon_id,
                            "node_id": node.id,
                            "pos": node.pos.copy(),
                            "severity": severity,
                            "edge": edge,
                        }
                    )

        # Print breach summary if any occurred
        total_breaches_this_step = sum(len(alerts) for alerts in breach_alerts.values())
        if total_breaches_this_step > 0:
            print(
                f"[Federation] Step {self.step_count}: {total_breaches_this_step} breaches detected"
            )
            for holon_id, alerts in breach_alerts.items():
                if alerts:
                    print(f"  Holon {holon_id}: {len(alerts)} nodes out of bounds")

        return breach_alerts

    def _calculate_violation_vector(
        self, pos: np.ndarray, partition: SpatialPartition, edge: str
    ) -> np.ndarray:
        """Calculate vector from boundary to node position."""
        x, y = pos[0], pos[1]

        if edge == "west":
            return np.array([x - partition.bounds_x[0], 0.0])
        elif edge == "east":
            return np.array([x - partition.bounds_x[1], 0.0])
        elif edge == "south":
            return np.array([0.0, y - partition.bounds_y[0]])
        elif edge == "north":
            return np.array([0.0, y - partition.bounds_y[1]])

        return np.zeros(2)

    def get_partition_assignments(self) -> Dict[int, SpatialPartition]:
        """Get mapping of holon_id -> SpatialPartition."""
        return {p.id: p for p in self.quadtree.get_all_partitions()}

    def step(self, holon_states: Dict[int, Dict[str, Any]]) -> Dict[int, List[Dict]]:
        """
        Execute one federation cycle.

        Args:
            holon_states: State summary from each holon

        Returns:
            Dict mapping holon_id -> List[breach_alert_dicts]
        """
        self.step_count += 1

        # Detect breaches
        breach_alerts = self.detect_breaches(holon_states)

        # Convert to dicts for holon consumption
        breach_dicts = {
            holon_id: [alert.to_dict() for alert in alerts]
            for holon_id, alerts in breach_alerts.items()
        }

        return breach_dicts

    def get_statistics(self) -> Dict[str, Any]:
        """Get federation metrics."""
        return {
            "step": self.step_count,
            "num_holons": self.num_holons,
            "total_breaches": self.total_breaches,
            "breaches_per_step": self.total_breaches / max(1, self.step_count),
            "grid_size": self.quadtree.grid_size,
        }

    def get_breach_heatmap(self) -> np.ndarray:
        """
        Generate heatmap of breach locations.
        Returns 2D array matching quadtree grid.
        """
        heatmap = np.zeros((self.quadtree.grid_size, self.quadtree.grid_size))

        for breach in self.breach_history[-1000:]:  # Last 1000 breaches
            pos = breach["pos"]
            partition = self.quadtree.get_partition_for_position(pos)
            if partition:
                row = partition.id // self.quadtree.grid_size
                col = partition.id % self.quadtree.grid_size
                heatmap[row, col] += 1

        return heatmap
