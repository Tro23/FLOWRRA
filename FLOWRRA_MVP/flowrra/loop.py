"""
loop.py

Manages the loop structure and connectivity between nodes.
Handles spring forces, break detection, and integrity metrics.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class Connection:
    """Represents a connection between two nodes in the loop."""

    node_a_id: int
    node_b_id: int
    ideal_distance: float
    is_broken: bool = False
    break_timestep: Optional[int] = None

    def __hash__(self):
        # Order-independent hash
        return hash(tuple(sorted([self.node_a_id, self.node_b_id])))

    def __eq__(self, other):
        if not isinstance(other, Connection):
            return False
        return sorted([self.node_a_id, self.node_b_id]) == sorted(
            [other.node_a_id, other.node_b_id]
        )


class LoopStructure:
    """
    Manages the loop connectivity and spring forces between nodes.
    """

    def __init__(
        self,
        ideal_distance: float = 0.6,
        stiffness: float = 0.5,
        break_threshold: float = 0.35,
        dimensions: int = 2,
    ):
        self.ideal_distance = ideal_distance
        self.stiffness = stiffness
        self.break_threshold = break_threshold
        self.dimensions = dimensions

        # NEW: The "Warning Zone". If distance passes this, spring becomes explonential.
        self.elastic_limit = self.break_threshold * 0.85

        # Loop topology: list of connections
        self.connections: List[Connection] = []

        # Metrics tracking
        self.integrity_history: List[float] = []
        self.break_count = 0
        self.total_breaks = 0

    def initialize_ring_topology(self, num_nodes: int):
        """
        Initialize a ring topology where each node connects to its neighbors.
        Node i connects to node (i+1) % num_nodes
        """
        self.connections.clear()
        for i in range(num_nodes):
            next_id = (i + 1) % num_nodes
            conn = Connection(
                node_a_id=i, node_b_id=next_id, ideal_distance=self.ideal_distance
            )
            self.connections.append(conn)

        print(
            f"[Loop] Initialized ring topology with {len(self.connections)} connections"
        )

    def get_connection(self, node_a_id: int, node_b_id: int) -> Optional[Connection]:
        """Find a connection between two nodes."""
        for conn in self.connections:
            if (conn.node_a_id == node_a_id and conn.node_b_id == node_b_id) or (
                conn.node_a_id == node_b_id and conn.node_b_id == node_a_id
            ):
                return conn
        return None

    def get_node_neighbors(self, node_id: int) -> List[int]:
        """Get IDs of all nodes connected to this node."""
        neighbors = []
        for conn in self.connections:
            if not conn.is_broken:
                if conn.node_a_id == node_id:
                    neighbors.append(conn.node_b_id)
                elif conn.node_b_id == node_id:
                    neighbors.append(conn.node_a_id)
        return neighbors

    def calculate_spring_forces(self, nodes: List[Any]) -> Dict[int, np.ndarray]:
        """
        Calculate spring forces for all nodes based on connections.

        Returns:
            Dict mapping node_id -> force_vector
        """
        forces = {node.id: np.zeros(self.dimensions) for node in nodes}

        for conn in self.connections:
            if conn.is_broken:
                continue

            # Find the two nodes
            node_a = next((n for n in nodes if n.id == conn.node_a_id), None)
            node_b = next((n for n in nodes if n.id == conn.node_b_id), None)

            if node_a is None or node_b is None:
                continue

            # Toroidal distance
            delta = node_b.pos - node_a.pos
            toroidal_delta = np.mod(delta + 0.5, 1.0) - 0.5
            distance = np.linalg.norm(toroidal_delta)

            if distance < 1e-6:
                continue

            # === NEW PHYSICS LOGIC ===
            force_direction = toroidal_delta / distance
            displacement = distance - self.ideal_distance

            if distance < self.elastic_limit:
                # ZONE 1: Linear Hooke's Law (Normal behavior)
                # Gentle guiding force
                force_magnitude = self.stiffness * displacement
            else:
                # ZONE 2: Non-Linear "Warning" Force
                # As we approach the break threshold, force spikes exponentially.
                # This tells the GNN: "TURN BACK NOW"
                excess = distance - self.elastic_limit
                scale = excess / (self.break_threshold - self.elastic_limit)

                # Base stiffness + Exponential spike
                force_magnitude = (self.stiffness * displacement) * (
                    1.0 + 5.0 * scale**2
                )

            force = force_magnitude * force_direction

            # Apply equal/opposite
            forces[node_a.id] += force
            forces[node_b.id] -= force

        return forces

    def check_breaks(
        self, nodes: List[Any], obstacle_manager, timestep: int
    ) -> List[Connection]:
        """
        Check for broken connections due to distance or obstacle interference.

        Returns:
            List of newly broken connections
        """
        newly_broken = []

        for conn in self.connections:
            if conn.is_broken:
                continue

            # Find nodes
            node_a = next((n for n in nodes if n.id == conn.node_a_id), None)
            node_b = next((n for n in nodes if n.id == conn.node_b_id), None)

            if node_a is None or node_b is None:
                continue

            # Check distance
            delta = node_b.pos - node_a.pos
            toroidal_delta = np.mod(delta + 0.5, 1.0) - 0.5
            distance = np.linalg.norm(toroidal_delta)

            # Break if too far
            if distance > self.break_threshold:
                conn.is_broken = True
                conn.break_timestep = timestep
                newly_broken.append(conn)
                self.break_count += 1
                self.total_breaks += 1
                print(
                    f"[Loop] Connection {conn.node_a_id}-{conn.node_b_id} broken: distance={distance:.3f}"
                )
                continue

            # Check obstacle intersection
            intersects, obs_ids = obstacle_manager.check_line_intersection(
                node_a.pos, node_b.pos
            )
            if intersects:
                conn.is_broken = True
                conn.break_timestep = timestep
                newly_broken.append(conn)
                self.break_count += 1
                self.total_breaks += 1
                print(
                    f"[Loop] Connection {conn.node_a_id}-{conn.node_b_id} broken: obstacle intersection {obs_ids}"
                )

        return newly_broken

    def attempt_reconnection(
        self, nodes: List[Any], obstacle_manager, timestep: int
    ) -> List[Connection]:
        """
        Attempt to reconnect broken connections if conditions are favorable.

        Returns:
            List of reconnected connections
        """
        reconnected = []

        for conn in self.connections:
            if not conn.is_broken:
                continue

            # Find nodes
            node_a = next((n for n in nodes if n.id == conn.node_a_id), None)
            node_b = next((n for n in nodes if n.id == conn.node_b_id), None)

            if node_a is None or node_b is None:
                continue

            # Check distance
            delta = node_b.pos - node_a.pos
            toroidal_delta = np.mod(delta + 0.5, 1.0) - 0.5
            distance = np.linalg.norm(toroidal_delta)

            # Reconnect if close enough and no obstacles
            if distance < self.ideal_distance * 1.2:
                intersects, _ = obstacle_manager.check_line_intersection(
                    node_a.pos, node_b.pos
                )
                if not intersects:
                    conn.is_broken = False
                    conn.break_timestep = None
                    reconnected.append(conn)
                    self.break_count -= 1
                    print(
                        f"[Loop] Connection {conn.node_a_id}-{conn.node_b_id} reconnected"
                    )

        return reconnected

    def calculate_integrity(self) -> float:
        """
        Calculate loop integrity as percentage of intact connections.

        Returns:
            Float between 0.0 (all broken) and 1.0 (all intact)
        """
        if len(self.connections) == 0:
            return 1.0

        intact_count = sum(1 for conn in self.connections if not conn.is_broken)
        integrity = intact_count / len(self.connections)
        self.integrity_history.append(integrity)
        return integrity

    def is_loop_coherent(self, min_integrity: float = 0.8) -> bool:
        """Check if loop is sufficiently intact."""
        return self.calculate_integrity() >= min_integrity

    def get_broken_connections(self) -> List[Connection]:
        """Get all currently broken connections."""
        return [conn for conn in self.connections if conn.is_broken]

    def get_intact_connections(self) -> List[Connection]:
        """Get all currently intact connections."""
        return [conn for conn in self.connections if not conn.is_broken]

    def repair_all_connections(self):
        """Force-repair all broken connections (used after WFC recovery)."""
        for conn in self.connections:
            if conn.is_broken:
                conn.is_broken = False
                conn.break_timestep = None
                self.break_count -= 1
        print(f"[Loop] All connections repaired")

    def get_statistics(self) -> Dict[str, Any]:
        """Get loop structure statistics."""
        intact = len(self.get_intact_connections())
        broken = len(self.get_broken_connections())
        current_integrity = self.calculate_integrity()

        return {
            "total_connections": len(self.connections),
            "intact_connections": intact,
            "broken_connections": broken,
            "current_integrity": current_integrity,
            "total_breaks_occurred": self.total_breaks,
            "avg_integrity": np.mean(self.integrity_history)
            if self.integrity_history
            else 1.0,
            "min_integrity": np.min(self.integrity_history)
            if self.integrity_history
            else 1.0,
        }
