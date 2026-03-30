"""
loop.py

Pure Topological Observer for FLOWRRA.
Calculates continuous 3D Euclidean distances to monitor swarm integrity.
No mechanical physics—only math.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import numpy as np

@dataclass
class Connection:
    node_a_id: int
    node_b_id: int
    ideal_distance: float
    is_broken: bool = False

    def __hash__(self):
        return hash(tuple(sorted([self.node_a_id, self.node_b_id])))

    def __eq__(self, other):
        if not isinstance(other, Connection): return False
        return sorted([self.node_a_id, self.node_b_id]) == sorted([other.node_a_id, other.node_b_id])

class LoopStructure:
    def __init__(self, ideal_distance: float = 2.0, break_threshold: float = 4.0, dimensions: int = 3):
        self.ideal_distance = ideal_distance
        self.break_threshold = break_threshold
        self.dimensions = dimensions
        self.connections: List[Connection] = []
        self.integrity_history: List[float] = []
        self.total_breaks = 0

    def initialize_ring_topology(self, num_nodes: int):
        """Creates the logical edges defining the swarm's desired shape."""
        self.connections.clear()
        for i in range(num_nodes):
            next_id = (i + 1) % num_nodes
            self.connections.append(Connection(i, next_id, self.ideal_distance))
        print(f"[Loop] Initialized ring topology with {len(self.connections)} target connections.")

    def check_breaks(self, nodes: List[Any], obstacle_manager: Any) -> List[Connection]:
        """
        Calculates pure 3D Euclidean distance. If the distance exceeds the threshold,
        or an obstacle breaks the line of sight, the edge is severed.
        """
        newly_broken = []

        # Create a quick lookup dictionary for active nodes
        node_dict = {n.id: n for n in nodes}

        for conn in self.connections:
            if conn.is_broken: continue

            if conn.node_a_id not in node_dict or conn.node_b_id not in node_dict:
                continue # Skip if a node is frozen/removed

            pos_a = node_dict[conn.node_a_id].pos
            pos_b = node_dict[conn.node_b_id].pos

            # Pure 3D Euclidean Distance (No Torus Modulo!)
            distance = np.linalg.norm(pos_b - pos_a)

            # 1. Check Distance Threshold
            if distance > self.break_threshold:
                conn.is_broken = True
                newly_broken.append(conn)
                self.total_breaks += 1
                continue

            # 2. Check Line of Sight (Obstacles)
            # Assuming obstacle_manager has a check_line_intersection method
            if obstacle_manager is not None:
                intersects, _ = obstacle_manager.check_line_intersection(pos_a, pos_b)
                if intersects:
                    conn.is_broken = True
                    newly_broken.append(conn)
                    self.total_breaks += 1

        return newly_broken

    def attempt_reconnection(self, nodes: List[Any], obstacle_manager: Any) -> List[Connection]:
        """If broken nodes drift back into the ideal range safely, heal the connection."""
        reconnected = []
        node_dict = {n.id: n for n in nodes}

        for conn in self.connections:
            if not conn.is_broken: continue

            if conn.node_a_id not in node_dict or conn.node_b_id not in node_dict:
                continue

            pos_a = node_dict[conn.node_a_id].pos
            pos_b = node_dict[conn.node_b_id].pos
            distance = np.linalg.norm(pos_b - pos_a)

            # Heal if they get close enough and have line of sight
            if distance < (self.ideal_distance * 1.5):
                intersects = False
                if obstacle_manager is not None:
                    intersects, _ = obstacle_manager.check_line_intersection(pos_a, pos_b)
                
                if not intersects:
                    conn.is_broken = False
                    reconnected.append(conn)

        return reconnected

    def calculate_integrity(self) -> float:
        """Returns % of intact connections (1.0 = perfect, 0.0 = total collapse)."""
        if not self.connections: return 1.0
        intact_count = sum(1 for conn in self.connections if not conn.is_broken)
        integrity = intact_count / len(self.connections)
        self.integrity_history.append(integrity)
        return integrity

    def is_loop_coherent(self, min_integrity: float = 0.8) -> bool:
        return self.calculate_integrity() >= min_integrity

    def repair_all_connections(self):
        """God-mode repair used by the Wave Function Collapse."""
        for conn in self.connections:
            conn.is_broken = False
        print(f"[Loop] Topology forcefully repaired by WFC.")
        
    def get_statistics(self) -> Dict[str, Any]:
        return {
            "total_connections": len(self.connections),
            "current_integrity": self.calculate_integrity(),
            "total_breaks_occurred": self.total_breaks,
        }