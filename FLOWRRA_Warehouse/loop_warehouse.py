"""
loop_Warehouse.py

Manages the global MQTT-based loop integrity for discrete warehouse AGV fleets.
Replaces continuous spring physics with a dual-zone collision and predictive deadlock monitor.
"""

from typing import Any, Dict, List, Set
import numpy as np


class WarehouseLoop:
    """
    Manages the holistic integrity of the warehouse fleet holon.
    Integrity scales based on clear flow (1.0), safety bubble overlap (0.5), 
    and fatal gridlock (0.0).
    """

    def __init__(self, collision_threshold: float = 1.0, warning_threshold: float = 2.0):
        # In a discrete Manhattan grid, <= collision_threshold is a crash.
        # > collision_threshold and <= warning_threshold is a predictive warning.
        self.collision_threshold = collision_threshold
        self.warning_threshold = warning_threshold
        
        # Metrics tracking
        self.integrity_history: List[float] = []
        self.total_collisions = 0
        self.current_integrity = 1.0
        
        # Track nodes currently in a deadlock/collision or warning state
        self.deadlocked_nodes: Set[str] = set()
        self.warning_nodes: Set[str] = set()

    def calculate_spring_forces(self, nodes: List[Any]) -> Dict[str, np.ndarray]:
        """
        Warehouse fleets do not use continuous spring forces; they use discrete routed actions.
        Returns an empty dictionary to maintain structural compatibility with the orchestrator.
        """
        return {}

    def check_integrity(self, nodes: List[Any], timestep: int, frozen_node_ids: Set[str]) -> float:
        """
        Evaluates the grid for fatal collisions (<=1 edge) and predictive warnings (<=2 edges).
        Updates current_integrity to 0.0 (Fatal), 0.5 (Warning/WFC Trigger), or 1.0 (Clear).
        
        Args:
            nodes: List of FleetNode objects (Active, Completed, or Damaged).
            timestep: Current simulation step (for logging).
            frozen_node_ids: Set of IDs for fleets that have parked at their goal.
            
        Returns:
            float: 1.0, 0.5, or 0.0.
        """
        # Clear state for the current step exactly as initialized
        self.deadlocked_nodes.clear()
        self.warning_nodes.clear()
        
        num_nodes = len(nodes)
        
        # Check pairwise Manhattan distances across all nodes in the holon
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                node_a = nodes[i]
                node_b = nodes[j]
                
                # --- FIX: IF EITHER FLEET IS PARKED, IT CEASES TO EXIST FOR COLLISIONS ---
                if node_a.id in frozen_node_ids or node_b.id in frozen_node_ids:
                    continue
                # -------------------------------------------------------------------------
                
                dist = np.sum(np.abs(node_a.current_pos - node_b.current_pos))
                
                # Zone 1: Fatal Crash / Deadlock
                if dist <= self.collision_threshold:
                    self.deadlocked_nodes.add(node_a.id)
                    self.deadlocked_nodes.add(node_b.id)
                    print(f"[Loop] CRITICAL: Fatal collision between Fleet {node_a.id} and Fleet {node_b.id} at step {timestep}.")
                
                # Zone 2: Predictive Collapse (Safety Bubble Overlap)
                elif self.collision_threshold < dist <= self.warning_threshold:
                    self.warning_nodes.add(node_a.id)
                    self.warning_nodes.add(node_b.id)
                    
        # Integrity Escalation Logic
        if len(self.deadlocked_nodes) > 0:
            self.current_integrity = 0.0  # Fatal Failure
            self.total_collisions += 1
        elif len(self.warning_nodes) > 0:
            self.current_integrity = 0.5  # Soft Collapse / WFC Trigger
        else:
            self.current_integrity = 1.0  # Clear Flow
            
        self.integrity_history.append(self.current_integrity)
        return self.current_integrity

    def calculate_integrity(self) -> float:
        """Returns the last calculated integrity."""
        return self.current_integrity

    def is_loop_coherent(self, min_integrity: float = 0.5) -> bool:
        """
        Check if the warehouse holon is free of fatal deadlocks.
        """
        return self.current_integrity >= min_integrity
        
    def force_repair(self):
        """
        Forces the loop back to a coherent state.
        Called by the orchestrator after a Wave Function Collapse (WFC) recovery event.
        """
        self.current_integrity = 1.0
        self.deadlocked_nodes.clear()
        self.warning_nodes.clear()
        print("[Loop] Integrity manually restored via Spatial-Temporal Recovery.")

    def get_statistics(self) -> Dict[str, Any]:
        """Get loop structure statistics for metrics tracking."""
        return {
            "current_integrity": self.current_integrity,
            "total_collisions_occurred": self.total_collisions,
            "avg_integrity": float(np.mean(self.integrity_history)) if self.integrity_history else 1.0,
            "deadlocked_fleets": list(self.deadlocked_nodes),
            "warning_fleets": list(self.warning_nodes)
        }