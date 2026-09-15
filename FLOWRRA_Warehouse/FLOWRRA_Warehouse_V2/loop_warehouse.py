"""
loop_Warehouse.py

Manages the global MQTT-based loop integrity for discrete warehouse AGV fleets.
Replaces continuous spring physics with a dual-zone collision and predictive deadlock monitor.

CHANGED 2026-09-13 -- check_integrity() no longer measures Manhattan distance on
raw coordinates. See proximity_warehouse.py for the full rationale; the short
version is that two fleets in adjacent aisles separated by a solid rack were
Manhattan distance 2 apart, scored a predictive warning, and dragged integrity
to 0.5 despite being physically unable to reach each other.

WHAT THE OLD BLOCK DID
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if node_a.id in frozen_node_ids or node_b.id in frozen_node_ids: continue
            dist = np.sum(np.abs(node_a.current_pos - node_b.current_pos))
            if   dist <= collision_threshold: -> fatal
            elif dist <= warning_threshold:   -> warning

  * O(N^2) numpy calls per step: ~20,000 pairwise subtractions at 200 fleets,
    essentially all between fleets nowhere near each other.
  * Straight-line distance THROUGH racks (the phantom pair).

WHAT IT DOES NOW
  A GraphProximity index, refreshed once per step by the orchestrator, returns
  only the pairs actually within warning_threshold graph-cells. The
  parked/stopped exemption is applied when the index is built, so it cannot be
  applied at one call site and forgotten at another.

  UNCHANGED: the escalation logic, the integrity values (1.0 / 0.5 / 0.0), the
  collision counter, and every previously existing field of get_statistics().
  Only the distance metric and the iteration order changed.

COMPATIBILITY
  check_integrity() keeps its original signature. With proximity=None it falls
  back to the verbatim O(N^2) Manhattan sweep, so an un-updated caller behaves
  exactly as before and published numbers can be reproduced.
"""

from typing import Any, Dict, List, Optional, Set
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

        # Diagnostic: how many pairs the graph metric rejects that Manhattan
        # would have flagged. This is the phantom-pair rate, measured live
        # instead of inferred from a one-off audit.
        self.phantom_pairs_rejected = 0
        self.pairs_evaluated = 0

    def reset_diagnostics(self):
        """Clear the per-episode phantom-pair counters."""
        self.phantom_pairs_rejected = 0
        self.pairs_evaluated = 0

    def calculate_spring_forces(self, nodes: List[Any]) -> Dict[str, np.ndarray]:
        """
        Warehouse fleets do not use continuous spring forces; they use discrete routed actions.
        Returns an empty dictionary to maintain structural compatibility with the orchestrator.
        """
        return {}

    def check_integrity(
        self,
        nodes: List[Any],
        timestep: int,
        frozen_node_ids: Set[str],
        proximity: Optional[Any] = None,
    ) -> float:
        """
        Evaluates the grid for fatal collisions (<= collision_threshold) and
        predictive warnings (<= warning_threshold).
        Updates current_integrity to 0.0 (Fatal), 0.5 (Warning/WFC Trigger), or 1.0 (Clear).

        Args:
            nodes: List of FleetNode objects (Active, Completed, or Damaged).
            timestep: Current simulation step (for logging).
            frozen_node_ids: Set of IDs for fleets that have parked at their goal.
            proximity: a GraphProximity ALREADY REFRESHED this step with the same
                exclusion set. None falls back to the original Manhattan sweep.

        Returns:
            float: 1.0, 0.5, or 0.0.
        """
        # Clear state for the current step exactly as initialized
        self.deadlocked_nodes.clear()
        self.warning_nodes.clear()

        if proximity is None:
            self._sweep_manhattan(nodes, timestep, frozen_node_ids)
        else:
            self._sweep_graph(proximity, timestep)

        # Integrity Escalation Logic -- UNCHANGED.
        if len(self.deadlocked_nodes) > 0:
            self.current_integrity = 0.0  # Fatal Failure
            self.total_collisions += 1
        elif len(self.warning_nodes) > 0:
            self.current_integrity = 0.5  # Soft Collapse / WFC Trigger
        else:
            self.current_integrity = 1.0  # Clear Flow

        self.integrity_history.append(self.current_integrity)
        return self.current_integrity

    def _sweep_graph(self, proximity: Any, timestep: int):
        """
        Graph-distance sweep. proximity.pairs() already excludes parked and
        stopped fleets on BOTH sides and truncates at the radius, so this only
        classifies.
        """
        for a_id, b_id, dist in proximity.pairs(radius=self.warning_threshold):
            self.pairs_evaluated += 1
            if dist <= self.collision_threshold:
                self.deadlocked_nodes.add(a_id)
                self.deadlocked_nodes.add(b_id)
                print(f"[Loop] CRITICAL: Fatal collision between Fleet {a_id} and Fleet {b_id} at step {timestep}.")
            else:
                self.warning_nodes.add(a_id)
                self.warning_nodes.add(b_id)

    def _sweep_manhattan(self, nodes: List[Any], timestep: int, frozen_node_ids: Set[str]):
        """
        The original O(N^2) coordinate sweep, kept verbatim so proximity=None
        reproduces every published number exactly.
        """
        num_nodes = len(nodes)
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                node_a = nodes[i]
                node_b = nodes[j]

                # IF EITHER FLEET IS PARKED, IT CEASES TO EXIST FOR COLLISIONS
                if node_a.id in frozen_node_ids or node_b.id in frozen_node_ids:
                    continue

                dist = np.sum(np.abs(node_a.current_pos - node_b.current_pos))
                self.pairs_evaluated += 1

                if dist <= self.collision_threshold:
                    self.deadlocked_nodes.add(node_a.id)
                    self.deadlocked_nodes.add(node_b.id)
                    print(f"[Loop] CRITICAL: Fatal collision between Fleet {node_a.id} and Fleet {node_b.id} at step {timestep}.")
                elif self.collision_threshold < dist <= self.warning_threshold:
                    self.warning_nodes.add(node_a.id)
                    self.warning_nodes.add(node_b.id)

    def measure_phantom_pairs(self, nodes: List[Any], frozen_node_ids: Set[str], proximity: Any) -> int:
        """
        How many pairs Manhattan flags this step that the graph metric rejects.
        Logs the phantom rate directly instead of inferring it. O(N^2), so this
        is a diagnostic to run on sampled steps, not every step.
        """
        if proximity is None:
            return 0
        graph_pairs = {
            (a, b) if a < b else (b, a)
            for a, b, _ in proximity.pairs(radius=self.warning_threshold)
        }
        phantom = 0
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                a, b = nodes[i], nodes[j]
                if a.id in frozen_node_ids or b.id in frozen_node_ids:
                    continue
                if float(np.sum(np.abs(a.current_pos - b.current_pos))) <= self.warning_threshold:
                    key = (a.id, b.id) if a.id < b.id else (b.id, a.id)
                    if key not in graph_pairs:
                        phantom += 1
        self.phantom_pairs_rejected += phantom
        return phantom

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
            "warning_fleets": list(self.warning_nodes),
            "phantom_pairs_rejected": self.phantom_pairs_rejected,
            "pairs_evaluated": self.pairs_evaluated,
        }