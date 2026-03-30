"""
recovery.py

Wave Function Collapse using Relative Shape Memory.
Saves the structural manifold of the swarm (ignoring absolute world coordinates)
to allow smooth, relative rewinds when integrity shatters.
"""

from typing import Any, Dict, List

import numpy as np


class Wave_Function_Collapse:
    def __init__(
        self, history_length: int = 200, collapse_threshold: float = 0.6, tau: int = 3
    ):
        self.history_length = history_length
        self.collapse_threshold = collapse_threshold
        self.tau = tau  # Consecutive unstable frames before collapse triggers

        # History stores the RELATIVE shape of the swarm
        self.history: List[Dict[str, Any]] = []

        self.total_collapses = 0
        self.successful_recoveries = 0

    def assess_loop_coherence(
        self, coherence: float, nodes: List[Any], loop_integrity: float
    ):
        """Records the relative shape of the swarm at this timestep."""

        # 1. Find the Swarm Centroid (Center of Mass)
        positions = np.array([n.pos for n in nodes])
        centroid = np.mean(positions, axis=0)

        # 2. Store RELATIVE positions
        snapshot = {
            "relative_positions": [n.pos - centroid for n in nodes],
            "velocities": [
                n.velocity() if hasattr(n, "velocity") else np.zeros(3) for n in nodes
            ],
            "coherence": coherence,
            "loop_integrity": loop_integrity,
        }

        self.history.append(snapshot)
        if len(self.history) > self.history_length:
            self.history.pop(0)

    def needs_recovery(self) -> bool:
        """Triggers if coherence stays below threshold for 'tau' steps, or loop critically shatters."""
        if len(self.history) < self.tau:
            return False

        recent = self.history[-self.tau :]
        is_unstable = all(h["coherence"] < self.collapse_threshold for h in recent)
        critically_broken = any(h["loop_integrity"] < 0.5 for h in recent)

        return is_unstable or critically_broken

    def collapse_and_reinitialize(self, nodes: List[Any]) -> Dict[str, Any]:
        """The Main Recovery Sequence: Snaps the swarm back into a stable relative shape."""
        self.total_collapses += 1
        print("[WFC] System shattered. Triggering Relative Shape Collapse...")

        best_shape = None
        best_score = -1.0

        # Look backwards through history for a safe relative shape
        for memory in reversed(self.history):
            if memory["coherence"] > best_score:
                best_score = memory["coherence"]
                best_shape = memory
            if best_score > 0.9:
                break

        # THE FIX: Local Hard Respawn
        if best_shape is None or best_score < self.collapse_threshold:
            print("[WFC] CRITICAL: No safe history. Rebuilding local topology.")

            # Find exactly where the tangled mess is right now
            current_positions = np.array([n.pos for n in nodes])
            current_centroid = np.mean(current_positions, axis=0)

            # Nudge the center up slightly so we don't spawn them inside the floor
            safe_z = max(current_centroid[2], 1.5)
            local_safe_center = np.array(
                [current_centroid[0], current_centroid[1], safe_z]
            )

            num_nodes = len(nodes)
            ideal_dist = 1.5
            radius = (ideal_dist * num_nodes) / (2 * np.pi)

            for i, node in enumerate(nodes):
                angle = i * (2 * np.pi / num_nodes)
                # Rebuild the perfect circle around their CURRENT location
                node.pos = local_safe_center + np.array(
                    [np.cos(angle) * radius, np.sin(angle) * radius, 0.0]
                )

                # Zero out velocity to stop momentum
                if hasattr(node, "velocity"):
                    node.velocity = np.zeros(3)

            self.history.clear()  # Wipe corrupt history
            return {"reinit_from": "local_hard_respawn", "success": True}

        # --- Normal Relative Rewind (If we have a good memory) ---
        current_positions = np.array([n.pos for n in nodes])
        current_centroid = np.mean(current_positions, axis=0)

        avg_velocity = np.mean(best_shape["velocities"], axis=0)
        safe_centroid = current_centroid - (avg_velocity * 0.5)

        for i, node in enumerate(nodes):
            node.pos = safe_centroid + best_shape["relative_positions"][i]
            if hasattr(node, "velocity"):
                node.velocity = np.zeros(3)

        self.successful_recoveries += 1
        self.history.clear()  # Wipe history to prevent instant re-trigger
        return {"reinit_from": "relative_shape_rewind", "success": True}
