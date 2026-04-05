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
        self,
        history_length: int = 150,
        collapse_threshold: float = 0.5,
        tau: int = 3,
    ):
        self.history_length = history_length
        self.collapse_threshold = collapse_threshold
        self.tau = tau

        self.history: List[Dict[str, Any]] = []
        self.total_collapses = 0
        self.successful_recoveries = 0

    def assess_loop_coherence(
        self, coherence: float, nodes: List[Any], loop_integrity: float
    ):
        """Records the relative shape of the swarm at this timestep."""
        positions = np.array([n.pos for n in nodes])
        centroid = np.mean(positions, axis=0)

        # FIX #2 — DummyNode exposes .vel (an ndarray), not .velocity() (a method).
        # The old code called n.velocity() which doesn't exist on DummyNode, so
        # hasattr(...) was always False and all stored velocities were zeros.
        # That made safe_centroid = current_centroid - 0*0.5 = current_centroid,
        # teleporting the swarm right back into whatever it crashed into.
        snapshot = {
            "relative_positions": [n.pos - centroid for n in nodes],
            "velocities": [getattr(n, "vel", np.zeros(3)).copy() for n in nodes],
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
        critically_broken = any(h["loop_integrity"] < 0.4 for h in recent)

        return is_unstable or critically_broken

    def collapse_and_reinitialize(self, nodes: List[Any]) -> Dict[str, Any]:
        """Pure Temporal Rewind. No hardcoded shapes. Only topological memory."""
        self.total_collapses += 1
        print("[WFC] Swarm shattered. Searching topological memory for stable shape...")

        best_shape = None
        best_score = -1.0

        for memory in reversed(self.history):
            if memory["coherence"] > best_score:
                best_score = memory["coherence"]
                best_shape = memory
            if best_score > 0.6:
                break

        current_positions = np.array([n.pos for n in nodes])

        if best_shape is None:
            print("[WFC] WARNING: No memory found! Freezing in place.")
            self.history.clear()
            return {
                "reinit_from": "emergency_freeze",
                "success": True,
                "target_positions": current_positions,
            }

        current_centroid = np.mean(current_positions, axis=0)
        # Now avg_velocity is non-zero because .vel is properly stored
        avg_velocity = np.mean(best_shape["velocities"], axis=0)
        safe_centroid = current_centroid - (avg_velocity * 0.5)

        target_safe_positions = [
            safe_centroid + best_shape["relative_positions"][i]
            for i in range(len(nodes))
        ]

        self.successful_recoveries += 1
        self.history.clear()

        return {
            "reinit_from": "relative_shape_rewind",
            "success": True,
            "target_positions": target_safe_positions,
        }
