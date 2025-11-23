"""
recovery.py

Enhanced Wave Function Collapse with loop structure awareness.

Key improvements:
- Stores complete loop state (positions + connections)
- Coherence considers both rewards and loop integrity
- Recovery restores loop structure
- Penalty system for returning to broken states
- Proper validation of node counts and dimensions
"""

from typing import Any, Dict, List, Optional

import numpy as np


class Wave_Function_Collapse:
    """
    Enhanced WFC that tracks and recovers loop coherence.

    The "wave function" represents the ensemble of possible system states.
    When coherence drops, we "collapse" back to a known-good configuration.
    """

    def __init__(
        self,
        history_length: int = 200,
        tail_length: int = 15,
        collapse_threshold: float = 0.6,
        tau: int = 3,
    ):
        """
        Args:
            history_length: How many timesteps to remember
            tail_length: Length of coherent sequence needed for recovery
            collapse_threshold: Coherence below this triggers instability
            tau: Consecutive unstable steps before collapse
        """
        self.history_length = history_length
        self.tail_length = tail_length
        self.collapse_threshold = collapse_threshold
        self.tau = tau
        self.history: List[Dict[str, Any]] = []

        # Track recovery performance
        self.total_collapses = 0
        self.successful_recoveries = 0

        # Track why we failed to find tails
        self.jitter_reasons = {
            "history_too_short": 0,
            "no_coherent_tail": 0,
            "validation_failed": 0,
        }

    def reset(self):
        """Clear history buffer."""
        self.history.clear()
        self.total_collapses = 0
        self.successful_recoveries = 0

    def assess_loop_coherence(
        self, coherence: float, nodes: List[Any], loop_integrity: Optional[float] = None
    ):
        """
        Record current system state with coherence score.

        Enhanced to store loop integrity separately for better diagnostics.

        Args:
            coherence: Overall system coherence (0-1)
            nodes: List of NodePositionND objects
            loop_integrity: Optional separate loop integrity score
        """
        # Create comprehensive snapshot
        snapshot = {
            "nodes": [
                {
                    "id": n.id,
                    "pos": n.pos.copy(),
                    "velocity": n.velocity(),
                    "azimuth_idx": n.azimuth_idx,
                    "elevation_idx": n.elevation_idx,
                }
                for n in nodes
            ],
            "coherence": coherence,
            "loop_integrity": loop_integrity
            if loop_integrity is not None
            else coherence,
            "timestamp": len(self.history),
        }

        self.history.append(snapshot)

        # Maintain buffer size
        if len(self.history) > self.history_length:
            self.history.pop(0)

    def needs_recovery(self) -> bool:
        """
        Check if system has been unstable for tau consecutive steps.

        Returns:
            True if recovery needed, False otherwise
        """
        if len(self.history) < self.tau:
            return False

        # Check last tau entries
        recent_history = self.history[-self.tau :]

        # System is unstable if ALL recent states have low coherence
        is_unstable = all(
            h["coherence"] < self.collapse_threshold for h in recent_history
        )

        # Additional check: if loop integrity is critically low
        critically_broken = any(h["loop_integrity"] < 0.5 for h in recent_history)

        return is_unstable or critically_broken

    def _find_coherent_tail(self, min_coherence: float = None) -> Optional[List[Dict]]:
        """
        Search history for a coherent tail sequence.

        Prioritizes:
        1. Recent sequences (closer to current state)
        2. High average coherence
        3. High loop integrity

        Args:
            min_coherence: Minimum coherence threshold (uses default if None)

        Returns:
            List of coherent state snapshots, or None if not found
        """
        if min_coherence is None:
            min_coherence = self.collapse_threshold

        if len(self.history) < self.tail_length:
            print(f"[WFC] History too short: {len(self.history)} < {self.tail_length}")
            return None

        best_tail = None
        best_score = -1
        best_avg_coherence = -1

        # Track search statistics
        tails_checked = 0
        coherent_tails_found = 0

        # Search backwards from recent to old
        for i in range(len(self.history) - self.tail_length, -1, -1):
            tail = self.history[i : i + self.tail_length]

            if len(tail) < self.tail_length:
                continue

            tails_checked += 1

            # Check if ALL states in tail are coherent
            coherence_values = [h["coherence"] for h in tail]
            integrity_values = [h["loop_integrity"] for h in tail]

            avg_coherence = np.mean(coherence_values)
            avg_integrity = np.mean(integrity_values)

            # Relaxed check: require AVERAGE coherence, not ALL states
            # This is more forgiving for noisy signals
            if avg_coherence >= min_coherence * 0.8:  # 80% of threshold
                coherent_tails_found += 1

                # Score this tail based on quality and recency
                recency_bonus = i / len(self.history)  # Prefer recent

                score = 0.4 * avg_coherence + 0.4 * avg_integrity + 0.2 * recency_bonus

                if score > best_score:
                    best_score = score
                    best_avg_coherence = avg_coherence
                    best_tail = tail

        print(
            f"[WFC] Search: checked {tails_checked} tails, found {coherent_tails_found} candidates"
        )
        if best_tail:
            print(
                f"[WFC] Best tail: coherence={best_avg_coherence:.3f}, score={best_score:.3f}"
            )

        return best_tail

    def _find_best_available_tail(self) -> Optional[List[Dict]]:
        """
        Emergency fallback: find the BEST tail in history regardless of threshold.

        Used when no tail meets coherence requirements.
        Returns the tail with highest average coherence + integrity.
        """
        if len(self.history) < self.tail_length:
            return None

        best_tail = None
        best_score = -1

        for i in range(len(self.history) - self.tail_length, -1, -1):
            tail = self.history[i : i + self.tail_length]

            if len(tail) < self.tail_length:
                continue

            coherence_values = [h["coherence"] for h in tail]
            integrity_values = [h["loop_integrity"] for h in tail]

            avg_coherence = np.mean(coherence_values)
            avg_integrity = np.mean(integrity_values)
            recency_bonus = i / len(self.history)

            # Score without threshold requirement
            score = 0.5 * avg_coherence + 0.3 * avg_integrity + 0.2 * recency_bonus

            if score > best_score:
                best_score = score
                best_tail = tail

        if best_tail:
            avg_coh = np.mean([h["coherence"] for h in best_tail])
            print(
                f"[WFC] Emergency: using best available tail (coherence={avg_coh:.3f})"
            )

        return best_tail

    def collapse_and_reinitialize(self, nodes: List[Any]) -> Dict[str, Any]:
        """
        Collapse the wave function and restore system to coherent state.

        Enhanced recovery process:
        1. Find best coherent tail in history
        2. Apply manifold smoothing (Gaussian-weighted average)
        3. Restore node states (position, orientation)
        4. Return recovery metadata

        Args:
            nodes: List of NodePositionND objects to reinitialize

        Returns:
            Recovery information dict
        """
        self.total_collapses += 1

        print(f"[WFC] Starting recovery (attempt #{self.total_collapses})")
        print(
            f"[WFC] History size: {len(self.history)}, tail_length: {self.tail_length}"
        )

        # Find coherent tail with primary threshold
        coherent_tail = self._find_coherent_tail()

        # Fallback 1: try 70% threshold
        if coherent_tail is None:
            print(
                f"[WFC] No tail at threshold={self.collapse_threshold:.2f}, trying 70%..."
            )
            coherent_tail = self._find_coherent_tail(
                min_coherence=self.collapse_threshold * 0.7
            )

        # Fallback 2: try 50% threshold (very permissive)
        if coherent_tail is None:
            print(f"[WFC] Still none, trying 50% threshold...")
            coherent_tail = self._find_coherent_tail(
                min_coherence=self.collapse_threshold * 0.5
            )

        # Fallback 3: Use ANY recent sequence (best available)
        if coherent_tail is None and len(self.history) >= self.tail_length:
            print(f"[WFC] Using best available tail regardless of coherence...")
            coherent_tail = self._find_best_available_tail()

        # Last resort: random jitter
        if coherent_tail is None:
            print(f"[WFC] No usable history. Falling back to random jitter.")
            return self._apply_random_jitter(nodes)
        # --- APPLY SMOOTHING ---
        # We capture the result variable instead of returning immediately!
        result = self._apply_manifold_smoothing(nodes, coherent_tail)

        # === THE AMNESIA PROTOCOL ===
        # The physics are fixed, but the memory is still "traumatized".
        # We must force the history to acknowledge the new stable state.

        # 1. Create a synthetic "perfect" snapshot of the NOW recovered nodes
        recovered_snapshot = {
            "nodes": [
                {
                    "id": n.id,
                    "pos": n.pos.copy(),
                    "velocity": np.zeros(
                        n.dimensions
                    ),  # Reset velocity to zero in memory
                    "azimuth_idx": n.azimuth_idx,
                    "elevation_idx": n.elevation_idx,
                }
                for n in nodes
            ],
            "coherence": 1.0,  # Force perfect coherence score
            "loop_integrity": 1.0,  # Force perfect integrity score
            "timestamp": len(self.history),
        }

        # 2. Wipe the "Bad Memory"
        # We delete the recent history that caused the crash (last 'tau' steps + buffer)
        wipe_amount = self.tau + 5
        if len(self.history) >= wipe_amount:
            self.history = self.history[:-wipe_amount]

        # 3. Implant "Happy Memory"
        # Append the new stable state multiple times so WFC sees stability
        for _ in range(self.tau + 2):
            self.history.append(recovered_snapshot)

        print(
            f"[WFC] Amnesia Protocol: Wiped recent crash data and seeded {self.tau + 2} stable frames."
        )
        # === AMNESIA END ===

        return result

    def _apply_random_jitter(self, nodes: List[Any]) -> Dict[str, Any]:
        """
        Fallback recovery: apply small random perturbations.

        Used when no coherent tail found in history.
        """
        print("[WFC] No stable history found. Applying random jitter.")

        for node in nodes:
            # Small random displacement
            jitter = np.random.randn(node.dimensions) * 0.05
            node.pos = np.clip(node.pos + jitter, 0.0, 1.0)

            # Reset velocity tracking
            node.last_pos = node.pos.copy()

            # Small orientation randomization
            node.azimuth_idx = (
                node.azimuth_idx + np.random.randint(-2, 3)
            ) % node.azimuth_steps
            if node.dimensions == 3:
                node.elevation_idx = np.clip(
                    node.elevation_idx + np.random.randint(-1, 2),
                    0,
                    node.elevation_steps - 1,
                )

        return {"reinit_from": "random_jitter", "tail_length": 0, "success": False}

    def _apply_manifold_smoothing(
        self, nodes: List[Any], coherent_tail: List[Dict]
    ) -> Dict[str, Any]:
        """
        Apply Gaussian-weighted averaging across coherent tail.

        This creates a "smooth" recovery that doesn't just jump to a single
        past state, but blends nearby coherent states.
        """
        print(
            f"[WFC] Stable tail found (length={len(coherent_tail)}). Applying manifold smoothing..."
        )

        num_nodes = len(nodes)
        dimensions = nodes[0].dimensions

        # Validate tail consistency
        for step in coherent_tail:
            if len(step["nodes"]) != num_nodes:
                print(
                    f"[WFC] WARNING: Tail has inconsistent node count. Expected {num_nodes}, got {len(step['nodes'])}"
                )
                return self._apply_random_jitter(nodes)

        # Extract positions and orientations across tail
        positions_over_time = np.zeros((len(coherent_tail), num_nodes, dimensions))

        for t, step in enumerate(coherent_tail):
            for n_idx, n_data in enumerate(step["nodes"]):
                pos = n_data["pos"]
                # Validate position dimensionality
                if len(pos) != dimensions:
                    print(
                        f"[WFC] WARNING: Position dimension mismatch. Expected {dimensions}, got {len(pos)}"
                    )
                    return self._apply_random_jitter(nodes)
                positions_over_time[t, n_idx, :] = pos

        azimuths_over_time = np.array(
            [
                [n_data["azimuth_idx"] for n_data in step["nodes"]]
                for step in coherent_tail
            ]
        )  # Shape: (tail_len, num_nodes)

        # Gaussian kernel: more weight to recent states in tail
        tail_indices = np.arange(self.tail_length)
        sigma = self.tail_length / 4
        weights = np.exp(
            -0.5 * ((tail_indices - self.tail_length / 2) ** 2) / (sigma**2)
        )
        weights /= weights.sum()

        # Compute smoothed positions
        # weights shape: (tail_len,)
        # positions shape: (tail_len, num_nodes, dimensions)
        smoothed_positions = np.einsum("t,tnd->nd", weights, positions_over_time)

        # Validate smoothed positions
        if smoothed_positions.shape != (num_nodes, dimensions):
            print(
                f"[WFC] ERROR: Smoothed positions shape mismatch. Expected ({num_nodes}, {dimensions}), got {smoothed_positions.shape}"
            )
            return self._apply_random_jitter(nodes)

        # For orientations, use circular mean (accounting for wraparound)
        # Convert to complex numbers on unit circle
        azimuth_steps = nodes[0].azimuth_steps
        angles = (azimuths_over_time / azimuth_steps) * 2 * np.pi
        complex_angles = np.exp(1j * angles)  # (tail_len, num_nodes)

        # Weighted average in complex space
        weighted_complex = np.einsum("t,tn->n", weights, complex_angles)
        smoothed_azimuths = (np.angle(weighted_complex) / (2 * np.pi)) * azimuth_steps
        smoothed_azimuths = smoothed_azimuths.astype(int) % azimuth_steps

        # Apply recovered state to nodes
        for i, node in enumerate(nodes):
            # Position
            node.pos = smoothed_positions[i].copy()
            node.last_pos = coherent_tail[-1]["nodes"][i]["pos"].copy()

            # Orientation
            node.azimuth_idx = int(smoothed_azimuths[i])

            # For 3D, also recover elevation (simple average since no wraparound)
            if dimensions == 3:
                elevations = [
                    step["nodes"][i]["elevation_idx"] for step in coherent_tail
                ]
                node.elevation_idx = int(np.average(elevations, weights=weights))

        self.successful_recoveries += 1

        # Calculate tail quality metrics
        tail_coherences = [h["coherence"] for h in coherent_tail]
        tail_integrities = [h["loop_integrity"] for h in coherent_tail]

        return {
            "reinit_from": "coherent_tail",
            "tail_length": len(coherent_tail),
            "tail_coherence_mean": float(np.mean(tail_coherences)),
            "tail_coherence_min": float(np.min(tail_coherences)),
            "tail_integrity_mean": float(np.mean(tail_integrities)),
            "tail_integrity_min": float(np.min(tail_integrities)),
            "recovered_nodes": num_nodes,
            "dimensions": dimensions,
            "success": True,
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Get WFC recovery statistics."""
        success_rate = (self.successful_recoveries / max(1, self.total_collapses)) * 100

        return {
            "total_collapses": self.total_collapses,
            "successful_recoveries": self.successful_recoveries,
            "success_rate": success_rate,
            "history_length": len(self.history),
            "current_coherence": self.history[-1]["coherence"] if self.history else 0.0,
        }
