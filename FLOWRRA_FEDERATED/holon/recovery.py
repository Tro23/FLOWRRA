"""
recovery.py - FIXED VERSION

Enhanced Wave Function Collapse with:
1. Spatial affordance collapse (forward-looking)
2. Temporal tail collapse (backward-looking)
3. Proper coordinate mapping for local grids
4. Differential amnesia protocol

Key Fixes:
- Corrected local_extent calculation using global_grid_shape
- Fixed _sample_affordance coordinate mapping
- Increased search radius and samples for better coverage
- Proper toroidal distance calculations
- Differential amnesia for spatial vs temporal recoveries
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class Wave_Function_Collapse:
    """
    Enhanced WFC with dual recovery modes:
    - Spatial: Forward-looking using current affordance field
    - Temporal: Backward-looking using historical coherent states
    """

    def __init__(
        self,
        history_length: int = 200,
        tail_length: int = 15,
        collapse_threshold: float = 0.6,
        tau: int = 3,
        global_grid_shape: Tuple[int, ...] = (60, 60),  # NEW
        local_grid_size: Tuple[int, ...] = (5, 5),  # NEW
    ):
        """
        Args:
            history_length: How many timesteps to remember
            tail_length: Length of coherent sequence needed for recovery
            collapse_threshold: Coherence below this triggers instability
            tau: Consecutive unstable steps before collapse
            global_grid_shape: World grid dimensions (for coordinate mapping)
            local_grid_size: Local grid dimensions (for extent calculation)
        """
        self.history_length = history_length
        self.tail_length = tail_length
        self.collapse_threshold = collapse_threshold
        self.tau = tau
        self.history: List[Dict[str, Any]] = []

        # NEW: Grid parameters for spatial collapse
        self.global_grid_shape = global_grid_shape
        self.local_grid_size = local_grid_size

        # Calculate local extent (how much world space the local grid covers)
        self.local_extent = (1.0 / max(global_grid_shape)) * max(local_grid_size)

        # Track recovery performance
        self.total_collapses = 0
        self.successful_recoveries = 0
        self.spatial_recoveries = 0  # NEW: Track spatial success
        self.temporal_recoveries = 0  # NEW: Track temporal success

    def reset(self):
        """Clear history buffer."""
        self.history.clear()
        self.total_collapses = 0
        self.successful_recoveries = 0
        self.spatial_recoveries = 0
        self.temporal_recoveries = 0

    def assess_loop_coherence(
        self, coherence: float, nodes: List[Any], loop_integrity: Optional[float] = None
    ):
        """Record current system state with coherence score."""
        snapshot = {
            "nodes": [
                {
                    "id": n.id,
                    "pos": n.pos.copy(),
                    "velocity": n.velocity(),
                    "azimuth_idx": n.azimuth_idx,
                    "elevation_idx": getattr(n, "elevation_idx", 0),
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

        if len(self.history) > self.history_length:
            self.history.pop(0)

    def needs_recovery(self) -> bool:
        """Check if system has been unstable for tau consecutive steps."""
        if len(self.history) < self.tau:
            return False

        recent_history = self.history[-self.tau :]
        is_unstable = all(
            h["coherence"] < self.collapse_threshold for h in recent_history
        )
        critically_broken = any(h["loop_integrity"] < 0.5 for h in recent_history)

        return is_unstable or critically_broken

    def _sample_affordance(
        self, candidate_pos: np.ndarray, node_pos: np.ndarray, local_grid: np.ndarray
    ) -> float:
        """
        FIXED: Sample affordance at candidate_pos from local_grid centered on node_pos.

        Key fix: Proper coordinate mapping using self.local_extent
        """
        grid_shape = np.array(local_grid.shape)

        # 1. Calculate relative displacement with toroidal wrapping
        rel_pos = np.mod(candidate_pos - node_pos + 0.5, 1.0) - 0.5

        # 2. Map to local grid coordinates using the ACTUAL local extent
        # rel_pos ranges from [-0.5, 0.5] in world coords
        # We need to map to [0, grid_shape] in grid coords
        grid_coords = (rel_pos / self.local_extent + 0.5) * grid_shape
        indices = grid_coords.astype(int)

        # 3. Bounds check
        if np.any(indices < 0) or np.any(indices >= grid_shape):
            return 0.0  # Outside local grid = unknown territory = low affordance

        return float(local_grid[tuple(indices)])

    def _calculate_virtual_integrity(
        self, positions: List[np.ndarray], ideal_dist: float = 0.6
    ) -> float:
        """
        Calculate loop integrity for proposed positions.
        Uses toroidal distance for wraparound world.
        """
        num_nodes = len(positions)
        if num_nodes < 3:
            return 0.0

        integrity_scores = []
        for i in range(num_nodes):
            p1 = positions[i]
            p2 = positions[(i + 1) % num_nodes]

            # Toroidal distance
            diff = np.mod(p2 - p1 + 0.5, 1.0) - 0.5
            dist = np.linalg.norm(diff)

            # Gaussian scoring: 1.0 at ideal_dist, drops off with deviation
            score = np.exp(-0.5 * ((dist - ideal_dist) ** 2) / 0.02)
            integrity_scores.append(score)

        return float(np.mean(integrity_scores))

    def _spatial_affordance_collapse(
        self,
        nodes: List[Any],
        local_grids: List[np.ndarray],
        ideal_dist: float = 0.6,
        config: dict = None,  # NEW: Accept config for tuning
    ) -> bool:
        """
        TUNABLE: Forward-looking collapse using current affordance field.

        Uses config parameters for search radius, samples, and thresholds.
        """
        # Get tuning parameters from config (with fallbacks)
        if config is None:
            search_radius_mult = 0.9
            samples = 32
            integrity_threshold = 0.7
            improvement_min = 1.05
        else:
            wfc_cfg = config.get("wfc", {})
            search_radius_mult = wfc_cfg.get("spatial_search_radius_mult", 0.9)
            samples = wfc_cfg.get("spatial_samples", 32)
            integrity_threshold = wfc_cfg.get("spatial_accept_threshold", 0.65)
            improvement_min = wfc_cfg.get("spatial_improvement_min", 1.03)

        # Search parameters
        search_radius = self.local_extent * search_radius_mult

        print(
            f"[WFC Spatial] radius={search_radius:.4f}, samples={samples}, "
            f"integrity_thresh={integrity_threshold:.2f}, improvement_min={improvement_min:.2f}"
        )

        new_positions = []
        improvement_scores = []

        for i, node in enumerate(nodes):
            best_pos = node.pos.copy()
            best_score = -1.0

            # Current affordance as baseline
            current_aff = self._sample_affordance(node.pos, node.pos, local_grids[i])

            for sample_idx in range(samples):
                # Generate candidate with uniform distribution over search radius
                angle = (sample_idx / samples) * 2 * np.pi
                radius = search_radius * np.sqrt(np.random.uniform(0, 1))

                if node.dimensions == 2:
                    jitter = np.array([radius * np.cos(angle), radius * np.sin(angle)])
                else:  # 3D
                    elevation = np.random.uniform(-search_radius, search_radius)
                    jitter = np.array(
                        [radius * np.cos(angle), radius * np.sin(angle), elevation]
                    )

                candidate_pos = np.mod(node.pos + jitter, 1.0)

                # 1. Affordance Score (from local grid)
                aff_score = self._sample_affordance(
                    candidate_pos, node.pos, local_grids[i]
                )

                # 2. Loop Integrity Constraint (distance to neighbors)
                integrity_score = 1.0
                neighbors = [(i - 1) % len(nodes), (i + 1) % len(nodes)]

                for n_idx in neighbors:
                    # Toroidal distance to neighbor
                    diff = np.mod(candidate_pos - nodes[n_idx].pos + 0.5, 1.0) - 0.5
                    dist = np.linalg.norm(diff)

                    # Gaussian penalty for deviation from ideal_dist
                    integrity_score *= np.exp(-0.5 * ((dist - ideal_dist) ** 2) / 0.01)

                # Combined score: prioritize affordance but respect integrity
                combined_score = (aff_score**0.6) * (integrity_score**0.4)

                if combined_score > best_score:
                    best_score = combined_score
                    best_pos = candidate_pos

            # Track improvement
            improvement = best_score / max(current_aff, 0.01)
            improvement_scores.append(improvement)
            new_positions.append(best_pos)

        # Check if the new configuration is actually better
        virtual_integrity = self._calculate_virtual_integrity(new_positions, ideal_dist)
        avg_improvement = np.mean(improvement_scores)

        print(
            f"[WFC Spatial] Virtual integrity: {virtual_integrity:.3f}, "
            f"Avg improvement: {avg_improvement:.3f}"
        )

        # NEW: Use configurable thresholds
        if (
            virtual_integrity > integrity_threshold
            and avg_improvement > improvement_min
        ):
            for i, node in enumerate(nodes):
                node.pos = new_positions[i].copy()
                node.last_pos = new_positions[i].copy()

            print(f"[WFC Spatial] ✅ SUCCESS! Moved to better configuration")
            self.spatial_recoveries += 1
            return True
        else:
            print(
                f"[WFC Spatial] ❌ REJECTED - "
                f"integrity={virtual_integrity:.3f} (need >{integrity_threshold:.2f}), "
                f"improvement={avg_improvement:.3f} (need >{improvement_min:.2f})"
            )
            return False

    def collapse_and_reinitialize(
        self,
        nodes: List[Any],
        local_grids: Optional[List[np.ndarray]] = None,
        ideal_dist: float = 0.6,
        config: dict = None,  # NEW: Accept config
    ) -> Dict[str, Any]:
        """
        Main recovery method with dual-mode collapse.
        """
        self.total_collapses += 1

        print(f"[WFC] === Recovery Attempt #{self.total_collapses} ===")

        # MODE 1: Spatial Affordance Collapse (if grids provided)
        if local_grids is not None and len(local_grids) == len(nodes):
            print(f"[WFC] Attempting SPATIAL collapse...")

            # NEW: Pass config for tunable parameters
            if self._spatial_affordance_collapse(
                nodes, local_grids, ideal_dist, config=config
            ):
                self.successful_recoveries += 1

                return {
                    "reinit_from": "spatial_affordance",
                    "success": True,
                    "spatial_recovery": True,
                }

        # MODE 2: Temporal Tail Collapse (backward-looking)
        print(f"[WFC] Spatial failed, attempting TEMPORAL collapse...")
        print(
            f"[WFC] History size: {len(self.history)}, tail_length: {self.tail_length}"
        )

        # Find coherent tail with cascading thresholds
        coherent_tail = self._find_coherent_tail()

        if coherent_tail is None:
            coherent_tail = self._find_coherent_tail(
                min_coherence=self.collapse_threshold * 0.7
            )

        if coherent_tail is None:
            coherent_tail = self._find_coherent_tail(
                min_coherence=self.collapse_threshold * 0.5
            )

        if coherent_tail is None and len(self.history) >= self.tail_length:
            coherent_tail = self._find_best_available_tail()

        if coherent_tail is None:
            print(f"[WFC] No usable history. Falling back to random jitter.")
            return self._apply_random_jitter(nodes)

        # Apply temporal recovery with manifold smoothing
        result = self._apply_manifold_smoothing(nodes, coherent_tail)

        # AMNESIA PROTOCOL (only for temporal jumps)
        print(f"[WFC] Applying AMNESIA protocol (temporal recovery)")

        recovered_snapshot = {
            "nodes": [
                {
                    "id": n.id,
                    "pos": n.pos.copy(),
                    "velocity": np.zeros(n.dimensions),
                    "azimuth_idx": n.azimuth_idx,
                    "elevation_idx": getattr(n, "elevation_idx", 0),
                }
                for n in nodes
            ],
            "coherence": 1.0,
            "loop_integrity": 1.0,
            "timestamp": len(self.history),
        }

        # Wipe recent bad memory
        wipe_amount = self.tau + 5
        if len(self.history) >= wipe_amount:
            self.history = self.history[:-wipe_amount]

        # Seed stable frames
        for _ in range(self.tau + 2):
            self.history.append(recovered_snapshot.copy())

        print(
            f"[WFC] Amnesia: Wiped {wipe_amount} frames, seeded {self.tau + 2} stable frames"
        )

        self.temporal_recoveries += 1
        return result

    # ===== TEMPORAL RECOVERY METHODS (unchanged) =====

    def _find_coherent_tail(self, min_coherence: float = None) -> Optional[List[Dict]]:
        """Search history for a coherent tail sequence."""
        if min_coherence is None:
            min_coherence = self.collapse_threshold

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

            if avg_coherence >= min_coherence * 0.8:
                recency_bonus = i / len(self.history)
                score = 0.4 * avg_coherence + 0.4 * avg_integrity + 0.2 * recency_bonus

                if score > best_score:
                    best_score = score
                    best_tail = tail

        return best_tail

    def _find_best_available_tail(self) -> Optional[List[Dict]]:
        """Emergency fallback: find the BEST tail regardless of threshold."""
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

            score = 0.5 * avg_coherence + 0.3 * avg_integrity + 0.2 * recency_bonus

            if score > best_score:
                best_score = score
                best_tail = tail

        return best_tail

    def _apply_random_jitter(self, nodes: List[Any]) -> Dict[str, Any]:
        """Fallback recovery: apply small random perturbations."""
        print("[WFC] Applying random jitter (last resort)")

        for node in nodes:
            jitter = np.random.randn(node.dimensions) * 0.05
            node.pos = np.clip(node.pos + jitter, 0.0, 1.0)
            node.last_pos = node.pos.copy()

        return {"reinit_from": "random_jitter", "success": False}

    def _apply_manifold_smoothing(
        self, nodes: List[Any], coherent_tail: List[Dict]
    ) -> Dict[str, Any]:
        """Apply Gaussian-weighted averaging across coherent tail."""
        print(f"[WFC] Applying manifold smoothing (tail length={len(coherent_tail)})")

        num_nodes = len(nodes)
        dimensions = nodes[0].dimensions

        # Validate tail
        for step in coherent_tail:
            if len(step["nodes"]) != num_nodes:
                return self._apply_random_jitter(nodes)

        # Extract positions over time
        positions_over_time = np.zeros((len(coherent_tail), num_nodes, dimensions))

        for t, step in enumerate(coherent_tail):
            for n_idx, n_data in enumerate(step["nodes"]):
                pos = n_data["pos"]
                if len(pos) != dimensions:
                    return self._apply_random_jitter(nodes)
                positions_over_time[t, n_idx, :] = pos

        # Gaussian weighting (prefer recent states in tail)
        tail_indices = np.arange(len(coherent_tail))
        sigma = len(coherent_tail) / 4
        weights = np.exp(
            -0.5 * ((tail_indices - len(coherent_tail) / 2) ** 2) / (sigma**2)
        )
        weights /= weights.sum()

        # Compute smoothed positions
        smoothed_positions = np.einsum("t,tnd->nd", weights, positions_over_time)

        # Apply to nodes
        for i, node in enumerate(nodes):
            node.pos = smoothed_positions[i].copy()
            node.last_pos = coherent_tail[-1]["nodes"][i]["pos"].copy()

        self.successful_recoveries += 1

        tail_coherences = [h["coherence"] for h in coherent_tail]
        tail_integrities = [h["loop_integrity"] for h in coherent_tail]

        return {
            "reinit_from": "coherent_tail",
            "tail_length": len(coherent_tail),
            "tail_coherence_mean": float(np.mean(tail_coherences)),
            "tail_integrity_mean": float(np.mean(tail_integrities)),
            "recovered_nodes": num_nodes,
            "dimensions": dimensions,
            "success": True,
        }

    def diagnose_spatial_failure(
        self, nodes: List[Any], local_grids: List[np.ndarray]
    ) -> Dict[str, Any]:
        """
        Diagnose why spatial collapse might be failing.
        Call this if spatial collapse consistently fails.
        """
        diagnostics = {
            "num_nodes": len(nodes),
            "num_grids": len(local_grids),
            "local_extent": self.local_extent,
            "grid_shapes": [g.shape for g in local_grids],
            "affordance_stats": [],
            "node_positions": [],
        }

        for i, (node, grid) in enumerate(zip(nodes, local_grids)):
            # Check affordance values in grid
            aff_stats = {
                "node_id": node.id,
                "position": node.pos.tolist(),
                "grid_mean": float(np.mean(grid)),
                "grid_max": float(np.max(grid)),
                "grid_min": float(np.min(grid)),
                "grid_nonzero_fraction": float(np.sum(grid > 0) / grid.size),
            }

            # Sample affordance at current position
            current_aff = self._sample_affordance(node.pos, node.pos, grid)
            aff_stats["current_affordance"] = float(current_aff)

            # Check if we can sample nearby positions
            test_offsets = [
                np.array([0.01, 0.0]),
                np.array([0.0, 0.01]),
                np.array([-0.01, 0.0]),
                np.array([0.0, -0.01]),
            ]

            nearby_samples = []
            for offset in test_offsets:
                test_pos = np.mod(node.pos + offset, 1.0)
                sample = self._sample_affordance(test_pos, node.pos, grid)
                nearby_samples.append(float(sample))

            aff_stats["nearby_affordances"] = nearby_samples
            diagnostics["affordance_stats"].append(aff_stats)
            diagnostics["node_positions"].append(node.pos.tolist())

        return diagnostics

    def get_statistics(self) -> Dict[str, Any]:
        """Get WFC recovery statistics."""
        success_rate = (self.successful_recoveries / max(1, self.total_collapses)) * 100
        spatial_rate = (self.spatial_recoveries / max(1, self.total_collapses)) * 100
        temporal_rate = (self.temporal_recoveries / max(1, self.total_collapses)) * 100

        return {
            "total_collapses": self.total_collapses,
            "successful_recoveries": self.successful_recoveries,
            "spatial_recoveries": self.spatial_recoveries,
            "temporal_recoveries": self.temporal_recoveries,
            "success_rate": success_rate,
            "spatial_success_rate": spatial_rate,
            "temporal_success_rate": temporal_rate,
            "history_length": len(self.history),
            "current_coherence": self.history[-1]["coherence"] if self.history else 0.0,
        }
