"""
density_Warehouse.py

Discrete Manhattan Affordance Field for FLOWRRA.

REWRITE (2026-08-28). Four changes from the Poisson version, in order of how
much they matter:

  1. STRUCTURE IS NOW IN THE FIELD.  The old field was 1331 dims (an 11^3 cube)
     that encoded nothing about the warehouse: an open aisle cell and a solid
     rack cell both read 1.0, and on an empty aisle the entire block was a
     constant 1.0 -- i.e. ~96.5% of the GNN's input vector carried no
     information most of the time. The field is now masked by two static,
     precomputed properties of the graph: is this cell a real node
     (`traversable`), and can this fleet actually reach it within
     `local_radius` moves THROUGH the graph rather than through a rack
     (`reachable`). A cell one aisle over is Manhattan-distance 2 but may be
     graph-distance 20; the old field called it open.

  2. NO MORE SATURATION.  The transform was A = clip(1 - R, 0, 1). Any R > 1
     collapsed to exactly 0, so a single fatal splat (severity 5.0) produced 63
     cells reading identical zeros -- no gradient anywhere inside, so Tier-1
     Spatial Escape had nothing to rank neighbors by and always fell through to
     Tier 2. It is now A = mask / (1 + R), which is monotone decreasing in R,
     never saturates, and preserves a usable gradient at any severity.

  3. MEMORY IS TRANSIENT, NOT A LANDMINE.  Splats used to ADD (+5.0 fatal,
     +2.0 warning) against a linear -0.1/step decay. The warning splat fired
     every single step integrity was 0.5, netting +1.9/step, pinning at the
     10.0 cap within 5 steps and taking 100 steps to clear -- which is exactly
     how a pair of fleets loitering near their (nearby) goals turned that goal
     region into a permanent no-go zone. Splats now take max() instead of
     summing, start well below saturation, and decay MULTIPLICATIVELY, so a
     fatal splat is materially gone in ~8 steps instead of ~47.

  4. PEERS ARE STAMPED AS SWEPT PATHS, NOT BALLS.  The old projection was one
     isotropic blob one cell ahead. A symmetric ball cannot express the most
     important fact in a crossing -- that a fleet moving AWAY is not a threat.
     Each active peer now stamps the cells it will occupy over the next
     `projection_steps` moves, with severity decaying along the trail. Head-on
     convergence lights up; receding traffic barely registers.

  Also: the returned vector is now the Manhattan DIAMOND (cells within
  local_radius) rather than the full cube. The cube's corners reach Manhattan
  distance 3*L and can never be reached in L moves, so they were permanently
  masked-out constants. At L=5 that is 231 cells instead of 1331. Both
  core_warehouse.py and main_runner_warehouse.py compute input_dim by calling
  len() on this function's output, so nothing downstream needs editing -- but
  any checkpoint trained on the old 1331-dim field will no longer load.
"""

import numpy as np
from collections import deque
from typing import Any, Dict, List, Optional, Set, Tuple


class WarehouseDensityField:
    """
    Manages the discrete 3D affordance field for the warehouse holon.

    affordance(cell) = traversable(cell) * reachable(cell) * 1 / (1 + repulsion(cell))

    1.0 means open, reachable and clear. 0.0 means either structurally
    impossible (wall / unreachable) or overwhelmed by repulsion.
    """

    def __init__(
        self,
        max_vision_range: int = 10,
        falloff_radius: float = 3.0,
        peer_severity: float = 0.7,
        projection_steps: int = 3,
        projection_falloff: float = 0.6,
        memory_decay_factor: float = 0.7,
        memory_floor: float = 0.05,
        memory_cap: float = 2.0,
        grid_pos_dict: Optional[Dict[Tuple[int, int, int], str]] = None,
        graph: Any = None,
        # --- accepted for drop-in compatibility with the Poisson constructor ---
        lambda_severity: Optional[float] = None,
        decay_rate: Optional[float] = None,
    ):
        self.local_radius = max_vision_range // 2
        L = self.local_radius

        self.falloff_radius = float(falloff_radius)
        self.peer_severity = float(peer_severity)
        self.projection_steps = int(projection_steps)
        self.projection_falloff = float(projection_falloff)
        self.memory_decay_factor = float(memory_decay_factor)
        self.memory_floor = float(memory_floor)
        self.memory_cap = float(memory_cap)

        # lambda_severity / decay_rate belonged to the Poisson kernel and have no
        # analogue here. Accepted silently so an un-updated core_warehouse.py
        # still constructs, but they do nothing -- see falloff_radius and
        # memory_decay_factor for the knobs that replaced them.
        self._legacy_args_used = (lambda_severity is not None) or (decay_rate is not None)

        # Spatial-Temporal Memory: (X, Y, Z) -> severity. Short-lived by design.
        self.collapse_memory: Dict[Tuple[int, int, int], float] = {}

        # ------------------------------------------------------------------
        # Precomputed local grid geometry (built once, reused every call)
        # ------------------------------------------------------------------
        self.grid_shape = (2 * L + 1, 2 * L + 1, 2 * L + 1)
        rel_x, rel_y, rel_z = np.meshgrid(
            np.arange(-L, L + 1), np.arange(-L, L + 1), np.arange(-L, L + 1), indexing="ij"
        )
        self._rel_x = rel_x.astype(np.float32)
        self._rel_y = rel_y.astype(np.float32)
        self._rel_z = rel_z.astype(np.float32)

        # Integer offsets of every cell in the cube, so a reachability BFS can be
        # written straight into grid coordinates.
        self._rel_int = np.stack([rel_x, rel_y, rel_z], axis=-1).astype(np.int32)

        # The Manhattan diamond: cells within local_radius MOVES of the centre.
        # Everything outside it is unreachable by construction (you cannot cover
        # Manhattan distance > L in L unit moves), so it is dropped from the
        # returned vector rather than shipped as a permanent zero.
        self._center_manhattan = (
            np.abs(self._rel_x) + np.abs(self._rel_y) + np.abs(self._rel_z)
        )
        self._diamond_mask = self._center_manhattan <= L
        self.output_dim = int(np.count_nonzero(self._diamond_mask))

        # Index of the centre cell WITHIN the flattened diamond output. Callers
        # that probe a single cell (recovery's Tier-1 check, braking) index
        # len(v)//2; the diamond is symmetric so that still lands on the centre,
        # but this is the honest way to say it.
        flat_center = np.zeros(self.grid_shape, dtype=bool)
        flat_center[L, L, L] = True
        self.center_index = int(np.argmax(flat_center[self._diamond_mask]))

        # Linear falloff kernel, indexed by Manhattan distance. Replaces the
        # Poisson survival curve: same monotone-decreasing shape, one legible
        # parameter. k(d) = max(0, 1 - d / falloff_radius).
        max_d = 3 * L + 2
        d_axis = np.arange(max_d + 1, dtype=np.float32)
        self._kernel = np.maximum(0.0, 1.0 - d_axis / self.falloff_radius).astype(np.float32)
        self._max_kernel_idx = max_d

        # ------------------------------------------------------------------
        # Static structure (optional but strongly recommended)
        # ------------------------------------------------------------------
        self.grid_pos_dict: Optional[Dict[Tuple[int, int, int], str]] = None
        self.graph = None
        self._coords_by_id: Dict[str, Tuple[int, int, int]] = {}
        # center cell -> boolean cube of structurally-usable cells. The graph never
        # mutates, so this is computed once per distinct centre and then free.
        self._structure_cache: Dict[Tuple[int, int, int], np.ndarray] = {}

        if grid_pos_dict is not None and graph is not None:
            self.set_structure(grid_pos_dict, graph)

    # ----------------------------------------------------------------------
    # Structure
    # ----------------------------------------------------------------------
    def set_structure(self, grid_pos_dict: Dict[Tuple[int, int, int], str], graph: Any):
        """
        Supplies the static warehouse topology. Without this the field falls back
        to the old behaviour (structure-blind: every cell treated as open), which
        keeps standalone tests and dimension-probing calls working, but gives up
        the single biggest information gain of this rewrite.
        """
        self.grid_pos_dict = grid_pos_dict
        self.graph = graph
        self._coords_by_id = {nid: coords for coords, nid in grid_pos_dict.items()}
        self._structure_cache.clear()

    def _structure_mask(self, center_idx: np.ndarray) -> np.ndarray:
        """
        Boolean cube: True where a cell is BOTH a real graph node AND reachable
        from the centre within local_radius graph hops.

        Every edge in this warehouse graph has length exactly 1.0, so one graph
        hop equals one grid step and a depth-L BFS is precisely "cells I could
        stand on within L moves". This is what stops a cell on the far side of a
        rack -- Manhattan-close, graph-distant -- from reading as open space.
        """
        if self.grid_pos_dict is None or self.graph is None:
            return np.ones(self.grid_shape, dtype=bool)

        key = (int(center_idx[0]), int(center_idx[1]), int(center_idx[2]))
        cached = self._structure_cache.get(key)
        if cached is not None:
            return cached

        mask = np.zeros(self.grid_shape, dtype=bool)
        start_id = self.grid_pos_dict.get(key)

        if start_id is None:
            # Fleet is mid-move between cells (speed 0.5 makes this common) and
            # rounded onto a non-node. Fail open rather than reporting the whole
            # neighbourhood as impassable.
            self._structure_cache[key] = np.ones(self.grid_shape, dtype=bool)
            return self._structure_cache[key]

        L = self.local_radius
        seen = {start_id: 0}
        queue = deque([(start_id, 0)])
        while queue:
            nid, depth = queue.popleft()
            coords = self._coords_by_id.get(nid)
            if coords is not None:
                rel = (coords[0] - key[0], coords[1] - key[1], coords[2] - key[2])
                if max(abs(rel[0]), abs(rel[1]), abs(rel[2])) <= L:
                    mask[rel[0] + L, rel[1] + L, rel[2] + L] = True
            if depth >= L:
                continue
            for nb in self.graph.neighbors(nid):
                if nb not in seen:
                    seen[nb] = depth + 1
                    queue.append((nb, depth + 1))

        self._structure_cache[key] = mask
        return mask

    # ----------------------------------------------------------------------
    # Spatial-temporal memory
    # ----------------------------------------------------------------------
    def splat_spatial_temporal_event(self, position: np.ndarray, severity_multiplier: float = 1.5):
        """
        Records a collapse at a coordinate as a SHORT-LIVED marker.

        Uses max() rather than +=. The old additive version, driven by a warning
        splat that fired every step integrity stayed at 0.5, ratcheted to its cap
        within ~5 steps and then needed ~100 steps to clear -- which is how two
        fleets pausing near adjacent goals bricked that whole region for the rest
        of the episode. max() means standing next to someone for 40 steps costs
        exactly the same as standing next to them for 1.
        """
        pos_tuple = tuple(np.round(position).astype(int))
        severity = float(min(severity_multiplier, self.memory_cap))
        current = self.collapse_memory.get(pos_tuple, 0.0)
        if severity > current:
            self.collapse_memory[pos_tuple] = severity

    def step_decay(self):
        """
        Multiplicative decay. At the default 0.7, a 1.5-severity fatal marker
        falls below the 0.05 floor in 8 steps -- roughly the temporal
        neighbourhood in which an encounter is actually still relevant, versus
        the ~47 steps of full blockade the linear-0.1 version produced.
        """
        if not self.collapse_memory:
            return
        decayed = {}
        for pos, sev in self.collapse_memory.items():
            new_sev = sev * self.memory_decay_factor
            if new_sev > self.memory_floor:
                decayed[pos] = new_sev
        self.collapse_memory = decayed

    # ----------------------------------------------------------------------
    # The field
    # ----------------------------------------------------------------------
    def get_local_affordance(
        self,
        center_pos: np.ndarray,
        all_fleets: List[Any],
        frozen_node_ids: Set[str],
        own_goal_pos: Any = None,
        frozen_obstacle_severity: float = 1.0,
        near_goal_radius: float = 3.0,
        own_id: Optional[str] = None,
        stopped_node_ids: Optional[Set[str]] = None,
        stopped_obstacle_severity: float = 1.4,
    ) -> np.ndarray:
        """
        Builds the local affordance vector for one fleet.

        Returns a flat float32 array of length self.output_dim (the Manhattan
        diamond of radius local_radius), ordered by the fixed C-order traversal
        of the cube restricted to that diamond -- so a given index always means
        the same relative offset for every fleet on every step.

        Frozen (parked) fleets keep the perspective-dependent severity ramp from
        the previous version: a parked peer is a real obstacle to through
        traffic, but is discounted toward 0 as it approaches the OBSERVING
        fleet's own goal, so a fleet whose destination neighbours a parked peer
        does not perceive its own goal as permanently blocked. own_goal_pos=None
        skips the discount entirely (fully transparent), so dimension-probing
        callers fail safe rather than silently seeing full-severity obstacles.
        """
        repulsion = np.zeros(self.grid_shape, dtype=np.float32)
        center_idx = np.round(center_pos).astype(int)

        def stamp(source_pos: np.ndarray, severity: float):
            """Add one linear-falloff peak. Vectorized over the whole cube."""
            if severity <= 0.0:
                return
            source_idx = np.round(source_pos).astype(int)
            rel = source_idx - center_idx
            # Anything further out than the kernel reaches contributes nothing.
            if np.max(np.abs(rel)) > self.local_radius + self.falloff_radius:
                return
            dist = (
                np.abs(self._rel_x - rel[0])
                + np.abs(self._rel_y - rel[1])
                + np.abs(self._rel_z - rel[2])
            )
            idx = np.minimum(dist, self._max_kernel_idx).astype(np.int64)
            repulsion[...] += self._kernel[idx] * severity

        # --- 1. Fleets -----------------------------------------------------
        for fleet in all_fleets:
            # Identify self by ID when the caller supplies one. The old
            # position-equality test had two failure modes it never recovered
            # from: during a fatal collision the fleet you just crashed INTO sits
            # at your exact coordinate and was silently skipped as "self" -- so
            # the field went blind to the obstacle precisely when it mattered
            # most; and recovery's Tier-1 check evaluates a NEIGHBOUR cell, where
            # the position never matches, so the fleet repelled itself out of
            # every escape it considered.
            if own_id is not None:
                if fleet.id == own_id:
                    continue
            elif np.array_equal(fleet.current_pos, center_pos):
                continue  # self (legacy positional fallback)

            if fleet.id in frozen_node_ids:
                if own_goal_pos is None:
                    continue

                # A STOPPED (errored) fleet stamps harder than a merely PARKED
                # one: a parked fleet is sitting where somebody wanted to be, so
                # goal cells are useful places, whereas a stopped fleet is an
                # unplanned hazard in the middle of an aisle. Still finite, so a
                # rescuer can push through it rather than being walled out.
                #
                # The near-goal discount below is reused deliberately and is
                # exactly right for handover: it scales severity down as the
                # obstacle approaches the OBSERVING fleet's own goal. A rescuer
                # dispatched to a pickup has goal_pos == the stopped fleet's
                # cell, so its discount goes to ~0 and the obstacle becomes
                # transparent to that one fleet, while every other fleet still
                # sees full severity and routes around. That mechanism was
                # written for parked peers blocking their neighbours' goals and
                # happens to give handover exactly the behaviour it needs for
                # free.
                sev = frozen_obstacle_severity
                if stopped_node_ids is not None and fleet.id in stopped_node_ids:
                    sev = stopped_obstacle_severity

                dist_to_own_goal = float(np.sum(np.abs(fleet.current_pos - own_goal_pos)))
                discount = float(np.clip(dist_to_own_goal / near_goal_radius, 0.0, 1.0))
                stamp(fleet.current_pos, sev * discount)
                continue

            # Active peer: stamp the SWEPT PATH it is about to occupy, not an
            # isotropic ball. node.direction is a unit cell delta, so step k of
            # the trail is k cells ahead. A peer travelling away from this fleet
            # therefore lays its trail away too, and barely registers here --
            # which an isotropic kernel could never express.
            stamp(fleet.current_pos, self.peer_severity)
            if np.any(fleet.direction != 0):
                severity = self.peer_severity
                for k in range(1, self.projection_steps + 1):
                    severity *= self.projection_falloff
                    stamp(fleet.current_pos + fleet.direction * k, severity)

        # --- 2. Spatial-temporal collapse memory ---------------------------
        for crash_pos_tuple, severity in self.collapse_memory.items():
            stamp(np.array(crash_pos_tuple, dtype=np.float32), severity)

        # --- 3. Transform + structural mask --------------------------------
        # 1/(1+R) instead of clip(1-R): monotone, bounded in (0, 1], and it never
        # flattens to a constant zero, so there is always a gradient telling the
        # policy which direction is LESS bad.
        affordance = 1.0 / (1.0 + repulsion)
        affordance = affordance * self._structure_mask(center_idx).astype(np.float32)

        return affordance[self._diamond_mask].astype(np.float32)

    # ----------------------------------------------------------------------
    def describe(self) -> str:
        """One-line summary, handy for a startup log line."""
        structured = "structure-aware" if self.grid_pos_dict is not None else "STRUCTURE-BLIND"
        return (
            f"[Density] radius={self.local_radius} dim={self.output_dim} ({structured}) "
            f"kernel=linear(r={self.falloff_radius}) memory=x{self.memory_decay_factor}/step"
        )