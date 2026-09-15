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
        projection_max_branches: int = 6,
        project_stationary: bool = True,
        projection_mode: str = "intended",
        kernel_metric: str = "graph",
        output_mode: str = "affordance",
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
        self.projection_max_branches = int(projection_max_branches)
        self.project_stationary = bool(project_stationary)
        # "intended"       -- greedy descent on the peer's goal BFS map (2026-09-13)
        # "dead_reckoning" -- current_pos + direction * k, the PRE-2026-09-13
        #                     behaviour, bit-for-bit including its two defects
        #                     (projects through racks, aliases on half-cells).
        #
        # This flag is not a fallback. It exists because a checkpoint trained
        # under dead reckoning is reading DIFFERENT VALUES in all 231 density
        # dims when evaluated under intended-path projection, and the size of
        # that shift scales with peer count. Reproducing the training-time
        # environment is the only way to separate "this feature is useless" from
        # "this checkpoint is out of distribution".
        if projection_mode not in ("intended", "dead_reckoning"):
            raise ValueError(
                f"projection_mode must be 'intended' or 'dead_reckoning', "
                f"got {projection_mode!r}")
        self.projection_mode = projection_mode

        # Memo for the intended-path descent, and a counter for how often the
        # descent failed and fell back to dead reckoning. See
        # _cached_intended_path() and the projection block in
        # get_local_affordance().
        self._projection_fallbacks = 0
        # Identity of THIS field, carried in the per-fleet memo token.
        #
        # BUG THIS FIXES, caught by test_projection immediately after the memo
        # moved onto the fleet: the memo is stored on the FLEET, so two density
        # fields sharing the same fleet objects -- an A/B comparing kernel
        # metrics, or the lesion harness -- would serve one field's cached path
        # to the other, silently, with no way to notice. Production has one
        # field per environment so it would never have shown up there, which is
        # exactly what makes it worth guarding.
        self._field_token = id(self)

        # When True, get_local_affordance() returns the raw 2-channel VOLUME
        # instead of the flattened 231-dim affordance. Set via get_local_volume(),
        # never left on -- the flat path is what the current encoder expects.
        self._return_volume = False

        # "affordance" -> 231 dims, mask/(1+R): what the flat encoder expects.
        # "channels"   -> 462 dims, mask and R packed separately, for the gated
        #                 convolution. The multiply is what makes 0 mean BOTH
        #                 "no track here" and "fully contested", and no
        #                 convolution can recover that distinction afterwards.
        if output_mode not in ("affordance", "channels"):
            raise ValueError(
                f"output_mode must be 'affordance' or 'channels', "
                f"got {output_mode!r}")
        self.output_mode = output_mode

        # Per-step drivers for the cost curve. Without these, a flat ms/step is
        # ambiguous: on the 2026-09-13 run active fleets fell 196 -> 137 (about
        # 22% fewer stamps) while cost ROSE 4%, so two effects were moving in
        # opposite directions and cancelling. That is not the same as "phase
        # does not matter", and only per-block drivers can tell them apart.
        self.stamps_this_step = 0
        self.stamp_early_outs = 0

        # GRAPH-DISTANCE KERNEL (2026-09-13). Cache of
        #   source cell -> [(cell coords, graph hops), ...]
        # for every cell within kernel reach of the source THROUGH THE GRAPH.
        # The graph never mutates, so each distinct source is paid for once per
        # run. _kernel_manhattan_fallbacks counts sources that could not be
        # placed on the graph (mid-move onto a non-node) and fell back to the
        # old Manhattan kernel.
        # "graph"     -- falloff measured in graph hops through real edges.
        # "manhattan"  -- the pre-2026-09-13 kernel, bit-for-bit, for
        #                 reproducing published runs. NOT a fallback for when
        #                 "graph" misbehaves; the fallback is per-source and
        #                 automatic (see stamp()).
        if kernel_metric not in ("graph", "manhattan"):
            raise ValueError(f"kernel_metric must be 'graph' or 'manhattan', got {kernel_metric!r}")
        self.kernel_metric = kernel_metric
        self._kernel_nbhd_cache: Dict[Tuple[int, int, int], Optional[List[Tuple[Tuple[int, int, int], int]]]] = {}
        self._kernel_manhattan_fallbacks = 0
        self._kernel_cache_cap = 200000

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
        # Consumers size input_dim from output_dim, so it must reflect the mode.
        self.diamond_cells = self.output_dim
        if output_mode == "channels":
            self.output_dim *= 2

        # Index of the centre cell WITHIN the flattened diamond output. Callers
        # that probe a single cell (recovery's Tier-1 check, braking) index
        # len(v)//2; the diamond is symmetric so that still lands on the centre,
        # but this is the honest way to say it.
        flat_center = np.zeros(self.grid_shape, dtype=bool)
        flat_center[L, L, L] = True
        self.center_index = int(np.argmax(flat_center[self._diamond_mask]))

        # Flat indices, within the 231-vector, of the six cells one step away
        # along each axis. Used by action_entropy() to read a fleet's immediate
        # options without recomputing the field. Precomputed because the
        # flattening is boolean-mask indexing in C order, so the flat position of
        # a cube cell is the count of live cells before it -- constant for a
        # given radius, and not worth deriving per call.
        _flat_pos = (np.cumsum(self._diamond_mask.ravel()) - 1).reshape(self.grid_shape)
        self._neighbour_flat_idx: List[int] = []
        for _d in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
            _i = (L + _d[0], L + _d[1], L + _d[2])
            if self._diamond_mask[_i]:
                self._neighbour_flat_idx.append(int(_flat_pos[_i]))

        # Linear falloff kernel, indexed by Manhattan distance. Replaces the
        # Poisson survival curve: same monotone-decreasing shape, one legible
        # parameter. k(d) = max(0, 1 - d / falloff_radius).
        max_d = 3 * L + 2
        d_axis = np.arange(max_d + 1, dtype=np.float32)
        self._kernel = np.maximum(0.0, 1.0 - d_axis / self.falloff_radius).astype(np.float32)
        self._max_kernel_idx = max_d

        # How far a stamp can reach at all: k(d) = max(0, 1 - d/falloff_radius)
        # is exactly 0 at d >= falloff_radius, so a BFS deeper than this can only
        # add zeros. At the default falloff_radius=3.0 the curve is
        # [1.0, 0.67, 0.33, 0.0] and reach is 2 -- so the per-stamp BFS settles
        # roughly 2.27^2 ~ 5-15 nodes, not a search.
        self._kernel_reach = int(np.ceil(self.falloff_radius)) - 1
        if self._kernel[min(self._kernel_reach + 1, max_d)] > 0.0:
            self._kernel_reach += 1

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
    # Graph-distance kernel neighbourhood
    # ----------------------------------------------------------------------
    def _kernel_neighbourhood(self, source_cell: Tuple[int, int, int]):
        """
        Every cell within `_kernel_reach` GRAPH hops of `source_cell`, with its
        hop count. None if the source is not a graph node.

        WHY THIS EXISTS. stamp() laid its falloff using Manhattan distance
        inside the cube:

            dist = |rel_x - sx| + |rel_y - sy| + |rel_z - sz|
            repulsion += kernel[dist] * severity

        That is the third instance of the same defect fixed this session in
        braking and in check_integrity, and the last one still live. A peer
        behind a rack at Manhattan 2 and graph 20 stamped 0.33 repulsion onto
        cells around the observer as though it were two cells away. The
        structural mask does NOT catch this: the mask zeroes cells the OBSERVER
        cannot reach, and says nothing about whether the SOURCE can reach them.
        An observer standing in a clear aisle has a fully-unmasked
        neighbourhood, and every cell in it was absorbing repulsion from fleets
        on the far side of a shelf.

        Note what this does NOT break: a peer genuinely around a corner at graph
        distance 2 still stamps at full kernel strength, because the BFS follows
        the aisle round the bend. Corners get MORE accurate here, not less --
        Manhattan under-counts a bend as much as it over-counts a rack.

        Cached per source cell. The graph is static, so each distinct cell is
        paid for once per run, and a stamp becomes ~5-15 scalar writes instead
        of a 1331-cell vectorized add.
        """
        if self.kernel_metric != "graph" or self.graph is None or not self.grid_pos_dict:
            return None
        cached = self._kernel_nbhd_cache.get(source_cell)
        if cached is not None or source_cell in self._kernel_nbhd_cache:
            return cached

        start_id = self.grid_pos_dict.get(source_cell)
        if start_id is None:
            self._kernel_nbhd_cache[source_cell] = None
            return None

        out: List[Tuple[Tuple[int, int, int], int]] = []
        seen = {start_id: 0}
        queue = deque([(start_id, 0)])
        while queue:
            nid, depth = queue.popleft()
            coords = self._coords_by_id.get(nid)
            if coords is not None:
                out.append((coords, depth))
            if depth >= self._kernel_reach:
                continue
            for nb in self.graph.neighbors(nid):
                if nb not in seen:
                    seen[nb] = depth + 1
                    queue.append((nb, depth + 1))

        if len(self._kernel_nbhd_cache) > self._kernel_cache_cap:
            self._kernel_nbhd_cache.clear()
        self._kernel_nbhd_cache[source_cell] = out
        return out

    # ----------------------------------------------------------------------
    # Intended path (BFS-gradient descent)
    # ----------------------------------------------------------------------
    def _arrival_node(self, fleet: Any) -> Optional[str]:
        """
        The grid node this fleet is on, or is arriving at if it is mid-move.

        Borrows the ceil/floor-along-the-axis-of-travel trick from
        FleetNode.get_goal_gradient(), for the reason documented there: at
        base_speed 0.5 a fleet spends half its life at x.5, and np.round uses
        banker's rounding, so round(10.5) == 10 but round(11.5) == 12. Rounding
        alone therefore names a DIFFERENT cell depending on parity.

        BUT NOT its "already exactly on a node, so step one further" branch, and
        this is worth stating because getting it wrong is easy and silent.
        get_goal_gradient() probes a CANDIDATE NEIGHBOUR for each of the six
        actions, so when its probe lands on the fleet's own cell it must push
        one further to name the neighbour. This function asks a different
        question -- WHERE IS THIS FLEET NOW -- and for that, a fleet sitting
        exactly on a node is on THAT node.

        The distinction matters because `direction` is the delta of the move the
        fleet LAST EXECUTED (apply_discrete_action writes it after the move), not
        the move it intends next. Extrapolating one cell past a fleet that is
        stationary on a node therefore projects its route from a cell it has not
        reached and may never reach. Caught by test_projection.py::test_turning_peer,
        which reported the route of a fleet at (3,0,0) as 4 -> 3 -> 2 instead of
        2 -> 1 -> 0.
        """
        if not self.grid_pos_dict:
            return None
        pos = np.asarray(fleet.current_pos, dtype=np.float64)
        direction = np.asarray(getattr(fleet, "direction", np.zeros(3)), dtype=np.float64)

        probe = np.round(pos).astype(np.float64)
        if np.any(direction != 0):
            axis = int(np.argmax(np.abs(direction)))
            # Only extrapolate when the fleet is genuinely BETWEEN cells.
            if abs(pos[axis] - round(float(pos[axis]))) > 1e-9:
                probe[axis] = (np.ceil(pos[axis]) if direction[axis] > 0
                               else np.floor(pos[axis]))

        return self.grid_pos_dict.get(tuple(int(v) for v in np.round(probe)))

    def _intended_path(self, fleet: Any) -> Optional[List[List[Tuple[Tuple[int, int, int], float]]]]:
        """
        The cells this fleet INTENDS to occupy over the next projection_steps
        moves, by greedy descent on its own precomputed goal BFS distance map.

        Returns one list per step, each holding (coords, weight) with the
        weights at each step summing to 1.0. Ties -- two neighbours equally
        closer to the goal -- split the weight, which is the analytic form of a
        "probability cloud" over the route: no rollout, no sampling, no
        trajectory tensor. On a degree-~2.27 corridor graph ties are rare, so
        this is almost always a single chain of k dictionary lookups.

        RETURN CONTRACT -- the two failure modes are NOT the same and must not
        be collapsed, which is how the first version of this produced 551
        spurious fallbacks in a 60-step smoke test:

          None  -> COULD NOT COMPUTE. No graph, no distance map, off-grid, or
                   unreachable from the goal. The caller falls back to dead
                   reckoning, because some projection beats none.
          []    -> COMPUTED, and there are no future cells: the fleet is sitting
                   on its goal. The caller must project NOTHING. Dead reckoning
                   here would stamp a trail straight ahead for a fleet that is
                   not going anywhere.
        """
        if self.graph is None or not self.grid_pos_dict:
            return None
        dmap = getattr(fleet, "goal_distance_map", None)
        if not dmap:
            return None

        start = self._arrival_node(fleet)
        if start is None or dmap.get(start) is None:
            return None

        # Generate one level more than needed, because the leading level is
        # dropped when it is the cell the fleet is already standing on.
        #
        # WHY: the caller stamps `fleet.current_pos` at full peer_severity before
        # laying the trail. For a fleet sitting exactly ON a node, `start` IS
        # that cell, so emitting it as trail step 1 would stamp it twice --
        # 0.7 + 0.42 instead of 0.7 -- silently inflating own-cell repulsion by
        # 60% for every stationary peer. For a MID-EDGE fleet, `start` is the
        # cell it is crossing into and has not been stamped, so it is kept.
        # The trail therefore always means "cells this peer will occupy NEXT".
        steps: List[List[Tuple[Tuple[int, int, int], float]]] = []
        frontier: Dict[str, float] = {start: 1.0}

        for k in range(self.projection_steps + 1):
            cells = []
            for nid, w in frontier.items():
                coords = self._coords_by_id.get(nid)
                if coords is not None:
                    cells.append((coords, w))
            if not cells:
                break
            steps.append(cells)

            nxt: Dict[str, float] = {}
            for nid, w in frontier.items():
                here = dmap.get(nid)
                if here is None:
                    continue
                descend = [nb for nb in self.graph.neighbors(nid)
                           if dmap.get(nb) is not None and dmap[nb] < here]
                if not descend:
                    continue  # at the goal, or in a pocket -- this branch ends
                share = w / len(descend)
                for nb in descend:
                    nxt[nb] = nxt.get(nb, 0.0) + share

            if not nxt:
                break
            if len(nxt) > self.projection_max_branches:
                keep = sorted(nxt.items(), key=lambda kv: -kv[1])[: self.projection_max_branches]
                total = sum(w for _, w in keep) or 1.0
                nxt = {nid: w / total for nid, w in keep}
            frontier = nxt

        own_cell = tuple(int(v) for v in np.round(np.asarray(fleet.current_pos)))
        if steps and len(steps[0]) == 1 and steps[0][0][0] == own_cell:
            steps = steps[1:]

        # NOT `steps or None`. An empty list means "computed, nothing ahead" and
        # must stay distinguishable from None ("could not compute").
        return steps[: self.projection_steps]

    def _fleet_cell(self, fleet: Any) -> Tuple[int, int, int]:
        """
        The fleet's rounded grid cell, memoised on the fleet for this position.

        get_local_affordance() runs once per OBSERVER and stamps every peer
        inside, so rounding a peer's position in that loop costs N^2 roundings
        per step for a quantity that depends only on the peer. Same fix, and the
        same shape, as the intended-path memo above.
        """
        pos = fleet.current_pos
        token = (self._field_token, float(pos[0]), float(pos[1]), float(pos[2]))
        memo = getattr(fleet, "_cell_memo", None)
        if memo is not None and memo[0] == token:
            return memo[1]
        cell = (int(round(float(pos[0]))),
                int(round(float(pos[1]))),
                int(round(float(pos[2]))))
        fleet._cell_memo = (token, cell)
        return cell

    def _cached_intended_path(self, fleet: Any):
        """
        Per-position memo for _intended_path.

        get_local_affordance() is called once per OBSERVER and loops over every
        peer inside, so an uncached descent would run O(N^2 * k * degree) times
        per step. The key is (id, arrival cell, direction, goal) -- everything
        the descent depends on -- so an entry can never be served to a fleet
        that has since moved or been re-targeted, and no explicit per-step reset
        hook is needed.
        """
        # REGRESSION FIXED 2026-09-13, introduced earlier the same session.
        #
        # WAS: a key built with np.round(pos * 1000).astype(np.int64) and
        #      np.sign(direction).astype(np.int8), then a lookup into a shared
        #      dict. get_local_affordance() calls this once per OBSERVER-PEER
        #      PAIR, so at 200 fleets that is 71,481 calls per step, each paying
        #      four numpy dispatches to build a key. The 2026-09-13 profile put
        #      it at 17.6 s cumulative, 14.4% of the step -- for a memo whose
        #      whole purpose was to make this cheap.
        #
        # NOW: the memo lives ON THE FLEET, and the token is a plain tuple of
        #      scalars, so a hit costs one getattr and one tuple compare. The
        #      path depends only on the fleet, so per-fleet storage is the
        #      natural home for it; the shared dict was never the right shape.
        #      It also bounds itself -- one entry per fleet, no 4096 cap, no
        #      periodic clear that threw away every live entry at once.
        pos = fleet.current_pos
        direction = getattr(fleet, "direction", None)
        token = (
            self._field_token,
            float(pos[0]), float(pos[1]), float(pos[2]),
            0.0 if direction is None else float(direction[0]),
            0.0 if direction is None else float(direction[1]),
            0.0 if direction is None else float(direction[2]),
            getattr(fleet, "current_goal_id", None),
        )
        memo = getattr(fleet, "_ip_memo", None)
        if memo is not None and memo[0] == token:
            return memo[1]

        val = self._intended_path(fleet)
        if val is None:
            # Counted HERE, on a cache miss, so the number means "distinct
            # fleet-steps whose route could not be computed". Counting at the
            # stamping site instead counted observer-peer PAIRS, inflating it by
            # roughly the fleet count and making it unreadable.
            self._projection_fallbacks += 1
        fleet._ip_memo = (token, val)
        return val

    # ----------------------------------------------------------------------
    # Spatial-temporal memory
    # ----------------------------------------------------------------------
    def action_entropy(self, affordance: np.ndarray) -> float:
        """
        How DECISIVE this fleet's immediate options are, in [0, 1].

        Normalise the affordance of the six neighbouring cells into a
        distribution and take Shannon entropy over it, scaled by log(n) so the
        result is comparable across cells with different numbers of live
        neighbours.

            1.0  every option looks equally good -- no reason to prefer any
            0.0  one option dominates -- the choice is made for you
            0.0  nothing is reachable at all (an isolated or walled-in cell)

        This is the `S` of `F = E - T*S`, computed rather than posited, and it is
        the LOCAL half: the global field entropy needs a partition function over
        a shared field, which does not exist yet.

        WHY OVER AFFORDANCE AND NOT OVER REPULSION. Entropy over R alone is
        undefined when nothing is contested, which at 0.24% occupancy is almost
        always. Affordance folds the reachability mask in, so a walled-in cell
        and a jammed cell both read low -- which is the correct reading for a
        DECISIVENESS measure, even though it would be the wrong one for a
        congestion measure. Do not reuse this as a congestion signal.

        Note the consequence: in open space every reachable neighbour has
        affordance 1.0, so entropy is exactly 1.0 and the feature is a constant.
        It only becomes informative under contention -- which is precisely where
        it is meant to be read.
        """
        if not self._neighbour_flat_idx:
            return 0.0
        vals = affordance[self._neighbour_flat_idx]
        total = float(vals.sum())
        if total <= 1e-9:
            return 0.0
        p = vals / total
        p = p[p > 1e-12]
        if p.size <= 1:
            return 0.0
        h = float(-(p * np.log(p)).sum())

        # Normalised by log(LIVE neighbours), not log(6).
        #
        # Dividing by log(6) would make the feature mostly a measure of DEGREE:
        # a mid-aisle cell has only 2 live neighbours, so its entropy could never
        # exceed log(2)/log(6) = 0.387 however undecided it was, while a junction
        # with 4 could reach 0.774. The policy would read "corridor vs junction",
        # which it already gets from the mask, instead of "decisive vs undecided",
        # which is the point.
        #
        # Against log(n_live) a corridor with two equally good options and a
        # junction with four both read 1.0, and both fall the same way as
        # contention concentrates. Degree-independent by construction.
        # Clipped: the quantity is mathematically bounded by [0, 1], but
        # float32 affordances promoted to float64 can overshoot by ~3e-9, which
        # is enough to fail a bounds assertion and to hand the encoder a value
        # outside the range every other feature respects.
        return float(min(1.0, max(0.0, h / np.log(p.size))))

    def get_local_volume(self, *args, **kwargs) -> np.ndarray:
        """
        The local field as a 2-channel VOLUME, shape (2, S, S, S) with S = 2L+1.

            channel 0 : structure mask   -- binary, 1 where a track exists and is
                                            reachable within L graph hops
            channel 1 : repulsion R      -- continuous, how contested each cell is

        WHY THIS EXISTS. get_local_affordance() returns `mask / (1 + R)`, and that
        multiply destroys information that cannot be recovered afterwards: a
        value of 0 means BOTH "no track here" and "infinitely contested here".
        Those are opposite facts. One says never go; the other says wait.

        Measured 2026-09-14 on the shipped field: at 0.25% occupancy **100%** of
        the 231 affordance dims are exactly 0.0 or 1.0 -- pure structure mask,
        zero congestion content. Even at 7.8% occupancy it is still 96%. So the
        block the network spends 79% of its input width on is, in practice, a
        binary reachability stamp.

        And that stamp is nearly free of information: on a 344-cell map there are
        **93 distinct mask shapes**, one of which covers 49% of all cells. About
        6.5 bits, encoded in 231 float32 = 7,392 bits.

        WHAT THE VOLUME IS, geometrically. Not a cube and not a ball. It is a
        geodesic ball B(v, L) under the GRAPH metric -- and on a degree-~2.27
        corridor graph that is a thin skeleton of corridor arms radiating from
        the observer. A line segment mid-aisle, a cross at a junction, a T at an
        intersection, a stub at a dead end. Typically ~35 live cells of 1,331,
        so the dense array is about 97% padding.

        The shape is itself the signal: it says what KIND of place the fleet is
        standing in, which is what the affordance encoding throws away by
        flattening it into an unordered list.

        Consumed by the Phase-2 convolutional encoder. The flat affordance path
        remains the default until that lands.
        """
        self._return_volume = True
        try:
            return self.get_local_affordance(*args, **kwargs)
        finally:
            self._return_volume = False

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
        static_obstacles: Optional[Set[Tuple[int, int, int]]] = None,
        static_obstacle_severity: float = 3.0,
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

        L = self.local_radius
        S = 2 * L + 1
        # Plain ints, hoisted out of stamp(). The early-out below runs on EVERY
        # stamp call including the ones it rejects -- 303,204 per step at 200
        # fleets -- and doing it with np.round / np.abs / np.max on a 3-element
        # array costs three numpy dispatches to compare three numbers. The
        # 2026-09-13 profile showed np.max at 1,517,575 calls, almost exactly
        # one per stamp, and np.round at 2,650,162.
        _cx, _cy, _cz = int(center_idx[0]), int(center_idx[1]), int(center_idx[2])
        _lim = L + self.falloff_radius

        def stamp(source_pos: np.ndarray, severity: float):
            """
            Add one linear-falloff peak, measured in GRAPH HOPS from the source.

            WAS (Manhattan in the cube, vectorized over all 1331 cells):
                dist = |rel_x - sx| + |rel_y - sy| + |rel_z - sz|
                idx  = minimum(dist, max_kernel_idx)
                repulsion[...] += kernel[idx] * severity

            The third and last live instance of the defect fixed this session in
            braking and check_integrity. A peer behind a rack at Manhattan 2 and
            graph 20 stamped 0.33 onto cells around the observer as though it
            were two cells away. The structural mask does not catch this: it
            zeroes cells the OBSERVER cannot reach and says nothing about whether
            the SOURCE can reach them, so an observer in a clear aisle had a
            fully-unmasked neighbourhood absorbing repulsion from fleets on the
            far side of a shelf.

            NOW: a cached BFS from the source cell out to _kernel_reach hops,
            following the aisle around bends. Corners get MORE accurate, not
            less -- Manhattan under-counts a bend exactly as it over-counts a
            rack. A stamp is now ~5-15 scalar writes instead of a 1331-cell
            vectorized add.

            Falls back to the Manhattan kernel, verbatim, when the source cannot
            be placed on the graph (mid-move onto a non-node). Counted, because
            a non-trivial fallback rate would mean the fallback is carrying the
            behaviour rather than backstopping it.
            """
            if severity <= 0.0:
                return
            # WAS: np.round(source_pos).astype(int), then rel = source_idx -
            #      center_idx, then np.max(np.abs(rel)) -- four numpy dispatches
            #      on 3-element arrays, per call, 303k times per step.
            # NOW: plain Python. Identical result; round() and int() on scalars
            #      match np.round().astype(int) for the half-integer positions
            #      this sees, because both use banker's rounding.
            sx = int(round(float(source_pos[0])))
            sy = int(round(float(source_pos[1])))
            sz = int(round(float(source_pos[2])))
            stamp_cell(sx, sy, sz, severity)

        def stamp_cell(sx: int, sy: int, sz: int, severity: float):
            """
            The integer-cell core of stamp().

            SPLIT OUT 2026-09-13. The 2026-09-13 profile showed
            builtins.round at 2,801,050 calls over 3 steps -- 933,000 per step
            -- and most of it was a pure round trip of my own making. The trail
            stamps did:

                stamp(np.asarray(coords, dtype=np.float32), severity * weight)

            where `coords` is ALREADY an integer tuple straight out of the BFS
            neighbourhood. It was converted to a float32 array so that stamp()
            could round it back to the integers it started as. np.asarray at
            536,141 calls and astype at 473,546 are the same round trip.

            Callers that already hold an integer cell now come here directly.
            """
            if severity <= 0.0:
                return
            rx = sx - _cx
            ry = sy - _cy
            rz = sz - _cz
            # Anything further out than the kernel reaches contributes nothing.
            if (rx > _lim or rx < -_lim or ry > _lim or ry < -_lim
                    or rz > _lim or rz < -_lim):
                self.stamp_early_outs += 1
                return
            self.stamps_this_step += 1

            nbhd = self._kernel_neighbourhood((sx, sy, sz))

            if nbhd is None:
                # Only a FALLBACK when the graph kernel was asked for and could
                # not place this source. Under kernel_metric="manhattan" this is
                # the intended path, so it must not inflate the counter.
                if self.kernel_metric == "graph":
                    self._kernel_manhattan_fallbacks += 1
                dist = (
                    np.abs(self._rel_x - rx)
                    + np.abs(self._rel_y - ry)
                    + np.abs(self._rel_z - rz)
                )
                idx = np.minimum(dist, self._max_kernel_idx).astype(np.int64)
                repulsion[...] += self._kernel[idx] * severity
                return

            for coords, hops in nbhd:
                w = self._kernel[hops] if hops <= self._max_kernel_idx else 0.0
                if w <= 0.0:
                    continue
                ix = coords[0] - _cx + L
                iy = coords[1] - _cy + L
                iz = coords[2] - _cz + L
                if 0 <= ix < S and 0 <= iy < S and 0 <= iz < S:
                    repulsion[ix, iy, iz] += w * severity

        # --- 0. Unregistered obstacles -------------------------------------
        # Humans, debris, a dropped pallet. Stamped HARDER than any fleet: a
        # stopped fleet is 1.4 and finite so a rescuer can push through it,
        # whereas nothing should ever want to be where a person is.
        #
        # NO near-goal discount, deliberately. That mechanism exists so an
        # obstacle sitting on the OBSERVER'S OWN GOAL becomes transparent to
        # that one fleet -- which is exactly right for a corpse being rescued
        # and exactly wrong for a human. Nobody's goal is ever a person.
        #
        # Repulsion alone is a preference, and a large enough reward can outbid
        # a preference. The hard veto lives in
        # FleetNode.get_valid_action_mask(), which refuses the move outright.
        if static_obstacles:
            for _cell in static_obstacles:
                stamp_cell(int(_cell[0]), int(_cell[1]), int(_cell[2]),
                           static_obstacle_severity)

        # --- 1. Fleets -----------------------------------------------------
        # Own goal hoisted to plain floats. The near-goal discount below runs
        # once per OBSERVER-PEER pair for every FROZEN peer, and the frozen count
        # only ever grows across an episode -- 9 at step 20, 160 at step 200 on
        # the 2026-09-13 run. At 400 state builds per step that is 64,000
        # evaluations per step by the end, and it was doing them with
        # np.sum(np.abs(a - b)) and np.clip: two numpy dispatches on 3-element
        # arrays to compare three numbers.
        #
        # This was the term that GREW while everything else shrank. Over 200
        # steps active fleets fell 191 -> 40 and stamps per observer fell 33 ->
        # 10, yet step cost rose 45%. The profile at step 200 shows np.sum at
        # 64,580 calls per step and np.clip at 65,295 -- almost exactly the
        # frozen count times the observer count.
        _has_goal = own_goal_pos is not None
        if _has_goal:
            _gx = float(own_goal_pos[0])
            _gy = float(own_goal_pos[1])
            _gz = float(own_goal_pos[2])
        _ngr = float(near_goal_radius) if near_goal_radius else 1.0

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
                if not _has_goal:
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

                # WAS:
                #   dist = float(np.sum(np.abs(fleet.current_pos - own_goal_pos)))
                #   discount = float(np.clip(dist / near_goal_radius, 0.0, 1.0))
                # Identical arithmetic, no numpy dispatch. The distance is
                # non-negative by construction, so the lower clip could never
                # bind; only the upper one is kept.
                _p = fleet.current_pos
                dist_to_own_goal = (abs(float(_p[0]) - _gx)
                                    + abs(float(_p[1]) - _gy)
                                    + abs(float(_p[2]) - _gz))
                discount = (1.0 if dist_to_own_goal >= _ngr
                            else dist_to_own_goal / _ngr)
                _c = self._fleet_cell(fleet)
                stamp_cell(_c[0], _c[1], _c[2], sev * discount)
                continue

            # Active peer: stamp the SWEPT PATH it is about to occupy.
            #
            # CHANGED 2026-09-13: the trail is now the peer's INTENDED route --
            # greedy descent on its own precomputed goal BFS map -- instead of
            # straight-line dead reckoning off node.direction.
            #
            # WHAT THE OLD VERSION DID: stamp(current_pos + direction * k) for
            # k = 1..projection_steps. Two defects, both silent:
            #
            #   a) It projects THROUGH RACKS. A peer one cell from a T-junction
            #      and about to turn had its next three cells stamped straight
            #      ahead into solid shelving -- repulsion laid down where no
            #      fleet can ever be, and none laid down on the cells it will
            #      actually occupy. The density field has known the graph since
            #      the August rewrite (_structure_mask); the projection did not
            #      use it. The structural mask then multiplies those in-rack
            #      cells by 0, so the stamp is not merely misplaced, it is
            #      DISCARDED -- a turning peer projected nothing at all.
            #
            #   b) It aliases on half-cells. At base_speed 0.5 a fleet is at x.5
            #      half the time, and stamp() rounds. np.round is banker's:
            #      round(11.5)=12, round(12.5)=12, round(13.5)=14. A peer at
            #      10.5 moving +X stamped 12, 12, 14 -- skipping 11 and 13 and
            #      double-stamping 12, so the "comet tail" had a hole in it.
            #
            # WHAT IT DOES NOW: walks the peer's goal_distance_map one cell at a
            # time through real edges. Ties split weight, which is the analytic
            # probability cloud over its route. Both defects go away because
            # every stamped cell is a graph node the peer can actually reach.
            #
            # BEHAVIOUR DELTA TO WATCH: a peer with direction == 0 (idle, or it
            # just bumped a wall, which zeroes direction in
            # apply_discrete_action) used to project NOTHING. It now projects
            # from its current cell, because a braked fleet still has an intent
            # and that intent is the whole point of this change. Set
            # CONFIG["density"]["project_stationary"] = False to restore the old
            # silence if this turns out to over-repel in congestion.
            # WAS stamp(fleet.current_pos, ...), which rounded the SAME peer
            # position once per OBSERVER -- N^2 roundings per step for a value
            # that depends only on the peer. Memoised on the fleet, same shape
            # as the intended-path memo.
            _c = self._fleet_cell(fleet)
            stamp_cell(_c[0], _c[1], _c[2], self.peer_severity)

            projecting = self.project_stationary or bool(np.any(fleet.direction != 0))
            if not projecting:
                continue

            path = (self._cached_intended_path(fleet)
                    if self.projection_mode == "intended" else None)
            if path is not None:
                # May be EMPTY: a peer standing on its goal has no future cells,
                # and dead-reckoning a trail for it would stamp a corridor it is
                # never going to enter.
                severity = self.peer_severity
                for cells in path:
                    severity *= self.projection_falloff
                    for coords, weight in cells:
                        # coords is already (int, int, int) from the BFS
                        # neighbourhood -- no array, no round trip.
                        stamp_cell(coords[0], coords[1], coords[2],
                                   severity * weight)
            elif np.any(fleet.direction != 0):
                # Reached either as the dead_reckoning MODE, or as a fallback
                # when the descent could not be computed.
                # Straight-line dead reckoning, exactly as before -- defects and
                # all -- because it is strictly better than projecting nothing,
                # and because a silent behaviour change here would be worse than
                # a known-imperfect one. The counter lives in
                # _cached_intended_path so it measures FLEET-STEPS, not pairs; if
                # it is not near zero on a real map the descent is failing and
                # that needs finding, not tolerating.
                severity = self.peer_severity
                for k in range(1, self.projection_steps + 1):
                    severity *= self.projection_falloff
                    stamp(fleet.current_pos + fleet.direction * k, severity)

        # --- 2. Spatial-temporal collapse memory ---------------------------
        for crash_pos_tuple, severity in self.collapse_memory.items():
            stamp_cell(crash_pos_tuple[0], crash_pos_tuple[1],
                       crash_pos_tuple[2], severity)

        # --- 3. Transform + structural mask --------------------------------
        # 1/(1+R) instead of clip(1-R): monotone, bounded in (0, 1], and it never
        # flattens to a constant zero, so there is always a gradient telling the
        # policy which direction is LESS bad.
        mask = self._structure_mask(center_idx).astype(np.float32)

        if self._return_volume:
            # TWO CHANNELS, NOT MULTIPLIED. See get_local_volume().
            return np.stack([mask, repulsion.astype(np.float32)], axis=0)

        if self.output_mode == "channels":
            # PACKED DIAMOND: mask then repulsion, each over the 231 live cells.
            # 462 numbers rather than the 2,662 of a dense cube -- at
            # buffer_capacity 15,000 and 60 fleets that is 3.9 GB of replay
            # instead of 19.8 GB, and the cube is ~97% padding anyway.
            #
            # The encoder scatters it back into a cube before convolving, and
            # the scatter is a fixed index assignment, so undoing the packing is
            # free. Crucially this leaves the PIPELINE UNCHANGED: still one flat
            # vector per fleet, so memory.push and _pad_transition need no edits.
            return np.concatenate([
                mask[self._diamond_mask],
                repulsion.astype(np.float32)[self._diamond_mask],
            ])

        affordance = 1.0 / (1.0 + repulsion)
        affordance = affordance * mask

        return affordance[self._diamond_mask].astype(np.float32)

    # ----------------------------------------------------------------------
    def describe(self) -> str:
        """One-line summary, handy for a startup log line."""
        structured = "structure-aware" if self.grid_pos_dict is not None else "STRUCTURE-BLIND"
        return (
            f"[Density] radius={self.local_radius} dim={self.output_dim} ({structured}) "
            f"falloff=linear(r={self.falloff_radius}) memory=x{self.memory_decay_factor}/step "
            f"projection={self.projection_mode}(k={self.projection_steps}, "
            f"stationary={self.project_stationary}) fallbacks={self._projection_fallbacks} "
            f"kernel={self.kernel_metric}(reach={self._kernel_reach}) "
            f"kernel_fallbacks={self._kernel_manhattan_fallbacks}"
        )