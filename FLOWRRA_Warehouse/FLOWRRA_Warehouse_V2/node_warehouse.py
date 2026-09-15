"""
node_Warehouse.py

Discrete 3D Manhattan Agent representation for FLOWRRA.
Handles state representation, 6-axis ray casting, relative goal displacement,
and peer-to-peer data integration for discrete warehouse graph navigation.
"""

import math
from bisect import bisect_left, bisect_right
from dataclasses import dataclass, field
from operator import itemgetter
from typing import Any, Dict, List, Optional, Tuple, Set
import numpy as np
import networkx as nx


# Single source of truth for the action encoding -- used by both
# apply_discrete_action() and get_valid_action_mask() below, so there's no risk
# of two copies of this mapping silently drifting apart from each other.
ACTION_DELTAS: Dict[int, np.ndarray] = {
    0: np.zeros(3, dtype=np.float32),
    1: np.array([1.0, 0.0, 0.0], dtype=np.float32),
    2: np.array([-1.0, 0.0, 0.0], dtype=np.float32),
    3: np.array([0.0, 1.0, 0.0], dtype=np.float32),
    4: np.array([0.0, -1.0, 0.0], dtype=np.float32),
    5: np.array([0.0, 0.0, 1.0], dtype=np.float32),
    6: np.array([0.0, 0.0, -1.0], dtype=np.float32),
}


# Shared immutables for is_structurally_valid()'s hot path: a module-level empty
# line so the miss case allocates nothing, and a C-level sort/bisect key so no
# Python lambda is called per element.
_EMPTY_LINE: List[Tuple[int, str]] = []
_FIRST = itemgetter(0)


def build_spatial_indices(
    grid_pos_dict: Dict[Tuple[int, int, int], str]
) -> Tuple[Dict[int, Dict[Tuple[int, int], List[Tuple[int, str]]]], Dict[str, Tuple[int, int, int]]]:
    """
    Precomputes fast lookup structures from grid_pos_dict ONCE per environment
    (it's built at construction and never mutated afterward), so per-step physics
    checks don't have to linearly rescan the entire warehouse every single call.

    BUG THIS FIXES: is_structurally_valid() and WarehouseRecovery's Tier-1 neighbor
    lookup both scanned the FULL grid_pos_dict (every node in the whole warehouse)
    on every call. is_structurally_valid() alone is called ~600 times per fleet per
    step via 6-axis ray casting, so at real-warehouse scale (thousands of grid
    nodes), profiling showed this one function eating ~75% of total step() time --
    pure single-threaded Python dict iteration that never touches numpy or the GPU,
    which is exactly why throwing a bigger instance at it didn't help at all.

    Returns:
      aisle_index: for each move_axis in {0,1,2}, a dict mapping
        (rounded value along the OTHER two axes) -> sorted list of
        (value along move_axis, node_id). Lets is_structurally_valid() find the
        two grid nodes bounding a proposed position along one axis by looking up
        one short aisle-line's worth of nodes instead of scanning the whole grid.
      coords_by_id: node_id -> (X, Y, Z), the reverse of grid_pos_dict. Lets
        WarehouseRecovery look up a neighbor's coordinates in O(1) instead of
        scanning grid_pos_dict for the matching node_id.
    """
    aisle_index: Dict[int, Dict[Tuple[int, int], List[Tuple[int, str]]]] = {0: {}, 1: {}, 2: {}}
    coords_by_id: Dict[str, Tuple[int, int, int]] = {}

    for coords_tuple, node_id in grid_pos_dict.items():
        coords_by_id[node_id] = coords_tuple
        for move_axis in (0, 1, 2):
            ax1, ax2 = [i for i in (0, 1, 2) if i != move_axis]
            key = (coords_tuple[ax1], coords_tuple[ax2])
            aisle_index[move_axis].setdefault(key, []).append((coords_tuple[move_axis], node_id))

    for move_axis in (0, 1, 2):
        for aisle_line in aisle_index[move_axis].values():
            aisle_line.sort(key=lambda t: t[0])

    return aisle_index, coords_by_id


def assign_goals_optimally(
    fleet_ids: List[str],
    fleet_node_ids: List[Optional[str]],
    goal_positions: Dict[str, np.ndarray],
    goal_distance_maps: Dict[str, Dict[str, int]],
) -> Dict[str, Optional[str]]:
    """
    One-time optimal 1:1 fleet->goal assignment, minimizing TOTAL graph distance
    across the whole fleet (the Hungarian / linear-sum-assignment problem).

    WHY: the alternative -- every fleet independently grabbing its own nearest
    goal -- is not an assignment at all, because nothing coordinates the picks.
    Measured over 25 random 25-fleet layouts, that leaves 9.5 fleets (38%) per
    layout committed to a goal another fleet will reach first, and leaves goals
    with nobody assigned to them. A perfect matching guarantees every fleet an
    uncontested goal and every goal an owner.

    Worth being explicit that this makes the benchmark HARDER, not easier: the
    optimal matching costs ~51% more total travel than everyone grabbing the
    nearest goal (177 vs 118 hops on the test layouts), precisely because it
    refuses to let three fleets share one cheap goal and strand two distant
    ones. What it removes is the free win, not the difficulty. Task allocation
    and collision-free execution are separate problems; solving the first with
    a standard algorithm means any fleet that still fails to arrive failed for
    navigation reasons, which is the thing actually under study here.

    Handles rectangular cases: with more fleets than goals, surplus fleets get
    None (caller should freeze them). With more goals than fleets, surplus goals
    are simply left unassigned.

    Returns {fleet_id: goal_id or None}.
    """
    n_f, n_g = len(fleet_ids), len(goal_positions)
    if n_f == 0 or n_g == 0:
        return {fid: None for fid in fleet_ids}

    goal_ids = list(goal_positions.keys())

    # Cost matrix. Unreachable pairs get a large FINITE sentinel -- np.inf makes
    # linear_sum_assignment raise instead of just avoiding that pairing.
    UNREACHABLE = 1e6
    cost = np.full((n_f, n_g), UNREACHABLE, dtype=np.float64)
    for i, node_id in enumerate(fleet_node_ids):
        if node_id is None:
            continue
        for j, gid in enumerate(goal_ids):
            dmap = goal_distance_maps.get(gid)
            if dmap is None:
                continue
            d = dmap.get(node_id)
            if d is not None:
                cost[i, j] = float(d)

    assignment: Dict[str, Optional[str]] = {fid: None for fid in fleet_ids}

    try:
        from scipy.optimize import linear_sum_assignment
        rows, cols = linear_sum_assignment(cost)
        for i, j in zip(rows, cols):
            if cost[i, j] < UNREACHABLE:
                assignment[fleet_ids[i]] = goal_ids[j]
    except ImportError:
        # No scipy: fall back to greedy matching over globally-sorted pairs.
        # Still a valid 1:1 matching with no contention -- just not guaranteed
        # minimal total cost.
        print("[Assign] scipy unavailable -- falling back to greedy 1:1 matching "
              "(valid, but not cost-optimal).")
        pairs = sorted(
            ((cost[i, j], i, j) for i in range(n_f) for j in range(n_g)
             if cost[i, j] < UNREACHABLE)
        )
        used_f, used_g = set(), set()
        for _, i, j in pairs:
            if i in used_f or j in used_g:
                continue
            assignment[fleet_ids[i]] = goal_ids[j]
            used_f.add(i); used_g.add(j)

    return assignment


class ArrayDistanceMap:
    """
    A BFS distance map backed by an int32 array instead of a dict.

    WHY THIS EXISTS. precompute_goal_distances returns one dict per goal. On a
    120k-node warehouse each of those is ~3.8 MB, so a 400-node goal bank retains
    ~1.54 GB for ONE map -- measured, not estimated. MapCache holds every map it
    has visited, so a 16-map curriculum accumulates several GB and either OOMs or
    swaps hard partway through a long run.

    The keys are identical across every map in a bank (they are that graph's
    nodes), so storing them 400 times over is pure waste. One shared
    node -> index dict plus one int32 array per goal costs ~480 KB per map:
    about 8x less, and the saving grows with bank size.

    Implements .get() / [] / in / len / bool so it is a drop-in for the dicts
    every consumer (get_goal_gradient, get_graph_distance_to_goal, the rescuer
    dispatcher) already expects. -1 encodes unreachable, matching dict-miss
    semantics.
    """

    __slots__ = ("_arr", "_idx")

    def __init__(self, arr: "np.ndarray", idx: Dict[str, int]):
        self._arr = arr
        self._idx = idx

    def get(self, key, default=None):
        i = self._idx.get(key)
        if i is None:
            return default
        v = int(self._arr[i])
        return default if v < 0 else v

    def __getitem__(self, key):
        v = self.get(key)
        if v is None:
            raise KeyError(key)
        return v

    def __contains__(self, key):
        return self.get(key) is not None

    def __len__(self):
        return int((self._arr >= 0).sum())

    def __bool__(self):
        # Callers test `if gm else ...` to distinguish "no map" from "map
        # present but this node unreachable", so a populated map must be truthy
        # even before any lookup.
        return self._arr.size > 0


def precompute_goal_distances_compact(
    G: Any,
    goal_nodes: List[str],
    node_index: Optional[Dict[str, int]] = None,
) -> Tuple[Dict[str, ArrayDistanceMap], Dict[str, int]]:
    """
    Same result as precompute_goal_distances, but array-backed (see
    ArrayDistanceMap) with one node -> index dict shared across every goal.

    Returns (maps, node_index) so the caller can reuse the index for any goal
    computed later -- for instance a pickup cell that was never in the bank.
    """
    if node_index is None:
        node_index = {n: i for i, n in enumerate(G.nodes())}
    n = len(node_index)

    out: Dict[str, ArrayDistanceMap] = {}
    for g in goal_nodes:
        if g not in node_index:
            continue
        arr = np.full(n, -1, dtype=np.int32)
        for node, dist in nx.single_source_shortest_path_length(G, g).items():
            arr[node_index[node]] = dist
        out[g] = ArrayDistanceMap(arr, node_index)
    return out, node_index


def precompute_goal_distances(
    G: Any,
    fleet_missions: List[Dict[str, Any]],
) -> Dict[str, Dict[str, int]]:
    """
    One-time precomputation: for every DISTINCT goal node among fleet_missions, runs
    a single-source BFS from that goal to get the true shortest-path distance (in
    graph edges) from every other reachable node. Goals are fixed for the whole
    training run (the same missions are reused every episode), so this only needs
    to run once, ever -- not once per episode, and definitely not once per step.

    BUG THIS FIXES: the reward function measured progress with straight-line
    Manhattan distance, not the actual path through the warehouse graph. Wherever
    the real layout forces a detour around a gap in the grid (missing edges --
    shelving, walls), a fleet's only available move can DECREASE its true
    graph-distance to the goal while INCREASING its straight-line distance -- so
    the old reward scored genuine progress as "wandering away." Checked against
    the real warehouse data: fleets 19, 11, 23, 9, 6 all need 1.4x-2.4x more graph
    hops than their Manhattan distance suggests, meaning a real chunk of their
    only-possible route was being actively punished.

    Returns: {goal_node_id: {node_id: hop_distance_to_goal, ...}, ...}
    """
    distance_maps: Dict[str, Dict[str, int]] = {}
    unique_goals = {m["goal_node"] for m in fleet_missions if "goal_node" in m}
    for goal_node in unique_goals:
        if goal_node in G:
            distance_maps[goal_node] = dict(nx.single_source_shortest_path_length(G, goal_node))
    return distance_maps

@dataclass
class FleetNode:
    id: str
    current_pos: np.ndarray
    goal_pos: np.ndarray
    warehouse_bounds: Tuple[float, float, float] = (50.0, 50.0, 10.0)
    speed: float = 0.5
    max_vision_range: int = 10
    # How far a ray walks, in CELLS. Decoupled from max_vision_range, which also
    # sets the density diamond radius (local_radius = max_vision_range // 2, so
    # 231 dims at 10) and cannot be lowered without gutting the field.
    #
    # 25 is not a tuning choice: ray_distances reports min(cells / 25.0, 1.0), so
    # the feature is pinned at 1.0 beyond 25 cells. The old trace ran to 50 --
    # half of every unobstructed ray was computed and then discarded by the clip.
    ray_range: int = 25
    # "clip25" -- min(cells / 25.0, 1.0). SATURATING: every ray clear beyond 25
    #             cells reads exactly 1.0, so the feature is a constant in open
    #             space. Measured saturation: 26% of casts at 0.24% occupancy.
    # "smooth"  -- cells / (cells + ray_softness). Monotone, never saturates,
    #             bounded in [0,1).
    #
    # This is the same defect class density_warehouse fixed in August, when
    # clip(1 - R) was replaced by 1/(1 + R) because "any R > 1 collapsed to
    # exactly 0" and 63 cells read identical zeros with no gradient anywhere
    # inside. The rays never got that treatment.
    ray_transform: str = "clip25"
    ray_softness: float = 8.0
    # Diagnostic counters for the ray-origin edge case. ray_origin_recovered
    # counts fleet-steps where round(current_pos) missed the graph and the
    # floor/ceil fallback rescued it -- i.e. steps that WOULD have had all six
    # rays read zero before 2026-09-13. ray_origin_blind counts steps that were
    # genuinely off-graph and still read zero.
    #
    # These exist to test a specific claim rather than argue about it: that
    # blind rays contributed to the dense-training regression. If recovered
    # steps are a fraction of a percent of fleet-steps, they did not.
    ray_origin_recovered: int = 0
    ray_origin_blind: int = 0
    # Ray information content, accumulated per cast. The question these answer:
    # ray_distances is min(cells / 25.0, 1.0), so on a sparse map with long
    # aisles most rays are PINNED AT 1.0 and carry no information at all. If so,
    # rays could lesion harmlessly on sparse maps simply because they are
    # constant there, while mattering under congestion -- and a null lesion
    # result would mean something quite different in each regime.
    #
    # ray_peer_hits vs ray_wall_hits also measures how often the wall/stopped-
    # fleet ambiguity actually arises: a ray reports the same distance for a
    # wall and for a stationary fleet, and peer_velocities only disambiguates
    # them when the peer is moving.
    ray_cells_sum: int = 0
    ray_casts: int = 0
    ray_saturated: int = 0
    ray_blocked_at_zero: int = 0
    ray_peer_hits: int = 0
    # True for the current step if round(current_pos) missed the graph -- i.e.
    # the state that produced six zero rays before the fallback existed. Read by
    # core_warehouse to cross-tabulate against collisions.
    ray_origin_offgrid: bool = False
    # When False, the floor/ceil fallback is skipped and the rays go blind
    # exactly as they did before 2026-09-13. This is the control arm: without
    # it, any post-fix measurement is of "fleet at a dead-end overshoot", not of
    # "fleet with blind rays", and those are different claims.
    ray_origin_fallback: bool = True
    use_orientation: bool = False
    
    G: Any = None 
    grid_pos_dict: Dict[Tuple[int, int, int], str] = field(default_factory=dict)

    direction: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    last_pos: Optional[np.ndarray] = field(default=None)
    last_action: int = 0
    trajectory_history: List[str] = field(default_factory=list)
    # NEW: Tabu memory for fatal actions. Maps (X,Y,Z) to a Set of banned action IDs.
    tabu_actions: Dict[Tuple[int, int, int], Set[int]] = field(default_factory=dict)

    # Precomputed once per environment by core_warehouse.py via build_spatial_indices()
    # and shared (same object) across every FleetNode -- see is_structurally_valid().
    # Left optional so a FleetNode can still be constructed standalone (e.g. in tests)
    # without them; it just falls back to the original full-grid scan in that case.
    aisle_index: Optional[Dict[int, Dict[Tuple[int, int], List[Tuple[int, str]]]]] = None
    # Class-level latch: the sortedness check below runs once per process, not
    # once per fleet. Set to False in a test to re-arm it.
    _aisle_index_checked: bool = False
    coords_by_id: Optional[Dict[str, Tuple[int, int, int]]] = None

    # Precomputed once (not per-episode) by main_runner_warehouse.py via
    # precompute_goal_distances(), specific to THIS fleet's own fixed goal node --
    # {node_id: hop_distance_to_goal}. Left optional so a FleetNode can still be
    # constructed standalone without it; get_graph_distance_to_goal() falls back to
    # straight-line Manhattan distance in that case.
    goal_distance_map: Optional[Dict[str, int]] = None

    # Which shared-pool goal this fleet is CURRENTLY pursuing -- None for the
    # original fixed-mission mode (goal_pos never changes, this is just unused).
    # Set and kept current by retarget_to_nearest_unclaimed() in the shared-pool
    # mode -- see core_warehouse.py's step() for where staleness gets checked.
    current_goal_id: Optional[str] = None

    # --- NEW: Store the distance at spawn for the aggressive deadline ---
    initial_graph_distance: float = field(init=False)
    

    def __post_init__(self):
        self.current_pos = np.array(self.current_pos, dtype=np.float32)
        self.goal_pos = np.array(self.goal_pos, dtype=np.float32)
        self.warehouse_bounds = np.array(self.warehouse_bounds, dtype=np.float32)
        self.direction = np.array(self.direction, dtype=np.float32)
        if self.last_pos is None:
            self.last_pos = self.current_pos.copy()

        # is_structurally_valid() bisects the aisle line instead of sorting it
        # per call. That is only correct if the index arrived sorted, which
        # build_spatial_indices() guarantees. A HAND-BUILT index would now give
        # silently WRONG answers -- letting a fleet through a shelf, or refusing
        # a legal move -- where the old code merely paid to re-sort it.
        #
        # A wrong answer here does not raise, so it has to be caught at
        # construction. Sampled, not exhaustive: a handful of lines is enough to
        # catch a builder that never sorts, and this runs once per fleet.
        if self.aisle_index is not None and not FleetNode._aisle_index_checked:
            FleetNode._aisle_index_checked = True
            for _ax in (0, 1, 2):
                for _i, _line in enumerate(self.aisle_index[_ax].values()):
                    if _i >= 8:
                        break
                    _vals = [v for v, _ in _line]
                    if _vals != sorted(_vals):
                        raise ValueError(
                            "aisle_index is not sorted by position along the move "
                            "axis. is_structurally_valid() bisects it and will "
                            "return wrong answers. Build it with "
                            "node_warehouse.build_spatial_indices(), which sorts, "
                            "rather than by hand."
                        )

        # --- NEW: Calculate and lock in the initial graph distance ---
        self.initial_graph_distance = self.get_graph_distance_to_goal()

    @property
    def velocity(self) -> np.ndarray:
        return self.direction * self.speed

    def get_relative_goal_displacement(self) -> np.ndarray:
        raw_displacement = self.goal_pos - self.current_pos
        scaled_displacement = raw_displacement / np.maximum(self.warehouse_bounds, 1e-6)
        return np.clip(scaled_displacement, -1.0, 1.0).astype(np.float32)

    def get_manhattan_distance_to_goal(self) -> float:
        return float(np.sum(np.abs(self.goal_pos - self.current_pos)))

    def get_progress_fraction(self) -> float:
        """
        How much of this fleet's ORIGINAL journey (per initial_graph_distance,
        locked in at spawn) is already behind it: 0.0 = just started, 1.0 = at
        the goal. Used for the "final approach" push -- see core_warehouse.py's
        step(), where a fleet this close to done gets extra help pushing through
        warning-zone caution instead of being throttled by it.

        In shared-pool mode, "spawn" here really means "when I last locked onto
        my current target" -- retarget_to_nearest_unclaimed() resets
        initial_graph_distance every time a fleet acquires a new target, so this
        always reflects progress toward whatever it's CURRENTLY pursuing, not a
        single value fixed for the whole episode.
        """
        if self.initial_graph_distance <= 1e-6:
            # Started at (or essentially at) the goal -- degenerate case, treat
            # as fully "arrived" rather than dividing by ~zero.
            return 1.0
        current = self.get_graph_distance_to_goal()
        return float(np.clip(1.0 - (current / self.initial_graph_distance), 0.0, 1.0))

    def retarget_to_nearest_unclaimed(
        self,
        goal_positions: Dict[str, np.ndarray],
        goal_distance_maps: Dict[str, Dict[str, int]],
        claimed_goal_ids: Set[str],
        reserved_goal_ids: Optional[Set[str]] = None,
    ) -> bool:
        """
        Shared-pool mode: scans every goal NOT in claimed_goal_ids and locks
        onto whichever is nearest by true graph distance from the fleet's
        CURRENT position. Called once at spawn, and again every time this fleet
        either claims a goal (needs its next one) or discovers its current
        target was claimed by a peer first (see core_warehouse.py's step() for
        the staleness check that triggers the latter).

        Deliberately reactive, not continuously re-optimizing: a fleet keeps
        its committed target for as long as it remains unclaimed, even if a
        marginally closer one opens up elsewhere. Re-evaluating every step
        risks thrashing -- a fleet equidistant between two goals could flip
        back and forth and never actually finish either one. Committing until
        forced to replan (target gone) gets the responsiveness without that
        failure mode.

        Reuses the exact same per-goal precomputed distance maps
        (precompute_goal_distances() in this file) that fixed-mission mode
        uses -- just looked up per pool-goal here instead of per-fleet, since
        goal LOCATIONS are still fixed and known in advance; only which fleet
        ends up assigned to which is dynamic now.

        Returns True if a target was found and set (goal_pos, goal_distance_map,
        current_goal_id, and initial_graph_distance all updated together).
        Returns False if every goal in the pool is already claimed -- caller
        should freeze the fleet in that case, there's nothing left to do.
        """
        # A goal is available only if it is neither already CLAIMED (a peer
        # physically arrived) nor RESERVED (a peer is currently en route to it).
        #
        # BUG THIS FIXES: this used to filter on claimed_goal_ids alone, so
        # nothing stopped N fleets from all committing to the same goal at once.
        # Measured on a 25-fleet layout: only 19 of 25 goals were targeted at
        # spawn, 6 fleets were chasing a goal they could not possibly win, and 6
        # goals had nobody heading for them -- before a single step was taken.
        # Averaged over 25 random layouts it was 9.5 of 25 fleets (38%) on a
        # doomed journey. Each loser then retargeted, often onto another
        # contested goal, and cascaded. It also silently corrupted the
        # mean_fraction_closed metric, because every retarget resets
        # initial_graph_distance and rebases the fraction to ~0.
        #
        # reserved_goal_ids=None preserves the old free-for-all behaviour for
        # any caller that hasn't been updated.
        reserved = reserved_goal_ids or set()
        unclaimed = [
            gid for gid in goal_positions
            if gid not in claimed_goal_ids and gid not in reserved
        ]
        if not unclaimed:
            return False

        curr_tuple = tuple(int(v) for v in np.round(self.current_pos).astype(int))
        curr_node_id = self.grid_pos_dict.get(curr_tuple) if self.grid_pos_dict else None

        best_gid: Optional[str] = None
        best_dist: Optional[float] = None
        for gid in unclaimed:
            dmap = goal_distance_maps.get(gid)
            dist = dmap.get(curr_node_id) if (dmap is not None and curr_node_id is not None) else None
            if dist is None:
                # Fall back to straight-line distance if graph distance isn't
                # available for this (position, goal) pair -- shouldn't
                # normally happen for a validly-positioned fleet, but keeps
                # this robust rather than crashing outright.
                dist = float(np.sum(np.abs(goal_positions[gid] - self.current_pos)))
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_gid = gid

        self.current_goal_id = best_gid
        self.goal_pos = goal_positions[best_gid].copy()
        self.goal_distance_map = goal_distance_maps.get(best_gid)
        self.initial_graph_distance = self.get_graph_distance_to_goal()
        return True

    def get_graph_distance_to_goal(self) -> float:
        """
        Distance-to-goal measured through the ACTUAL warehouse graph (shortest path
        in edges), not straight-line Manhattan distance -- see
        precompute_goal_distances() for why this matters.

        Since current_pos can be fractional (mid-move, especially under affordance
        braking), this linearly interpolates between the two grid nodes bounding
        the fleet's position along whichever single axis it's currently traversing
        (at most one axis is ever fractional, since movement is always along one
        axis at a time). That keeps this exactly as responsive to every single
        step -- including tiny braked ones -- as the old Manhattan metric was,
        instead of only updating once every couple of steps when a full edge
        completes.
        """
        if self.goal_distance_map is None:
            # No precomputed map available (e.g. a FleetNode built standalone,
            # outside core_warehouse.py) -- fall back rather than crash.
            return self.get_manhattan_distance_to_goal()

        pos = self.current_pos
        floor_pos = np.floor(pos).astype(int)
        ceil_pos = np.ceil(pos).astype(int)

        node_floor = self.grid_pos_dict.get(tuple(int(v) for v in floor_pos))
        node_ceil = self.grid_pos_dict.get(tuple(int(v) for v in ceil_pos))

        dist_floor = self.goal_distance_map.get(node_floor) if node_floor is not None else None
        dist_ceil = self.goal_distance_map.get(node_ceil) if node_ceil is not None else None

        if dist_floor is None and dist_ceil is None:
            # Neither bounding node is in the precomputed map -- shouldn't happen
            # for a validly-positioned fleet, but fall back safely if it ever does.
            return self.get_manhattan_distance_to_goal()
        if dist_floor is None:
            return float(dist_ceil)
        if dist_ceil is None:
            return float(dist_floor)
        if node_floor == node_ceil:
            # Exactly on a grid node -- no interpolation needed.
            return float(dist_floor)

        frac = float(np.max(pos - floor_pos))  # the one nonzero fractional component
        return dist_floor + frac * (dist_ceil - dist_floor)

    def is_structurally_valid(self, proposed_pos: np.ndarray, action: int) -> bool:
        """
        ULTRA-STRICT CONTINUOUS PHYSICS ENGINE.
        Verifies if the proposed continuous coordinate lies on a valid NetworkX edge.
        """
        if action == 0:
            return True
            
        move_axis = 0 if action in [1, 2] else 1 if action in [3, 4] else 2
        ax1, ax2 = [i for i in [0, 1, 2] if i != move_axis]
        
        if self.aisle_index is not None:
            # FAST PATH: O(1) dict lookup + a scan of just this one aisle line,
            # instead of scanning every node in the entire warehouse.
            #
            # Grid coordinates are integers, and the tolerance band below (0.1) is
            # far narrower than the 1.0 spacing between integers, so at most ONE
            # integer can ever be within 0.1 of a given proposed_pos value. That
            # integer, if it exists, is exactly round(proposed_pos[axis]) -- so
            # checking that single rounded candidate against the same "< 0.1" test
            # the original scan used is provably equivalent to the full scan: it
            # can never match something the scan would have rejected, or miss
            # something the scan would have found.
            cand_ax1 = int(round(float(proposed_pos[ax1])))
            cand_ax2 = int(round(float(proposed_pos[ax2])))
            
            if abs(cand_ax1 - proposed_pos[ax1]) < 0.1 and abs(cand_ax2 - proposed_pos[ax2]) < 0.1:
                # NO list() COPY, and NO re-sort below.
                #
                # PROFILE 2026-09-13, 200 fleets / 120k nodes / 80 steps:
                #   {method 'sort' of 'list'}    6,217,503 calls   1450.6 s
                #   node_warehouse:520 <lambda>  1,078,227,146 calls  1358.5 s
                # = 2809 s of 4760 s total, FIFTY-NINE PERCENT of step time,
                # spent re-sorting a list that build_spatial_indices() had
                # already sorted at construction (see its final loop) and that
                # is never mutated afterwards. The copy then paid ~173 element
                # copies per call on top, 6.2 M times, for a read-only scan.
                #
                # The billion lambda calls are the sort KEY being evaluated once
                # per element per sort: 1.078e9 / 6.22e6 = 173, which is the
                # mean aisle-line length on this map.
                aligned_nodes = self.aisle_index[move_axis].get((cand_ax1, cand_ax2), _EMPTY_LINE)
            else:
                aligned_nodes = _EMPTY_LINE
        else:
            # Fallback: original full-grid scan (used only if no index was supplied,
            # e.g. a FleetNode built standalone outside core_warehouse.py).
            aligned_nodes = []
            for coords_tuple, node_id in self.grid_pos_dict.items():
                if abs(coords_tuple[ax1] - proposed_pos[ax1]) < 0.1 and \
                   abs(coords_tuple[ax2] - proposed_pos[ax2]) < 0.1:
                    aligned_nodes.append((coords_tuple[move_axis], node_id))
            # This path builds a FRESH unsorted list, so it still has to sort.
            # It is the no-index fallback for a standalone FleetNode and is not
            # on any hot path.
            aligned_nodes.sort(key=_FIRST)

        if not aligned_nodes:
            return False # Strayed off the grid lines into a shelf

        # Find the two nodes bounding our proposed position.
        #
        # WAS a linear scan of the whole aisle line (~173 entries on this map):
        #     for val, nid in aligned_nodes:
        #         if val <= p_val + 0.1: node_behind = nid
        #         if val >= p_val - 0.1 and node_ahead is None: node_ahead = nid
        # which is O(n) for what is, on a sorted list, two O(log n) lookups.
        # node_behind is the LAST entry with val <= p+0.1 and node_ahead is the
        # FIRST with val >= p-0.1, which is exactly bisect_right / bisect_left.
        # itemgetter is C-level, so the key costs ~8 calls per lookup rather
        # than the 173 the old sort key cost per call.
        p_val = proposed_pos[move_axis]

        i_behind = bisect_right(aligned_nodes, p_val + 0.1, key=_FIRST) - 1
        if i_behind < 0:
            return False # Flew out of bounds
        i_ahead = bisect_left(aligned_nodes, p_val - 0.1, key=_FIRST)
        if i_ahead >= len(aligned_nodes):
            return False # Flew out of bounds

        node_behind = aligned_nodes[i_behind][1]
        node_ahead = aligned_nodes[i_ahead][1]
            
        # If we are between two nodes, there MUST be an edge between them
        if node_behind != node_ahead:
            if not self.G.has_edge(node_behind, node_ahead):
                return False
                
        return True

    def apply_discrete_action(self, action: int):
        self.last_action = action
        self.last_pos = self.current_pos.copy()

        proposed_delta = ACTION_DELTAS.get(action, np.zeros(3, dtype=np.float32))
        proposed_pos = self.current_pos + (proposed_delta * self.speed)
        
        if not self.is_structurally_valid(proposed_pos, action):
            # Hit a wall! Kill momentum.
            self.direction = np.zeros(3, dtype=np.float32)
        else:
            # Move is valid
            self.direction = proposed_delta.copy()
            self.current_pos = proposed_pos
            
        # VDA 5050 Logging
        actual_tuple = tuple(np.round(self.current_pos).astype(int))
        if actual_tuple in self.grid_pos_dict:
            node_id_string = self.grid_pos_dict[actual_tuple]
            if not self.trajectory_history or self.trajectory_history[-1] != node_id_string:
                self.trajectory_history.append(node_id_string)

    def get_valid_action_mask(self, reference_speed: float = 0.5) -> np.ndarray:
        """
        Which of the 7 actions are actually structurally valid FROM the fleet's
        CURRENT position -- reuses the exact same is_structurally_valid() check
        that governs whether a chosen action actually moves the fleet, and the
        same shared ACTION_DELTAS apply_discrete_action() uses, so "valid for
        masking" and "valid when applied" can never disagree.

        BUG THIS FIXES: exploration previously sampled uniformly across all 7
        actions with no idea which were structurally possible from the current
        node. On the real warehouse graph (11230 edges over 9900 nodes, average
        degree ~2.27), a typical node only has ~2 of 6 possible directions
        actually open -- so roughly 70% of random exploratory actions were
        landing on invalid edges, doing nothing, and teaching the network
        nothing about real navigation while still costing a full step and the
        idle penalty.

        reference_speed: fixed at the system's max (base_speed) rather than this
        node's current (possibly braked) speed, which isn't finalized until
        later in step() -- action selection happens before per-node braking is
        computed. This doesn't change the verdict: whether an edge exists in a
        given direction at all is a fact about the graph, not about how far
        along that edge a given step's speed would travel, and no speed this
        system ever uses (0.05 to 0.5) is large enough to overshoot past the
        immediately adjacent node (grid nodes are 1 unit apart).

        Idle (action 0) is always valid -- it's never structurally constrained.
        """
        mask = np.zeros(len(ACTION_DELTAS), dtype=bool)
        mask[0] = True
        for action in range(1, len(ACTION_DELTAS)):
            proposed_pos = self.current_pos + (ACTION_DELTAS[action] * reference_speed)
            mask[action] = self.is_structurally_valid(proposed_pos, action)
        return mask

    def sense_6_axis_rays(self, all_fleets: List["FleetNode"]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Six axis-aligned rays: how far the corridor is clear, and what stopped it.

        REWRITTEN 2026-09-13 to walk the GRAPH instead of tracing coordinates.

        WHAT IT USED TO DO
            trace_pos = current_pos.copy()
            for _ in range(int(max_vision_range * (5.0 / speed))):   # 100 iters
                next_pos = trace_pos + r_dir * speed                 # 0.5 cells
                if not self.is_structurally_valid(next_pos, ray_idx+1): break
                distance_found += speed
                ...

        Three problems, found by the 2026-09-13 profile at 200 fleets:

        1. IT WAS THE LARGEST TERM, and it GREW as the map emptied. Rays stop
           when they hit something, so fewer fleets means longer traces. Over 200
           warmup steps active fleets fell 190 -> 40 and field work fell ~70%,
           while step cost ROSE 22%: this function was absorbing the slack.
           16.9s of 32.6s profiled, 52%.

        2. HALF THE WORK WAS DISCARDED. The loop ran to 50 cells but the feature
           is min(cells / 25.0, 1.0), pinned at 1.0 beyond 25. Everything past
           halfway was computed and thrown away by the clip. Capping the walk at
           ray_range = 25 therefore costs NO information.

        3. TWO is_structurally_valid CALLS PER CELL. At speed 0.5 it stepped in
           half-cells, and each call does two bisects over the aisle line plus an
           edge check -- to answer a question the graph answers with one dict
           lookup.

        WHAT IT DOES NOW: walks node to node through real edges. One dict lookup
        and one has_edge per cell, no validity check at all, capped at ray_range.

        Note that naive unit-stepping in COORDINATES would have been wrong, not
        just fast: landing exactly on a node makes is_structurally_valid's
        `behind` and `ahead` the same node, so the connecting edge is never
        checked and a ray would tunnel straight through a gap. Walking the graph
        cannot do that -- it only moves where an edge exists.

        SEMANTIC DELTA: distance is now counted in whole cells from the fleet's
        cell, where it used to accumulate in 0.5 increments from its continuous
        position. Sub-cell precision on "how far is this aisle clear" was noise.
        Dimensions are unchanged (6 + 18 + 18), so checkpoints still load.
        """
        ray_dirs = ((1, 0, 0), (-1, 0, 0),
                    (0, 1, 0), (0, -1, 0),
                    (0, 0, 1), (0, 0, -1))

        fleet_pos_map = {
            tuple(np.round(fleet.current_pos).astype(int)): fleet
            for fleet in all_fleets if fleet.id != self.id
        }

        ray_distances = np.zeros(6, dtype=np.float32)
        peer_velocities = np.zeros((6, 3), dtype=np.float32)
        peer_displacements = np.zeros((6, 3), dtype=np.float32)

        pos = self.current_pos
        start_cell = (int(round(float(pos[0]))),
                      int(round(float(pos[1]))),
                      int(round(float(pos[2]))))
        grid = self.grid_pos_dict
        start_id = grid.get(start_cell) if grid else None

        self.ray_origin_offgrid = (start_id is None)

        if start_id is None and grid and self.ray_origin_fallback:
            # ROUNDING MISSED THE GRAPH. Fall back to the nearest node the fleet
            # is actually straddling.
            #
            # BUG THIS FIXES, found by the equivalence test: a fleet can sit half
            # a cell PAST the last column. is_structurally_valid permits it --
            # moving +X from x=59 proposes 59.5, whose bounding nodes are both
            # node 59, so `behind == ahead` and the move is legal. But
            # round(59.5) is 60 under banker's rounding, 60 is not a node, and
            # every ray went blind. Measured on a 60-wide map: a fleet at x=59.5
            # reported 0 in all six directions while the original correctly saw
            # 25 cells down -X.
            #
            # Blind rays at the map edge are exactly where a fleet most needs to
            # see, so this is worth the two extra dict lookups on a rare path.
            frac = (abs(float(pos[0]) - round(float(pos[0]))),
                    abs(float(pos[1]) - round(float(pos[1]))),
                    abs(float(pos[2]) - round(float(pos[2]))))
            axis = 0 if frac[0] >= frac[1] and frac[0] >= frac[2] else (
                   1 if frac[1] >= frac[2] else 2)
            for _cand in (int(math.floor(float(pos[axis]))),
                          int(math.ceil(float(pos[axis])))):
                probe = list(start_cell)
                probe[axis] = _cand
                probe_t = (probe[0], probe[1], probe[2])
                if probe_t in grid:
                    start_cell = probe_t
                    start_id = grid[probe_t]
                    self.ray_origin_recovered += 1
                    break

        if start_id is None:
            self.ray_origin_blind += 1
            # Genuinely off-graph: inside a rack. Every ray reads 0, which is
            # what the original produced too -- its first is_structurally_valid
            # would have failed immediately.
            return ray_distances, peer_velocities, peer_displacements

        G = self.G
        reach = int(self.ray_range)

        # OFF-AXIS GUARD -- reproduces the original semantics exactly, and its
        # absence was the one real regression the equivalence test found.
        #
        # is_structurally_valid() refuses any move whose two PERPENDICULAR
        # coordinates are not within 0.1 of an integer. So a fleet at x = 0.5,
        # straddling an edge in X, cannot move in Y at all -- and the old ray,
        # which called is_structurally_valid on its first step, reported 0 for
        # both Y rays. get_valid_action_mask() agrees: it masks those actions out.
        #
        # Rounding the position to a cell and walking from there would report
        # "17 cells clear in -Y" for a fleet that cannot take a single step that
        # way. The ray must agree with the movement engine about which
        # directions exist, or the network reads clearance in directions its own
        # action mask forbids.
        off = (abs(float(pos[0]) - round(float(pos[0]))),
               abs(float(pos[1]) - round(float(pos[1]))),
               abs(float(pos[2]) - round(float(pos[2]))))

        for ray_idx in range(6):
            move_axis = ray_idx >> 1          # 0,1 -> X   2,3 -> Y   4,5 -> Z
            if any(off[a] >= 0.1 for a in (0, 1, 2) if a != move_axis):
                continue                      # ray stays 0, as before

            dx, dy, dz = ray_dirs[ray_idx]
            cur = start_cell
            cur_id = start_id
            cells = 0
            peer_hit = None

            for _ in range(reach):
                nxt = (cur[0] + dx, cur[1] + dy, cur[2] + dz)
                nxt_id = grid.get(nxt)
                if nxt_id is None:
                    break                      # no node there: wall
                if G is not None and not G.has_edge(cur_id, nxt_id):
                    break                      # node exists but no track to it
                cells += 1
                cur = nxt
                cur_id = nxt_id
                peer = fleet_pos_map.get(nxt)
                if peer is not None:
                    peer_hit = peer
                    break

            if self.ray_transform == "smooth":
                ray_distances[ray_idx] = cells / (cells + self.ray_softness)
            else:
                ray_distances[ray_idx] = cells / 25.0 if cells < 25 else 1.0

            self.ray_casts += 1
            self.ray_cells_sum += cells
            if cells >= 25:
                self.ray_saturated += 1
            elif cells == 0:
                self.ray_blocked_at_zero += 1
            if peer_hit is not None:
                self.ray_peer_hits += 1

            if peer_hit is not None:
                peer_velocities[ray_idx] = peer_hit.direction.copy()
                peer_displacements[ray_idx] = peer_hit.get_relative_goal_displacement()

        return ray_distances, peer_velocities, peer_displacements

    def get_goal_gradient(self) -> np.ndarray:
        """
        THE ROUTING SIGNAL. Six numbers, one per movement action (+X, -X, +Y, -Y,
        +Z, -Z): how much this fleet's TRUE graph-distance-to-goal would drop if
        it took that action. +1 = a full step closer, -1 = a full step further,
        0 = structurally invalid, off the map, or exactly neutral.

        WHY THIS EXISTS: core_warehouse.py rewards movement as
        (old_dist - new_dist) * movement_reward_multiplier, measured on
        get_graph_distance_to_goal(). But nothing in the state vector could
        observe that quantity. The closest thing was
        get_relative_goal_displacement(), which is the EUCLIDEAN offset to the
        goal -- and precompute_goal_distances()'s own docstring already
        establishes that these two disagree badly in this warehouse (fleets 19,
        11, 23, 9, 6 need 1.4x-2.4x more graph hops than their Manhattan
        distance implies). So the policy was being rewarded for reducing a
        number it had no way to perceive, and had to infer routing around
        shelving from a 5-cell local view of a 9900-node graph.

        Observed consequence in the 11-episode run: the ONLY fleets that ever
        reached a goal were the ones that spawned essentially on top of one --
        fleets 14, 5, 20, 1, 3 completing at byte-identical coordinates in
        episode after episode, while 13 of 25 fleets never completed once, in
        any episode, at any epsilon from 0.49 down to 0.010. That is the
        signature of a random walk, not a policy.

        The distance map this reads is a full BFS from every goal, already
        precomputed once per run by precompute_goal_distances(). This is a
        dictionary lookup per action, not a search.

        What it changes conceptually: the GNN stops having to solve routing --
        a problem BFS already solved optimally, and not what FLOWRRA is about --
        and starts doing what it is actually for, which is resolving multi-agent
        contention on top of a route it can already see.
        """
        grad = np.zeros(6, dtype=np.float32)
        if self.goal_distance_map is None or not self.grid_pos_dict:
            return grad

        here = self.get_graph_distance_to_goal()

        for i, action in enumerate(range(1, 7)):
            delta = ACTION_DELTAS[action]

            # Validity is checked against the move the fleet would ACTUALLY make
            # (one base_speed increment, which may be a half-edge)...
            if not self.is_structurally_valid(self.current_pos + delta * self.speed, action):
                continue  # wall -> 0.0, indistinguishable from "no help", which is correct

            # ...but the distance lookup probes the bounding GRID CELL in that
            # direction, because the BFS map is keyed by grid node and a
            # half-step (base_speed=0.5) lands between two of them.
            #
            # TWO ROUNDING BUGS THIS AVOIDS, both silent:
            #   a) probing `current_pos + delta * speed` and rounding: at x=10.5
            #      np.round gives 10, not 11 (numpy rounds halves to even), so
            #      every probe resolved back to the cell the fleet was already
            #      standing on and the whole gradient came out as six zeros.
            #   b) probing `round(current_pos) + delta`: at (20, 19.5) the fleet
            #      is BETWEEN cells 19 and 20, but rounding snaps it onto 20 --
            #      the goal itself -- so the +Y probe looked at cell 21, fell off
            #      the grid, and every action scored <= 0. The fleet then froze
            #      permanently half a step from home.
            # Taking ceil/floor along the axis of travel is exact in both cases:
            # it always names the cell the fleet would actually arrive at next.
            axis = int(np.argmax(np.abs(delta)))
            probe = np.round(self.current_pos).astype(np.float64)
            if delta[axis] > 0:
                probe[axis] = np.ceil(self.current_pos[axis])
                if probe[axis] == self.current_pos[axis]:
                    probe[axis] += 1          # already exactly on a node
            else:
                probe[axis] = np.floor(self.current_pos[axis])
                if probe[axis] == self.current_pos[axis]:
                    probe[axis] -= 1

            node_id = self.grid_pos_dict.get(tuple(int(v) for v in np.round(probe)))
            if node_id is None:
                continue
            there = self.goal_distance_map.get(node_id)
            if there is None:
                continue  # unreachable from this goal's BFS tree

            # Adjacent grid cells differ by exactly one hop, so this is already
            # in [-1, 1] for a fleet sitting on a node; the clip only guards the
            # mid-move case, where `here` is interpolated.
            grad[i] = float(np.clip(here - float(there), -1.0, 1.0))

        return grad

    def get_situation_features(self) -> np.ndarray:
        """
        Six flags describing the fleet's CURRENT SITUATION rather than its geometry.

        These are set by the orchestrator each step (see core_warehouse.py) and
        default to zero, so a FleetNode used outside the orchestrator still
        produces a correctly-sized state vector.

        nearest_peer_proximity is 0 when the nearest mobile peer is at or beyond
        the warning threshold and approaches 1 at the collision boundary -- the
        same scaling the safety reward uses, so the feature and its reward speak
        the same language.
        """
        return np.array([
            float(getattr(self, "sf_is_rescuer", 0.0)),
            float(getattr(self, "sf_orders_carried", 0.0)),
            float(getattr(self, "sf_on_pickup_cell", 0.0)),
            float(getattr(self, "sf_in_warning", 0.0)),
            float(getattr(self, "sf_in_deadlock", 0.0)),
            float(getattr(self, "sf_peer_proximity", 0.0)),
        ], dtype=np.float32)

    def get_state_vector(self, all_fleets: List["FleetNode"]) -> np.ndarray:
        self_direction = self.direction.copy()
        self_displacement = self.get_relative_goal_displacement()

        ray_dists, peer_vels, peer_disps = self.sense_6_axis_rays(all_fleets=all_fleets)

        state_components = [
            self_direction,                     
            self_displacement,       # EUCLIDEAN offset -- kept, but see below
            ray_dists,                          
            peer_vels.flatten(),                
            peer_disps.flatten(),
            # +6 dims. The true graph-distance gradient over the 6 movement
            # actions -- the single piece of information the reward function
            # measures and the state vector previously did not contain. See
            # get_goal_gradient()'s docstring for the full diagnosis.
            self.get_goal_gradient(),
            # +6 dims. SITUATION FEATURES, without which two of the five reward
            # heads are learning constants.
            #
            # The rescue head cannot learn anything if a handover mission is
            # indistinguishable from an ordinary delivery -- the +25 would arrive
            # at a state that looks like every other arrival, so there is nothing
            # for it to attach the value to. Likewise the safety head needs to
            # know it is IN danger, not merely infer it from raw ray distances.
            #
            # Order: [is_rescuer, orders_carried, on_pickup_cell,
            #         in_warning_zone, in_deadlock, nearest_peer_proximity]
            self.get_situation_features(),
        ]

        if self.use_orientation:
            heading = np.arctan2(self.direction[1], self.direction[0]) / np.pi
            state_components.append(np.array([heading], dtype=np.float32))

        return np.concatenate(state_components).astype(np.float32)

    def state_layout(self) -> Dict[str, Tuple[int, int]]:
        """
        Named [start, end) index ranges into the BASE state vector, derived from
        the same component widths get_state_vector() concatenates.

        ADDED 2026-09-13 for the lesion harness. Deliberately computed rather
        than hardcoded: the base vector has gained components twice already
        (the 6 goal-gradient dims, then the 6 situation features), and a
        hardcoded table of offsets would have silently pointed at the wrong
        dimensions after either change. Hardcoding these indices in the harness
        would make a lesion experiment report confident numbers about whichever
        dims happened to sit at those offsets.

        The density block is NOT covered here -- it is concatenated onto the end
        by core_warehouse.py, so the harness adds it as
        ("density", base_len, base_len + density_dim).
        """
        widths = [
            ("self_direction", 3),
            ("self_displacement", 3),
            ("ray_distances", 6),
            ("peer_velocities", 18),
            ("peer_displacements", 18),
            ("goal_gradient", 6),
            ("situation_features", 6),
        ]
        if self.use_orientation:
            widths.append(("heading", 1))

        layout: Dict[str, Tuple[int, int]] = {}
        cursor = 0
        for name, w in widths:
            layout[name] = (cursor, cursor + w)
            cursor += w

        # Convenience group: everything derived from the 6-axis ray cast. This
        # is the block the "how much does the policy use its rays" lesion
        # switches off, and it is 42 of the 60 base dims.
        layout["rays_all"] = (layout["ray_distances"][0], layout["peer_displacements"][1])
        layout["_base_len"] = (0, cursor)
        return layout