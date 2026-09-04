"""
node_Warehouse.py

Discrete 3D Manhattan Agent representation for FLOWRRA.
Handles state representation, 6-axis ray casting, relative goal displacement,
and peer-to-peer data integration for discrete warehouse graph navigation.
"""

from dataclasses import dataclass, field
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
                aligned_nodes = list(self.aisle_index[move_axis].get((cand_ax1, cand_ax2), []))
            else:
                aligned_nodes = []
        else:
            # Fallback: original full-grid scan (used only if no index was supplied,
            # e.g. a FleetNode built standalone outside core_warehouse.py).
            aligned_nodes = []
            for coords_tuple, node_id in self.grid_pos_dict.items():
                if abs(coords_tuple[ax1] - proposed_pos[ax1]) < 0.1 and \
                   abs(coords_tuple[ax2] - proposed_pos[ax2]) < 0.1:
                    aligned_nodes.append((coords_tuple[move_axis], node_id))
                
        if not aligned_nodes:
            return False # Strayed off the grid lines into a shelf
            
        aligned_nodes.sort(key=lambda x: x[0])
        
        # Find the two nodes bounding our proposed position
        p_val = proposed_pos[move_axis]
        node_behind = None
        node_ahead = None
        
        for val, nid in aligned_nodes:
            if val <= p_val + 0.1:
                node_behind = nid
            if val >= p_val - 0.1 and node_ahead is None:
                node_ahead = nid
                
        if node_behind is None or node_ahead is None:
            return False # Flew out of bounds
            
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
        ray_dirs = [
            np.array([1, 0, 0]), np.array([-1, 0, 0]), 
            np.array([0, 1, 0]), np.array([0, -1, 0]), 
            np.array([0, 0, 1]), np.array([0, 0, -1])
        ]

        fleet_pos_map = {
            tuple(np.round(fleet.current_pos).astype(int)): fleet
            for fleet in all_fleets if fleet.id != self.id
        }

        ray_distances = np.zeros(6, dtype=np.float32)
        peer_velocities = np.zeros((6, 3), dtype=np.float32)
        peer_displacements = np.zeros((6, 3), dtype=np.float32)

        for ray_idx, r_dir in enumerate(ray_dirs):
            distance_found = 0.0
            peer_hit = None
            
            trace_pos = self.current_pos.copy()
            # Step forward in increments of speed to trace the physical graph continuously
            # max(speed, eps): this divides by speed to size the ray budget, so a
            # stationary fleet would raise ZeroDivisionError. Nothing set speed
            # to exactly 0 before (affordance braking floors it at 0.1), but a
            # stopped or held fleet legitimately can be.
            _ray_speed = max(float(self.speed), 1e-3)
            for _ in range(int(self.max_vision_range * (5.0 / _ray_speed))): 
                next_pos = trace_pos + (r_dir * self.speed)
                
                if not self.is_structurally_valid(next_pos, action=ray_idx+1):
                    break # Hit a wall
                    
                distance_found += self.speed
                trace_pos = next_pos
                
                check_tuple = tuple(np.round(trace_pos).astype(int))
                if check_tuple in fleet_pos_map:
                    peer_hit = fleet_pos_map[check_tuple]
                    break
                    
            # Normalize distance (Assuming max typical ray length ~ 25m)
            ray_distances[ray_idx] = float(min(distance_found / 25.0, 1.0))

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