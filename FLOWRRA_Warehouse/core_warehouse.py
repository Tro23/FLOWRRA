"""
core_Warehouse.py

Streamlined FLOWRRA Orchestrator for discrete warehouse AGV fleets.
Integrates dual-zone loop integrity, Poisson density fields, affordance braking,
and highly targeted Manhattan reward gradients.
"""

from typing import Any, Dict, List, Optional, Set, Tuple
import random
import numpy as np

from config_warehouse import CONFIG
from agent_warehouse import GNNAgent
from node_warehouse import (FleetNode, build_spatial_indices, precompute_goal_distances,
                             assign_goals_optimally)
from loop_warehouse import WarehouseLoop
from density_warehouse import WarehouseDensityField
from recovery_warehouse import WarehouseRecovery


class FLOWRRA:
    """
    Command center for the discrete warehouse holon.
    """

    def __init__(
        self, 
        G: Any, 
        grid_pos_dict: Dict[Tuple[int, int, int], str], 
        fleet_missions: List[Dict[str, Any]], 
        mode: str = "training",
        goal_distance_maps: Optional[Dict[str, Dict[str, int]]] = None,
        shared_pool_mode: bool = False,
        goal_pool: Optional[Dict[str, np.ndarray]] = None,
    ):
        self.G = G
        # --- THE DICTIONARY FIX ---
        # Convert the Pandas string-keyed dictionary into the Tuple-keyed format 
        # that the physics engine, ray-casters, and WFC recoveries actually expect!
        self.grid_pos_dict = {}
        for node_id, coords in grid_pos_dict.items():
            if isinstance(coords, dict) and 'X' in coords:
                self.grid_pos_dict[(int(coords['X']), int(coords['Y']), int(coords['Z']))] = str(node_id)
            else:
                self.grid_pos_dict[node_id] = coords
        # --------------------------

        # Built ONCE here and shared (same objects) across every FleetNode below --
        # see build_spatial_indices()'s docstring for why this matters.
        self.aisle_index, self.coords_by_id = build_spatial_indices(self.grid_pos_dict)

        # PER-MAP normalization bounds, replacing CONFIG's hardcoded (50, 50, 10).
        # See _derive_bounds() and config_warehouse.py's "bounds" comment.
        self.warehouse_bounds = self._derive_bounds()

        # Same one-time-computation principle as above, for the graph-distance-to-goal
        # reward fix (see precompute_goal_distances()'s docstring). main_runner_warehouse.py
        # computes this ONCE (goals are fixed across the whole training run) and passes
        # it in; this fallback only exists so FLOWRRA can still be constructed standalone
        # (tests, scripts) without requiring the caller to precompute it first.
        if goal_distance_maps is not None:
            self.goal_distance_maps = goal_distance_maps
        elif shared_pool_mode:
            # Pool mode: fleet_missions has no per-fleet "goal_node" field to read
            # (goals aren't pre-assigned), so build the same {goal_node_id: {node:
            # hop_distance}} shape from the pool's goal IDs instead -- goal
            # LOCATIONS are still fixed and known in advance, only which fleet
            # ends up at which is dynamic.
            self.goal_distance_maps = precompute_goal_distances(
                G, [{"goal_node": gid} for gid in (goal_pool or {})]
            )
        else:
            self.goal_distance_maps = precompute_goal_distances(G, fleet_missions)

        self.mode = mode

        # Shared-pool mode (see node_warehouse.py's retarget_to_nearest_unclaimed
        # for the actual mechanism): fleets aren't pre-assigned a fixed goal_pos --
        # they claim goals from a shared pool on a first-arrival basis, and keep
        # going ("one or more") until every goal is claimed or the episode ends.
        # goal_pool: {goal_node_id: position} -- REQUIRED when shared_pool_mode=True.
        self.shared_pool_mode = shared_pool_mode
        self.goal_pool: Dict[str, np.ndarray] = goal_pool if goal_pool is not None else {}
        # Which pool goals have been claimed so far this episode -- the actual
        # "not-dones" tracking the cartons-on-aisles framing describes. Coverage
        # (episode success) means this equals the full pool, not "all fleets frozen".
        self.claimed_goals: Set[str] = set()

        # RESERVATION LEDGER {goal_id: fleet_id}. A goal is reserved the moment a
        # fleet commits to it and released when that fleet retargets or claims.
        # claimed_goals alone was not enough: it only records goals a fleet has
        # physically ARRIVED at, so any number of fleets could be simultaneously
        # en route to the same goal, with all but one guaranteed to waste the
        # trip. See node_warehouse.retarget_to_nearest_unclaimed() for the
        # measured cost of not having this.
        self.targeted_goals: Dict[str, str] = {}

        # Per-episode record of whether the policy's chosen action matched the
        # goal gradient's best move. A fresh FLOWRRA is built per episode, so
        # this resets naturally. Read via get_gradient_agreement().
        self._grad_agree: List[float] = []

        # Manhattan radius within which two fleets are linked in the GAT.
        # See _build_adjacency() for why a complete graph was actively harmful.
        self.interaction_radius = float(
            CONFIG["gnn"].get("interaction_radius", 10.0)
        )

        # Per-fleet motion accounting. Separates the two failure modes that both
        # show up as "mean_fraction_closed ~= 0": a fleet that travels hundreds of
        # cells but loops back on itself (wandering -> policy/reward problem) and
        # a fleet that barely moves at all (stalling -> idle or affordance-braking
        # problem). These need opposite fixes, and the journey metric cannot tell
        # them apart. Read via get_fleet_diagnostics().
        self._fleet_travel: Dict[str, float] = {}
        self._fleet_idle: Dict[str, int] = {}
        self._fleet_steps: Dict[str, int] = {}
        self._fleet_speed: Dict[str, float] = {}

        # LIVELOCK ESCAPE. Once every other fleet has frozen, the environment is
        # stationary and a greedy (epsilon=0) policy becomes a deterministic map
        # from a fleet's own position to an action. A deterministic map on a
        # finite set of positions MUST eventually cycle -- and if that cycle does
        # not contain the goal, the fleet orbits forever. This is not a
        # hypothetical: the 400-episode checkpoint deploys to 23/25 with ZERO
        # collisions and ZERO recovery events, byte-identical across repeated
        # runs, with one fleet stranded TWO HOPS from its goal. Training never
        # shows it because eps_min=0.01 never reaches zero, so roughly one random
        # action per 100 steps eventually breaks any cycle.
        #
        # The escape defers to get_goal_gradient(), which is a BFS oracle: greedy
        # argmax on it was verified to reach the goal by an optimal-length path
        # from every tested start. So a stalled fleet handed back to the gradient
        # is guaranteed to make progress, and the override releases as soon as it
        # does.
        self._best_dist: Dict[str, float] = {}
        self._stall_steps: Dict[str, int] = {}
        self._override_until: Dict[str, int] = {}
        self.livelock_patience = int(CONFIG["recovery"].get("livelock_patience", 40))
        self.livelock_escape_steps = int(CONFIG["recovery"].get("livelock_escape_steps", 20))
        self.livelock_overrides = 0
        
        # Initialize Discrete Components
        # NOTE: previously these hardcoded their own literals (1.0, 10, 2.0, ...) instead
        # of reading CONFIG, so editing config_warehouse.py silently had no effect at all.
        self.loop = WarehouseLoop(
            collision_threshold=CONFIG["warehouse"]["collision_threshold"],
            warning_threshold=CONFIG["warehouse"]["warning_threshold"],
        )
        self.density = WarehouseDensityField(
            max_vision_range=CONFIG["warehouse"]["max_vision_range"],
            falloff_radius=CONFIG["density"]["falloff_radius"],
            peer_severity=CONFIG["density"]["peer_severity"],
            projection_steps=CONFIG["density"]["projection_steps"],
            projection_falloff=CONFIG["density"]["projection_falloff"],
            memory_decay_factor=CONFIG["density"]["memory_decay_factor"],
            memory_floor=CONFIG["density"]["memory_floor"],
            memory_cap=CONFIG["density"]["memory_cap"],
            grid_pos_dict=self.grid_pos_dict,   # both already built above, at lines 36 and 40-45
            graph=self.G,
        )
        # The 3-Tier Spatial-Temporal recovery protocol (recovery_warehouse.py) was fully
        # implemented but never instantiated or called anywhere -- fatal collisions were
        # only ever "repaired" by resetting the integrity flag, without ever actually
        # moving the crashed fleets apart. See step() below for where it's now invoked.
        self.frozen_obstacle_severity = CONFIG["density"]["frozen_obstacle_severity"]
        self.frozen_near_goal_radius = CONFIG["density"]["frozen_near_goal_radius"]
        self.recovery = WarehouseRecovery(
            history_length=CONFIG["recovery"]["history_length"],
            spatial_safe_threshold=CONFIG["recovery"]["spatial_safe_threshold"],
            collision_threshold=CONFIG["warehouse"]["collision_threshold"],
            base_yield_steps=CONFIG["recovery"]["base_yield_steps"],
            yield_escalation_per_repeat=CONFIG["recovery"]["yield_escalation_per_repeat"],
            pair_escalation_threshold=CONFIG["recovery"]["pair_escalation_threshold"],
            frozen_obstacle_severity=self.frozen_obstacle_severity,
            frozen_near_goal_radius=self.frozen_near_goal_radius,
        )
        self.fatal_splat_multiplier = CONFIG["density"]["fatal_splat_multiplier"]
        self.warning_splat_multiplier = CONFIG["density"]["warning_splat_multiplier"]

        # Maps node_id -> the step_count at which its forced Tier-3 yield expires.
        # A node yields for as long as self.step_count < this value -- see step()'s
        # action-application loop below. Replaces the old one-step-only
        # forced_yield_ids, which let a loser immediately retry the same losing
        # move the very next step.
        self._yield_until: Dict[str, int] = {}

        # Reward shaping -- previously these were bare literals inside step()'s
        # reward calculation, not configurable from CONFIG at all.
        r = CONFIG["rewards"]
        self.reward_mission_complete = r["mission_complete"]
        self.baseline_reward = r["baseline_reward"]
        self.movement_reward_multiplier = r["movement_reward_multiplier"]
        self.idle_penalty = r["idle_penalty"]
        self.reward_fatal_collision = r["fatal_collision"]
        self.reward_warning_zone = r["warning_zone"]
        self.warning_zone_cooldown = r["warning_zone_cooldown"]
        self.final_approach_bonus = r["final_approach_bonus"]
        self.final_approach_threshold = CONFIG["warehouse"]["final_approach_threshold"]
        self.final_approach_speed_floor = CONFIG["warehouse"]["final_approach_speed_floor"]
        self.overtime_threshold_steps = r["overtime_threshold_steps"]
        self.overtime_max_penalty = r["overtime_max_penalty"]
        # Needed here (not just by main_runner_warehouse.py's training loop) to
        # compute the overtime ramp's fraction-of-the-way-through-overtime below.
        self.max_steps_per_episode = CONFIG["training"]["max_steps_per_episode"]
        # Tracks the step_count each node last actually EARNED the warning_zone
        # bonus, so it can be granted once per distinct encounter instead of
        # once per step -- see the cooldown check in step() below.
        self._last_warning_reward_step: Dict[str, int] = {}
        
        # Frozen / Parked Node Tracking
        self.frozen_nodes: Set[str] = set()

        # ---- VDA 5050 ERROR STOPS & HANDOVER ------------------------------
        # A THIRD fleet state, distinct from both active and frozen.
        #
        # frozen_nodes means "arrived at its goal and parked" and is load-bearing
        # in five places: collision exemption in check_integrity, the replay
        # active_mask, the episode-done condition, density obstacle handling, and
        # the agent-side forced idle. A stopped fleet needs FOUR of those five to
        # behave identically -- it has no agency, must not train the network, and
        # is an obstacle to route around -- but must NOT count toward
        # episode-done, because an episode where one fleet died is not an episode
        # where everyone finished. Hence a separate set, unioned with
        # frozen_nodes at every site except that one.
        e = CONFIG.get("errors", {})
        self.errors_enabled = bool(e.get("enabled", False))
        self.error_prob_per_step = float(e.get("prob_per_step", 0.0))
        self.error_max_per_episode = int(e.get("max_per_episode", 0))
        self.error_min_step = int(e.get("min_step", 0))
        self.error_max_step_fraction = float(e.get("max_step_fraction", 1.0))
        self.error_min_progress = float(e.get("min_progress_fraction", 0.0))
        self.stop_confirm_steps = int(e.get("stop_confirm_steps", 15))
        self.pickup_dwell_steps = int(e.get("pickup_dwell_steps", 4))
        self.pickup_reward = float(e.get("pickup_reward", 25.0))
        self.stopped_obstacle_severity = float(e.get("stopped_obstacle_severity", 1.4))
        self.recall_retired = bool(e.get("recall_retired", True))
        self.recall_distance_slack = float(e.get("recall_distance_slack", 1.5))

        # Fleets that have errored and stopped moving. Immobile from the moment
        # they enter this set; the mission response waits for confirmation.
        self.stopped_nodes: Set[str] = set()
        # fleet_id -> step at which it errored (for the confirmation timer).
        self._error_step: Dict[str, int] = {}
        # Confirmed dead, order cancelled, goal released back to the pool.
        self.stopped_confirmed: Set[str] = set()

        # OPEN PICKUPS: pickup_id -> {"pos", "goal_id", "source_fleet"}.
        # A pickup is created when a stopped fleet is confirmed, and consumed
        # when a rescuer completes its dwell there.
        self.open_pickups: Dict[str, Dict[str, Any]] = {}
        # rescuer_id -> pickup_id it is currently dispatched to.
        self._pickup_assignment: Dict[str, str] = {}
        # rescuer_id -> steps dwelt on the pickup cell so far.
        self._pickup_dwell: Dict[str, int] = {}
        # fleet_id -> further goals it still owes after its current one. A
        # rescuer that collects a failed peer's load carries BOTH orders and
        # must deliver both before retiring.
        self._pending_goals: Dict[str, List[str]] = {}

        self.total_errors = 0
        self.total_handovers = 0
        self.total_recalls = 0
        self.actions_overridden = 0
        self.actions_total = 0

        # ---- DECOMPOSED REWARDS --------------------------------------------
        rd = CONFIG.get("reward_decomposition", {})
        self.reward_heads = list(rd.get("heads", ["total"]))
        self.K = len(self.reward_heads)
        self.HEAD = {h: i for i, h in enumerate(self.reward_heads)}

        # ---- POLICY-INVOKED RECOVERY ---------------------------------------
        rp = CONFIG.get("recovery_policy", {})
        self.recovery_policy_enabled = bool(rp.get("enabled", False))
        self.recovery_invocation_cost = float(rp.get("invocation_cost", -4.0))
        self.recovery_forced_after = int(rp.get("forced_fallback_steps", 6))
        self.recovery_resolution_bonus = float(rp.get("resolution_bonus", 6.0))
        self.recovery_preemptive_bonus = float(rp.get("preemptive_bonus", 9.0))
        # Fleets a preemptive separation just moved, pending verification
        # on the NEXT step that they actually came clear.
        self._preemptive_watch = set()
        self.recovery_preemptive_success = 0
        # OPPORTUNITY DENOMINATOR. An invocation is only useful when
        # something is actually at risk, so the raw count is
        # uninterpretable on its own: 'rec inv 2' is excellent if there
        # were 2 chances and poor if there were 50. These count the
        # chances, so intervention becomes a RATE.
        self.risk_steps = 0        # steps with any deadlocked/warning fleet
        self.risk_steps_acted = 0  # of those, steps the policy invoked
        self.warning_steps = 0     # steps with a warning but no deadlock yet
        self._deadlock_streak = 0
        self._step_deadlocked = set()
        self._step_warning = set()
        self._recovery_ran_this_step = False
        self.recovery_wasted = 0
        self.recovery_preemptive = 0
        self.recovery_invocations = 0      # policy asked for it
        self.recovery_forced = 0           # orchestrator had to step in
        self.recovery_resolved = 0         # invocation actually cleared the deadlock
        self.last_recovery_mode = 0        # 0 none, 1 spatial, 2 temporal
        self._pending_integrity_reward = 0.0
        
        # Spawn the discrete fleets
        self.nodes = self._initialize_fleets(fleet_missions)
        
        # Calculate GNN input dimension based on node state vector shape
        input_dim = len(self.nodes[0].get_state_vector(self.nodes)) + len(self.density.get_local_affordance(self.nodes[0].current_pos, self.nodes, self.frozen_nodes))
        action_size = CONFIG["gnn"]["action_size"] # {-1, 0, 1} across 3 axes + idle
        
        
        self.gnn = None ## main_runner will inject it through the shared agent env.gnn = shared_agent.
        
        self.step_count = 0

    def _derive_bounds(self) -> Tuple[float, float, float]:
        """
        Normalization bounds taken from this map's actual coordinate extents.

        BUG THIS FIXES: every FleetNode was constructed with
        CONFIG["warehouse"]["bounds"] = (50, 50, 10), and
        get_relative_goal_displacement() divides the raw goal offset by that and
        clips to +-1. On any map wider than 50 cells, every goal further than 50
        away saturates to exactly 1.0. The DIRECTION of the goal survives; the
        MAGNITUDE is destroyed -- and destroyed exactly for the long journeys the
        multi-map curriculum is meant to teach. A fleet 51 cells out and one 400
        cells out present the network with an identical feature.

        Using the per-map span also makes the feature scale-INVARIANT, which is
        what you want for topology transfer: "60% of the way across this
        warehouse" should mean the same thing on a 25-wide map and a 200-wide
        one. That is a better input to a policy meant to generalise than a raw
        cell count, which has no consistent meaning across maps.

        Falls back to the CONFIG literal if coordinates are missing or a span is
        degenerate (a single-level map has zero Z extent, and dividing by zero
        there would produce inf/NaN features rather than a useful signal).
        """
        fallback = tuple(float(v) for v in CONFIG["warehouse"]["bounds"])
        if CONFIG.get("ablation", {}).get("hardcoded_bounds", False):
            print(f"[ABLATION] Using hardcoded bounds {fallback} instead of map extents.")
            return fallback
        coords = [c for c in self.grid_pos_dict.keys() if isinstance(c, tuple)]
        if not coords:
            print(f"[Core] No tuple coordinates found; using fallback bounds {fallback}.")
            return fallback

        arr = np.asarray(coords, dtype=np.float32)
        spans = arr.max(axis=0) - arr.min(axis=0)

        out = []
        for axis, (span, fb) in enumerate(zip(spans, fallback)):
            # A flat axis (e.g. a single-level warehouse: max Z == min Z) has
            # span 0. Every displacement along it is 0 too, so the divisor is
            # arbitrary -- but it must not be 0, and it must not be so small
            # that float noise blows up. 1.0 keeps that axis's feature at a
            # clean 0.0.
            out.append(float(span) if span >= 1.0 else 1.0)

        print(f"[Core] Derived per-map bounds {tuple(round(v, 1) for v in out)} "
              f"(CONFIG fallback was {fallback}).")
        return (out[0], out[1], out[2])

    def _initialize_fleets(self, fleet_missions: List[Dict[str, Any]]) -> List[FleetNode]:
        """
        Translates the Pandas mission data into active FleetNodes.

        Pool mode: missions only need "id" and "start_pos" -- there's no
        per-fleet goal to assign. goal_pos is set to the fleet's own start_pos
        as a placeholder (it's immediately overwritten below by each fleet's
        first retarget_to_nearest_unclaimed() call, before anything else ever
        reads it).
        """
        nodes = []
        for mission in fleet_missions:
            if self.shared_pool_mode:
                placeholder_goal = mission["start_pos"]
                node = FleetNode(
                    id=str(mission["id"]),
                    current_pos=mission["start_pos"],
                    goal_pos=placeholder_goal,
                    G=self.G,
                    grid_pos_dict=self.grid_pos_dict,
                    warehouse_bounds=self.warehouse_bounds,
                    speed=CONFIG["warehouse"]["base_speed"],
                    max_vision_range=CONFIG["warehouse"]["max_vision_range"],
                    aisle_index=self.aisle_index,
                    coords_by_id=self.coords_by_id,
                )
            else:
                goal_node = mission.get("goal_node")
                node = FleetNode(
                    id=str(mission["id"]),
                    current_pos=mission["start_pos"],
                    goal_pos=mission["goal_pos"],
                    G=self.G,
                    grid_pos_dict=self.grid_pos_dict,
                    warehouse_bounds=self.warehouse_bounds,
                    speed=CONFIG["warehouse"]["base_speed"],
                    max_vision_range=CONFIG["warehouse"]["max_vision_range"],
                    aisle_index=self.aisle_index,
                    coords_by_id=self.coords_by_id,
                    goal_distance_map=self.goal_distance_maps.get(goal_node),
                )
            nodes.append(node)
        print(f"[Core] Spawned {len(nodes)} discrete fleets into the warehouse.")

        if self.shared_pool_mode:
            # Initial assignment. The previous version had every fleet
            # independently grab its own nearest goal, on the reasoning that a
            # sensible spread should EMERGE from the claiming mechanism. It
            # doesn't: with nothing coordinating the picks, only 19 of 25 goals
            # were targeted at spawn, 6 fleets were committed to a goal they
            # could not win, and 6 goals had no one heading for them. Emergence
            # needs a mechanism that can express exclusion, and independent
            # argmin cannot.
            if CONFIG["assignment"]["mode"] == "hungarian":
                fleet_ids = [n.id for n in nodes]
                fleet_node_ids = [
                    self.grid_pos_dict.get(tuple(int(v) for v in np.round(n.current_pos).astype(int)))
                    for n in nodes
                ]
                plan = assign_goals_optimally(
                    fleet_ids, fleet_node_ids, self.goal_pool, self.goal_distance_maps
                )
                unassigned = 0
                for node in nodes:
                    gid = plan.get(node.id)
                    if gid is None:
                        # More fleets than reachable goals -- this fleet has
                        # nothing to do, so retire it immediately.
                        #
                        # BUG THIS FIXES: this used to leave the fleet active on
                        # the assumption that step()'s pool-exhaustion check
                        # would catch it. That check is
                        # `if node.current_goal_id in self.claimed_goals`, and an
                        # unassigned fleet's current_goal_id is None, which is
                        # never in that set -- so it never fired. The fleet
                        # wandered the entire episode with goal_distance_map=None
                        # (get_goal_gradient() returns six zeros, so it had no
                        # routing signal at all), purely as a moving obstacle to
                        # everyone else. Only reachable with fleets > goals, but
                        # silent when it happened.
                        unassigned += 1
                        self.frozen_nodes.add(node.id)
                        # getattr: _initialize_fleets() runs from __init__ BEFORE
                        # self.gnn is assigned, so a direct attribute access
                        # raises here.
                        _gnn = getattr(self, "gnn", None)
                        if _gnn is not None:
                            _gnn.freeze_node(node.id, node.current_pos)
                        continue
                    node.current_goal_id = gid
                    node.goal_pos = self.goal_pool[gid].copy()
                    node.goal_distance_map = self.goal_distance_maps.get(gid)
                    node.initial_graph_distance = node.get_graph_distance_to_goal()
                    self.targeted_goals[gid] = node.id
                print(f"[Core] Optimal 1:1 assignment: {len(self.targeted_goals)}/{len(self.goal_pool)} "
                      f"goals assigned, {unassigned} fleets unassigned, 0 contested.")
            else:
                # Legacy: independent nearest-pick, now at least reservation-aware
                # so two fleets can't commit to the same goal.
                for node in nodes:
                    self._retarget(node)
                print(f"[Core] {len(self.goal_pool)} goals in the shared pool, all fleets have an initial target.")

        return nodes

    def _reserved_by_others(self, node: FleetNode) -> Set[str]:
        """Goals reserved by some OTHER fleet. A fleet never blocks itself."""
        if not CONFIG["assignment"].get("reserve_targets", True):
            return set()
        return {gid for gid, fid in self.targeted_goals.items() if fid != node.id}

    def _release_target(self, node: FleetNode):
        """Drop this fleet's reservation, if it holds one."""
        held = [gid for gid, fid in self.targeted_goals.items() if fid == node.id]
        for gid in held:
            del self.targeted_goals[gid]

    def _retarget(self, node: FleetNode) -> bool:
        """
        Single entry point for retargeting: release the old reservation, pick a
        goal that is neither claimed nor reserved by a peer, reserve it.

        Every retarget path in step() funnels through here so the ledger can
        never drift out of sync with what the fleets are actually chasing.
        """
        self._release_target(node)
        found = node.retarget_to_nearest_unclaimed(
            self.goal_pool,
            self.goal_distance_maps,
            self.claimed_goals,
            reserved_goal_ids=self._reserved_by_others(node),
        )
        if found and node.current_goal_id is not None:
            self.targeted_goals[node.current_goal_id] = node.id
        return found

    def _build_adjacency(self) -> np.ndarray:
        """
        Proximity adjacency for the GAT: a fleet attends to itself plus any fleet
        within interaction_radius (Manhattan), instead of to all 25 unconditionally.

        BUG THIS FIXES -- this was the oscillation. The adjacency used to be
        np.ones((N, N)), a COMPLETE graph, so after the attention softmax every
        fleet's representation was a weighted average over the entire fleet. The
        six goal-gradient features are fleet-SPECIFIC and DIRECTIONAL: fleet 7's
        "+X is downhill" averages against fleet 12's "-X is downhill" and
        cancels. Measured on the real network: under the complete adjacency,
        perturbing one fleet's gradient features changed that fleet's own
        Q-values exactly as much as it changed every other fleet's (ratio 1.0x,
        own response 0.00008). Under proximity adjacency the same perturbation
        moves its own Q-values 28x more, with effectively zero leakage.

        That is precisely the signature seen in training: the policy picked the
        gradient's best move 42.6% of the time and the EXACT OPPOSITE 42.1% of
        the time, with only 15% left for all other actions. It could see that the
        gradient dimension was salient but not whose gradient it was, so it could
        not resolve the sign -- and fleets travelled 328 cells to cover 19.6 hops,
        netting zero.

        Self-loops are always present, so an isolated fleet attends only to
        itself and its own features pass through cleanly.
        """
        n = len(self.nodes)
        adj = np.eye(n, dtype=np.float32)
        if n > 1:
            positions = np.array([node.current_pos for node in self.nodes], dtype=np.float32)
            for i in range(n):
                within = np.sum(np.abs(positions - positions[i]), axis=1) <= self.interaction_radius
                adj[i, within] = 1.0
        return adj

    def get_fleet_diagnostics(self) -> List[Dict[str, Any]]:
        """
        Per-fleet motion breakdown for the episode. The field that matters is
        'efficiency' = |net progress toward goal| / cells actually travelled:

          ~1.0  fleet drove more or less straight at its goal
          ~0.0  with HIGH travelled -> wandering; it moved a long way and ended
                up where it started. The policy is the problem.
          ~0.0  with LOW travelled  -> stalling; it never went anywhere. Look at
                idle_frac and mean_speed: high idle means the policy is choosing
                action 0, low mean_speed means affordance braking is throttling
                it to a crawl.
        """
        out = []
        for node in self.nodes:
            steps = self._fleet_steps.get(node.id, 0)
            if steps == 0:
                continue
            travelled = self._fleet_travel.get(node.id, 0.0)
            initial = float(max(node.initial_graph_distance, 1.0))
            final = float(node.get_graph_distance_to_goal())
            progress = initial - final
            out.append({
                "id": node.id,
                "finished": node.id in self.frozen_nodes,
                "initial_hops": initial,
                "final_hops": final,
                "travelled_cells": travelled,
                "efficiency": (progress / travelled) if travelled > 1e-6 else 0.0,
                "idle_frac": self._fleet_idle.get(node.id, 0) / steps,
                "mean_speed": self._fleet_speed.get(node.id, 0.0) / steps,
            })
        return out

    def get_gradient_agreement(self) -> float:
        """
        Fraction of policy actions this episode that matched argmax(goal_gradient).

        The single cleanest read on whether the routing signal added to the state
        vector is being used. Independent of contention, collisions and recovery
        noise: it asks only whether the network moves the way the precomputed BFS
        says it should. Returns 0.0 before any action has been recorded.
        """
        return float(np.mean(self._grad_agree)) if self._grad_agree else 0.0

    # ======================================================================
    # VDA 5050 ERROR STOPS & PACKAGE HANDOVER
    # ======================================================================

    @property
    def immobile_nodes(self) -> Set[str]:
        """
        Every fleet that will not move this step, for whatever reason.

        This is what belongs at four of the five sites frozen_nodes is currently
        used: collision exemption, the density obstacle branch, the replay
        active_mask, and the agent-side forced idle. Those all ask "is this thing
        a static object rather than an agent", and a stopped fleet answers yes to
        all four.

        The FIFTH site -- the episode-done check in main_runner -- must keep
        using frozen_nodes alone. "Every fleet is immobile" is not "every fleet
        succeeded"; an episode where one fleet died and its package was never
        picked up should run on so a rescuer has time to reach it.
        """
        return self.frozen_nodes | self.stopped_nodes

    def _maybe_inject_error(self):
        """
        Roll for a VDA 5050 error stop on some active fleet.

        Rate-based rather than scheduled, so longer episodes see proportionally
        more errors and the policy cannot learn "errors happen at step N".
        """
        if not self.errors_enabled:
            return
        if self.total_errors >= self.error_max_per_episode:
            return
        if self.step_count < self.error_min_step:
            return
        # Late errors are structurally unrecoverable -- see max_step_fraction.
        if self.step_count > self.error_max_step_fraction * self.max_steps_per_episode:
            return
        if random.random() >= self.error_prob_per_step:
            return

        # Only fleets genuinely under way. Erroring one still sitting on its
        # spawn makes the pickup and the start the same cell, which teaches
        # nothing about mid-mission recovery.
        candidates = [
            n for n in self.nodes
            if n.id not in self.immobile_nodes
            and n.current_goal_id is not None
            and n.get_progress_fraction() >= self.error_min_progress
        ]
        if not candidates:
            return

        victim = random.choice(candidates)
        self.stopped_nodes.add(victim.id)
        self._error_step[victim.id] = self.step_count
        self.total_errors += 1

        # If the victim was itself en route to a pickup, that pickup is now
        # ORPHANED AGAIN and must go back on the board.
        #
        # BUG THIS FIXES: _dispatch_rescuers() skips any pickup that already
        # appears in _pickup_assignment.values(). Leaving a dead rescuer's entry
        # in that dict therefore makes the pickup look permanently serviced, so
        # no replacement is ever dispatched and the load is stranded for the rest
        # of the episode. Rescuers are ordinary fleets and are themselves valid
        # error candidates, so this is not a corner case -- it showed up as
        # "3 pickups open, 0 handovers" the first time three errors landed in one
        # episode.
        stranded = self._pickup_assignment.pop(victim.id, None)
        self._pickup_dwell.pop(victim.id, None)
        self._yield_until.pop(victim.id, None)
        if stranded is not None:
            print(f"[Core] Fleet {victim.id} errored while servicing {stranded}; "
                  f"that pickup is back on the board.")
        # Physically stop it dead, this step, before anything else reads its
        # direction: a stopped AGV holds position and stops projecting a swept
        # path into the density field.
        # Zero the DIRECTION only, not the speed. Immobility is enforced by
        # membership in immobile_nodes (the action loop skips it entirely), and
        # zeroing direction is what stops it projecting a swept path into the
        # density field. Setting speed = 0.0 additionally would look harmless but
        # crashes sense_6_axis_rays(), which divides by speed to size its ray
        # budget -- a latent division that never fired before because affordance
        # braking floors speed at 0.1.
        victim.direction = np.zeros(3, dtype=np.float32)
        print(f"[Core] *** VDA5050 ERROR *** Fleet {victim.id} stopped at step "
              f"{self.step_count} ({victim.get_progress_fraction()*100:.0f}% through "
              f"its journey to {victim.current_goal_id}).")

    def _confirm_stops_and_open_pickups(self):
        """
        Promote errored fleets to CONFIRMED dead once the timer expires, release
        their goals, and publish their cells as pickups.

        The delay gates the MISSION response, not the physics -- nothing about a
        dead robot changes at step 15. What changes is the fleet manager's
        decision to stop waiting for it to clear and reassign its order. That
        maps onto VDA 5050's errorLevel escalation from WARNING (recoverable,
        hold) to FATAL (order cancelled, master reassigns).
        """
        for node in self.nodes:
            if node.id not in self.stopped_nodes:
                continue
            if node.id in self.stopped_confirmed:
                continue
            if self.step_count - self._error_step.get(node.id, 0) < self.stop_confirm_steps:
                continue

            self.stopped_confirmed.add(node.id)
            # A fleet can die while carrying more than one order (it may itself
            # have been a rescuer). All of them are stranded at its cell, so the
            # pickup has to carry the whole queue or the extras vanish silently.
            orphan_goals = [g for g in ([node.current_goal_id]
                                        + self._pending_goals.pop(node.id, []))
                            if g is not None and g not in self.claimed_goals]

            # Cancel the order: the reservation goes back so a rescuer can take
            # it. NOT added to claimed_goals -- nothing was delivered.
            self._release_target(node)

            # Clear the fleet's OWN view of its order too, not just the ledger.
            # _release_target only drops the targeted_goals entry, so without
            # this a confirmed-dead fleet keeps advertising current_goal_id ==
            # the orphaned goal forever. Anything scanning nodes by goal id then
            # finds the corpse alongside (or before) the fleet actually carrying
            # the load -- which is exactly what the handover test caught. The
            # order is cancelled; the state should say so.
            node.current_goal_id = None

            if not orphan_goals:
                # Nothing left to hand over (its goal was claimed by someone
                # else while it sat there). It stays an obstacle, no pickup.
                print(f"[Core] Fleet {node.id} confirmed dead; no recoverable order.")
                continue

            pickup_id = f"pickup_{node.id}_{self.step_count}"
            self.open_pickups[pickup_id] = {
                "pos": node.current_pos.copy(),
                "goal_ids": list(orphan_goals),
                "source_fleet": node.id,
            }
            print(f"[Core] Fleet {node.id} confirmed dead. Order(s) "
                  f"{orphan_goals} cancelled; pickup {pickup_id} opened at its cell.")

    def _dispatch_rescuers(self):
        """
        Send one fleet to each open pickup.

        Chooses the nearest available fleet by TRUE GRAPH distance to the pickup
        cell, not Manhattan: on a warehouse this sparse (average degree ~2.27)
        the straight-line-nearest fleet is regularly on the far side of a rack
        run and a long way round in hops.

        A rescuer keeps its own current goal reserved while it diverts. It has
        not abandoned that goal, it is servicing the pickup first and will
        inherit the orphaned goal on arrival -- so releasing it here would let a
        peer take a goal this fleet is still committed to.
        """
        for pickup_id, pickup in list(self.open_pickups.items()):
            if pickup_id in self._pickup_assignment.values():
                continue  # already has a rescuer en route

            pickup_pos = pickup["pos"]
            pickup_node_id = self.grid_pos_dict.get(
                tuple(int(round(v)) for v in pickup_pos)
            )
            if pickup_node_id is None:
                continue

            # Distance map FOR THE PICKUP CELL, built before ranking.
            #
            # BUG THIS FIXES: candidates used to be ranked by their distance to
            # the orphaned FINAL GOAL, on the reasoning that the rescuer has to
            # reach it eventually anyway. That is the wrong quantity and it
            # selects badly: who is nearest the destination has almost nothing to
            # do with who can reach the package. Observed in training -- a fleet
            # 118 hops from the pickup was dispatched while ~30 others were still
            # active, and the episode ended with it still 104 hops short, so the
            # handover never happened at all.
            #
            # Building the pickup's own map first costs nothing: the dispatcher
            # already had to build it to hand to the winner. It is one BFS
            # either way, now used for the decision as well as the assignment.
            pickup_map = self.goal_distance_maps.get(pickup_node_id)
            if pickup_map is None:
                from node_warehouse import (precompute_goal_distances_compact,
                                            ArrayDistanceMap)
                import networkx as nx
                try:
                    sample_map = next(iter(self.goal_distance_maps.values()), None)
                    if isinstance(sample_map, ArrayDistanceMap):
                        built, _ = precompute_goal_distances_compact(
                            self.G, [pickup_node_id], node_index=sample_map._idx)
                        pickup_map = built.get(pickup_node_id)
                    else:
                        pickup_map = nx.single_source_shortest_path_length(
                            self.G, pickup_node_id)
                    if pickup_map is None:
                        continue
                    self.goal_distance_maps[pickup_node_id] = pickup_map
                except Exception:
                    continue

            # Rank BOTH working and idle fleets. A retired fleet is not out of
            # the game -- it is an idle vehicle parked at a delivered goal, and
            # recalling one costs nothing that matters: GNNAgent.unfreeze_node()
            # simply drops it from the frozen set, its claimed goal stays
            # claimed, and it starts producing training transitions again.
            #
            # This is the scenario the "retire immediately" comment in the
            # mission-complete branch explicitly anticipated and set aside as
            # not yet existing. It exists now.
            best_active, best_active_d = None, float("inf")
            best_idle, best_idle_d = None, float("inf")
            for node in self.nodes:
                if node.id in self.stopped_nodes:
                    continue  # dead; cannot rescue anything
                if node.id in self._pickup_assignment:
                    continue  # already servicing a different pickup
                node_id = self.grid_pos_dict.get(
                    tuple(int(round(v)) for v in node.current_pos)
                )
                if node_id is None:
                    continue
                d = pickup_map.get(node_id)
                if d is None:
                    continue  # unreachable from there
                if node.id in self.frozen_nodes:
                    if self.recall_retired and d < best_idle_d:
                        best_idle, best_idle_d = node, float(d)
                elif d < best_active_d:
                    best_active, best_active_d = node, float(d)

            # Prefer the idle fleet unless it is disproportionately further.
            # Diverting a working fleet delays an order already in progress;
            # an idle one has nothing to delay, so it earns some slack.
            best, best_d, recalled = best_active, best_active_d, False
            if best_idle is not None and (
                best_active is None
                or best_idle_d <= best_active_d * self.recall_distance_slack
            ):
                best, best_d, recalled = best_idle, best_idle_d, True

            if best is None:
                continue

            # Don't dispatch a rescuer that cannot possibly arrive in time.
            # At base_speed a fleet covers base_speed hops per step AT BEST,
            # before any detour, yield or braking, so this is a generous bound
            # and anything failing it is hopeless rather than merely tight.
            # Leaving the pickup open lets a better-placed fleet take it later.
            steps_left = self.max_steps_per_episode - self.step_count
            _speed = max(float(CONFIG["warehouse"]["base_speed"]), 1e-6)
            if steps_left > 0 and best_d / _speed > steps_left:
                continue

            self._pickup_assignment[best.id] = pickup_id
            self._pickup_dwell[best.id] = 0

            if recalled:
                # Back into service. Order matters: clear the frozen state
                # BEFORE retargeting, or the forced-idle path in step() would
                # hold it stationary on the first move.
                self.frozen_nodes.discard(best.id)
                self.gnn.unfreeze_node(best.id)
                self.total_recalls += 1
                # Its own goal was delivered and stays in claimed_goals, so
                # completion accounting is untouched by the recall -- deliveries
                # are counted from claimed_goals, not from who happens to be
                # parked at the end.

            # Retarget onto the pickup CELL. goal_distance_map is the map for the
            # pickup's own node so the BFS gradient routes there properly, and
            # initial_graph_distance is rebased so get_progress_fraction()
            # measures leg 1 rather than a stale leg from the old goal.
            best.goal_pos = pickup_pos.copy()
            best.goal_distance_map = pickup_map
            best.initial_graph_distance = best.get_graph_distance_to_goal()
            self._best_dist.pop(best.id, None)
            self._stall_steps[best.id] = 0
            print(f"[Core] Fleet {best.id} {'RECALLED from retirement and ' if recalled else ''}"
                  f"dispatched to {pickup_id} "
                  f"({best.initial_graph_distance:.0f} hops, {steps_left} steps left)."
                  + (f" [idle {best_idle_d:.0f}h vs working {best_active_d:.0f}h]"
                     if recalled and best_active is not None else ""))

    def _service_pickups(self, node: FleetNode) -> float:
        """
        Advance the dwell timer for a rescuer standing on its pickup, and hand
        the package over once the transfer completes.

        Returns any reward earned this step (the leg-1 pickup bonus, once).

        The dwell exists so a handover is an EVENT WITH A DURATION rather than
        something that happens in the instant a fleet happens to pass over a
        cell. Without it, any fleet crossing the pickup en route to somewhere
        else would silently inherit the order.
        """
        pickup_id = self._pickup_assignment.get(node.id)
        if pickup_id is None:
            return 0.0
        pickup = self.open_pickups.get(pickup_id)
        if pickup is None:
            # Pickup vanished (shouldn't happen, but don't strand the fleet).
            del self._pickup_assignment[node.id]
            return 0.0

        dist = float(np.sum(np.abs(node.current_pos - pickup["pos"])))
        if dist > 0.5:
            # Not there yet. Reset the dwell so it must be CONTINUOUS -- a fleet
            # that touches the cell, wanders off and comes back starts over.
            self._pickup_dwell[node.id] = 0
            return 0.0

        self._pickup_dwell[node.id] = self._pickup_dwell.get(node.id, 0) + 1

        # HOLD THE VEHICLE STILL FOR THE TRANSFER, using the same forced-idle
        # mechanism Tier 3 uses for yields.
        #
        # Without this the dwell is a policy decision, and an untrained policy
        # will not make it: it has to choose idle for pickup_dwell_steps
        # CONSECUTIVE steps on one cell, having never been rewarded for doing so,
        # and the dwell counter resets the moment it steps off. Measured on the
        # synthetic map, an untrained rescuer reached the pickup reliably but
        # bounced off before completing the transfer more often than not -- so
        # the handover reward it needs in order to learn any of this almost never
        # arrived. Classic chicken-and-egg.
        #
        # Forcing the hold is also the more faithful model: a physical load
        # transfer immobilises the vehicle for its duration. That is not
        # something the AGV decides, so it should not be something the policy has
        # to discover. The policy's job is to GET there; the transfer is
        # mechanical. The idle-penalty exemption in step() is scoped to exactly
        # this window.
        # DWELL IS NOW A POLICY CHOICE, not a forced hold.
        #
        # It used to set _yield_until here to pin the vehicle in place for the
        # transfer. That guaranteed handovers completed, but it also meant the
        # network never CHOSE to complete one -- so the rescue reward attached to
        # no decision it had made, and the whole mechanic was unlearnable. Since
        # the rescue head now gets a clean signal, the policy can learn to hold
        # position on the pickup, and holding is already available to it as the
        # idle action.

        if self._pickup_dwell[node.id] < self.pickup_dwell_steps:
            return 0.0

        # Transfer done -- release the hold immediately so leg 2 can start on the
        # very next step rather than idling out the remainder of the window.
        self._yield_until.pop(node.id, None)

        # ---- TRANSFER COMPLETE: inherit every orphaned order ----
        orphan_goals = [g for g in pickup["goal_ids"] if g not in self.claimed_goals]
        del self.open_pickups[pickup_id]
        del self._pickup_assignment[node.id]
        self._pickup_dwell.pop(node.id, None)
        self.total_handovers += 1

        if not orphan_goals:
            # Somebody else delivered it while the transfer was in progress.
            # Fall back to a normal retarget rather than chasing a done goal.
            self._retarget(node)
            return self.pickup_reward

        # ---- THE RESCUER NOW CARRIES TWO LOADS ----
        #
        # It does NOT abandon its own order. A physical AGV that collects a
        # second package from a failed peer is carrying both, and owes both
        # destinations; the handover is a recovery, so the system should absorb
        # the failure completely rather than trading one undelivered order for
        # another.
        #
        # BUG THIS FIXES: this used to call _release_target(node) here, which
        # drops EVERY reservation the fleet holds, and then assign the orphaned
        # goal -- a swap, not an addition. The rescuer's own goal went back into
        # the pool, and because a handover typically completes late in an episode
        # when most peers have already frozen, there was usually nobody left able
        # to take it. Net effect: one error still cost one undelivered order, so
        # the whole handover mechanism recovered nothing measurable.
        #
        # Both goals stay reserved to this fleet in targeted_goals. The nearer
        # one is delivered first (cheaper total travel, and it banks one delivery
        # sooner in case the rescuer itself errors before finishing).
        own_goal = node.current_goal_id
        queue = [g for g in ([own_goal] + orphan_goals + self._pending_goals.get(node.id, []))
                 if g is not None and g not in self.claimed_goals]
        queue = list(dict.fromkeys(queue))  # dedupe, preserve order

        def _hops(gid):
            gm = self.goal_distance_maps.get(gid)
            if not gm:
                return float("inf")
            nid = self.grid_pos_dict.get(tuple(int(round(v)) for v in node.current_pos))
            d = gm.get(nid) if nid is not None else None
            return float(d) if d is not None else float("inf")

        queue.sort(key=_hops)
        first, rest = queue[0], queue[1:]

        self._pending_goals[node.id] = rest
        for gid in queue:
            self.targeted_goals[gid] = node.id

        node.current_goal_id = first
        node.goal_pos = (self.goal_pool[first].copy() if self.shared_pool_mode
                         else node.goal_pos)
        node.goal_distance_map = self.goal_distance_maps.get(first)
        node.initial_graph_distance = node.get_graph_distance_to_goal()

        # Rebase the stall tracker too. _best_dist is keyed to the OLD leg's
        # distances, so leaving it would make every step of the next leg look
        # like a failure to improve and trip livelock_patience within 40 steps.
        self._best_dist.pop(node.id, None)
        self._stall_steps[node.id] = 0

        print(f"[Core] *** HANDOVER *** Fleet {node.id} picked up the load from "
              f"{pickup['source_fleet']}. Now carrying {len(queue)} order(s): "
              f"delivering {first} first ({node.initial_graph_distance:.0f} hops)"
              + (f", then {rest[0]}." if rest else "."))
        return self.pickup_reward

    def _run_recovery(self, forced: bool) -> Dict[str, Any]:
        """
        Execute a collapse. `forced` distinguishes the orchestrator stepping in
        from the policy asking.

        The distinction IS the metric. Forced invocations mean the policy sat in
        a deadlock without calling for help; the count falling toward zero over
        training is the evidence that recovery has actually been learned rather
        than merely survived.
        """
        self._recovery_ran_this_step = True
        n_dead_before = len(self._step_deadlocked)

        # PREEMPTIVE MODE. collapse_and_reinitialize() separates the DEADLOCKED
        # set and explicitly leaves warning-zone fleets undisturbed. So an
        # invocation made before anything has crashed -- fleets in the warning
        # band, no collision yet -- arrived with an empty set, did nothing, and
        # still printed "Tier 1: Spatial Escape Successful". Preemptive recovery
        # was a no-op by construction, which is precisely the behaviour that was
        # supposed to change: Tier 1 has to fire BEFORE the collision, not after.
        #
        # When there is no deadlock but fleets are at risk, the at-risk set
        # becomes the set to separate. Same machinery, applied one step earlier,
        # which is the entire point of letting the policy call it.
        dead = set(self._step_deadlocked)
        warn = set(self._step_warning)
        preemptive = (not dead) and bool(warn)
        if preemptive:
            dead, warn = warn, set()

        result = self.recovery.collapse_and_reinitialize(
            nodes=self.nodes,
            warning_nodes=warn,
            deadlocked_nodes=dead,
            density_field=self.density,
            frozen_node_ids=self.immobile_nodes,
        )
        if preemptive:
            self.recovery_preemptive += 1
            # Verified next step: did these fleets actually come clear?
            self._preemptive_watch = set(dead)
        self.loop.force_repair()
        self._deadlock_streak = 0
        if forced:
            self.recovery_forced += 1
        else:
            self.recovery_invocations += 1
        if n_dead_before > 0:
            self.recovery_resolved += 1
        return result

    def _policy_recovery_step(self):
        """
        Let the POLICY decide whether to collapse, before any collision forces it.

        The old trigger lived inside the deadlocked branch -- it could only fire
        after check_integrity had already returned 0.0, i.e. after a crash. It
        was structurally incapable of acting first.

        Tier 1 (spatial escape) and Tier 3 (yield) do not need an invocation:
        they are an ordinary move and an ordinary idle, both already in the
        per-fleet action space, and with the warning-zone penalty now graded the
        policy has a reason to use them early. Tier 2 does need one, because a
        temporal rewind restores MANY fleets at once and cannot be expressed as
        one fleet's action -- hence a graph-level head.

        The forced fallback is not optional. An untrained policy never invokes,
        so without it every early episode deadlocks permanently and produces no
        usable experience to learn from.
        """
        if not self.recovery_policy_enabled:
            return
        H = self.HEAD

        mode = 0
        if self.gnn is not None and hasattr(self.gnn, "choose_recovery"):
            mode = int(self.gnn.choose_recovery())
        self.last_recovery_mode = mode

        if self.loop.current_integrity <= 0.0:
            self._deadlock_streak += 1
        else:
            self._deadlock_streak = 0

        forced = (self._deadlock_streak >= self.recovery_forced_after)
        if mode == 0 and not forced:
            return

        # NOTHING AT RISK -> charge, but do not execute.
        #
        # Recovery on an empty conflict set is a no-op that still called
        # force_repair(), producing "Initiating Spatial-Temporal Collapse for 0
        # nodes" followed by "Tier 1: Spatial Escape Successful" -- a success
        # report for having done nothing, and a wipe of integrity state that was
        # already clean.
        #
        # The COST is still charged. The policy asked for a collapse it did not
        # need, and that has to be learnable or the head never stops asking.
        # Preemptive invocation stays legitimate: warning-zone fleets count as
        # at-risk, so calling a collapse BEFORE a collision -- which is the whole
        # point of moving this out of the deadlocked branch -- still executes.
        at_risk = self._step_deadlocked or self._step_warning
        if at_risk:
            self.risk_steps += 1
            if self._step_warning and not self._step_deadlocked:
                self.warning_steps += 1
            if mode != 0:
                self.risk_steps_acted += 1

        if not at_risk and not forced:
            self.recovery_wasted += 1
            self._pending_integrity_reward += self.recovery_invocation_cost
            return

        n_dead_before = len(self._step_deadlocked)
        self._run_recovery(forced=(mode == 0 and forced))

        # Cost and outcome land on the INTEGRITY head, charged to every fleet
        # that was party to the deadlock. Without a cost the policy learns to
        # crash and undo: a rewind erases a collision, so a free rewind makes
        # ignoring safety optimal. The goal head separately penalises the lost
        # travel distance, so the two heads pull against each other and the
        # policy has to resolve that tension -- which is the decision we want it
        # to learn.
        self._pending_integrity_reward += self.recovery_invocation_cost
        if n_dead_before > 0:
            self._pending_integrity_reward += self.recovery_resolution_bonus

    def is_episode_over(self) -> bool:
        """
        True when no fleet can still act.

        Deliberately NOT `len(frozen_nodes) == len(nodes)`, which was the old
        early-exit condition and now means something different from termination.
        With error stops those two come apart:

          * all frozen           -> everyone succeeded. Over, and a success.
          * all frozen or stopped -> nothing left that can move, but a package
            was abandoned. Over, and NOT a success.
          * some active, a pickup still open -> NOT over, even if every other
            fleet has parked. The rescuer needs the clock to reach the pickup,
            and cutting the episode here would delete exactly the transitions
            that teach handover.

        main_runner logs completion off frozen_nodes, so success reporting is
        unaffected by this.
        """
        return len(self.immobile_nodes) == len(self.nodes)

    def get_error_statistics(self) -> Dict[str, Any]:
        """Error/handover counters for the training log."""
        return {
            "errors_injected": self.total_errors,
            "stops_confirmed": len(self.stopped_confirmed),
            "handovers_completed": self.total_handovers,
            "retired_fleets_recalled": self.total_recalls,
            "action_override_rate": (self.actions_overridden / self.actions_total
                                     if self.actions_total else 0.0),
            "recovery_invocations": self.recovery_invocations,
            "recovery_forced": self.recovery_forced,
            "recovery_resolved": self.recovery_resolved,
            "recovery_wasted": self.recovery_wasted,
            "recovery_preemptive": self.recovery_preemptive,
            "recovery_preemptive_success": self.recovery_preemptive_success,
            "risk_steps": self.risk_steps,
            "risk_steps_acted": self.risk_steps_acted,
            "warning_steps": self.warning_steps,
            "intervention_rate": (self.risk_steps_acted / self.risk_steps
                                  if self.risk_steps else 0.0),
            "pickups_open": len(self.open_pickups),
            "stopped_fleets": sorted(self.stopped_nodes),
        }

    def get_active_nodes(self) -> List[FleetNode]:
        return [n for n in self.nodes if n.id not in self.immobile_nodes]

    def step(self, episode_step: int, total_episodes: int) -> float:
        """
        Executes one discrete simulation step for the warehouse environment.
        """
        # 0. VDA 5050 ERROR STOPS. Rolled BEFORE integrity so a fleet that
        # errors this step is already out of the collision system when integrity
        # is evaluated, rather than spending one step as a phantom conflict.
        self._maybe_inject_error()
        self._confirm_stops_and_open_pickups()
        self._dispatch_rescuers()

        # 1. Check Loop Integrity (Dual-Zone Logic)
        #
        # immobile_nodes, not frozen_nodes: a stopped fleet leaves the collision
        # system entirely. It is not a party to a conflict, it is furniture, and
        # it is represented purely as a density stamp. That single choice is what
        # makes the whole handover work -- no fatal collision when the rescuer
        # arrives at distance 0, no integrity drop to 0.5 (and so no spurious WFC
        # recovery) every time one happens, and no warning_splat slowly poisoning
        # the exact cell the next rescuer has to reach.
        current_integrity = self.loop.check_integrity(self.nodes, self.step_count, self.immobile_nodes)

        # SNAPSHOT the conflict sets before anything can clear them.
        #
        # BUG THIS FIXES: force_repair() empties deadlocked_nodes and
        # warning_nodes, and _policy_recovery_step() below may call it -- but the
        # reward loop at the end of this same step reads those sets to assign the
        # collision and proximity penalties. So on every step where recovery
        # fired, the safety penalties silently vanished: a fleet could crash,
        # trigger a collapse, and be charged nothing for the crash.
        #
        # That is the same shape as the action-logging bug -- the consequence is
        # erased before it is recorded -- and it removes exactly the transitions
        # the safety head needs in order to learn avoidance at all.
        self._step_deadlocked = set(self.loop.deadlocked_nodes)
        self._step_warning = set(self.loop.warning_nodes)
        self._recovery_ran_this_step = False

        # PAY OFF A PREEMPTIVE SEPARATION MADE LAST STEP, if it worked.
        # "Worked" = every fleet it moved is now clear of both the warning band
        # and any collision. Deferred by one step because that is the earliest
        # point the outcome is observable -- the separation happens after the
        # integrity check that would show its effect.
        self._pending_integrity_reward = 0.0
        if self._preemptive_watch:
            still_at_risk = self._preemptive_watch & (
                self._step_deadlocked | self._step_warning)
            if not still_at_risk:
                self._pending_integrity_reward += self.recovery_preemptive_bonus
                self.recovery_preemptive_success += 1
            self._preemptive_watch = set()

        # 1a. Publish situation flags onto each fleet so the rescue and safety
        # heads have something to condition on. Without these a handover is
        # indistinguishable from an ordinary delivery in the state vector, and
        # the rescue head can only learn a constant.
        span = max(1e-6, self.loop.warning_threshold - self.loop.collision_threshold)
        for n in self.nodes:
            n.sf_is_rescuer = 1.0 if n.id in self._pickup_assignment else 0.0
            n.sf_orders_carried = float(
                (1 if n.current_goal_id else 0) + len(self._pending_goals.get(n.id, [])))
            n.sf_on_pickup_cell = 0.0
            pid = self._pickup_assignment.get(n.id)
            if pid is not None and pid in self.open_pickups:
                d = float(np.sum(np.abs(n.current_pos - self.open_pickups[pid]["pos"])))
                n.sf_on_pickup_cell = 1.0 if d <= 0.5 else 0.0
            n.sf_in_warning = 1.0 if n.id in self._step_warning else 0.0
            n.sf_in_deadlock = 1.0 if n.id in self._step_deadlocked else 0.0
            nearest = min(
                (float(np.sum(np.abs(n.current_pos - o.current_pos)))
                 for o in self.nodes
                 if o.id != n.id and o.id not in self.immobile_nodes),
                default=self.loop.warning_threshold)
            n.sf_peer_proximity = float(np.clip(
                (self.loop.warning_threshold - nearest) / span, 0.0, 1.0))

        # 1b. POLICY-INVOKED RECOVERY, evaluated every step -- including steps
        # where integrity is still 1.0, which is the whole point: the policy can
        # now collapse BEFORE a collision instead of only after one.
        self._policy_recovery_step()
        
        # Record this step's positions/integrity so Tier 2 (Temporal Rewind) has
        # something to rewind to if a fatal collision happens down the line.
        self.recovery.assess_loop_coherence(self.nodes, current_integrity)
        
        # Handle Spatial-Temporal Collapses
        tier3_winner_id: Optional[str] = None
        if current_integrity == 0.0:
            # Fatal Crash: Splat the deadlocked locations with max severity.
            # Iterates the SNAPSHOT, not loop.deadlocked_nodes: if the policy
            # already invoked a collapse earlier this step, force_repair() has
            # emptied that set and the crash would go unsplatted and unrecorded
            # in the tabu list -- losing both the density warning and the
            # "never take this action from this cell again" memory.
            for deadlocked_id in self._step_deadlocked:
                crashed_node = next(n for n in self.nodes if n.id == deadlocked_id)
                self.density.splat_spatial_temporal_event(crashed_node.current_pos, severity_multiplier=self.fatal_splat_multiplier)

                # NEW: Record the exact mistake! 
                # Block the action that brought them from their last safe position into this crash.
                safe_pos_tuple = tuple(np.round(crashed_node.last_pos).astype(int))
                if safe_pos_tuple not in crashed_node.tabu_actions:
                    crashed_node.tabu_actions[safe_pos_tuple] = set()
                crashed_node.tabu_actions[safe_pos_tuple].add(crashed_node.last_action)
            
            # Actually try to separate the crashed fleets (Spatial Escape -> Temporal
            # Rewind -> Right-of-Way Yield) before declaring the loop repaired. Previously
            # force_repair() ran on its own: it only resets the integrity flag/deadlock
            # set, so crashed fleets stayed stacked on the exact same cell and re-triggered
            # a "Fatal collision" on every subsequent step forever.
            # DOUBLE-RECOVERY GUARD. current_integrity is the value captured at
            # the top of this step, but _policy_recovery_step() ran since then and
            # may already have collapsed and called force_repair(). Without this
            # check the legacy path fires a SECOND collapse on the now-empty
            # conflict set -- which is what produced the stream of "Initiating
            # Spatial-Temporal Collapse for 0 nodes" followed by "Tier 1: Spatial
            # Escape Successful": a success report for recovering nothing, and a
            # spurious increment of the forced-invocation counter that is supposed
            # to measure how often the policy FAILED to act.
            if self._recovery_ran_this_step:
                recovery_result = {"reinit_from": "already_recovered_this_step"}
            else:
                recovery_result = self._run_recovery(forced=True)

            # Tier 3 ("yield") declares a winner and expects every OTHER conflicted
            # fleet to hold still -- now for an escalating number of steps (see
            # WarehouseRecovery's Tier 3 for why), not just this one step. Enforced
            # below in the action-application loop via self._yield_until, since
            # otherwise the GNN action chosen a few lines down would simply
            # overwrite the loser's direction again and the "yield" would have no
            # actual effect.
            # Read yield_durations regardless of WHICH tier resolved the geometry.
            # It used to be gated on mode == "yield", so a hold could only ever be
            # imposed by Tier 3 -- and Tier 3 never fired, because Tier 1 or 2
            # always "succeeded" first. That is exactly how fleets 9 and 24 kept
            # colliding at steps 107/142/177/189/203 in a single episode while
            # tier3_recoveries stayed at 0 for 30 straight episodes. Tiers 1 and 2
            # now attach a hold themselves on a repeat offence, so the pair is
            # separated in TIME as well as space.
            if recovery_result.get("winner") is not None:
                tier3_winner_id = recovery_result.get("winner")
            for loser_id, duration in recovery_result.get("yield_durations", {}).items():
                self._yield_until[loser_id] = self.step_count + duration
            
        if current_integrity == 0.5:
            # Predictive Warning (Safety Bubble): Splat mild severity between fleets
            warning_list = list(self.loop.warning_nodes)
            if len(warning_list) >= 2:
                n1 = next(n for n in self.nodes if n.id == warning_list[0])
                n2 = next(n for n in self.nodes if n.id == warning_list[1])
                midpoint = (n1.current_pos + n2.current_pos) / 2.0
                self.density.splat_spatial_temporal_event(midpoint, severity_multiplier=self.warning_splat_multiplier)

        # Decay old Spatial-Temporal memory
        self.density.step_decay()

        # Pool mode: a fleet's committed target may have been claimed by a peer
        # since the last time this fleet checked (retarget_to_nearest_unclaimed()
        # only runs when THIS fleet claims a goal or is first spawned -- it has
        # no way to know a peer got there first except by being told here).
        # Checked before state-building so both what the GNN perceives THIS step
        # and this step's reward calculation use the fleet's actual current
        # target, not a stale claimed one it's still pointlessly traveling toward.
        if self.shared_pool_mode:
            for node in self.nodes:
                if node.id in self.immobile_nodes:
                    continue
                # A rescuer on leg 1 has no pool goal committed -- its target is
                # a pickup cell -- so the staleness check does not apply to it.
                if node.id in self._pickup_assignment:
                    continue
                if node.current_goal_id in self.claimed_goals:
                    found = self._retarget(node)
                    if not found:
                        # Every goal claimed while this fleet was en route to one
                        # that got taken -- nothing left for it to do.
                        self._release_target(node)
                        self.frozen_nodes.add(node.id)
                        self.gnn.freeze_node(node.id, node.current_pos)
                        print(f"[Core] Fleet {node.id} frozen -- pool exhausted while retargeting.")

        # 2. Build States & Get GNN Actions
        node_features = []
        node_ids = []
        valid_action_masks = []
        
        for node in self.nodes:
            node_ids.append(node.id)
            # Base ray-cast state
            base_state = node.get_state_vector(self.nodes)
            # Local Poisson affordance
            local_affordance = self.density.get_local_affordance(
                node.current_pos, self.nodes, self.immobile_nodes,
                own_goal_pos=node.goal_pos,
                frozen_obstacle_severity=self.frozen_obstacle_severity,
                near_goal_radius=self.frozen_near_goal_radius,
                own_id=node.id,
                stopped_node_ids=self.stopped_nodes,
                stopped_obstacle_severity=self.stopped_obstacle_severity,
            )
            # Concatenate
            full_state = np.concatenate([base_state, local_affordance])
            node_features.append(full_state)

            # Which of the 7 actions actually lead somewhere from here -- see
            # get_valid_action_mask()'s docstring for why this matters on a
            # graph this sparse (~2.27 average degree of 6 possible directions).
            # Uses CONFIG's base_speed directly (not a local variable) since
            # per-node braking hasn't been computed yet at this point in
            # step() -- and validity doesn't depend on the exact speed anyway,
            # see the docstring.
            valid_action_masks.append(
                node.get_valid_action_mask(reference_speed=CONFIG["warehouse"]["base_speed"])
            )

        node_features_array = np.array(node_features, dtype=np.float32)
        valid_action_masks_array = np.array(valid_action_masks, dtype=bool)
        
        # Dummy adjacency for active nodes (since GNN relies on p2p connections)
        adj_mat = self._build_adjacency()
        
        # GNN Action Selection
        actions = self.gnn.choose_actions(
            node_features=node_features_array,
            adj_matrix=adj_mat,
            episode_number=episode_step,
            total_episodes=total_episodes,
            node_ids=node_ids,
            valid_action_masks=valid_action_masks_array,
        )

        # 3. Apply Actions, Affordance Braking, and Calculate Rewards
        step_rewards = []
        
        for i, node in enumerate(self.nodes):
            if node.id in self.immobile_nodes:
                step_rewards.append(np.zeros(self.K, dtype=np.float32))
                continue
                
            action_id = actions[i] if actions is not None else 0

            # ---- DIAGNOSTIC: is the policy actually reading the goal gradient? ----
            # Measured on the RAW GNN choice, before the Tier-3 yield override and
            # tabu masking below, because those are not policy decisions and would
            # pollute the signal.
            #
            # Interpretation: 6 movement actions + idle, so a policy ignoring the
            # gradient entirely scores ~1/6 = 0.17. Epsilon-random actions dilute
            # this, so the realistic floor is epsilon/6 + (1-epsilon)*p_policy --
            # compare against epsilon in the same log line, don't read it absolutely.
            # Climbing well past 0.17 as epsilon decays means the six gradient
            # features have landed. Flat at 0.17 with epsilon near 0 means the
            # network is ignoring them, and no amount of extra episodes will help.
            _g = node.get_goal_gradient()
            if np.any(_g != 0):
                self._grad_agree.append(1.0 if action_id == int(np.argmax(_g)) + 1 else 0.0)

            # Livelock escape: hand a stalled fleet back to the BFS gradient, and
            # KEEP it there for a burst of steps rather than a single one.
            #
            # A single-step override is not enough: the cycling policy reclaims
            # control immediately and the fleet needs another full patience window
            # before it gets one more assisted step. Measured against an
            # adversarial always-oscillating policy, single-step gave 9 overrides
            # in 400 steps and left the fleet short; a sustained burst walks it
            # clear of the cycle in one intervention.
            if self.step_count >= self._override_until.get(node.id, -1):
                if self._stall_steps.get(node.id, 0) >= self.livelock_patience:
                    self._override_until[node.id] = self.step_count + self.livelock_escape_steps
                    self._stall_steps[node.id] = 0

            if self.step_count < self._override_until.get(node.id, -1):
                _gl = node.get_goal_gradient()
                if np.any(_gl > 0):
                    action_id = int(np.argmax(_gl)) + 1
                    self.livelock_overrides += 1

            if self.step_count < self._yield_until.get(node.id, -1):
                # Tier 3 Right-of-Way Yield: this fleet is still serving an active
                # yield (possibly escalated from a repeat offense -- see
                # WarehouseRecovery's Tier 3), so it must hold still regardless of
                # what the GNN picked, for as long as the yield window lasts.
                action_id = 0

            # NEW: Tabu Action Masking
            # BUG FIX: this used to be indented one level deeper, inside the
            # "if node.id in forced_yield_ids" block above -- so it only ever ran
            # for Tier-3 yield LOSERS, whose action was already hardcoded to 0
            # regardless (checking whether 0 is tabu there is moot). For every other
            # fleet -- i.e. anyone resolved via Tier 1/2, which your log shows is
            # nearly all of them -- this mistake-blocking check never executed at
            # all, which is exactly why the same fatal action kept getting repeated.
            curr_pos_tuple = tuple(np.round(node.current_pos).astype(int))
            if curr_pos_tuple in node.tabu_actions and action_id in node.tabu_actions[curr_pos_tuple]:
                # The GNN tried to repeat a fatal mistake! Force exploration.
                valid_actions = [a for a in range(7) if a not in node.tabu_actions[curr_pos_tuple]]
                action_id = random.choice(valid_actions) if valid_actions else 0
            
            # BUG THIS FIXES: this used to be get_manhattan_distance_to_goal(), which
            # measures straight-line distance, not the actual path through the
            # warehouse graph. Wherever the real layout forces a detour around a gap
            # in the grid, a fleet's only available move can reduce its true
            # graph-distance to goal while INCREASING straight-line distance -- so
            # the old reward scored that forced, correct move as "wandering away".
            # See precompute_goal_distances() and get_graph_distance_to_goal() in
            # node_warehouse.py for the full mechanism and a worked example.
            old_dist = node.get_graph_distance_to_goal()
            old_pos = node.current_pos.copy()  # NEW: for genuine-idle detection below
            
            # Affordance Braking: throttle speed based on GENUINE proximity to the
            # nearest other fleet, using the same collision_threshold /
            # warning_threshold zones the rest of the system (loop integrity,
            # recovery, reward shaping) already validates against.
            #
            # BUG THIS FIXES: this used to read the *summed* local affordance
            # field's center value. The density field's "Active Peer" repulsion is
            # summed across EVERY nearby peer (0.7 severity each, unbounded). At
            # realistic fleet counts, just ~8 OTHER fleets sitting at a harmless
            # distance of 2-4 cells -- nowhere near collision_threshold=1.0 or even
            # warning_threshold=2.0 -- was already enough to sum past 1.0 and floor
            # local_safety, which floors speed to 0.05 units/step (10% of
            # base_speed). With ~20-25 fleets sharing a warehouse, that was true
            # almost everywhere, almost always: fleets were being throttled to a
            # crawl by the mere PRESENCE of distant, harmless peers, not by any
            # genuine nearby threat -- which is why they barely moved across an
            # entire episode. Using the single nearest peer's real distance (with a
            # linear ramp between the two existing thresholds) fixes that, and also
            # removes a second, redundant get_local_affordance() call that was
            # happening here on top of the one already done for the state vector.
            base_speed = node.speed
            # BUG FIX: this used to be `(node.id in self.loop.deadlocked_nodes) and
            # (node.id not in forced_yield_ids)`, which is True for EVERY deadlocked
            # fleet whenever recovery resolved via Tier 1/2 (forced_yield_ids stays
            # empty in that case) -- not just the one specific fleet Tier 3 actually
            # declared the winner. Per your log, Tier 2 succeeds almost every time,
            # so in practice *both* fleets in a pair were getting slammed with
            # speed=1.1 on the very same step they'd just been rewound to a safe
            # distance apart -- quite possibly walking them right back together.
            is_tier3_winner = (node.id == tier3_winner_id)

            # If they are the Tier 3 winner, give them a physical bump to help
            # clear the 1.0 threshold and break the loop.
            if is_tier3_winner:
                node.speed = 1.1 
            else:
                # Only fleets that can actually COLLIDE with this one brake it.
                #
                # BUG THIS FIXES: this used to measure distance to every other
                # fleet, including parked and stopped ones -- which
                # check_integrity() explicitly exempts from collisions. So a
                # fleet could be throttled to speed 0.05 (local_safety floor 0.1
                # x base_speed 0.5) by a neighbour it is physically incapable of
                # crashing into, with no safety benefit whatsoever, and stay
                # throttled for as long as it remained nearby. Measured: a
                # rescuer heading for a pickup sat at ~0.15 cells/step and
                # oscillated for 385 consecutive steps three cells short of its
                # target, held there by a single parked peer. It also never
                # escaped, because at that speed it could not improve its best
                # distance enough to reset the stall counter cleanly.
                #
                # This is the same class of error as the summed-affordance bug
                # already documented above: braking driven by the mere PRESENCE
                # of a harmless peer rather than by a genuine collision risk. The
                # density field still repels around immobile fleets, so routing
                # around them is unaffected -- what changes is that they no
                # longer apply a speed penalty they cannot justify.
                if CONFIG.get("ablation", {}).get("brake_on_immobile", False):
                    other_positions = [n.current_pos for n in self.nodes
                                       if n.id != node.id]
                else:
                    other_positions = [n.current_pos for n in self.nodes
                                       if n.id != node.id and n.id not in self.immobile_nodes]
                if other_positions:
                    min_dist = float(np.min(np.sum(np.abs(np.array(other_positions) - node.current_pos), axis=1)))
                else:
                    min_dist = float('inf')

                if min_dist <= self.loop.collision_threshold:
                    # Floor at a small positive value instead of letting this hit
                    # exactly 0 (same anti-deadlock reasoning as before: exactly-zero
                    # speed can never resolve a collision on its own, and a crashed
                    # fleet needs to always be able to crawl away, however slowly).
                    # NOTE: final-approach never overrides this -- genuine collision
                    # distance stays safety-first regardless of how close to done a
                    # fleet is.
                    local_safety = 0.1
                elif min_dist >= self.loop.warning_threshold:
                    # Nothing genuinely nearby -- no need to throttle at all.
                    local_safety = 1.0
                else:
                    # Linear ramp between the two existing thresholds.
                    frac = (min_dist - self.loop.collision_threshold) / (self.loop.warning_threshold - self.loop.collision_threshold)
                    local_safety = 0.1 + frac * 0.9

                    # Final-approach override: many fleets in this dataset have
                    # NEARBY goals, so ending up in the warning zone right as BOTH
                    # fleets are about to finish is common and often not a real
                    # conflict -- but the ramp above would still throttle this
                    # fleet right when it should be pushing through. Once it's
                    # covered final_approach_threshold of its ORIGINAL journey,
                    # raise its floor instead of leaving it to the normal ramp.
                    # Only relaxes the warning-zone case above -- the fatal branch
                    # above this one is untouched, so real collision risk still
                    # always wins over "almost done."
                    if node.get_progress_fraction() >= self.final_approach_threshold:
                        local_safety = max(local_safety, self.final_approach_speed_floor)

                node.speed = base_speed * local_safety
            
            # Execute physical move
            _pos_before = node.current_pos.copy()
            # RECORD WHAT WAS EXECUTED, not what the network picked. action_id is a
            # local that four paths overwrite above (livelock escape, Tier-3 yield,
            # forced dwell, tabu masking) and none wrote back into `actions`, which
            # is what reaches the replay buffer. Every overridden step therefore
            # stored (state, action_CHOSEN, reward_from_action_TAKEN) -- mislabelled
            # data, not off-policy data. Measured at 13-24% of transitions.
            self.actions_total += 1
            if int(actions[i]) != int(action_id):
                self.actions_overridden += 1
            actions[i] = action_id

            node.apply_discrete_action(action_id)

            # --- motion accounting (see _fleet_travel in __init__) ---
            _moved = float(np.sum(np.abs(node.current_pos - _pos_before)))
            self._fleet_travel[node.id] = self._fleet_travel.get(node.id, 0.0) + _moved
            self._fleet_steps[node.id] = self._fleet_steps.get(node.id, 0) + 1
            self._fleet_speed[node.id] = self._fleet_speed.get(node.id, 0.0) + float(node.speed)
            if _moved < 1e-6:
                self._fleet_idle[node.id] = self._fleet_idle.get(node.id, 0) + 1
            
            # Restore base speed for next calculation
            node.speed = base_speed

            # 4. Dense Reward Calculation
            # REWORKED (2026-08-25): unconditional baseline + reward PROPORTIONAL
            # to actual progress, replacing the old flat +1.0/-1.5/-2.0 scheme --
            # see config_warehouse.py's "rewards" section for the full rationale
            # (borrowed from the FLOWRRA-GNN swarm version, which never had a
            # frozen-fleet problem).
            # Reward is a K-VECTOR now, one slot per head, not a scalar. Nothing
            # is summed before it reaches the buffer -- that summing is what made
            # the rare components (rescue at ~0.1% of episode reward mass)
            # unrecoverable from the learning signal.
            rvec = np.zeros(self.K, dtype=np.float32)
            H = self.HEAD
            rvec[H["goal"]] += self.baseline_reward
            new_dist = node.get_graph_distance_to_goal()

            # A fleet counts as progressing only when it beats its BEST distance
            # so far this episode, not merely its previous step -- otherwise an
            # oscillation between two cells resets the counter every other step
            # and the stall is never detected, which is exactly the failure mode
            # this guards against.
            best = self._best_dist.get(node.id)
            if best is None or new_dist < best - 1e-6:
                self._best_dist[node.id] = new_dist
                self._stall_steps[node.id] = 0
            else:
                self._stall_steps[node.id] = self._stall_steps.get(node.id, 0) + 1
            
            # A0. HANDOVER SERVICING. Must run BEFORE the mission-complete check
            # below, because a rescuer on leg 1 has goal_pos set to the PICKUP
            # CELL -- so new_dist < 0.1 would otherwise fire the normal arrival
            # path and try to claim a pool goal that isn't there.
            _on_pickup_mission = node.id in self._pickup_assignment
            if _on_pickup_mission:
                rvec[H["rescue"]] += self._service_pickups(node)
                if node.id not in self._pickup_assignment:
                    # Transfer completed THIS step: the fleet's goal just changed
                    # to leg 2. old_dist was measured against the retired leg, so
                    # the proportional movement term below would read the leg
                    # switch as one enormous jump. Rebase both to the new leg so
                    # the transfer step scores 0 movement -- the pickup bonus is
                    # what pays for it.
                    new_dist = node.get_graph_distance_to_goal()
                    old_dist = new_dist

            # A. Mission Complete
            if new_dist < 0.1 and not _on_pickup_mission:
                if self.shared_pool_mode:
                    if node.current_goal_id is not None and node.current_goal_id not in self.claimed_goals:
                        # First arrival -- claim it, and retire immediately.
                        # BUG THIS FIXES (design change, not a defect): "keep
                        # going" (claim, then retarget to the next-nearest,
                        # indefinitely) kept active fleet count fixed at 25
                        # while the pool shrank -- so the ratio of competing
                        # fleets to remaining goals only grew as the episode
                        # progressed, concentrating more fleets onto fewer
                        # points right when contention should be easing, not
                        # worsening. "Keep going" was meant as a resilience
                        # mechanism for a fleet-breakdown scenario that doesn't
                        # exist yet in this simulation -- right now it only
                        # supplied that downside without the upside (covering
                        # for a disabled peer) ever coming into play. One claim
                        # per fleet keeps active count shrinking in step with
                        # the pool instead.
                        self.claimed_goals.add(node.current_goal_id)
                        delivered = node.current_goal_id
                        rvec[H["goal"]] += self.reward_mission_complete

                        # A rescuer carrying two loads still owes the second
                        # destination -- it must not retire after the first.
                        pending = [g for g in self._pending_goals.get(node.id, [])
                                   if g not in self.claimed_goals]
                        if pending:
                            nxt = pending[0]
                            self._pending_goals[node.id] = pending[1:]
                            # Release only the goal just delivered, keeping the
                            # queued reservation intact: _release_target drops
                            # EVERY reservation this fleet holds, which would
                            # hand the second order to a peer mid-delivery.
                            self.targeted_goals.pop(delivered, None)
                            self.targeted_goals[nxt] = node.id
                            node.current_goal_id = nxt
                            node.goal_pos = (self.goal_pool[nxt].copy()
                                             if self.shared_pool_mode else node.goal_pos)
                            node.goal_distance_map = self.goal_distance_maps.get(nxt)
                            node.initial_graph_distance = node.get_graph_distance_to_goal()
                            self._best_dist.pop(node.id, None)
                            self._stall_steps[node.id] = 0
                            print(f"[Core] Fleet {node.id} delivered {delivered} "
                                  f"({len(self.claimed_goals)}/{len(self.goal_pool)} claimed); "
                                  f"still carrying {nxt}, "
                                  f"{node.initial_graph_distance:.0f} hops on.")
                        else:
                            # Reservation has served its purpose; the goal is now
                            # permanently claimed, so drop it from the ledger to keep
                            # targeted_goals meaning "actively being pursued".
                            self._release_target(node)
                            self._pending_goals.pop(node.id, None)
                            self.frozen_nodes.add(node.id)
                            self.gnn.freeze_node(node.id, node.current_pos)
                            print(f"[Core] Fleet {node.id} claimed goal {delivered} and retired! "
                                  f"({len(self.claimed_goals)}/{len(self.goal_pool)} claimed)")
                    else:
                        # Arrived at a goal a peer claimed first (this same
                        # step, or while this fleet was still en route) --
                        # nothing was actually accomplished, so no bonus and no
                        # retirement, just retarget and keep trying.
                        found = self._retarget(node)
                        if not found:
                            self.frozen_nodes.add(node.id)
                            self.gnn.freeze_node(node.id, node.current_pos)
                            print(f"[Core] Fleet {node.id} frozen -- pool fully claimed "
                                  f"({len(self.claimed_goals)}/{len(self.goal_pool)}).")
                else:
                    rvec[H["goal"]] += self.reward_mission_complete
                    self.frozen_nodes.add(node.id)
                    self.gnn.freeze_node(node.id, node.current_pos)
                    print(f"[Core] Fleet {node.id} reached goal and crystallized!")
            else:
                # B. Movement: proportional to actual progress, not a flat
                # bonus/penalty regardless of magnitude. A full-speed step
                # toward goal earns much more than a heavily-braked one; a
                # forced detour (the only available progress when the direct
                # path is blocked) earns something close to zero instead of
                # being punished as hard as a genuinely wrong move.
                if np.array_equal(old_pos, node.current_pos):
                    # Genuinely didn't move at all this step (idle action, or
                    # an invalid/blocked move) -- NOT "moved but zero net
                    # progress", which the proportional term below already
                    # handles correctly on its own (nets to ~0 without needing
                    # a separate check).
                    #
                    # EXCEPTION: a rescuer part-way through its pickup dwell is
                    # holding still ON PURPOSE, for a fixed number of steps we
                    # imposed. Penalising it would teach the policy to leave the
                    # cell before the transfer completes, which is the one thing
                    # the dwell exists to prevent.
                    if self._pickup_dwell.get(node.id, 0) == 0:
                        rvec[H["time"]] += self.idle_penalty
                else:
                    rvec[H["goal"]] += (old_dist - new_dist) * self.movement_reward_multiplier

                    # Final-approach push: reinforce that pushing through the
                    # warning zone late in the journey is GOOD, not risky --
                    # directly countering the natural pull of warning-zone
                    # caution right when a fleet should be finishing. Scoped to
                    # genuine progress specifically, same as before.
                    if (new_dist < old_dist and node.id in self.loop.warning_nodes
                            and node.get_progress_fraction() >= self.final_approach_threshold):
                        rvec[H["goal"]] += self.final_approach_bonus

                # C. Overtime pressure: escalating penalty for still not being
                # home once the episode runs past overtime_threshold_steps --
                # ramps linearly from 0 at that step to overtime_max_penalty at
                # max_steps_per_episode. Only reached here (not in the mission-
                # complete branch above), so a fleet that finishes exactly
                # during overtime still gets a clean completion bonus, not a
                # muddled one.
                if self.step_count >= self.overtime_threshold_steps:
                    span = max(1, self.max_steps_per_episode - self.overtime_threshold_steps)
                    overtime_frac = min(1.0, (self.step_count - self.overtime_threshold_steps) / span)
                    rvec[H["time"]] += overtime_frac * self.overtime_max_penalty
                    
            # D. Spatial-Temporal Consequences
            if node.id in self._step_deadlocked:
                rvec[H["safety"]] += self.reward_fatal_collision
            elif node.id in self._step_warning:
                # GRADED PROXIMITY PENALTY, replacing what used to be a +0.5
                # BONUS for occupying the exact band that precedes a collision.
                #
                # Nothing in the reward discouraged approach and something
                # actively paid for it, which is the mechanical reason Tier 1
                # only ever fired AFTER impact -- the policy had no gradient
                # pointing away from danger until the danger had happened.
                #
                # The penalty scales with how far INTO the band the fleet is:
                # ~0 at warning_threshold, full magnitude just outside a crash.
                # So Q-values for closing actions fall while there is still room
                # to act, which is preemptive avoidance learned rather than
                # overridden. In a Manhattan grid where d <= collision_threshold
                # IS a crash, there are no legitimate close passes to protect --
                # the only pass-through in the system is over an IMMOBILE fleet,
                # and those are exempt from check_integrity entirely.
                #
                # No cooldown: the old one existed only to stop two fleets
                # farming a repeated BONUS by loitering. A penalty cannot be
                # farmed, and every step spent near a crash is a step of risk.
                nearest = min(
                    (float(np.sum(np.abs(node.current_pos - o.current_pos)))
                     for o in self.nodes
                     if o.id != node.id and o.id not in self.immobile_nodes),
                    default=self.loop.warning_threshold)
                span = max(1e-6, self.loop.warning_threshold - self.loop.collision_threshold)
                closeness = float(np.clip(
                    (self.loop.warning_threshold - nearest) / span, 0.0, 1.0))
                rvec[H["safety"]] += self.reward_warning_zone * closeness

            # E. Integrity: a per-step signal on how coherent the holon is, so
            # the integrity head has something dense to learn from rather than
            # only the sparse invocation events.
            rvec[H["integrity"]] += (self.loop.current_integrity - 1.0)
            rvec[H["integrity"]] += self._pending_integrity_reward

            step_rewards.append(rvec)

        self._pending_integrity_reward = 0.0

        # [N, K] -- one row per fleet, one column per reward head.
        step_rewards_array = np.asarray(step_rewards, dtype=np.float32)
        
        # 5. Push to GNN Memory (Training Mode)
        # BUG THIS FIXES: this used to be a no-op comment ("assuming ... handled
        # internally") -- nothing, anywhere in the codebase, ever called
        # self.gnn.memory.push(...). That meant len(shared_agent.memory) was
        # always 0, so main_runner_warehouse.py's `if len(shared_agent.memory) >=
        # shared_agent.batch_size: shared_agent.learn(...)` could never fire.
        # The network's weights were never once updated by gradient descent, for
        # the entire training run -- which fully explains a reward that never
        # improves no matter how many episodes run: there was no learning
        # happening at all, just a fixed (randomly-initialized) policy plus
        # whatever fraction of actions epsilon made random that episode.
        if self.gnn is not None and hasattr(self.gnn, "memory") and actions is not None:
            next_node_features = []
            for node in self.nodes:
                base_state = node.get_state_vector(self.nodes)
                local_affordance = self.density.get_local_affordance(
                    node.current_pos, self.nodes, self.immobile_nodes,
                    own_goal_pos=node.goal_pos,
                    frozen_obstacle_severity=self.frozen_obstacle_severity,
                    near_goal_radius=self.frozen_near_goal_radius,
                    own_id=node.id,
                    stopped_node_ids=self.stopped_nodes,
                    stopped_obstacle_severity=self.stopped_obstacle_severity,
                )
                next_node_features.append(np.concatenate([base_state, local_affordance]))
            next_node_features_array = np.array(next_node_features, dtype=np.float32)
            next_adj_mat = self._build_adjacency()

            # TERMINAL means "no fleet can still act", which with error stops is
            # NOT the same as "every fleet succeeded". A permanently stopped
            # fleet whose pickup nobody can service is a genuine terminal state
            # and should bootstrap as one; main_runner tracks SUCCESS separately
            # off frozen_nodes.
            done = self.is_episode_over()

            # Per-node active mask, recorded AT THIS TIMESTEP. Parked fleets
            # push reward 0.0 with an arbitrary action from a position sitting on
            # a goal; without this they train the network to value goal-adjacent
            # states at ~0. See GNNAgent.learn()'s masked-loss comment.
            active_mask = np.array(
                [n.id not in self.immobile_nodes for n in self.nodes], dtype=np.float32
            )

            self.gnn.memory.push(
                node_features_array,
                adj_mat,
                actions,
                step_rewards_array,
                next_node_features_array,
                next_adj_mat,
                done,
                current_integrity,
                active_mask,
                self.last_recovery_mode,
            )
        
        self.step_count += 1
        # Scalar return value is the WEIGHTED SUM across heads, for logging
        # only -- the buffer receives the full [N, K] matrix.
        return float(step_rewards_array.sum()) if step_rewards_array.size else 0.0