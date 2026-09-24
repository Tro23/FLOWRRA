"""
core_Warehouse.py

Streamlined FLOWRRA Orchestrator for discrete warehouse AGV fleets.
Integrates dual-zone loop integrity, Poisson density fields, affordance braking,
and highly targeted Manhattan reward gradients.
"""

from typing import Any, Dict, List, Optional, Set, Tuple
import random
import zlib
import numpy as np

from config_warehouse import CONFIG
from agent_warehouse import GNNAgent
from node_warehouse import (FleetNode, build_spatial_indices, precompute_goal_distances,
                             assign_goals_optimally)
from loop_warehouse import WarehouseLoop
from proximity_warehouse import GraphProximity
from density_warehouse import WarehouseDensityField
from recovery_warehouse import WarehouseRecovery


# Reward layouts. LEGACY: the five heads every reward line is written against.
# PRIORITY: the three heads of REWARD_DESIGN.md -- safety > delivery > efficiency.
LEGACY_HEADS = ("goal", "safety", "integrity", "rescue", "time")
PRIORITY_HEADS = ("safety", "delivery", "efficiency")

import node_warehouse as _node_module

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
        self.arrival_radius = float(
            CONFIG["warehouse"].get("arrival_radius", 0.1))
        # Doorstep diagnostic: what a fleet within one hop of its goal actually
        # does. Reward at that point is +101.5 to step in, -0.5 to idle, -1.5 to
        # retreat -- a 103-point gap -- and fleets still step in only 6% of the
        # time. Reward, gradient, speed and action mask have all been ruled out,
        # so the remaining question is what the NETWORK ranks there.
        self._doorstep: Dict[str, int] = {
            "steps": 0, "arrived": 0, "closer": 0, "idle": 0, "away": 0,
            "overridden": 0, "peer_within_warning": 0}
        self._best_dist: Dict[str, float] = {}
        self._stall_steps: Dict[str, int] = {}
        self._override_until: Dict[str, int] = {}
        self.livelock_patience = int(CONFIG["recovery"].get("livelock_patience", 40))
        self.livelock_escape_steps = int(CONFIG["recovery"].get("livelock_escape_steps", 20))
        self.livelock_overrides = 0
        # Tabu masking accounting. tabu_overrides_stuck counts the degenerate
        # case where every structurally legal action from a cell is tabu; if it
        # is ever non-trivial, the tabu list is over-growing and needs a decay.
        self.tabu_overrides = 0
        self.tabu_overrides_stuck = 0

        # Braking accumulators, read and reset per block by profile_step.py.
        # The ray budget used to be 1/speed, so how hard fleets brake was the
        # leading hypothesis for the scaling wall; it turned out not to be, but
        # only because the number was finally measured at the right moment.
        self._braked_speed_sum = 0.0
        self._braked_speed_n = 0
        self._braked_speed_min = float("inf")

        # [normal, edge] mobile fleet-steps, and how many of each were fatal or
        # in the warning band that step.
        self._edge_steps = [0, 0]
        self._edge_collisions = [0, 0]
        self._edge_warnings = [0, 0]

        _percfg = CONFIG.get("perception", {})
        self._idle_mode = _percfg.get("idle_mode", "full")
        if self._idle_mode not in ("full", "cached", "skip"):
            raise ValueError(f"idle_mode must be full|cached|skip, got {self._idle_mode!r}")
        self._idle_refresh_every = int(_percfg.get("idle_refresh_every", 10))
        self._idle_perception_reused = 0
        self._idle_perception_computed = 0
        
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
            projection_max_branches=CONFIG["density"].get("projection_max_branches", 6),
            project_stationary=CONFIG["density"].get("project_stationary", True),
            kernel_metric=CONFIG["density"].get("kernel_metric", "graph"),
            output_mode=CONFIG["density"].get("output_mode", "affordance"),
            slow_channel=CONFIG["density"].get("slow_channel", False),
            slow_decay_factor=CONFIG["density"].get("slow_decay_factor", 0.987),
            slow_severity_scale=CONFIG["density"].get("slow_severity_scale", 1.0),
            projection_mode=CONFIG["density"].get("projection_mode", "intended"),
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
            max_yield_steps=CONFIG["recovery"].get("max_yield_steps", 30),
            escape_clearance=CONFIG["recovery"].get("escape_clearance", 0.0),
            warning_threshold=CONFIG["warehouse"]["warning_threshold"],
            escalate_on_recurrence=CONFIG["recovery"].get("escalate_on_recurrence", False),
            recurrence_radius=CONFIG["recovery"].get("recurrence_radius", None),
            pair_escalation_threshold=CONFIG["recovery"]["pair_escalation_threshold"],
            frozen_obstacle_severity=self.frozen_obstacle_severity,
            frozen_near_goal_radius=self.frozen_near_goal_radius,
        )
        self.fatal_splat_multiplier = CONFIG["density"]["fatal_splat_multiplier"]
        self.warning_splat_multiplier = CONFIG["density"]["warning_splat_multiplier"]

        # ---- GRAPH PROXIMITY (2026-09-13) ----------------------------------
        # One index, refreshed once per step, feeding all four sites that used to
        # each compute Manhattan distance on raw coordinates independently:
        # sf_peer_proximity, braking, the safety reward, and check_integrity.
        # Having four copies of a metric is how they drift apart; having one is
        # how the state feature, the reward and the mechanism keep speaking the
        # same language. See proximity_warehouse.py.
        _pcfg = CONFIG.get("proximity", {})
        self.adjacency_metric = _pcfg.get("adjacency_metric", "manhattan")
        # 0 disables edge features entirely and the adjacency stays [N, N], which
        # is what every previous run used. Kept as a working identity path so the
        # mechanism is ablatable after the retrain rather than assumed.
        self.edge_feature_dim = int(CONFIG["gnn"].get("edge_feature_dim", 0))
        # The index must reach as far as its FURTHEST consumer. Braking and
        # integrity clamp at warning_threshold (2.0), but the attention graph
        # uses interaction_radius (10.0), so a radius-4 index would silently
        # truncate the adjacency. Raising it costs a deeper BFS per fleet --
        # depth 10 instead of 4 on a degree-~2.27 graph, so tens of nodes rather
        # than ten -- and it WIDENS the censoring point of mean_peer_gap, which
        # is reported at search_radius. That metric is therefore not comparable
        # across this change.
        _radius = max(
            float(_pcfg.get("search_radius", 4.0)),
            float(CONFIG["warehouse"]["warning_threshold"]),
        )
        if self.adjacency_metric == "graph":
            _radius = max(_radius, self.interaction_radius)
        self.proximity = GraphProximity(
            graph=self.G,
            grid_pos_dict=self.grid_pos_dict,
            search_radius=_radius,
            metric=_pcfg.get("metric", "graph"),
        )
        # Sampled phantom-pair measurement: every Nth step, count how many pairs
        # Manhattan would have flagged that the graph metric rejects. 0 disables.
        self.phantom_audit_every = int(_pcfg.get("phantom_audit_every", 0))

        # ---- LESION (2026-09-13) -------------------------------------------
        # Zero named blocks of the state vector at INFERENCE time, on an already
        # trained checkpoint, to measure how much the policy actually uses them.
        # This is a lesion, not an ablation: it does not retrain, so it answers
        # "does the trained policy read this?" and NOT "could a policy trained
        # without it have done as well?". Those are different questions and the
        # harness says so in its own header.
        #
        # Applied at the single point where the base and density blocks are
        # concatenated, so the two state builds in step() cannot diverge.
        # RESOLVED LAZILY, not here.
        #
        # BUG THIS FIXES (2026-09-14): this block used to call
        # self.nodes[0].state_layout() at this point in __init__ -- but fleets
        # are not spawned until line ~405, so self.nodes does not exist yet.
        # It raised AttributeError on the FIRST construction with a lesion
        # configured.
        #
        # It went undetected through an entire session because the guard is
        # `if self._lesion_names`, and every run until now used only the
        # `control` arm, where that list is empty. The lesion hook had literally
        # never executed. A code path gated behind a config flag is a code path
        # that is not being tested.
        self._lesion_names = list(CONFIG.get("lesion", {}).get("zero_blocks", []) or [])
        self._lesion_slices: Optional[List[Tuple[int, int]]] = None

        # Second index with NO exclusions, built only when the brake_on_immobile
        # ablation is active. That ablation asks whether parked fleets should
        # still throttle traffic, which needs a different exclusion set, not a
        # different metric. Kept as a separate object refreshed once per step
        # rather than rebuilt inside the per-fleet loop, which would have been
        # N rebuilds per step.
        self._prox_all = (
            GraphProximity(
                graph=self.G,
                grid_pos_dict=self.grid_pos_dict,
                search_radius=self.proximity.search_radius,
                metric=self.proximity.metric,
            )
            if CONFIG.get("ablation", {}).get("brake_on_immobile", False)
            else None
        )

        # Maps node_id -> the step_count at which its forced Tier-3 yield expires.
        # A node yields for as long as self.step_count < this value -- see step()'s
        # action-application loop below. Replaces the old one-step-only
        # forced_yield_ids, which let a loser immediately retry the same losing
        # move the very next step.
        self._yield_until: Dict[str, int] = {}
        # Cumulative steps each fleet spent HELD by recovery or voluntarily
        # WAITING, across the whole episode -- not just its state at the end.
        self._steps_held: Dict[str, int] = {}
        rc = CONFIG["recovery"]
        self.preempt_skip_convoys = bool(rc.get("preempt_skip_convoys", False))
        # See _drop_handled_pairs: a pair with one fleet already held is not a
        # new conflict -- the winner passing its held loser is what the hold is
        # FOR. False: old behaviour, exactly reproducible.
        self.preempt_skip_held_pairs = bool(rc.get("preempt_skip_held_pairs", False))
        self.convoy_alignment = float(rc.get("convoy_alignment", 0.7))
        self.convoy_fleets_exempted = 0
        self.convoy_invocations_skipped = 0
        self.held_pair_invocations_skipped = 0
        self.held_pair_fleets_exempted = 0
        self.convoy_splats_skipped = 0
        # Order-independent step: phase B reads snapshots, simultaneous arrivals
        # are resolved by distance, retargets are deferred, and splats are
        # computed from the committed configuration. See SIMULTANEOUS_STEP.md.
        self.simultaneous_step = bool(
            CONFIG.get("step", {}).get("simultaneous", False))
        # Recovery escapes fleets one at a time and picks a right-of-way winner;
        # both used to inherit the order of self.nodes. See WarehouseRecovery.
        self.recovery.order_independent = self.simultaneous_step
        # Steps on which recovery moved a fleet and the index was re-taken.
        self.post_recovery_refreshes = 0
        # Dead fleets whose handover completed, and every order taken over from
        # a dead fleet. Measurement only.
        self._rescued_dead: Set = set()
        self._inherited_orders: Set = set()
        self.inherited_orders_delivered = 0
        self.warning_splats = 0
        self.warning_splat_per_pair = bool(
            CONFIG["density"].get("warning_splat_per_pair", False))
        self._steps_waiting: Dict[str, int] = {}

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

        # ---- WAITING (Phase 2) ---------------------------------------------
        # A fleet is WAITING when it held still BECAUSE it was blocked, as
        # distinct from idling in open space, from being parked at its goal, and
        # from being dead. Those four look identical through peer_velocities --
        # all have direction 0 -- which is why a peer could never tell whether
        # the fleet in front of it would ever move again.
        #
        # This is a peer-VISIBLE state, not just an internal flag: it is what
        # ray_hit_waiting reads, and what a VDA 5050 state report would carry.
        # ---- UNREGISTERED OBSTACLES ----------------------------------------
        # Humans, debris, a dropped pallet. Cells that are blocked by something
        # NOBODY TOLD THE ORCHESTRATOR ABOUT.
        #
        # They are categorically different from every fleet obstacle. A fleet is
        # in the registry, so its position is reported, and the near-goal
        # discount can make it transparent to whoever needs to reach it. Nobody's
        # goal is ever a human, so an unregistered obstacle gets NO transparency
        # rule -- it is simply in the way, for everyone, always.
        #
        # Humans and debris are one category deliberately. They differ in how
        # they arrive and how long they last, not in what a fleet should do
        # about them, and a single channel keeps the state vector honest until
        # there is evidence the distinction matters.
        #
        # Detected only by RAY CAST. Position reports cover the fleet registry;
        # the graph covers static structure; a ray is the only channel in which
        # something unregistered can exist at all.
        # ---- DESPAWN ON DELIVERY (Phase 4, default off) ---------------------
        _ecfg = CONFIG.get("episode", {})
        self.despawn_on_delivery = bool(_ecfg.get("despawn_on_delivery", False))
        self._despawning: Dict[str, str] = {}
        self.despawned_nodes: Set[str] = set()
        self.exit_nodes: List[str] = []
        self._exit_distance_maps: Dict[str, Dict[str, int]] = {}
        self._coords_by_id = {v: k for k, v in self.grid_pos_dict.items()}
        if self.despawn_on_delivery:
            self._init_exits(_ecfg)

        self.static_obstacles: Set[Tuple[int, int, int]] = set()
        # Optional population of humans and debris. None when the feature is
        # off, which is the default. See obstacles_warehouse.py.
        from obstacles_warehouse import from_config as _obstacles_from_config
        self.obstacle_field = _obstacles_from_config(
            self.G, self.grid_pos_dict, CONFIG, seed=CONFIG.get("seed"))
        if self.obstacle_field is not None:
            # Keep clear of fleet starts and every goal in the pool: an episode
            # must not begin with a fleet standing inside a human, or with a
            # goal nobody can reach.
            _avoid = {tuple(np.round(m["start_pos"]).astype(int))
                      for m in fleet_missions if "start_pos" in m}
            _avoid |= {tuple(np.round(p).astype(int))
                       for p in (goal_pool or {}).values()}
            self.obstacle_field.reset(avoid=_avoid)
            self.static_obstacles = self.obstacle_field.occupied_cells()
        self.static_obstacle_severity = float(
            CONFIG["density"].get("static_obstacle_severity", 3.0))

        self.waiting_nodes: Set[str] = set()
        self._wait_steps: Dict[str, int] = {}
        self._wait_cap: Dict[str, int] = {}
        _wcfg = CONFIG.get("waiting", {})
        self.waiting_enabled = bool(_wcfg.get("enabled", False))
        self.max_wait_steps = int(_wcfg.get("max_wait_steps", 12))
        self.wait_block_threshold = float(
            _wcfg.get("block_threshold", CONFIG["warehouse"]["warning_threshold"]))
        self.waits_started = 0
        self.wait_steps_total = 0
        self.waits_capped = 0
        # Denominator for throughput pressure. Read from CONFIG rather than
        # threaded in, because step() has no view of the runner's step budget.
        self._max_steps_hint = int(
            CONFIG["training"].get("max_steps_per_episode", 780))
        self._throughput_t = 1.0
        self.mutual_waits = 0
        self.mutual_wait_steps = int(_wcfg.get("mutual_wait_steps", 3))
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
        # MODE. "legacy" pushes the five heads the reward lines are written
        # against. "priority" still COMPUTES on those five -- every reward line
        # and instrument is written against them -- but PUSHES the three
        # priority heads built from their terms: attributed integrity and
        # recovery, the corrected warning, no baseline, today's progress form,
        # each divided by a fixed value scale. The learner sees only the push.
        self.reward_mode = str(rd.get("mode", "legacy"))
        self.push_heads = list(rd.get("heads", ["total"]))
        if self.reward_mode == "priority":
            if tuple(self.push_heads) != PRIORITY_HEADS:
                raise ValueError(
                    f"reward_decomposition.mode='priority' needs heads "
                    f"{list(PRIORITY_HEADS)}, got {self.push_heads}")
            self.reward_heads = list(LEGACY_HEADS)
        elif self.reward_mode == "legacy":
            self.reward_heads = list(self.push_heads)
        else:
            raise ValueError(f"unknown reward_decomposition.mode {self.reward_mode!r}")
        _sc = rd.get("scales", {}) or {}
        self.priority_scales = {h: float(_sc.get(h, 1.0)) for h in PRIORITY_HEADS}
        if self.reward_mode == "priority" and min(self.priority_scales.values()) <= 0:
            raise ValueError(f"priority scales must be positive: {self.priority_scales}")
        self.K = len(self.reward_heads)
        self.HEAD = {h: i for i, h in enumerate(self.reward_heads)}
        # What the learner actually received, per priority head (active fleets).
        self._live = {h: {"pos": 0.0, "neg": 0.0} for h in PRIORITY_HEADS}

        # "LOOK AT EVERYONE": every fleet perceives the holon's integrity. A
        # module-level switch, so fleets created later -- and the probe the
        # runner uses to measure the state width -- all carry the same width.
        self.holon_perception = bool(
            CONFIG.get("perception", {}).get("holon_integrity", False))
        _node_module.HOLON_PERCEPTION = self.holon_perception

        # REWARD COMPOSITION, per head, positives and negatives kept apart.
        # Measurement only. The runner logs just the episode total, which
        # cannot say whether penalties drown the delivery signal: a total of
        # +3000 could be +5000 of delivery minus 2000 of penalty, or +3100 minus
        # 100. Two views --
        #   _all     : every fleet. Sums back to the logged R exactly.
        #   _learned : only fleets ACTIVE at action time -- the rewards that
        #              reach the gradient. A reward on an inactive fleet is
        #              multiplied by a zero mask and teaches nothing.
        self._rwd = {view: {sign: {h: 0.0 for h in self.reward_heads}
                            for sign in ("pos", "neg")}
                     for view in ("all", "learned")}

        # PER TERM, inside each head -- measurement only. The per-head view
        # cannot separate delivery from progress shaping inside "goal", or a
        # collision from a warning inside "safety". Each reward line in the step
        # also records its value under its term name; the re-attribution mirrors
        # its moves onto the collision and warning terms. Terms within a head
        # sum to that head exactly.
        self._TERMS = [("goal", "baseline"), ("goal", "delivery"),
                       ("goal", "progress"), ("goal", "final_approach"),
                       ("safety", "collision"), ("safety", "warning"),
                       ("integrity", "coherence"), ("integrity", "recovery_events"),
                       ("rescue", "pickup"), ("rescue", "rescue_delivery"),
                       ("time", "idle"), ("time", "overtime")]
        self._TI = {t: i for i, t in enumerate(self._TERMS)}
        self._term_pos = np.zeros(len(self._TERMS))
        self._term_neg = np.zeros(len(self._TERMS))
        # Per-step spread of each head over ACTIVE fleet-steps -- the scale a
        # normalisation would have to use.
        self._spread_n = 0
        self._spread_sum = np.zeros(self.K)
        self._spread_sq = np.zeros(self.K)
        # Warning re-attribution diagnostic -- see the re-attribution block.
        self._warnundo_n = 0
        self._warnundo_mismatched = 0
        self._warnundo_escaped_kept = 0
        self._warnundo_error = 0.0
        # SHADOW MODE -- the proposed three-head reward (REWARD_DESIGN.md s.10),
        # computed every step alongside the live one and NEVER learned from.
        # Its per-step spread is what fixes the normalisation scales.
        # "nz"/"abs": how OFTEN each head speaks and how LOUDLY when it does. A
        # head of rare large events (collisions) has its std inflated by the
        # spikes; normalising by that std would shrink its everyday signal.
        # Frequency and typical nonzero magnitude give a robust scale instead.
        self._shadow = {h: {"pos": 0.0, "neg": 0.0, "sum": 0.0, "sq": 0.0,
                            "nz": 0, "abs": 0.0}
                        for h in ("safety", "delivery", "efficiency")}
        self._shadow_n = 0

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
        self._goal_claim_step: Dict[str, int] = {}   # goal id -> step it was delivered
        self._brake_eval_steps = 0   # (fleet, step) pairs where braking was evaluated
        self._braked_fleet_steps = 0 # of those, how many were inside the warning band
        self._min_dist_sum = 0.0     # running sum, for the mean gap
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
        self._pending_integrity_by_fleet = {}
        self._pending_integrity_system = 0.0

        # WHICH ACTION GETS CHARGED FOR A COLLISION. check_integrity() runs at
        # the START of a step, so the set it produces describes the PREVIOUS
        # action's geometry -- and the -50 then lands on the CURRENT action.
        # Measured: 100% misattributed, and 83% of the penalised actions had
        # just RESOLVED the overlap.
        self.attribute_collisions_post_action = bool(
            CONFIG["rewards"].get("attribute_collisions_post_action", False))
        self._post_action_deadlocked = set()
        self._post_action_warning = set()

        # Orders a fleet inherited from a rescue, per fleet, cleared as each is
        # delivered. Only populated when rescue_delivery_bonus is non-zero.
        self._rescued_orders: Dict[str, set] = {}
        self.rescue_delivery_bonus = float(
            CONFIG["rewards"].get("rescue_delivery_bonus", 0.0))
        self.rescued_orders_delivered = 0

        # Snapshot active_mask BEFORE the action loop instead of after it, so a
        # fleet that delivers during the step is still active in the transition
        # that earned the reward. False reproduces the old behaviour exactly.
        self.mask_active_before_actions = bool(
            CONFIG["rewards"].get("mask_active_before_actions", False))
        self._active_at_action_time = None
        
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
            ray_origin_fallback=CONFIG.get("ablation", {}).get(
                "ray_origin_fallback", True),
            ray_transform=CONFIG["density"].get("ray_transform", "clip25"),
            ray_softness=float(CONFIG["density"].get("ray_softness", 8.0)),
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
            ray_origin_fallback=CONFIG.get("ablation", {}).get(
                "ray_origin_fallback", True),
            ray_transform=CONFIG["density"].get("ray_transform", "clip25"),
            ray_softness=float(CONFIG["density"].get("ray_softness", 8.0)),
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

    def _init_exits(self, ecfg: Dict[str, Any]) -> None:
        """
        Pick the exit nodes and precompute a distance map to each.

        "boundary" takes cells sitting at the minimum or maximum of any axis --
        the edge of the layout, where a dock or a charging bay would be. An
        explicit list overrides it, because a real warehouse's doors are not
        wherever the coordinate extremes happen to fall.

        Distance maps are computed ONCE at construction, exactly like
        goal_distance_maps. There are only a handful of exits, so this costs a
        few BFS runs at startup and nothing per step.
        """
        import networkx as nx

        explicit = ecfg.get("exit_nodes")
        if explicit:
            self.exit_nodes = [n for n in explicit if n in self.G]
        else:
            cells = list(self.grid_pos_dict)
            if not cells:
                return
            lo = [min(c[a] for c in cells) for a in range(3)]
            hi = [max(c[a] for c in cells) for a in range(3)]
            edge = [c for c in cells
                    if any(c[a] == lo[a] or c[a] == hi[a] for a in range(3))]
            # Cap the count: every exit costs a stored BFS map, and a fleet only
            # ever needs the nearest one.
            cap = int(ecfg.get("max_exits", 8))
            if len(edge) > cap:
                step = max(1, len(edge) // cap)
                edge = edge[::step][:cap]
            self.exit_nodes = [self.grid_pos_dict[c] for c in edge]

        for ex in self.exit_nodes:
            try:
                self._exit_distance_maps[ex] = dict(
                    nx.single_source_shortest_path_length(self.G, ex))
            except Exception:
                continue
        self.exit_nodes = [e for e in self.exit_nodes
                           if e in self._exit_distance_maps]
        if not self.exit_nodes:
            # No usable exit: fall back to the original freeze-on-delivery
            # rather than leaving fleets with a goal they can never reach.
            self.despawn_on_delivery = False
            print("[Core] despawn_on_delivery requested but no reachable exit "
                  "node was found; falling back to freeze-on-delivery.")
        else:
            print(f"[Core] {len(self.exit_nodes)} exit node(s) for despawn.")

    def _nearest_exit(self, node) -> Optional[str]:
        """
        Closest exit by GRAPH distance, not straight line -- an exit on the far
        side of a rack is not near.
        """
        best, best_d = None, float("inf")
        cell = self.density._fleet_cell(node)
        nid = self.grid_pos_dict.get(cell)
        if nid is None:
            return None
        for ex in self.exit_nodes:
            dm = self._exit_distance_maps.get(ex)
            if not dm:
                continue
            d = dm.get(nid)
            if d is not None and d < best_d:
                best, best_d = ex, d
        return best

    def _collect_despawned(self) -> int:
        """
        Remove fleets that have reached their exit.

        Genuinely removed from self.nodes, not flagged -- the point is that they
        stop costing anything. Every per-step structure is keyed by id or rebuilt
        from self.nodes, and the replay buffer already pads a varying fleet
        count, so a shrinking roster needs no special handling anywhere else.
        """
        if not self._despawning:
            return 0
        gone = []
        for node in self.nodes:
            ex = self._despawning.get(node.id)
            if ex is None:
                continue
            if self.grid_pos_dict.get(self.density._fleet_cell(node)) == ex:
                gone.append(node)
        for node in gone:
            self._despawning.pop(node.id, None)
            self.despawned_nodes.add(node.id)
            self.nodes.remove(node)
            self.frozen_nodes.discard(node.id)
            self.stopped_nodes.discard(node.id)
            self.waiting_nodes.discard(node.id)
            print(f"[Core] Fleet {node.id} left the floor "
                  f"({len(self.nodes)} still on it).")
        return len(gone)

    def incomplete_postmortem(self) -> Dict[str, Any]:
        """
        WHY did each fleet that never delivered fail to?

        Three runs have now established that everything measurable improves
        while roughly eight fleets per episode quietly do not arrive, and
        nothing records the reason. Errors explain 23% of the gap;
        mean_hops_remaining is UNCORRELATED with completion (-0.02), so
        "they ran out of distance" is not it either.

        Categories are mutually exclusive and checked in order of how
        definitively they end a fleet's episode:

          no_goal          never assigned one -- more fleets than reachable goals
          dead_unrescued   errored, nobody was ever dispatched
          dead_rescue_late a rescuer was assigned and did not finish in time
          rescuer_busy     alive, but spent the episode fetching somebody else
          frozen           parked and never recalled
          yielding         still held by recovery when the episode ended
          waiting          voluntarily holding for a blocked corridor
          livelocked       no improvement on its best distance for >= patience
          moving           genuinely still travelling when time ran out

        The point is to replace a guess with a distribution. If it comes back
        mostly `livelocked`, yield and recovery matter. Mostly `moving`, it is
        speed and braking. Mostly `rescuer_busy`, the rescue subsystem is
        cannibalising deliveries to save fleets -- and that would be a
        REWARD-DESIGN question, not a routing one.
        """
        import collections
        buckets = collections.Counter()
        detail = collections.defaultdict(list)
        start_detail = collections.defaultdict(list)
        held_detail = collections.defaultdict(list)
        wait_detail = collections.defaultdict(list)
        patience = int(CONFIG["recovery"].get("livelock_patience", 40))

        for node in self.nodes:
            if node.current_goal_id in self.claimed_goals and node.id in self.frozen_nodes:
                continue                                   # delivered
            hops = node.get_graph_distance_to_goal()
            hops = float(hops) if hops is not None and np.isfinite(hops) else float("nan")

            # DEAD IS CHECKED FIRST, and the order is load-bearing.
            #
            # A fleet that errors has its goal CANCELLED (current_goal_id = None
            # at the error-confirmation site), so checking no_goal first would
            # file every dead fleet under "never got an assignment" -- turning
            # the rescue subsystem's failures into a phantom dispatch bug. The
            # first version of this method did exactly that.
            if node.id in self.stopped_nodes:
                # A DEAD FLEET, split four ways.
                #
                # This used to test `node.id in self.open_pickups` and
                # `node.id in self._pickup_assignment.values()`. Both compare a
                # FLEET id against PICKUP ids ("pickup_49_190") -- open_pickups
                # is keyed by pickup id, and _pickup_assignment maps rescuer ->
                # pickup id -- so neither could ever match. "dead_rescue_late"
                # never fired, and EVERY dead fleet landed in "dead_unrescued",
                # rescued or not. That is why the bucket equalled the number of
                # deaths exactly (2.30 vs 2.30 on cold_run13) while 1.45
                # handovers per episode had in fact completed.
                #
                # Pickups are removed in exactly one place -- a completed
                # handover -- so an open pickup at episode end means the rescue
                # never finished. Match pickups to this fleet by source_fleet.
                _mine = [pid for pid, p in self.open_pickups.items()
                         if p.get("source_fleet") == node.id]
                _assigned = set(self._pickup_assignment.values())
                if node.id in self._rescued_dead:
                    k = "dead_rescued"          # handover completed
                elif any(pid in _assigned for pid in _mine):
                    k = "dead_rescue_late"      # a rescuer was on it, ran out of time
                elif _mine:
                    k = "dead_abandoned"        # pickup open, nobody assigned
                else:
                    k = "dead_no_pickup"        # nothing was ever opened for it
            elif node.current_goal_id is None:
                # Genuinely never assigned: more fleets than UNIQUE goals.
                # goal_pool is keyed by goal node, so duplicate goals in a
                # scenario collapse and the surplus fleets are frozen at spawn.
                # Worth checking a scenario's unique-goal count before reading
                # anything into this bucket.
                k = "no_goal"
            elif node.id in self._pickup_assignment:
                k = "rescuer_busy"
            elif node.id in self.frozen_nodes:
                k = "frozen"
            elif self._yield_until.get(node.id, 0) > self.step_count:
                k = "yielding"
            elif node.id in self.waiting_nodes:
                k = "waiting"
            elif self._stall_steps.get(node.id, 0) >= patience:
                k = "livelocked"
            else:
                k = "moving"

            buckets[k] += 1
            if np.isfinite(hops):
                detail[k].append(hops)

            # WHERE IT STARTED. The end distance alone cannot distinguish four
            # very different failures:
            #   start ~ end, both high  -> never made progress
            #   start >> end            -> progressing, just too slowly
            #   start < end             -> pushed AWAY from its goal
            # initial_graph_distance is set when the goal is assigned, so it is
            # read in-process and cannot be contaminated by log ordering.
            _s = getattr(node, "initial_graph_distance", None)
            if _s is not None and np.isfinite(_s):
                start_detail[k].append(float(_s))
            held_detail[k].append(self._steps_held.get(node.id, 0))
            wait_detail[k].append(self._steps_waiting.get(node.id, 0))

        out = {f"incomplete_{k}": v for k, v in buckets.items()}
        out["incomplete_total"] = int(sum(buckets.values()))
        for k, v in detail.items():
            out[f"hops_{k}"] = round(float(np.mean(v)), 2)
        for k, v in start_detail.items():
            out[f"start_hops_{k}"] = round(float(np.mean(v)), 2)
        for k, v in held_detail.items():
            out[f"steps_held_{k}"] = round(float(np.mean(v)), 1)
        for k, v in wait_detail.items():
            out[f"steps_waiting_{k}"] = round(float(np.mean(v)), 1)

        # The same, for fleets that DID deliver -- the comparison group. Without
        # it, "incomplete fleets were held 40 steps" means nothing: every fleet
        # might have been.
        _dl = [n for n in self.nodes
               if n.current_goal_id in self.claimed_goals and n.id in self.frozen_nodes]
        if _dl:
            _ds = [float(n.initial_graph_distance) for n in _dl
                   if getattr(n, "initial_graph_distance", None) is not None
                   and np.isfinite(n.initial_graph_distance)]
            if _ds:
                out["start_hops_delivered"] = round(float(np.mean(_ds)), 2)
            out["steps_held_delivered"] = round(float(np.mean(
                [self._steps_held.get(n.id, 0) for n in _dl])), 1)
            out["steps_waiting_delivered"] = round(float(np.mean(
                [self._steps_waiting.get(n.id, 0) for n in _dl])), 1)
        return out

    def separation_report(self) -> Dict[str, Any]:
        """
        Do fleets actually get APART after a recovery, or just shuffle?

        Tier 1 picks from G.neighbors(), so ONE HOP is the furthest it can ever
        move anyone. Measured over 11,076 relocations in cold_run5: mean
        displacement 1.02 cells against a warning_threshold of 2.0. So a
        "successful escape" lands a fleet still inside the warning zone of the
        peer it collided with -- outside collision_threshold, and nowhere near
        clear. The pair re-converges, which is exactly the "they come back
        again" pattern.

        This reports the live pairwise picture so the claim can be checked per
        run rather than re-derived from logs.
        """
        act = [n for n in self.nodes if n.id not in self.immobile_nodes]
        if len(act) < 2:
            return {"sep_pairs": 0}
        pos = np.array([n.current_pos for n in act], dtype=float)
        d = np.abs(pos[:, None, :] - pos[None, :, :]).sum(-1)
        iu = np.triu_indices(len(act), k=1)
        d = d[iu]
        warn = float(self.loop.warning_threshold)
        coll = float(self.loop.collision_threshold)
        return {
            "sep_pairs": int(d.size),
            "sep_min": round(float(d.min()), 2),
            "sep_mean_nearest": round(float(np.min(
                np.where(np.eye(len(act), dtype=bool), np.inf,
                         np.abs(pos[:, None, :] - pos[None, :, :]).sum(-1)),
                axis=1).mean()), 2),
            "sep_within_collision": int((d <= coll).sum()),
            "sep_within_warning": int((d <= warn).sum()),
            "sep_pct_within_warning": round(100.0 * float((d <= warn).mean()), 2),
        }

    def _path_cells(self, node) -> Dict[Tuple[int, int, int], Tuple[int, float]]:
        """
        Where this fleet expects to be over the next few steps: {cell: (t, w)}.

        t = 0 is the cell it is standing on. Later levels come from the same BFS
        descent the density field already computes and memoises per fleet, so
        this is a dictionary build, not new work. Where the descent branches the
        weight splits, which is the analytic probability over its route.
        """
        cells: Dict[Tuple[int, int, int], Tuple[int, float]] = {}
        c0 = self.density._fleet_cell(node)
        cells[c0] = (0, 1.0)
        path = self.density._cached_intended_path(node)
        if path:
            for t, level in enumerate(path, start=1):
                for coords, w in level:
                    prev = cells.get(coords)
                    if prev is None or w > prev[1]:
                        cells[coords] = (t, float(w))
        return cells

    def _build_edge_features(self, adj: np.ndarray) -> np.ndarray:
        """
        Turn the binary adjacency into [N, N, 1 + E].

        Channel 0 stays the mask. The rest are PAIRWISE facts the GAT has never
        had: it knew only that two fleets were neighbours, not how far apart,
        nor whether their routes cross. Every pairwise fact had to be inferred
        from two node vectors that do not contain each other's positions.

            1  closeness      1 - d / interaction_radius, graph distance. 1 when
                              adjacent, 0 at the edge of the neighbourhood.
            2  path conflict  max over shared cells of w_i * w_j, discounted by
                              how soon. High means "our routes want the same
                              cell, soon".
            3  head-on        1.0 when each fleet's next cell is the other's
                              current one. A swap is invisible to a cell-overlap
                              test -- neither ever occupies the other's target at
                              the same t -- and it is the collision that matters
                              most, because neither fleet can yield by continuing.

            4  arrival order  +1 if I reach the contested cell FIRST, -1 if the
                              other does, 0 if simultaneous. ANTISYMMETRIC, and
                              the only channel that is.

        WHY ORDER NEEDS ITS OWN CHANNEL. Conflict is symmetric -- "our routes
        cross" is a property of the pair. But WHO YIELDS is not, and without a
        sign both fleets read the identical number and have no basis to behave
        differently. That is how two fleets converge on a cell, both hold, and
        deadlock: the wait cap then breaks the tie by remaining graph distance,
        which is a fallback, not a decision.

        With the sign, "conflict is high AND I arrive second" is a single
        learnable condition, and the answer to it is usually ONE STEP of
        holding -- let them through, then proceed.

        WHY THE TEMPORAL TERM IS SHARP. Two fleets crossing the same cell at
        DIFFERENT times do not collide: one passes through and leaves. The first
        version discounted by 1/(1 + max(t) + |dt|), giving 0.25 for simultaneous
        arrival and 0.167 for one a step apart -- a 33% reduction for a case that
        should barely register. overlap = 1/(1 + |dt|) halves per step of
        separation instead, so inserting one step of delay genuinely resolves the
        crossing rather than merely softening it.

        Channels 1-3 are symmetric; only arrival order flips.

        This is also the release signal waiting has been missing. "Blocked"
        means our projections intersect; "clearing" means they no longer do --
        visible BEFORE the blocker has physically moved away, which a
        nearest-peer distance cannot give you.
        """
        n = len(self.nodes)
        out = np.zeros((n, n, 1 + self.edge_feature_dim), dtype=np.float32)
        out[:, :, 0] = adj
        if self.edge_feature_dim == 0 or n <= 1:
            return out

        paths = [self._path_cells(node) for node in self.nodes]
        index = {node.id: i for i, node in enumerate(self.nodes)}
        radius = max(1e-6, self.interaction_radius)

        for i, node in enumerate(self.nodes):
            for peer_id, d in self.proximity.peers_within(node.id, self.interaction_radius):
                j = index.get(peer_id)
                if j is None or j <= i:
                    continue            # symmetric: fill both halves at once
                close = max(0.0, 1.0 - float(d) / radius)

                pi, pj = paths[i], paths[j]
                conflict = 0.0
                order = 0.0
                swapped = False
                if len(pj) < len(pi):
                    pi, pj = pj, pi
                    swapped = True
                for cell, (ti, wi) in pi.items():
                    hit = pj.get(cell)
                    if hit is None:
                        continue
                    tj, wj = hit
                    # Discount by how soon: a shared cell 3 steps out matters
                    # less than the next one, and the two fleets arriving at
                    # different times is a weaker conflict than a simultaneous
                    # one.
                    # Two terms doing two jobs:
                    #   overlap -- 1.0 when both want the cell at the SAME step,
                    #              halving per step of separation, because
                    #              different times is not a collision.
                    #   soon    -- a contested cell one step out matters more
                    #              than one three steps out.
                    overlap = 1.0 / (1.0 + abs(ti - tj))
                    soon = 1.0 / (1.0 + min(ti, tj))
                    score = wi * wj * overlap * soon
                    if score > conflict:
                        conflict = score
                        order = 0.0 if ti == tj else (1.0 if ti < tj else -1.0)

                ci = self.density._fleet_cell(self.nodes[i])
                cj = self.density._fleet_cell(self.nodes[j])
                head_on = float(
                    pj.get(ci, (99, 0.0))[0] == 1 and pi.get(cj, (99, 0.0))[0] == 1)

                # The loop above may have swapped pi/pj for efficiency, which
                # flips whose perspective `order` was computed from.
                if swapped:
                    order = -order

                vals = (close, conflict, head_on, order)[: self.edge_feature_dim]
                for k, v in enumerate(vals, start=1):
                    out[i, j, k] = v
                    # Channels 1-3 symmetric; arrival order ANTISYMMETRIC.
                    out[j, i, k] = -v if k == 4 else v
        return out

    def _build_adjacency_graph(self) -> np.ndarray:
        """
        Attention graph built on GRAPH distance, not Manhattan on coordinates.

        THE FOURTH AND LAST INSTANCE of the phantom-pair defect. We fixed it in
        braking, in check_integrity, in the density falloff kernel and in
        sf_peer_proximity on 2026-09-13 -- and missed this one, which is
        arguably the worst place for it. The line it replaces was

            within = np.sum(np.abs(positions - positions[i]), axis=1) <= radius

        so the GAT was deciding WHO ATTENDS TO WHOM using straight-line distance
        through solid racks. Two fleets in adjacent aisles, graph-20 apart and
        unable to reach each other, were exchanging messages every layer.

        Every other consumer of a bad distance produced a bad number. This one
        produced a bad TOPOLOGY -- the message-passing structure of the network
        itself was wrong.

        Self-loops are preserved exactly as before, so an isolated fleet still
        attends to itself and its own features pass through cleanly.
        """
        n = len(self.nodes)
        adj = np.eye(n, dtype=np.float32)
        if n <= 1:
            return adj
        index = {node.id: i for i, node in enumerate(self.nodes)}
        for node in self.nodes:
            i = index[node.id]
            for peer_id, _d in self.proximity.peers_within(
                    node.id, self.interaction_radius):
                j = index.get(peer_id)
                if j is not None:
                    adj[i, j] = 1.0
                    adj[j, i] = 1.0
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

        # Which dead fleet this rescue saved, and which orders it took over --
        # recorded here because this is the ONE place a pickup closes. Tracked
        # regardless of rescue_delivery_bonus, which only affects the reward.
        self._rescued_dead.add(pickup.get("source_fleet"))
        self._inherited_orders.update(orphan_goals)

        # Remember WHICH orders came from a rescue, so delivering one can be paid
        # separately from an ordinary delivery. Without this they are
        # indistinguishable at delivery time: _pickup_assignment is cleared the
        # moment the dwell completes, so leg 2 takes the ordinary
        # mission_complete path and the rescue head never learns that finishing
        # the job is part of the job.
        if self.rescue_delivery_bonus:
            self._rescued_orders.setdefault(node.id, set()).update(orphan_goals)

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

    def _splat_warnings(self) -> None:
        """Stamp mild repulsion for predictive warnings. See the notes inside."""
        # Predictive Warning (Safety Bubble): splat mild repulsion between
        # fleets that are close.
        #
        # THE OLD VERSION PICKED TWO ARBITRARY FLEETS. It took
        # list(warning_nodes) -- a SET, so the order is arbitrary -- and
        # splatted at the midpoint of elements [0] and [1]. With one warning
        # pair that was correct. With two or more anywhere on the map it
        # usually took one fleet from each, stamping repulsion halfway
        # between two unrelated fleets -- often empty floor -- while every
        # real conflict but at most one got nothing. And it splatted ONCE per
        # step however many conflicts existed.
        #
        # warning_splat_per_pair: one splat per actual warning PAIR, at that
        # pair's midpoint. And with preempt_skip_convoys, a pair that is
        # simply following -- same axis, same direction, gap not closing --
        # is not splatted: stamping repulsion along a shared route pushes
        # both the convoy and everyone behind it off a route that was fine.
        if self.warning_splat_per_pair:
            by_id = {n.id: n for n in self.nodes}
            for a_id, b_id, dist in self.proximity.pairs(
                    radius=self.loop.warning_threshold):
                if dist <= self.loop.collision_threshold:
                    continue                    # collisions splat on their own path
                a, b = by_id.get(a_id), by_id.get(b_id)
                if a is None or b is None:
                    continue
                if self.preempt_skip_convoys and self._is_following(a, b):
                    self.convoy_splats_skipped += 1
                    continue
                midpoint = (a.current_pos + b.current_pos) / 2.0
                self.density.splat_spatial_temporal_event(
                    midpoint, severity_multiplier=self.warning_splat_multiplier)
                self.warning_splats += 1
        else:
            warning_list = list(self.loop.warning_nodes)
            if len(warning_list) >= 2:
                n1 = next(n for n in self.nodes if n.id == warning_list[0])
                n2 = next(n for n in self.nodes if n.id == warning_list[1])
                midpoint = (n1.current_pos + n2.current_pos) / 2.0
                self.density.splat_spatial_temporal_event(midpoint, severity_multiplier=self.warning_splat_multiplier)
                self.warning_splats += 1

    def _is_following(self, a, b) -> bool:
        """
        True when a and b are travelling together rather than converging:
        both moved on their last action, along the same axis in the same
        direction, and the distance between them did not shrink.

        node.direction is the unit axis vector of the last SUCCESSFUL move (zero
        when idle or blocked), and node.last_pos is where the fleet stood before
        it -- so both describe the most recent action, even after a teleport.
        """
        da = np.asarray(a.direction, dtype=np.float32)
        db = np.asarray(b.direction, dtype=np.float32)
        if not (np.any(da) and np.any(db)):
            return False                      # someone idled or was blocked
        if float(np.dot(da, db)) < self.convoy_alignment:
            return False                      # head-on or crossing
        if a.last_pos is None or b.last_pos is None:
            return False
        gap_now = float(np.sum(np.abs(a.current_pos - b.current_pos)))
        gap_was = float(np.sum(np.abs(a.last_pos - b.last_pos)))
        return gap_now >= gap_was - 0.05      # steady or opening, not closing

    def _drop_following_pairs(self, at_risk: Set[str]) -> Set[str]:
        """
        Keep only the at-risk fleets that have at least one warning partner
        they are NOT simply following. A fleet whose every close neighbour is a
        convoy partner needs no intervention.
        """
        by_id = {n.id: n for n in self.nodes}
        needs: Set[str] = set()
        for a_id, b_id, _dist in self.proximity.pairs(radius=self.loop.warning_threshold):
            if a_id not in at_risk and b_id not in at_risk:
                continue
            a, b = by_id.get(a_id), by_id.get(b_id)
            if a is None or b is None or not self._is_following(a, b):
                needs.add(a_id)
                needs.add(b_id)
        return set(at_risk) & needs

    def _attr_integrity(self, fleets, value: float) -> None:
        """
        SHADOW MODE ledger. A recovery event's value is charged to the fleets
        involved in it -- "pay by your own actions" -- instead of being added in
        full to every fleet. An event with nobody involved (a wasted invocation:
        the policy asked for recovery with nobody at risk) is a whole-holon
        decision and goes to the holon, split evenly across active fleets.
        Measurement only: the live reward still uses _pending_integrity_reward.
        """
        fleets = set(fleets or ())
        if not fleets:
            self._pending_integrity_system += value
            return
        for fid in fleets:
            self._pending_integrity_by_fleet[fid] = (
                self._pending_integrity_by_fleet.get(fid, 0.0) + value)

    def _priority_rewards(self, T, shadow, sh_warn, include_potential: bool):
        """
        The three priority heads, per fleet, from the legacy terms T [N, terms].
        Shared by shadow mode and the live priority reward so they cannot drift.

        safety     : collision + warning + own integrity state + own recovery share
        delivery   : arrival + rescue pickup + rescue delivery
        efficiency : progress + idle + overtime + final approach   (no baseline)

        Warning and integrity use the POST-ACTION view when it exists -- what the
        action produced. include_potential adds the exact potential-shaping term
        m*(1-gamma)*distance (shadow only; the live reward keeps today's form).
        """
        ti = self._TI
        if sh_warn is not None:
            warn = sh_warn
            dl, wn = self._post_action_deadlocked, self._post_action_warning
        else:
            warn = T[:, ti[("safety", "warning")]]
            dl, wn = self._step_deadlocked, self._step_warning
        coh = np.array([-1.0 if n.id in dl else (-0.5 if n.id in wn else 0.0)
                        for n in self.nodes], dtype=np.float64)
        eff = (T[:, ti[("goal", "progress")]] + T[:, ti[("time", "idle")]]
               + T[:, ti[("time", "overtime")]] + T[:, ti[("goal", "final_approach")]])
        if include_potential:
            eff = eff + shadow[:, 0]
        return {
            "safety": T[:, ti[("safety", "collision")]] + warn + coh + shadow[:, 1],
            "delivery": (T[:, ti[("goal", "delivery")]] + T[:, ti[("rescue", "pickup")]]
                         + T[:, ti[("rescue", "rescue_delivery")]]),
            "efficiency": eff,
        }

    def _stamp_holon(self) -> None:
        """
        Write the holon's integrity onto every fleet, for the configuration
        ABOUT TO BE PERCEIVED: 0 if any collision, 0.5 if any warning, 1 if
        clear. Read-only (peek_conflicts touches no counters). Computed fresh
        rather than reusing the start-of-step check, which would describe the
        world before recovery -- and before the action, for the next state.
        """
        if not self.holon_perception:
            return
        d, w = self.loop.peek_conflicts(self.proximity)
        v = 0.0 if d else (0.5 if w else 1.0)
        for n in self.nodes:
            n.sf_holon_integrity = v

    def _drop_handled_pairs(self, at_risk: Set[str]) -> Set[str]:
        """
        Keep only the at-risk fleets that have at least one warning partner
        that is BOTH not being followed (when convoys are skipped) AND not
        currently held.

        This must be ONE combined test, not a convoy pass followed by a
        held pass. A fleet whose two warning partners are one convoy partner
        and one held fleet needs no recovery -- but a convoy pass keeps it
        (the held partner is not a convoy) and a held pass keeps it (the convoy
        partner is not held). Only testing both conditions per pair drops it.

        "Held" is read from _yield_until, the authoritative record, rather than
        the copy handed to recovery.
        """
        held = {fid for fid, until in self._yield_until.items()
                if until > self.step_count}
        by_id = {n.id: n for n in self.nodes}
        needs: Set[str] = set()
        for a_id, b_id, _dist in self.proximity.pairs(radius=self.loop.warning_threshold):
            if a_id not in at_risk and b_id not in at_risk:
                continue
            a, b = by_id.get(a_id), by_id.get(b_id)
            if a is None or b is None:
                needs.add(a_id); needs.add(b_id)
                continue
            if self.preempt_skip_convoys and self._is_following(a, b):
                continue
            if a_id in held or b_id in held:
                continue
            needs.add(a_id); needs.add(b_id)
        return set(at_risk) & needs

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

        # CONVOYS ARE NOT CONFLICTS.
        #
        # Measured on cold_run12: 742 of 769 recovery invocations (96%) were
        # PREEMPTIVE -- no collision, just fleets inside the warning band -- and
        # only 148 of them (20%) actually separated anyone. The cause, found by
        # reading coordinates in the log: two fleets travelling the SAME route,
        # one about a cell behind the other, down the x=5 shaft and along the
        # corridor. With collision_threshold 0.5 and warning_threshold 2.0 a
        # normal following gap is permanently a warning, so preemptive recovery
        # fired every single step, seized both fleets from the policy, charged
        # the integrity head, and teleported them one cell forward -- the pair
        # re-formed the next step and it happened again, thirty times running.
        #
        # Intervene on CONVERGENCE, not proximity. A pair counts as FOLLOWING
        # when both moved last step, along the same axis in the same direction,
        # and the gap between them did not shrink. Such pairs are dropped from
        # the preemptive set. Everything else still goes to recovery: head-on
        # (opposite directions), crossing (perpendicular), a leader that idled
        # with a follower closing, and every ACTUAL collision -- this filter
        # only ever touches the preemptive path, never a real deadlock.
        #
        # If nothing is left, recovery does not run at all: no teleport, no
        # force_repair, no invocation cost, no counter. The warning stays on the
        # books, because the fleets really are close -- they just are not in
        # conflict.
        if preemptive and self.preempt_skip_convoys:
            before = len(dead)
            dead = self._drop_following_pairs(dead)
            self.convoy_fleets_exempted += before - len(dead)
            if not dead:
                self.convoy_invocations_skipped += 1
                return {"reinit_from": "skipped_convoy"}

        # HELD PAIRS ARE NOT NEW CONFLICTS EITHER.
        #
        # Right-of-way alternates on repeat offences (anti-starvation: whoever
        # yielded last gets priority next). That assumes the loser's hold lets
        # the winner get THROUGH before the next conflict. But to pass a held
        # fleet at a junction the winner must come within warning distance of
        # it -- which fired a preemptive recovery MID-PASS, flipped the winner,
        # released the loser and held the fleet that was almost through.
        # Neither ever finished a turn. cold_run16: 54 such ping-pong runs, 334
        # recoveries, one pair flipping 26 times in a row. Escalating preemptive
        # pairs (escalate_on_recurrence) fed far more pairs into this path:
        # cold_run15 had 9 runs.
        #
        # So a pair with one fleet currently held is dropped from the PREEMPTIVE
        # set -- the pass is supposed to happen. A fleet stays at risk only if it
        # has a warning partner that is neither followed nor held. Real
        # collisions never reach this path and are always recovered.
        if preemptive and self.preempt_skip_held_pairs:
            before = len(dead)
            dead = self._drop_handled_pairs(dead)
            self.held_pair_fleets_exempted += before - len(dead)
            if not dead:
                self.held_pair_invocations_skipped += 1
                return {"reinit_from": "skipped_held_pair"}

        # Current obstacle cells, handed over every call: this set is REASSIGNED
        # whenever humans move, so a reference stored once would go stale.
        self.recovery.static_obstacles = set(self.static_obstacles)
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
            self._attr_integrity(self._step_deadlocked or self._step_warning, self.recovery_invocation_cost)
            return

        n_dead_before = len(self._step_deadlocked)
        # Capture the result!
        recovery_result = self._run_recovery(forced=(mode == 0 and forced))
        
        # Apply the yields!
        if recovery_result and recovery_result.get("yield_durations"):
            for loser_id, duration in recovery_result.get("yield_durations", {}).items():
                self._yield_until[loser_id] = self.step_count + duration
                
            self.recovery.currently_yielding = {
                fid for fid, until in self._yield_until.items()
                if until > self.step_count
            }

        # Cost and outcome land on the INTEGRITY head...
        self._pending_integrity_reward += self.recovery_invocation_cost
        self._attr_integrity(self._step_deadlocked or self._step_warning, self.recovery_invocation_cost)
        if n_dead_before > 0:
            self._pending_integrity_reward += self.recovery_resolution_bonus
            self._attr_integrity(self._step_deadlocked or self._step_warning, self.recovery_resolution_bonus)

        # Cost and outcome land on the INTEGRITY head, charged to every fleet
        # that was party to the deadlock. Without a cost the policy learns to
        # crash and undo: a rewind erases a collision, so a free rewind makes
        # ignoring safety optimal. The goal head separately penalises the lost
        # travel distance, so the two heads pull against each other and the
        # policy has to resolve that tension -- which is the decision we want it
        # to learn.
        self._pending_integrity_reward += self.recovery_invocation_cost
        self._attr_integrity(self._step_deadlocked or self._step_warning, self.recovery_invocation_cost)
        if n_dead_before > 0:
            self._pending_integrity_reward += self.recovery_resolution_bonus
            self._attr_integrity(self._step_deadlocked or self._step_warning, self.recovery_resolution_bonus)

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
        # If there are open pickups waiting for a rescuer, keep running.
        if len(self.open_pickups) > 0:
            return False
            
        # If any fleet is actively assigned to a rescue, keep running.
        if len(self._pickup_assignment) > 0:
            return False

        # Otherwise, check if everything is physically immobile.
        return len(self.immobile_nodes) == len(self.nodes)
        

    def get_error_statistics(self) -> Dict[str, Any]:
        """Error/handover counters for the training log."""
        return {
            "errors_injected": self.total_errors,
            "stops_confirmed": len(self.stopped_confirmed),
            "handovers_completed": self.total_handovers,
            "retired_fleets_recalled": self.total_recalls,
            "mean_braked_speed": (self._braked_speed_sum / self._braked_speed_n
                                  if self._braked_speed_n else float("nan")),
            "min_braked_speed": (self._braked_speed_min
                                 if self._braked_speed_n else float("nan")),
            # Fleet-steps whose rays would have read zero in ALL six directions
            # before the 2026-09-13 ray-origin fallback. Reported as a rate over
            # fleet-steps so it can be compared against a claim like "this is why
            # dense training regressed".
            "ray_origin_recovered": sum(getattr(n, "ray_origin_recovered", 0)
                                        for n in self.nodes),
            "ray_origin_blind": sum(getattr(n, "ray_origin_blind", 0)
                                    for n in self.nodes),
            "ray_origin_recovered_rate": (
                sum(getattr(n, "ray_origin_recovered", 0) for n in self.nodes)
                / max(1, self.step_count * max(1, len(self.nodes)) * 2)),
            "edge_steps_normal": self._edge_steps[0],
            "edge_steps_offgrid": self._edge_steps[1],
            "edge_collisions_normal": self._edge_collisions[0],
            "edge_collisions_offgrid": self._edge_collisions[1],
            "collision_rate_normal": (self._edge_collisions[0] / self._edge_steps[0]
                                      if self._edge_steps[0] else float("nan")),
            "collision_rate_offgrid": (self._edge_collisions[1] / self._edge_steps[1]
                                       if self._edge_steps[1] else float("nan")),
            # >1 means off-grid fleet-steps collide MORE. nan when no off-grid
            # step ever arose, which is itself the answer.
            "edge_collision_rate_ratio": (
                (self._edge_collisions[1] / self._edge_steps[1])
                / (self._edge_collisions[0] / self._edge_steps[0])
                if self._edge_steps[0] and self._edge_steps[1]
                and self._edge_collisions[0] else float("nan")),
            "idle_perception_reused": self._idle_perception_reused,
            "idle_perception_computed": self._idle_perception_computed,
            "idle_perception_reuse_rate": (
                self._idle_perception_reused
                / max(1, self._idle_perception_reused + self._idle_perception_computed)),
            # Ray information content. saturated_rate near 1.0 means the rays
            # are a constant on this map and carry nothing -- which makes a null
            # lesion result uninformative rather than reassuring.
            "ray_casts": sum(getattr(n, "ray_casts", 0) for n in self.nodes),
            "ray_mean_cells": (
                sum(getattr(n, "ray_cells_sum", 0) for n in self.nodes)
                / max(1, sum(getattr(n, "ray_casts", 0) for n in self.nodes))),
            "ray_saturated_rate": (
                sum(getattr(n, "ray_saturated", 0) for n in self.nodes)
                / max(1, sum(getattr(n, "ray_casts", 0) for n in self.nodes))),
            "ray_blocked_rate": (
                sum(getattr(n, "ray_blocked_at_zero", 0) for n in self.nodes)
                / max(1, sum(getattr(n, "ray_casts", 0) for n in self.nodes))),
            "ray_peer_hit_rate": (
                sum(getattr(n, "ray_peer_hits", 0) for n in self.nodes)
                / max(1, sum(getattr(n, "ray_casts", 0) for n in self.nodes))),
            "waits_started": self.waits_started,
            "wait_steps_total": self.wait_steps_total,
            "waits_capped": self.waits_capped,
            "mutual_waits": self.mutual_waits,
            "static_obstacles": len(self.static_obstacles),
            **self.incomplete_postmortem(),
            **self.separation_report(),
            # Tier-1 escape counters live on the recovery object, so they have
            # to be merged in explicitly -- they were in the runner's CSV filter
            # but never in this dict, so the column silently never appeared.
            # tier2_ and recurrence_ were missing here: tier2_rejected_obstacle
            # existed but was filtered out before reaching the CSV -- the same
            # silent-missing-column bug as the doorstep columns once had.
            **{k: v for k, v in self.recovery.get_statistics().items()
               if k.startswith(("tier1_", "tier2_", "recurrence_"))},
            **{f"doorstep_{k}": v for k, v in self._doorstep.items()},
            "doorstep_step_in_rate": round(
                self._doorstep["closer"] / max(1, self._doorstep["steps"]), 4),
            "doorstep_idle_rate": round(
                self._doorstep["idle"] / max(1, self._doorstep["steps"]), 4),
            "rescued_orders_delivered": self.rescued_orders_delivered,
            "convoy_fleets_exempted": self.convoy_fleets_exempted,
            "convoy_invocations_skipped": self.convoy_invocations_skipped,
            "held_pair_invocations_skipped": self.held_pair_invocations_skipped,
            # rwd_term_{pos|neg}_{head}_{term}: per term, active fleets.
            **{f"rwd_term_{sg}_{h}_{t}": round(float(v), 3)
               for (h, t), vp, vn in zip(self._TERMS, self._term_pos, self._term_neg)
               for sg, v in (("pos", vp), ("neg", vn))},
            # rwd_spread_{mean|std}_{head}: per active fleet-step.
            **{f"rwd_spread_{st}_{h}": round(float(v), 5)
               for h, k in self.HEAD.items()
               for st, v in (
                   ("mean", self._spread_sum[k] / max(1, self._spread_n)),
                   ("std", np.sqrt(max(0.0, self._spread_sq[k] / max(1, self._spread_n)
                                       - (self._spread_sum[k] / max(1, self._spread_n)) ** 2))))},
            **{f"rwd_shadow_{k}_{h}": round(v, 3)
               for h, d in self._shadow.items() for k, v in (("pos", d["pos"]), ("neg", d["neg"]))},
            **{f"rwd_shadow_{st}_{h}": round(float(v), 5)
               for h, d in self._shadow.items()
               for st, v in (("mean", d["sum"] / max(1, self._shadow_n)),
                             ("std", np.sqrt(max(0.0, d["sq"] / max(1, self._shadow_n)
                                                 - (d["sum"] / max(1, self._shadow_n)) ** 2))))},
            **{f"rwd_shadow_{st}_{h}": round(float(v), 5)
               for h, d in self._shadow.items()
               for st, v in (("freq", d["nz"] / max(1, self._shadow_n)),
                             ("typical", d["abs"] / max(1, d["nz"])))},
            **{f"rwd_live_{sg}_{h}": round(d[sg], 4)
               for h, d in self._live.items() for sg in ("pos", "neg")},
            "rwd_warnundo_fleetsteps": self._warnundo_n,
            "rwd_warnundo_mismatched": self._warnundo_mismatched,
            "rwd_warnundo_escaped_kept": self._warnundo_escaped_kept,
            "rwd_warnundo_error": round(self._warnundo_error, 3),
            # rwd_{all|learned}_{pos|neg}_{head}: see the accumulators in __init__.
            **{f"rwd_{v}_{sg}_{h}": round(x, 3)
               for v, byv in self._rwd.items()
               for sg, bys in byv.items()
               for h, x in bys.items()},
            "held_pair_fleets_exempted": self.held_pair_fleets_exempted,
            "convoy_splats_skipped": self.convoy_splats_skipped,
            "post_recovery_refreshes": self.post_recovery_refreshes,
            # ORDERS, not fleets. The post-mortem counts FLEETS, so a dead fleet
            # whose order a rescuer delivered still appears there as an
            # undelivered fleet. These count what actually reached a goal.
            "orders_total": len(self.goal_pool) if self.goal_pool else 0,
            "orders_delivered": len(self.claimed_goals),
            "orders_undelivered": (len(self.goal_pool) - len(self.claimed_goals))
                                  if self.goal_pool else 0,
            "orders_inherited": len(self._inherited_orders),
            "orders_inherited_delivered": self.inherited_orders_delivered,
            "orders_stranded_in_open_pickups": sum(
                len([g for g in p["goal_ids"] if g not in self.claimed_goals])
                for p in self.open_pickups.values()),
            "sep_warning_splats": self.warning_splats,
            "despawned": len(self.despawned_nodes),
            "despawning": len(self._despawning),
            **({f"obstacle_{k}": v
                for k, v in self.obstacle_field.statistics().items()}
               if self.obstacle_field is not None else {}),
            "waiting_now": len(self.waiting_nodes),
            "tabu_overrides": self.tabu_overrides,
            "tabu_overrides_stuck": self.tabu_overrides_stuck,
            "projection_fallbacks": getattr(self.density, "_projection_fallbacks", 0),
            "kernel_manhattan_fallbacks": getattr(self.density, "_kernel_manhattan_fallbacks", 0),
            "memory_offgrid_rejected": getattr(
                self.density, "_memory_offgrid_rejected", 0),
            "kernel_fallback_cells": list(
                getattr(self.density, "_kernel_fallback_cells", [])),
            "phantom_pairs_rejected": self.loop.phantom_pairs_rejected,
            "proximity_off_grid": self.proximity.off_grid_fleets,
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
            # DISTANCE, COUNTED. soc_hops is sum_of_costs (a TIME) multiplied by
            # base_speed, which answers "how far could it have gone at nominal
            # speed?" -- and FLOWRRA never sustains nominal speed. Affordance
            # braking floors it as low as 0.05, the final-approach override only
            # lifts the ramp to 0.7, and dwelling at a pickup moves it not at all.
            # So the conversion overstates FLOWRRA's distance by an unknown
            # margin. _fleet_travel already sums the Manhattan distance actually
            # moved each step; reporting it removes the assumption entirely and
            # makes the number directly comparable to a baseline hop count.
            "distance_travelled": float(sum(self._fleet_travel.values())),
            "brake_duty_cycle": (self._braked_fleet_steps / self._brake_eval_steps
                                 if self._brake_eval_steps else 0.0),
            "mean_peer_gap": (self._min_dist_sum / self._brake_eval_steps
                              if self._brake_eval_steps else float("nan")),
            "pickups_open": len(self.open_pickups),
            "stopped_fleets": sorted(self.stopped_nodes),
        }

    def _perceive(self, node) -> np.ndarray:
        """
        The full state vector for one fleet, with optional idle caching.

        Width is never hardcoded anywhere: the runner, the lesion harness and
        the correlation script all compute input_dim as
        len(get_state_vector) + density.output_dim, so a change to either half
        propagates on its own. What does NOT propagate is a saved checkpoint --
        load_state_dict is strict, so any width change invalidates it and forces
        the retrain. That is by design; the bundle was always going to need one.

        WHY THIS EXISTS. Both state-build loops iterate self.nodes, not
        get_active_nodes(), so a PARKED fleet casts six rays, builds a 231-dim
        affordance field and gets a Q-forward pass -- twice per step -- and then
        active_mask zeroes its contribution to the loss. Nothing consumes it. At
        step 200 of the 2026-09-13 run that was 160 of 200 fleets: 80% of all
        perception computed and discarded.

        In an episodic benchmark that is the last third of one episode. In
        CONTINUOUS operation, where orders arrive in bursts and most of the fleet
        idles between them, it is the steady state.

        WHY IT IS NOT FREE, and why "full" is the default. _build_adjacency()
        includes immobile fleets, so the GAT attends over them: an ACTIVE fleet's
        embedding depends on its idle neighbours' features. Reusing a stale
        vector for an idle fleet therefore changes what the active fleets see,
        and their Q-values with them. This is a behaviour change wearing a
        compute-saving costume, and it has to be measured rather than assumed.

        MODES (CONFIG["perception"]["idle_mode"]):
          "full"   -- recompute every fleet every build. Current behaviour, default.
          "cached" -- recompute an idle fleet every idle_refresh_every steps and
                      reuse in between. Staleness is bounded by that number.
          "skip"   -- reuse indefinitely while idle. Maximum saving, maximum
                      staleness; really an ablation, not a setting.

        An idle fleet does not move, so the only thing that goes stale is what
        is happening AROUND it -- which is exactly what its neighbours attend to.
        Hence the refresh interval rather than a permanent cache.
        """
        if self._idle_mode != "full" and node.id in self.immobile_nodes:
            memo = getattr(node, "_perception_memo", None)
            if memo is not None and (
                    self._idle_mode == "skip"
                    or (self.step_count - memo[0]) < self._idle_refresh_every):
                self._idle_perception_reused += 1
                return memo[1]

        # Affordance FIRST, then the base vector. The order matters: the local
        # entropy scalar is computed from the affordance and read back inside
        # get_state_vector(), so building the base vector first would fold in
        # last step's value.
        local_affordance = self.density.get_local_affordance(
            node.current_pos, self.nodes, self.immobile_nodes,
            own_goal_pos=node.goal_pos,
            frozen_obstacle_severity=self.frozen_obstacle_severity,
            near_goal_radius=self.frozen_near_goal_radius,
            own_id=node.id,
            stopped_node_ids=self.stopped_nodes,
            stopped_obstacle_severity=self.stopped_obstacle_severity,
            static_obstacles=self.static_obstacles,
            static_obstacle_severity=self.static_obstacle_severity,
        )
        node.sf_local_entropy = self.density.action_entropy(local_affordance)
        node.sf_throughput_t = self._throughput_t
        base_state = node.get_state_vector(self.nodes)
        full_state = self._apply_lesion(
            np.concatenate([base_state, local_affordance]))
        self._idle_perception_computed += 1

        if self._idle_mode != "full" and node.id in self.immobile_nodes:
            node._perception_memo = (self.step_count, full_state)
        elif hasattr(node, "_perception_memo"):
            # Became mobile again: drop the cache so it cannot be served later
            # from a position the fleet has since left.
            del node._perception_memo
        return full_state

    def _resolve_lesion_slices(self) -> List[Tuple[int, int]]:
        """
        Turn the configured block NAMES into index ranges, once, on first use.

        Deferred out of __init__ because it needs a spawned fleet to read
        state_layout() from, and fleets are built later in __init__. Resolving
        on first use is also the only point at which the density field and the
        fleets are both guaranteed to exist.
        """
        layout = dict(self.nodes[0].state_layout())
        base_len = layout["_base_len"][1]
        layout["density"] = (base_len, base_len + self.density.output_dim)
        slices: List[Tuple[int, int]] = []
        for name in self._lesion_names:
            if name not in layout:
                raise KeyError(
                    f"lesion block {name!r} is not a state block. "
                    f"Known: {sorted(k for k in layout if not k.startswith('_'))}"
                )
            slices.append(layout[name])
        print(f"[Core] LESION ACTIVE -- zeroing {self._lesion_names} "
              f"= {slices} of {base_len + self.density.output_dim} dims")
        return slices

    def _apply_lesion(self, state: np.ndarray) -> np.ndarray:
        """
        Zero the configured state blocks. No-op when no lesion is configured,
        so the normal path costs one branch.
        """
        if not self._lesion_names:
            return state
        if self._lesion_slices is None:
            self._lesion_slices = self._resolve_lesion_slices()
        for lo, hi in self._lesion_slices:
            state[lo:hi] = 0.0
        return state

    def get_active_nodes(self) -> List[FleetNode]:
        return [n for n in self.nodes if n.id not in self.immobile_nodes]

    def step(self, episode_step: int, total_episodes: int) -> float:
        """
        Executes one discrete simulation step for the warehouse environment.
        """
        # 0. VDA 5050 ERROR STOPS. Rolled BEFORE integrity so a fleet that
        # errors this step is already out of the collision system when integrity
        # is evaluated, rather than spending one step as a phantom conflict.
        # Remove fleets that reached their exit LAST step, before anything this
        # step reads the roster.
        #
        # Removing them mid-step was a shape bug: perception had already built
        # feature vectors for 24 fleets when the adjacency came back 23 wide, and
        # torch reported it as "size of tensor a (23) must match tensor b (24)".
        # Anything that changes len(self.nodes) has to happen at a step boundary,
        # where every downstream structure is rebuilt from the new roster.
        self._collect_despawned()

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
        # Refresh the proximity index BEFORE anything reads a peer distance this
        # step. Every consumer below (integrity, sf_peer_proximity, braking, the
        # safety reward) must see the same snapshot, or the state feature and the
        # reward that scores it can disagree about where the fleets are.
        self.proximity.refresh(self.nodes, excluded_ids=self.immobile_nodes)
        # A fleet that became immobile or finished while waiting is no longer
        # waiting -- leaving it in the set would advertise "I will move when
        # clear" to peers about a fleet that never will, which is precisely the
        # ambiguity this state exists to remove.
        # Peer-visible flags, refreshed before any state vector is built. Rays
        # read these off the peer they hit, which avoids threading four more
        # sets through get_state_vector -> sense_6_axis_rays.
        # Advance humans BEFORE perception, so the cell set a fleet reacts to is
        # the one it is actually standing next to this step, not last step's.
        if self.obstacle_field is not None:
            self.obstacle_field.step()
            self.static_obstacles = self.obstacle_field.occupied_cells()

        for _n in self.nodes:
            _n.sf_is_immobile = 1.0 if _n.id in self.immobile_nodes else 0.0
            _n.static_obstacles = self.static_obstacles

        # ---- THROUGHPUT PRESSURE, the T of F = E - T*S ---------------------
        #     urgency = orders_remaining / steps_remaining
        #     T       = 1 / (1 + urgency)
        #
        # Many orders and little time -> urgency high -> T LOW -> push, accept
        # risk. Plenty of slack -> T HIGH -> hold, stay clean.
        #
        # Bounded in (0, 1], one number, and it becomes order-queue depth
        # unchanged when a continuous order stream replaces the fixed pool --
        # which is the point of defining it this way rather than as a step
        # counter. Global for now; per-fleet would let different zones run at
        # different pressures, and is a later question.
        _orders_left = max(0, len(self.goal_pool) - len(self.claimed_goals))
        _steps_left = max(1, self._max_steps_hint - self.step_count)
        _urgency = _orders_left / _steps_left
        self._throughput_t = 1.0 / (1.0 + _urgency)

        if self.waiting_enabled and self.waiting_nodes:
            for _wid in list(self.waiting_nodes):
                if _wid in self.immobile_nodes:
                    self.waiting_nodes.discard(_wid)
                    self._wait_steps.pop(_wid, None)
        if self._prox_all is not None:
            self._prox_all.refresh(self.nodes, excluded_ids=set())

        current_integrity = self.loop.check_integrity(
            self.nodes, self.step_count, self.immobile_nodes, proximity=self.proximity)

        if (self.phantom_audit_every
                and self.step_count % self.phantom_audit_every == 0):
            self.loop.measure_phantom_pairs(
                self.nodes, self.immobile_nodes, self.proximity)

        # ---- EDGE-STATE x COLLISION CONTINGENCY TABLE ----------------------
        # Tests one claim: that fleets whose rays went blind -- because
        # round(current_pos) missed the graph after overshooting a dead end by
        # half a cell -- collided more than fleets that could see.
        #
        # Counted over MOBILE fleet-steps only; immobile fleets are exempt from
        # collisions and would dilute both cells of the table.
        #
        # READ IT AS A RATE RATIO, not a raw count. Edge states are rare, so the
        # raw collision count in that cell will always be small. The question is
        # whether P(collision | edge) exceeds P(collision | normal).
        #
        # AND READ IT AS CORRELATION. Dead ends are where fleets queue and get
        # stuck for reasons that have nothing to do with rays. A high ratio makes
        # blind rays a candidate, not a cause. Separating them needs both arms of
        # CONFIG["ablation"]["ray_origin_fallback"]: with the fallback ON, the
        # fleet is at the same dead end but CAN see, so whatever excess remains
        # belongs to the dead end rather than the blindness.
        _dead = self.loop.deadlocked_nodes
        _warn = self.loop.warning_nodes
        for _n in self.nodes:
            if _n.id in self.immobile_nodes:
                continue
            _edge = 1 if getattr(_n, "ray_origin_offgrid", False) else 0
            self._edge_steps[_edge] += 1
            if _n.id in _dead:
                self._edge_collisions[_edge] += 1
            if _n.id in _warn:
                self._edge_warnings[_edge] += 1

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

        # WHO WAS ACTIVE WHEN THE ACTION WAS CHOSEN.
        #
        # Snapshotted HERE, before the action loop, because that loop FREEZES a
        # fleet the moment it delivers -- and active_mask used to be built after
        # the loop, from immobile_nodes. So the fleet that had just earned
        # mission_complete was marked inactive in the very transition carrying
        # it, and learn()'s masked TD loss multiplied that sample by zero.
        #
        # The +100 arrival reward therefore never reached the gradient. Not once,
        # in any run. Every other action near a goal was trained -- stepping
        # sideways, idling, backing off -- and the one action that completes the
        # mission was the only one systematically excluded. Measured at the
        # doorstep: 38.2% step in, 29.0% idle, 32.7% move away, with a peer
        # nearby only 1.3% of the time. That is what no signal looks like.
        #
        # It also explains completion pinned near 0.80 across every run
        # regardless of schedule, architecture or fix; gradient_agreement rising
        # while completion stayed flat (approach is trained, arrival is not);
        # and mission_complete 125 not helping, since 125 x 0 is still 0.
        #
        # A fleet already parked at the START of the step stays masked -- that
        # was the original and correct intent, since it pushes reward 0.0 with an
        # arbitrary action from a goal cell. A fleet that freezes DURING the step
        # took a real action and earned a real reward, and is masked from the
        # NEXT transition onward, not this one.
        self._active_at_action_time = np.array(
            [n.id not in self.immobile_nodes for n in self.nodes], dtype=np.float32
        ) if self.mask_active_before_actions else None
        self._step_warning = set(self.loop.warning_nodes)
        self._recovery_ran_this_step = False

        # PAY OFF A PREEMPTIVE SEPARATION MADE LAST STEP, if it worked.
        # "Worked" = every fleet it moved is now clear of both the warning band
        # and any collision. Deferred by one step because that is the earliest
        # point the outcome is observable -- the separation happens after the
        # integrity check that would show its effect.
        self._pending_integrity_reward = 0.0
        self._pending_integrity_by_fleet = {}
        self._pending_integrity_system = 0.0
        if self._preemptive_watch:
            still_at_risk = self._preemptive_watch & (
                self._step_deadlocked | self._step_warning)
            if not still_at_risk:
                self._pending_integrity_reward += self.recovery_preemptive_bonus
                self._attr_integrity(self._preemptive_watch, self.recovery_preemptive_bonus)
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
            # WAS: an O(N^2) generator over every other fleet, measuring
            #      float(np.sum(np.abs(n.current_pos - o.current_pos))) --
            #      Manhattan through racks, recomputed from scratch for every
            #      fleet, 40,000 iterations per step at 200 fleets.
            # NOW: a lookup into the index refreshed once at the top of step().
            #      Same exclusion set, same clip, same span. Distances beyond
            #      search_radius report inf, which clips to 0 exactly as the old
            #      `default=warning_threshold` did.
            nearest = self.proximity.nearest(n.id)
            n.sf_peer_proximity = float(np.clip(
                (self.loop.warning_threshold - nearest) / span, 0.0, 1.0))

        # 1b. POLICY-INVOKED RECOVERY, evaluated every step -- including steps
        # where integrity is still 1.0, which is the whole point: the policy can
        # now collapse BEFORE a collision instead of only after one.
        #
        # Positions before recovery, so we can tell afterwards whether it moved
        # anyone -- see the post-recovery refresh before perception below.
        _pos_pre_recovery = ({n.id: n.current_pos.copy() for n in self.nodes}
                             if self.simultaneous_step else None)
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

            # Tell recovery WHO IS CURRENTLY HELD, so it stops counting their
            # continued overlap as fresh offences. A held fleet cannot separate
            # itself; escalating because it has not is a feedback loop, and it
            # ran the 2026-09-17 cold run into a 600-step freeze.
            self.recovery.currently_yielding = {
                fid for fid, until in self._yield_until.items()
                if until > self.step_count
            }
            
        # With the simultaneous step, warnings are splatted in the JUDGE phase,
        # from the configuration this step's actions produced. Splatting here as
        # well would mark the previous action's geometry a second time.
        if current_integrity == 0.5 and not self.simultaneous_step:
            self._splat_warnings()

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
                # Nor does it apply to a fleet on its way OUT. Its target is an
                # exit node, not a pool goal -- but exits are boundary cells and
                # goals can be boundary cells too, so the overlap made
                # `current_goal_id in claimed_goals` fire on a goal that was
                # never its goal. _retarget then found nothing (the pool really
                # was exhausted, which is why the fleet was leaving) and froze
                # it mid-drive.
                #
                # This is the "4 fleets frozen WITH a goal" left open earlier.
                # It reproduced three times out of three once the test asserted
                # on frozen_nodes instead of immobile_nodes -- the first version
                # of that assertion was too broad and hid it behind an unrelated
                # VDA error stop.
                if node.id in self._despawning:
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

        # ---- POST-RECOVERY SNAPSHOT (step.simultaneous) --------------------
        #
        # The proximity index is refreshed once at the top of the step, and the
        # network's "nearest peer" input (sf_peer_proximity) is written from it
        # -- both BEFORE recovery runs. Recovery then teleports fleets and never
        # refreshes the index. So everything downstream -- this step's
        # perception, braking in phase A, the waiting check in phase B -- read
        # positions from before the teleport.
        #
        # Measured over 64 fleets moved by recovery: the stale index put the
        # nearest peer at a mean of 0.49 cells, 88% still inside the collision
        # band; the true distance after the move was 1.46, only 2% still
        # colliding. So a just-repaired fleet was told a peer was on top of it,
        # and braked toward the speed floor, despite being clear.
        #
        # Refresh only when recovery actually moved someone, and rewrite the
        # peer-proximity feature from the fresh index. sf_in_warning and
        # sf_in_deadlock are deliberately left alone: whether they mean "you are
        # in conflict now" or "you were involved in one this step" is a design
        # question, not a stale read.
        if _pos_pre_recovery is not None and any(
                not np.array_equal(_pos_pre_recovery[n.id], n.current_pos)
                for n in self.nodes):
            self.proximity.refresh(self.nodes, excluded_ids=self.immobile_nodes)
            _span = max(1e-6, self.loop.warning_threshold - self.loop.collision_threshold)
            for n in self.nodes:
                n.sf_peer_proximity = float(np.clip(
                    (self.loop.warning_threshold - self.proximity.nearest(n.id)) / _span,
                    0.0, 1.0))
            self.post_recovery_refreshes += 1

        self._stamp_holon()

        # 2. Build States & Get GNN Actions
        node_features = []
        node_ids = []
        valid_action_masks = []
        
        for node in self.nodes:
            node_ids.append(node.id)
            node_features.append(self._perceive(node))

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
        adj_mat = (self._build_adjacency_graph()
                   if self.adjacency_metric == "graph"
                   else self._build_adjacency())
        if self.edge_feature_dim > 0:
            adj_mat = self._build_edge_features(adj_mat)
        
        # GNN Action Selection
        actions = self.gnn.choose_actions(
            node_features=node_features_array,
            adj_matrix=adj_mat,
            episode_number=episode_step,
            total_episodes=total_episodes,
            node_ids=node_ids,
            valid_action_masks=valid_action_masks_array,
        )

        # The network's masked Q-values for THIS step, [num_nodes, 7], or None
        # if no forward pass ran (every fleet explored). Used by the tabu veto
        # below to take the network's second choice instead of a random one.
        action_ranking = getattr(self.gnn, "last_q_values", None)

        # 3. Apply Actions, Affordance Braking, and Calculate Rewards
        step_rewards = []
        step_terms = []
        step_shadow = []   # per fleet: [potential-shaping correction, attributed recovery]
        
        # ======================================================================
        # THE STEP, IN TWO PHASES
        #
        # Phase A: every fleet decides and moves. Movement never reads another
        #          fleet's position, so this is order-independent on its own.
        # Phase B: consequences -- waiting, arrivals and claims, rewards --
        #          judged on the configuration everyone has already committed to.
        #
        # This used to be ONE loop that moved fleet i and then immediately judged
        # it, before fleet i+1 had moved. Shared state written mid-loop -- goal
        # claims, the waiting set -- was then read by later fleets, so outcomes
        # depended on where a fleet sat in self.nodes. Measured with identical
        # actions, forwards vs reversed: 2 of 40 fleets in a different place
        # after one step, 5 of 40 after five.
        #
        # The split itself is always on and changes nothing by itself. With
        # step.simultaneous False, phase B reads live state exactly as before.
        # With it True, phase B reads snapshots taken between the phases, so no
        # fleet's outcome can depend on list order. See SIMULTANEOUS_STEP.md.
        # ======================================================================
        _carry = {}
        # The pre-split loop refreshed the proximity index once, partway through
        # the FIRST fleet's turn -- after that fleet moved and ran its waiting
        # check, before the second fleet braked. Restored for step.simultaneous
        # False so the old code is reproduced exactly; see the end of phase A.
        _leftover_i = None
        _leftover_peers = None
        for i, node in enumerate(self.nodes):
            if node.id in self.immobile_nodes:
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

            # ---- DOORSTEP ----------------------------------------------------
            # Recorded only within one hop of the goal, where every explanation
            # we could test has already been eliminated.
            _dd = node.get_graph_distance_to_goal()
            if _dd is not None and np.isfinite(_dd) and 0.05 < float(_dd) <= 1.05:
                self._doorstep["steps"] += 1
                if action_id == 0:
                    self._doorstep["idle"] += 1
                elif action_id == int(np.argmax(_g)) + 1:
                    self._doorstep["closer"] += 1
                else:
                    self._doorstep["away"] += 1
                if self.proximity.nearest(node.id) <= self.loop.warning_threshold:
                    self._doorstep["peer_within_warning"] += 1

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
                # Time LOST to yielding, per fleet, over the whole episode. The
                # post-mortem only sees each fleet's state at step 780, so a fleet
                # held a dozen times earlier and moving freely by the end looked
                # unaffected. This is the cumulative cost.
                self._steps_held[node.id] = self._steps_held.get(node.id, 0) + 1

            if node.id in self.waiting_nodes:
                self._steps_waiting[node.id] = self._steps_waiting.get(node.id, 0) + 1

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
                # WHAT THIS USED TO DO:
                #     valid_actions = [a for a in range(7)
                #                      if a not in node.tabu_actions[curr_pos_tuple]]
                #     action_id = random.choice(valid_actions) if valid_actions else 0
                #
                # "valid" there meant ONLY "not tabu". It never consulted
                # structural validity, and valid_action_masks_array -- computed
                # thirty lines above for exactly this purpose and handed to the
                # GNN -- was sitting in scope unused. On this graph the average
                # node degree is ~2.27 of 6 possible directions, so roughly 70%
                # of the actions it could pick walk into a rack: the fleet burns
                # the step, apply_discrete_action zeroes its direction, and the
                # replacement for a fatal move is a wall.
                #
                # CORRECTION TO AN EARLIER DIAGNOSIS, recorded here so it is not
                # repeated: I previously claimed that recording the EXECUTED
                # action (line ~1631) turned overridden steps into "training the
                # network to imitate noise". That is wrong. GNNAgent.learn() is
                # off-policy DQN -- smooth_l1_loss(q_taken, r + gamma * max
                # q_target) with no imitation term -- so a transition
                # (s, a_executed, r, s') is CORRECT training data whatever chose
                # a_executed. Recording the executed action is right and stays.
                #
                # The defect that IS real: the substitute was uniform over
                # non-tabu actions, so most overrides produced a wall-bump
                # transition. That teaches Q(s, wall) correctly and teaches
                # nothing about which legal move was better.
                #
                # WHAT IT DOES NOW: restrict to actions that are both non-tabu
                # AND structurally valid, then take the network's OWN ranking
                # over that set -- its second choice, not a coin flip. Falls back
                # to the goal gradient, then to idle. Random selection is kept
                # only for the degenerate case where nothing is ranked.
                tabu = node.tabu_actions[curr_pos_tuple]
                structurally_ok = valid_action_masks_array[i]
                candidates = [a for a in range(7)
                              if a not in tabu and bool(structurally_ok[a])]

                if not candidates:
                    # Every legal move is tabu. Idle is always structurally
                    # valid, so this only triggers when idle itself is tabu --
                    # in which case holding still is still the least-bad option.
                    action_id = 0
                    self.tabu_overrides_stuck += 1
                elif action_ranking is not None:
                    action_id = int(max(candidates, key=lambda a: action_ranking[i][a]))
                else:
                    _gt = node.get_goal_gradient()
                    ranked = [a for a in candidates if 1 <= a <= 6 and _gt[a - 1] > 0]
                    if ranked:
                        action_id = int(max(ranked, key=lambda a: _gt[a - 1]))
                    elif self.simultaneous_step:
                        # The global generator is consumed fleet by fleet, so a
                        # draw here depended on how many fleets before this one
                        # had drawn -- reverse the list and the same fleet gets a
                        # different number. Seed by fleet and step instead.
                        # zlib.crc32, not hash(): str hashes change per process.
                        _rng = random.Random(zlib.crc32(f"{node.id}:{self.step_count}".encode()))
                        action_id = _rng.choice(candidates)
                    else:
                        action_id = random.choice(candidates)
                self.tabu_overrides += 1
            
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
                # WAS: build a list of every other fleet's position, then
                #      np.min(np.sum(np.abs(...))) -- Manhattan through racks,
                #      O(N) list construction per fleet per step.
                # NOW: the shared graph-distance index. Two fleets in adjacent
                #      aisles behind a rack, or both mid-edge on parallel tracks,
                #      no longer throttle each other for a collision they cannot
                #      have. See proximity_warehouse.py and
                #      test_loop_equivalence.py for both phantom species.
                #
                # The brake_on_immobile ablation is preserved: it asks whether
                # parked fleets should still throttle traffic, which is a
                # question about the EXCLUSION SET, not the metric.
                min_dist = (self._prox_all.nearest(node.id)
                            if self._prox_all is not None
                            else self.proximity.nearest(node.id))

                # BRAKE DUTY CYCLE. Occupancy (agents/nodes) is a convenient
                # axis but not a physical one: it treats a degree-2 corridor
                # cell and a junction as equivalent, ignores that traffic
                # concentrates on routes, and says nothing about how close
                # vehicles actually are. What throttles a fleet is THIS number --
                # distance to the nearest live peer -- so the honest density
                # measure is the fraction of fleet-steps spent inside the
                # warning band, i.e. the fraction of operating time under brake.
                # Dimensionless, comparable across any map or fleet count, and
                # it is the mechanism rather than a proxy for it.
                self._brake_eval_steps += 1
                # BUG THIS FIXES (pre-existing): min_dist was float('inf')
                # whenever every other fleet was immobile -- routine in the last
                # third of an episode -- and this line summed it, poisoning
                # mean_peer_gap to inf for the remainder of the run. The value is
                # now CENSORED at proximity.search_radius, so mean_peer_gap is a
                # censored mean and has to be reported as one.
                self._min_dist_sum += float(
                    self.proximity.search_radius
                    if not np.isfinite(min_dist)
                    else min(min_dist, self.proximity.search_radius))
                if min_dist <= self.loop.warning_threshold:
                    self._braked_fleet_steps += 1
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
                # Sampled HERE, not after step() returns.
                #
                # BUG THIS FIXES: the 200-step warmup on 2026-09-13 reported
                # mean_speed and min_speed pinned at 0.500 for every block,
                # while collisions were occurring -- which cannot both be true.
                # Cause: `node.speed = base_speed` at the end of this loop
                # restores the unbraked value, and the profiler sampled after
                # step() had returned. It was measuring the restored number.
                # The braked speed only exists inside this loop, so it has to be
                # recorded inside it.
                self._braked_speed_sum += node.speed * 1.0
                self._braked_speed_n += 1
                if node.speed < self._braked_speed_min:
                    self._braked_speed_min = float(node.speed)
            
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
            _carry[i] = (_moved, base_speed, old_dist, old_pos)

            # RESTORED, flag off only. In the old single loop this refresh sat in
            # the first fleet's consequences: its waiting check read the
            # UNREFRESHED index, then the refresh ran, then every later fleet
            # braked and waited against the refreshed one. The split puts all
            # braking before any waiting, so no single refresh point reproduces
            # both. Instead: record the first fleet's waiting neighbours from the
            # unrefreshed index, THEN refresh; phase B hands that fleet the
            # recorded answer. With step.simultaneous True it does not run --
            # it is an order leak, and the flag exists to remove those.
            if (not self.simultaneous_step and self.attribute_collisions_post_action
                    and _leftover_i is None):
                if self.waiting_enabled:
                    _leftover_peers = self.proximity.peers_within(
                        node.id, self.wait_block_threshold)
                self.proximity.refresh(self.nodes, excluded_ids=self.immobile_nodes)
                (self._post_action_deadlocked,
                 self._post_action_warning) = self.loop.peek_conflicts(self.proximity)
                _leftover_i = i

        # ---- BETWEEN THE PHASES: snapshot, and resolve simultaneous arrivals ---
        _claimed_snap = frozenset(self.claimed_goals)
        _waiting_snap = frozenset(self.waiting_nodes)
        _goal_winner = {}
        if self.simultaneous_step and self.shared_pool_mode:
            # Two fleets reaching one goal in the same step: the closer one wins,
            # ties broken by id. Never by position in the list -- that is exactly
            # the dependence this removes.
            _cands = {}
            for _i in _carry:
                _n = self.nodes[_i]
                _g = _n.current_goal_id
                if _g is None or _g in _claimed_snap or _n.id in self._pickup_assignment:
                    continue
                _d = _n.get_graph_distance_to_goal()
                if _d is None or not np.isfinite(_d) or _d >= self.arrival_radius:
                    continue
                _cands.setdefault(_g, []).append((float(_d), str(_n.id), _n.id))
            for _g, _lst in _cands.items():
                _goal_winner[_g] = min(_lst)[2]

        def _may_claim(goal, fid):
            if self.simultaneous_step:
                return goal not in _claimed_snap and _goal_winner.get(goal) == fid
            return goal not in self.claimed_goals

        _pending_retarget = []
        for i, node in enumerate(self.nodes):
            if i not in _carry:
                step_rewards.append(np.zeros(self.K, dtype=np.float32))
                step_terms.append(np.zeros(len(self._TERMS), dtype=np.float32))
                step_shadow.append((0.0, 0.0))
                continue
            _moved, base_speed, old_dist, old_pos = _carry[i]

            # ---- WAITING vs IDLING ------------------------------------------
            # The distinction is WHY the fleet did not move, and it is decided
            # here rather than by a separate action, deliberately.
            #
            # An eighth HOLD action was the obvious alternative and is worse:
            # it grows the network's output layer, and since HOLD would be
            # reward-exempt while IDLE is not, HOLD strictly dominates and IDLE
            # becomes a dead action the policy never selects. Same behaviour,
            # one more dimension, one less usable action.
            #
            # So: held still AND a mobile peer is within block_threshold graph
            # cells -> WAITING. Held still in open space -> idling, penalised as
            # before. Persistence is not forced; it emerges for as long as the
            # blocking condition persists, and the policy keeps its per-step say.
            if self.waiting_enabled:
                _peers = (_leftover_peers if i == _leftover_i
                          else self.proximity.peers_within(
                              node.id, self.wait_block_threshold))
                if _moved < 1e-6 and _peers:
                    if node.id not in self.waiting_nodes:
                        self.waiting_nodes.add(node.id)
                        self.waits_started += 1
                    n_wait = self._wait_steps.get(node.id, 0) + 1
                    self._wait_steps[node.id] = n_wait
                    self.wait_steps_total += 1
                    # ---- THE CAP DEPENDS ON WHY YOU ARE BLOCKED -------------
                    # A fixed timer gets two opposite cases wrong: a fleet
                    # queueing behind a MOVING peer is penalised at step 13 for
                    # correct behaviour, while a fleet in a mutual standoff is
                    # exempt for 12 steps of futility.
                    #
                    # Three cases, and one never reaches this branch:
                    #
                    #   blocker DEAD or PARKED -- never reaches here, and the
                    #     reason is that THEY ARE NOT BLOCKING ANYTHING. Neither
                    #     is a hard obstacle: check_integrity exempts all of
                    #     immobile_nodes, so passing through a dead or parked
                    #     fleet costs no collision, and the density field only
                    #     expresses a soft preference. Better still, the
                    #     near-goal discount makes a dead fleet COMPLETELY
                    #     TRANSPARENT to its own rescuer -- whose goal_pos IS
                    #     that cell, so the discount goes to zero -- while every
                    #     other fleet still sees full severity. So nobody should
                    #     ever wait on one: a rescuer drives straight through,
                    #     and anyone else goes around or pushes past.
                    #
                    #   blocker MOVING -- productive queueing. Long cap.
                    #
                    #   blocker ALSO WAITING -- mutual standoff. Waiting longer
                    #     is strictly worse, so the exemption lapses fast.
                    _blocker = _peers[0][0]
                    _mutual = _blocker in (_waiting_snap if self.simultaneous_step
                                          else self.waiting_nodes)
                    _cap = self.mutual_wait_steps if _mutual else self.max_wait_steps

                    # SYMMETRY BREAKING. Two fleets sharing a cap lapse on the
                    # same step, both move, and re-collide -- the
                    # [19,49]-colliding-314-times pattern in a different costume.
                    # The fleet FURTHER from its goal keeps the long cap and
                    # holds; the one closer is pushed to move first. Ties break
                    # on id, deterministically.
                    if _mutual:
                        self.mutual_waits += 1
                        _peer_node = next((p for p in self.nodes
                                           if p.id == _blocker), None)
                        _mine = node.get_graph_distance_to_goal()
                        _theirs = (_peer_node.get_graph_distance_to_goal()
                                   if _peer_node is not None else _mine)
                        if _mine > _theirs or (_mine == _theirs and node.id > _blocker):
                            _cap = self.max_wait_steps

                    if n_wait == _cap + 1:
                        self.waits_capped += 1

                    self._wait_cap[node.id] = _cap
                    node.sf_is_waiting = 1.0
                    node.sf_wait_steps = min(1.0, n_wait / max(1, _cap))
                else:
                    self.waiting_nodes.discard(node.id)
                    self._wait_steps.pop(node.id, None)
                    self._wait_cap.pop(node.id, None)
                    node.sf_is_waiting = 0.0
                    node.sf_wait_steps = 0.0
            
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
            tvec = np.zeros(len(self._TERMS), dtype=np.float32)
            H = self.HEAD
            _rterm = self.baseline_reward
            rvec[H["goal"]] += _rterm
            tvec[self._TI[("goal", "baseline")]] += _rterm
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
                _rterm = self._service_pickups(node)
                rvec[H["rescue"]] += _rterm
                tvec[self._TI[("rescue", "pickup")]] += _rterm
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
            # ARRIVAL RADIUS. get_graph_distance_to_goal() INTERPOLATES: a fleet
            # mid-edge between the goal cell and its neighbour reads 0.5, not 0
            # or 1. At base_speed 0.5 a fleet is mid-edge half the time, so a
            # threshold of 0.1 means "within a tenth of a cell" -- far stricter
            # than everything else in the system, which rounds position to cells.
            #
            # Measured: non-arriving fleets reach a best distance of 1.0 and
            # then drift back out to 5.0, and 2-4 active fleets per episode get
            # inside 0.5 of their goal without ever being credited. 0.5 is the
            # consistent definition -- is the fleet's CELL the goal cell.
            if new_dist < self.arrival_radius and not _on_pickup_mission:
                if self.shared_pool_mode:
                    if node.current_goal_id is not None and _may_claim(node.current_goal_id, node.id):
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
                        # Stamp WHEN each goal was claimed. Needed to measure
                        # rescue latency -- steps from the failure that orphaned
                        # an order to the delivery that recovered it. The naive
                        # baseline reports this because its rescuer is retargeted
                        # explicitly; FLOWRRA's rescues run through the
                        # orchestrator and produced no latency figure at all,
                        # which made the comparison one-sided for no reason
                        # other than missing instrumentation.
                        self._goal_claim_step[node.current_goal_id] = self.step_count
                        delivered = node.current_goal_id
                        _rterm = self.reward_mission_complete
                        rvec[H["goal"]] += _rterm
                        tvec[self._TI[("goal", "delivery")]] += _rterm
                        if delivered in self._inherited_orders:
                            self.inherited_orders_delivered += 1

                        # DELIVERING A RESCUED ORDER also pays the rescue head.
                        # The sum the bootstrap argmax sees becomes
                        # mission_complete + this, so a rescued delivery outranks
                        # an ordinary one -- which is the intent.
                        #
                        # It goes in RESCUE rather than being added to
                        # mission_complete because the sum is identical either
                        # way and the head is not: in "goal" it muddies what that
                        # head means; in "rescue" it gives a head that fires 1.36
                        # times an episode a SECOND event, tied to an outcome the
                        # network already predicts well.
                        _ro = self._rescued_orders.get(node.id)
                        if _ro and delivered in _ro:
                            _rterm = self.rescue_delivery_bonus
                            rvec[H["rescue"]] += _rterm
                            tvec[self._TI[("rescue", "rescue_delivery")]] += _rterm
                            _ro.discard(delivered)
                            self.rescued_orders_delivered += 1

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

                            exit_node = (self._nearest_exit(node)
                                         if self.despawn_on_delivery else None)
                            if exit_node is not None:
                                # DRIVE OUT instead of parking on the goal cell.
                                #
                                # A frozen fleet is excluded from the proximity
                                # index, so it stops being a blocker for waiting
                                # and a candidate for collisions -- it leaves
                                # only a static density stamp, forever, on a cell
                                # somebody wanted. A despawning fleet is FULLY
                                # ACTIVE for the drive out: counted in proximity,
                                # in collisions, in waiting, in path conflict.
                                # More real congestion for longer, then none.
                                #
                                # It also removes the one state Phase 4 could not
                                # express honestly. ray_hit_permanent means "this
                                # will never move again", which is true of a dead
                                # fleet and false of a parked one that might be
                                # recalled. Despawning deletes the ambiguous case
                                # rather than encoding it.
                                self._despawning[node.id] = exit_node
                                node.current_goal_id = exit_node
                                node.goal_pos = np.array(
                                    self._coords_by_id[exit_node], dtype=np.float32)
                                node.goal_distance_map = self._exit_distance_maps.get(exit_node)
                                node.initial_graph_distance = node.get_graph_distance_to_goal()
                                self._best_dist.pop(node.id, None)
                                self._stall_steps[node.id] = 0
                                print(f"[Core] Fleet {node.id} claimed goal {delivered} "
                                      f"({len(self.claimed_goals)}/{len(self.goal_pool)} claimed); "
                                      f"heading out via {exit_node}.")
                            else:
                                self.frozen_nodes.add(node.id)
                                self.gnn.freeze_node(node.id, node.current_pos)
                                print(f"[Core] Fleet {node.id} claimed goal {delivered} and retired! "
                                      f"({len(self.claimed_goals)}/{len(self.goal_pool)} claimed)")
                    else:
                        # Arrived at a goal a peer claimed first (this same
                        # step, or while this fleet was still en route) --
                        # nothing was actually accomplished, so no bonus and no
                        # retirement, just retarget and keep trying.
                        if self.simultaneous_step:
                            # Deferred: retargeting reads claimed_goals and
                            # targeted_goals, which other fleets are still
                            # changing in this pass. Resolved after the loop in
                            # a fixed order that does not depend on list position.
                            _pending_retarget.append(node)
                        else:
                            found = self._retarget(node)
                            if not found:
                                self.frozen_nodes.add(node.id)
                                self.gnn.freeze_node(node.id, node.current_pos)
                                print(f"[Core] Fleet {node.id} frozen -- pool fully claimed "
                                      f"({len(self.claimed_goals)}/{len(self.goal_pool)}).")
                else:
                    _rterm = self.reward_mission_complete
                    rvec[H["goal"]] += _rterm
                    tvec[self._TI[("goal", "delivery")]] += _rterm
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
                    # SECOND EXCEPTION: a fleet holding still because it is
                    # BLOCKED is doing the right thing. Penalising that is what
                    # taught the policy to push into jams.
                    #
                    # Bounded on purpose. The exemption lapses after
                    # max_wait_steps so the penalty resumes and the policy is
                    # pushed to try something else -- otherwise a fleet could
                    # park beside a peer and farm a free ride forever, which is
                    # what makes a POSITIVE reward for inaction unsafe. An
                    # exemption can only ever be worth zero.
                    _waiting_exempt = (
                        self.waiting_enabled
                        and node.id in self.waiting_nodes
                        and self._wait_steps.get(node.id, 0)
                            <= self._wait_cap.get(node.id, self.max_wait_steps)
                    )
                    if self._pickup_dwell.get(node.id, 0) == 0 and not _waiting_exempt:
                        _rterm = self.idle_penalty
                        rvec[H["time"]] += _rterm
                        tvec[self._TI[("time", "idle")]] += _rterm
                else:
                    _rterm = (old_dist - new_dist) * self.movement_reward_multiplier
                    rvec[H["goal"]] += _rterm
                    tvec[self._TI[("goal", "progress")]] += _rterm

                    # Final-approach push: reinforce that pushing through the
                    # warning zone late in the journey is GOOD, not risky --
                    # directly countering the natural pull of warning-zone
                    # caution right when a fleet should be finishing. Scoped to
                    # genuine progress specifically, same as before.
                    if (new_dist < old_dist and node.id in self.loop.warning_nodes
                            and node.get_progress_fraction() >= self.final_approach_threshold):
                        _rterm = self.final_approach_bonus
                        rvec[H["goal"]] += _rterm
                        tvec[self._TI[("goal", "final_approach")]] += _rterm

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
                    _rterm = overtime_frac * self.overtime_max_penalty
                    rvec[H["time"]] += _rterm
                    tvec[self._TI[("time", "overtime")]] += _rterm
                    
            # D. Spatial-Temporal Consequences
            if node.id in self._step_deadlocked:
                _rterm = self.reward_fatal_collision
                rvec[H["safety"]] += _rterm
                tvec[self._TI[("safety", "collision")]] += _rterm
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
                # WAS: a second O(N^2) generator, identical in intent to the
                #      sf_peer_proximity one above but written out again. Same
                #      Manhattan metric, same exclusion, independently maintained.
                # NOW: the same index both of them should always have shared, so
                #      the safety FEATURE and the safety REWARD cannot disagree.
                nearest = self.proximity.nearest(node.id)
                span = max(1e-6, self.loop.warning_threshold - self.loop.collision_threshold)
                closeness = float(np.clip(
                    (self.loop.warning_threshold - nearest) / span, 0.0, 1.0))
                _rterm = self.reward_warning_zone * closeness
                rvec[H["safety"]] += _rterm
                tvec[self._TI[("safety", "warning")]] += _rterm

            # E. Integrity: a per-step signal on how coherent the holon is, so
            # the integrity head has something dense to learn from rather than
            # only the sparse invocation events.
            _rterm = (self.loop.current_integrity - 1.0)
            rvec[H["integrity"]] += _rterm
            tvec[self._TI[("integrity", "coherence")]] += _rterm
            _rterm = self._pending_integrity_reward
            rvec[H["integrity"]] += _rterm
            tvec[self._TI[("integrity", "recovery_events")]] += _rterm
            # SHADOW: this fleet's own share of recovery events, and the term
            # that turns progress into exact potential-based shaping:
            # m*(old - gamma*new) = m*(old - new) + m*(1 - gamma)*new. A fleet
            # that delivered this step is terminal (potential 0), so it gets 0.
            _sh_rec = (self._pending_integrity_by_fleet.get(node.id, 0.0)
                       + self._pending_integrity_system / max(1, len(_carry)))
            _sh_prog = (0.0 if node.id in self.immobile_nodes else
                        self.movement_reward_multiplier
                        * (1.0 - float(getattr(self.gnn, "gamma", 0.99))) * float(new_dist))

            step_rewards.append(rvec)
            step_terms.append(tvec)
            step_shadow.append((_sh_prog, _sh_rec))

        # Deferred retargets, in a fixed order independent of list position.
        for node in sorted(_pending_retarget, key=lambda n: str(n.id)):
            found = self._retarget(node)
            if not found:
                self.frozen_nodes.add(node.id)
                self.gnn.freeze_node(node.id, node.current_pos)
                print(f"[Core] Fleet {node.id} frozen -- pool fully claimed "
                      f"({len(self.claimed_goals)}/{len(self.goal_pool)}).")

        self._pending_integrity_reward = 0.0
        self._pending_integrity_by_fleet = {}
        self._pending_integrity_system = 0.0

        # [N, K] -- one row per fleet, one column per reward head.
        step_rewards_array = np.asarray(step_rewards, dtype=np.float32)
        step_terms_array = (np.asarray(step_terms, dtype=np.float32) if step_terms
                            else np.zeros((0, len(self._TERMS)), dtype=np.float32))
        step_shadow_array = (np.asarray(step_shadow, dtype=np.float64) if step_shadow
                             else np.zeros((0, 2), dtype=np.float64))
        _sh_warn = None      # corrected warning, filled by the re-attribution

        # ---- SECOND PASS: charge the collision to the action that caused it ---
        #
        # The loop above applies each fleet's action AND computes its reward in
        # the same iteration, so fleet 0's reward is evaluated against everyone
        # else's OLD positions -- conflict tests inside it are order-dependent.
        # And self._step_deadlocked comes from check_integrity(), which runs at
        # the START of the step, so it describes the PREVIOUS action's geometry.
        #
        # Measured 2026-09-20 on a real episode: 100% of -50 penalties were for
        # overlaps that existed before the penalised action ran, and 83% of those
        # actions had just RESOLVED the overlap. The action that causes a
        # collision was charged nothing; the action that fixed it was charged the
        # largest penalty in the system. Not a delay -- an inversion.
        #
        # So: subtract what the loop charged from the stale sets, re-check the
        # geometry ONCE now that every fleet has moved, and charge that instead.
        # Read-only -- peek_conflicts touches no counter, because
        # total_collisions, current_integrity and integrity_history all belong to
        # the start-of-step check that recovery and every published metric use.
        if self.attribute_collisions_post_action and len(step_rewards_array):
            self.proximity.refresh(self.nodes, excluded_ids=self.immobile_nodes)
            (self._post_action_deadlocked,
             self._post_action_warning) = self.loop.peek_conflicts(self.proximity)
            _sh = self.HEAD["safety"]
            _sh_warn = np.zeros(len(step_rewards_array), dtype=np.float64)
            _tc = self._TI[("safety", "collision")]
            _tw = self._TI[("safety", "warning")]
            for i, node in enumerate(self.nodes):
                if i >= len(step_rewards_array):
                    break
                # undo the stale charge
                if node.id in self._step_deadlocked:
                    step_rewards_array[i, _sh] -= self.reward_fatal_collision
                    step_terms_array[i, _tc] -= self.reward_fatal_collision
                elif node.id in self._step_warning:
                    nearest = self.proximity.nearest(node.id)
                    span = max(1e-6, self.loop.warning_threshold - self.loop.collision_threshold)
                    _undo = self.reward_warning_zone * float(
                        np.clip((self.loop.warning_threshold - nearest) / span, 0.0, 1.0))
                    # DIAGNOSTIC, read-only. The charge being undone was computed
                    # in the step from the fleet's position BEFORE its move; this
                    # undo recomputes closeness from the index refreshed AFTER it.
                    # They differ whenever the fleet moved, so the stale charge is
                    # not removed: a fleet that stays in the zone pays its
                    # starting closeness regardless of its action, and one that
                    # ESCAPES keeps its full starting penalty. Collisions are
                    # unaffected -- their undo subtracts the same constant.
                    _orig = float(step_terms_array[i, _tw])
                    self._warnundo_n += 1
                    self._warnundo_error += _orig - _undo
                    if abs(_orig - _undo) > 1e-6:
                        self._warnundo_mismatched += 1
                        if (node.id not in self._post_action_warning
                                and node.id not in self._post_action_deadlocked):
                            self._warnundo_escaped_kept += 1
                    step_rewards_array[i, _sh] -= _undo
                    step_terms_array[i, _tw] -= _undo
                # charge what this step's actions actually produced
                if node.id in self._post_action_deadlocked:
                    step_rewards_array[i, _sh] += self.reward_fatal_collision
                    step_terms_array[i, _tc] += self.reward_fatal_collision
                elif node.id in self._post_action_warning:
                    nearest = self.proximity.nearest(node.id)
                    span = max(1e-6, self.loop.warning_threshold - self.loop.collision_threshold)
                    _recharge = self.reward_warning_zone * float(
                        np.clip((self.loop.warning_threshold - nearest) / span, 0.0, 1.0))
                    step_rewards_array[i, _sh] += _recharge
                    step_terms_array[i, _tw] += _recharge
                    # SHADOW: the warning this action actually produced.
                    _sh_warn[i] = _recharge

        # ---- JUDGE: mark consequences in the SAME transition as their penalty --
        #
        # A collision caused by a(t) used to be charged -50 in transition t but
        # splatted only at the start of step t+1, so the next state the penalty
        # is bootstrapped from showed no trace of it. Measured on fresh
        # collisions: the splat was ABSENT from next_s(t) 52% of the time (mean
        # 0.334 against 1.050 one step later). Splatting here -- after every
        # fleet has moved and before next_s(t) is perceived -- puts the penalty
        # and its visible trace in one transition. Splats use max(), so the
        # start-of-step re-detection at t+1 is idempotent, not double-counted.
        if self.simultaneous_step and len(step_rewards_array):
            if not self.attribute_collisions_post_action:
                self.proximity.refresh(self.nodes, excluded_ids=self.immobile_nodes)
                (self._post_action_deadlocked,
                 self._post_action_warning) = self.loop.peek_conflicts(self.proximity)
            _by_id = {n.id: n for n in self.nodes}
            for _fid in self._post_action_deadlocked:
                _n = _by_id.get(_fid)
                if _n is not None:
                    self.density.splat_spatial_temporal_event(
                        _n.current_pos, severity_multiplier=self.fatal_splat_multiplier)
            if self._post_action_warning:
                self._splat_warnings()
        
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
            self._stamp_holon()
            next_node_features = []
            for node in self.nodes:
                next_node_features.append(self._perceive(node))
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
            # See the snapshot taken before the action loop. The post-loop view
            # excludes every fleet that delivered this step, which is exactly the
            # sample carrying mission_complete.
            if (self._active_at_action_time is not None
                    and len(self._active_at_action_time) == len(self.nodes)):
                active_mask = self._active_at_action_time
            else:
                active_mask = np.array(
                    [n.id not in self.immobile_nodes for n in self.nodes],
                    dtype=np.float32
                )

            # PRIORITY MODE: the learner receives the three priority heads,
            # each divided by its fixed value scale. Legacy: the five heads.
            if self.reward_mode == "priority":
                _pr = self._priority_rewards(step_terms_array.astype(np.float64),
                                             step_shadow_array, _sh_warn,
                                             include_potential=False)
                # Fourth column, for the RECOVERY HEAD only: the legacy integrity head
                # (holon coherence + every recovery event at full size, identical on
                # every fleet), unscaled -- exactly the signal it trained on before.
                _push_rewards = np.column_stack(
                    [_pr[h] / self.priority_scales[h] for h in PRIORITY_HEADS]
                    + [step_rewards_array[:, self.HEAD["integrity"]]]).astype(np.float32)
                _am = np.asarray(active_mask) > 0
                for _j, _h in enumerate(PRIORITY_HEADS):
                    _c = _push_rewards[_am, _j]
                    self._live[_h]["pos"] += float(_c[_c > 0].sum())
                    self._live[_h]["neg"] += float(_c[_c < 0].sum())
            else:
                _push_rewards = step_rewards_array
            self.gnn.memory.push(
                node_features_array,
                adj_mat,
                actions,
                _push_rewards,
                next_node_features_array,
                next_adj_mat,
                done,
                current_integrity,
                active_mask,
                self.last_recovery_mode,
                # Structural validity in the NEXT state. Without it the target
                # maxes over impossible actions whose Q-values no transition
                # ever corrects: loss_goal 0.39 -> 384.5 in cold_run5, and
                # r(episode) = +0.97 in warm_run7, with completion FLAT in both
                # because the action mask kept the rot out of the behaviour.
                #
                # Recomputed, not reused: valid_action_masks_array was built
                # BEFORE the action loop, so it describes where fleets were.
                next_valid_mask=np.array(
                    [n.get_valid_action_mask() for n in self.nodes], dtype=bool),
            )
        
        # ---- REWARD COMPOSITION (measurement only) -------------------------
        # The final [N, K] matrix -- after post-action re-attribution, exactly
        # what was pushed. The learned view uses the same active mask the push
        # used: the pre-action snapshot, else the immobile set.
        if step_rewards_array.size:
            if (self._active_at_action_time is not None
                    and len(self._active_at_action_time) == len(self.nodes)):
                _act = np.asarray(self._active_at_action_time) > 0
            else:
                _act = np.array([n.id not in self.immobile_nodes
                                 for n in self.nodes], dtype=bool)
            for _h, _k in self.HEAD.items():
                _col = step_rewards_array[:, _k]
                _lrn = _col[_act]
                self._rwd["all"]["pos"][_h] += float(_col[_col > 0].sum())
                self._rwd["all"]["neg"][_h] += float(_col[_col < 0].sum())
                self._rwd["learned"]["pos"][_h] += float(_lrn[_lrn > 0].sum())
                self._rwd["learned"]["neg"][_h] += float(_lrn[_lrn < 0].sum())
            # per term (active fleets), and per-head per-step spread
            if len(step_terms_array) == len(_act):
                _tl = step_terms_array[_act]
                self._term_pos += np.where(_tl > 0, _tl, 0.0).sum(axis=0)
                self._term_neg += np.where(_tl < 0, _tl, 0.0).sum(axis=0)
            # ---- SHADOW MODE: the three-head reward, never learned from ----
            if (len(step_terms_array) == len(_act) == len(step_shadow_array)):
                _T = step_terms_array.astype(np.float64)
                _shadow = self._priority_rewards(_T, step_shadow_array, _sh_warn,
                                                 include_potential=True)
                for _hn, _v in _shadow.items():
                    _va = _v[_act]
                    self._shadow[_hn]["pos"] += float(_va[_va > 0].sum())
                    self._shadow[_hn]["neg"] += float(_va[_va < 0].sum())
                    self._shadow[_hn]["sum"] += float(_va.sum())
                    self._shadow[_hn]["sq"] += float((_va ** 2).sum())
                    _nzm = np.abs(_va) > 1e-9
                    self._shadow[_hn]["nz"] += int(_nzm.sum())
                    self._shadow[_hn]["abs"] += float(np.abs(_va[_nzm]).sum())
                self._shadow_n += int(_act.sum())
            _hl = step_rewards_array[_act].astype(np.float64)
            self._spread_n += _hl.shape[0]
            self._spread_sum += _hl.sum(axis=0)
            self._spread_sq += (_hl ** 2).sum(axis=0)

        self.step_count += 1
        # Scalar return value is the WEIGHTED SUM across heads, for logging
        # only -- the buffer receives the full [N, K] matrix.
        return float(step_rewards_array.sum()) if step_rewards_array.size else 0.0