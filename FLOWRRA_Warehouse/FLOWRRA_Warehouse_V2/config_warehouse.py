"""
config_Warehouse.py

Centralized configuration payload for the DhaaRn FLOWRRA orchestrator.
Manages hyper-parameters for the discrete warehouse GNN, density fields, and training loops.
"""

CONFIG = {
    # ==========================================
    # 1. HARDWARE & WAREHOUSE LIMITS
    # ==========================================
    "warehouse": {
        "bounds": (50.0, 50.0, 10.0),      # FALLBACK ONLY. Maximum X, Y, Z dimensions for
                                            # normalization. FLOWRRA now derives these per
                                            # map from the actual node coordinate extents
                                            # (see core_warehouse.py's _derive_bounds), and
                                            # only falls back to this literal if the
                                            # coordinate dict is empty or degenerate.
                                            #
                                            # WHY: get_relative_goal_displacement() divides
                                            # the goal offset by these bounds and clips to
                                            # +-1. Hardcoded at 50, every goal more than 50
                                            # cells away on a 100x100 map saturates to
                                            # exactly 1.0 -- direction survives, MAGNITUDE
                                            # is destroyed, and it is destroyed precisely
                                            # for the long journeys the multi-map curriculum
                                            # exists to teach. Deriving per map also makes
                                            # the feature scale-invariant, which is what you
                                            # want for transfer: "60% of the way across this
                                            # warehouse" should mean the same thing on every
                                            # map.
        "base_speed": 0.5,                 # Discrete movement speed scalar
        "max_vision_range": 10,            # Edges an AGV can see down an aisle
        "collision_threshold": 0.5,        # Manhattan distance triggering a fatal crash.
                                            # Was 1.0 -- but every edge in the real graph
                                            # is exactly 1.0 long (confirmed: 11230/11230),
                                            # so a threshold of 1.0 meant "same cell" (dist=0,
                                            # a true overlap) and "adjacent cell" (dist=1, one
                                            # aisle over) were scored identically as fatal.
                                            # 0.5 makes only true overlaps fatal; adjacent-node
                                            # proximity now falls into the warning zone instead.
        "warning_threshold": 2.0,          # Manhattan distance triggering safety yield

        # Final-approach override: many fleets in this dataset have NEARBY goals
        # (confirmed by inspecting the missions), so it's common and expected --
        # not a sign of real conflict -- for two fleets to end up in each other's
        # warning zone right as they're both about to finish. Without this, the
        # normal warning-zone speed throttle (see step()'s affordance braking)
        # slows a fleet down exactly when it should be pushing through the last
        # stretch. Once a fleet has covered final_approach_threshold of its
        # ORIGINAL journey (get_progress_fraction() in node_warehouse.py), its
        # warning-zone speed floor is raised to final_approach_speed_floor instead
        # of the normal ramp's minimum -- but ONLY in the warning zone, never in
        # a genuine fatal-distance situation, where safety still floors it at 0.1
        # regardless of progress.
        "final_approach_threshold": 0.85,
        "final_approach_speed_floor": 0.7,
    },
    
    # ==========================================
    # 1b. GOAL ASSIGNMENT (shared-pool mode only)
    # ==========================================
    # Ignored entirely when USE_SHARED_POOL_MODE is False -- fixed-mission mode
    # has a pre-assigned 1:1 pairing from the CSV already.
    "assignment": {
        # "hungarian": one-time optimal 1:1 fleet->goal matching at spawn,
        #     minimizing TOTAL graph distance across the fleet
        #     (scipy.optimize.linear_sum_assignment, with a greedy 1:1 fallback
        #     if scipy is missing). Every goal gets exactly one owner and every
        #     fleet an uncontested target.
        # "greedy": the previous behaviour -- each fleet independently picks its
        #     own nearest available goal, in fleet-list order. Now at least
        #     reservation-aware, so it can no longer double-book, but the picks
        #     are still uncoordinated and the total is not optimal.
        #
        # Measured on 25 random 25-fleet layouts: uncoordinated picking left an
        # average of 9.5 fleets (38%) chasing a goal another fleet would reach
        # first. Note that "hungarian" costs ~51% MORE total travel (177 vs 118
        # hops) -- it refuses to let three fleets share one cheap goal while two
        # distant goals go unserved. It removes the free win, not the difficulty.
        "mode": "hungarian",

        # Reservation ledger (core_warehouse.py's targeted_goals). Applies to
        # BOTH modes above and to every mid-episode retarget: a goal another
        # fleet is currently en route to is unavailable, not just one that has
        # already been physically claimed. Keeps the 1:1 property that the
        # spawn assignment establishes from decaying over the episode.
        "reserve_targets": True,
    },

    # ==========================================
    # 2. DISCRETE PHYSICS & DENSITY
    # ==========================================
    "density": {
        # REWORKED (2026-08-28). The Poisson kernel is gone -- see
        # density_warehouse.py's module docstring for the full rationale. Short
        # version: the Poisson survival curve was just a monotone-decreasing
        # falloff with an illegible shape parameter, and none of the actual
        # Poisson structure was ever used. A linear ramp does the same job with a
        # number you can reason about.
        #
        # lambda_severity / decay_rate REMOVED. WarehouseDensityField still
        # ACCEPTS them (so an un-updated caller constructs without error) but
        # ignores them entirely.

        "falloff_radius": 3.0,             # Repulsion reaches zero at this Manhattan
                                            # distance. k(d) = max(0, 1 - d/r), so at 3.0
                                            # the curve is [1.0, 0.67, 0.33, 0.0]. Replaces
                                            # lambda_severity. Bigger = wider, softer
                                            # bubbles; smaller = tighter, more local.
        "peer_severity": 0.7,              # Repulsion an active peer stamps at its own
                                            # cell. Unchanged in value from the old
                                            # hardcoded 0.7, now configurable.
        "projection_steps": 3,             # SWEPT PATH length, in CELLS (node.direction is
                                            # a unit cell delta). Was effectively 1, and
                                            # isotropic. Stamping the cells a peer is about
                                            # to occupy is what lets the field distinguish
                                            # a fleet closing head-on from one receding --
                                            # a symmetric ball scores both identically.
                                            # At base_speed=0.5, 3 cells is ~6 sim steps of
                                            # lookahead, which is the "5 steps ahead"
                                            # horizon that actually matters for a crossing.
        "projection_falloff": 0.6,         # Severity multiplier per cell along the trail:
                                            # 0.42, 0.25, 0.15. The comet tail fades.

        # ---- Spatial-Temporal memory: SHORT-LIVED, never saturating ----
        # The old scheme ADDED +5.0 (fatal) / +2.0 (warning) against a linear
        # -0.1/step decay. Measured consequences: one fatal splat produced 63
        # cells reading exactly 0.0, the crash cell did not rise above zero for
        # 41 steps, and no adjacent cell passed the Tier-1 gate for 47 steps
        # (~23 cells of travel at base_speed=0.5). Worse, the warning splat fired
        # EVERY step integrity was 0.5, netting +1.9/step, pinning at the 10.0
        # cap in 5 steps and needing 100 steps to clear -- which is exactly how
        # two fleets pausing near adjacent goals turned that goal region into a
        # permanent no-go zone for the rest of the episode.
        "fatal_splat_multiplier": 1.5,     # Down from 5.0. Splats now take max(), not +=,
                                            # so re-splatting an occupied cell is a true
                                            # no-op instead of a ratchet.
        "warning_splat_multiplier": 0.6,   # Down from 2.0. A close pass is a hint, not a
                                            # crater.
        "memory_decay_factor": 0.7,        # MULTIPLICATIVE per step (replaces the linear
                                            # decay_rate=0.1). A 1.5 fatal marker falls
                                            # below memory_floor in 8 steps: affordance at
                                            # the crash cell goes 0.40 -> 0.49 -> 0.58 ->
                                            # 0.66 -> 0.74 -> 0.80 -> 0.85 -> 0.89 -> gone.
                                            # That is the temporal neighbourhood in which
                                            # an encounter is still relevant.
        "memory_floor": 0.05,              # Below this a marker is deleted outright.
        "memory_cap": 2.0,                 # Hard ceiling on any single marker. Down from
                                            # 10.0, and now genuinely load-bearing: under
                                            # A = 1/(1+R) nothing saturates, but this keeps
                                            # a marker from dominating the local field.
        "frozen_obstacle_severity": 1.0,   # Base severity a parked (frozen) fleet
                                            # presents to OTHER fleets passing through --
                                            # full strength at near_goal_radius+ away,
                                            # smoothly discounted to 0 right at a fleet's
                                            # OWN goal (see get_local_affordance's
                                            # docstring for why: without this, a fleet
                                            # whose goal neighbors an already-parked
                                            # peer would see its own destination as a
                                            # permanent no-go zone and never finish).
        "frozen_near_goal_radius": 3.0,    # Distance (graph units) within which a
                                            # frozen peer's obstacle severity ramps down
                                            # toward 0, scaled by how close it is to the
                                            # OBSERVING fleet's own goal.

        # ---- Peer projection: INTENDED PATH, not dead reckoning (2026-09-13) ----
        # The trail a peer stamps is now its greedy descent on its own goal BFS
        # map, not current_pos + direction * k. See density_warehouse.py's
        # projection block for the two defects that replaced: projecting through
        # racks (where the structural mask then discards the stamp entirely, so a
        # turning peer projected NOTHING), and half-cell aliasing under banker's
        # rounding (a peer at 10.5 stamped 12, 12, 14 -- skipping 11 and 13).
        "projection_max_branches": 6,      # Cap on simultaneous branches when the
                                            # descent ties. On a degree-~2.27 corridor
                                            # graph ties are rare, so this is a bound on
                                            # pathological cases, not a tuning knob.
        # ---- Falloff kernel: GRAPH HOPS, not Manhattan (2026-09-13) ----
        # The third and last instance of the Manhattan-vs-graph defect fixed
        # this session. stamp() measured falloff as Manhattan distance inside
        # the cube, so a peer behind a shelf at Manhattan 2 / graph 20 stamped
        # 0.33 repulsion onto the observer as though it were two cells away.
        # _structure_mask does NOT catch this: it zeroes cells the OBSERVER
        # cannot reach, and says nothing about whether the SOURCE can reach them.
        #
        # Corners get MORE accurate, not less: a BFS follows the aisle round a
        # bend, so a peer at graph distance 2 around a corner still stamps at
        # full kernel strength, where Manhattan under-counted it.
        #
        # Verified identical to the old kernel down an unobstructed corridor
        # (test_kernel.py::test_straight_corridor_matches_manhattan_exactly).
        # "manhattan" reproduces published runs bit-for-bit.
        # "affordance" -> 231 dims, mask/(1+R), for the flat encoder.
        # "channels"   -> 462 dims, mask and repulsion packed separately, for
        #                 the gated 3D convolution. Must be set together with
        #                 gnn.encoder = "conv": the encoder needs the two
        #                 channels apart to gate on the mask, and the flat path
        #                 has no idea what to do with 462 dims.
        "output_mode": "affordance",       # "affordance" | "channels"

        "kernel_metric": "graph",          # "graph" | "manhattan"

        # Passed through to FleetNode. See its ray_transform field: "clip25"
        # saturates at 25 cells, "smooth" is d/(d+softness) and never does.
        "ray_transform": "clip25",         # "clip25" | "smooth"
        "ray_softness": 8.0,

        # "intended" | "dead_reckoning". Set BOTH this and kernel_metric to
        # their pre-2026-09-13 values ("dead_reckoning", "manhattan") to
        # reproduce the environment a checkpoint was TRAINED under. A policy
        # trained on one set of density values and evaluated on another is out
        # of distribution in all 231 dims, and the size of that shift grows with
        # peer count -- invisible at 0.24% occupancy, large at 4.18%.
        "projection_mode": "intended",

        "project_stationary": True,        # Project peers whose direction is 0 (idle,
                                            # braked, or just wall-bumped -- which zeroes
                                            # direction in apply_discrete_action).
                                            # BEHAVIOUR DELTA: the old dead-reckoning
                                            # version projected nothing for these. Set
                                            # False to restore that if intent-splatting
                                            # stationary fleets over-repels in congestion.
    },

    # ==========================================
    # 2. WAITING
    # ==========================================
    # Standing still is available (action 0, always structurally valid) and was
    # unconditionally punished: idle_penalty fires on every non-moving step
    # while progress pays movement_reward_multiplier * distance_closed. The
    # policy was taught, at every density, that waiting costs and pushing pays.
    #
    # Measured at 4.18% occupancy on 2026-09-14: 13,041 collapse events, one
    # fleet pair colliding 314 separate times, and 25.7% of Tier-1 "successful"
    # escapes moving the fleet to its own position.
    #
    # A real multi-step hold already existed (_yield_until) but ONLY Tier-3
    # recovery could invoke it, at a single call site. Its own comment records
    # that it replaced a one-step version because that "let a loser immediately
    # retry the same losing move the very next step." The lesson was learned
    # once, for recovery. The policy had no equivalent.
    #
    # WAITING is a peer-VISIBLE state, not just an internal flag. A waiting
    # fleet, a dead fleet and a parked fleet all have direction 0 and are
    # indistinguishable through peer_velocities -- so a fleet could never tell
    # whether the one in front of it would ever move again. ray_hit_waiting
    # reads this set; a VDA 5050 state report would carry it.
    "waiting": {
        "enabled": False,                  # default OFF until the retrain
        "max_wait_steps": 12,              # the idle-penalty exemption LAPSES after
                                            # this many consecutive waiting steps, so
                                            # the penalty resumes and the policy is
                                            # pushed to try something else. Bounded on
                                            # purpose: an exemption can only ever be
                                            # worth zero, where a positive reward for
                                            # inaction could be farmed by parking
                                            # beside a peer forever.
        "mutual_wait_steps": 3,            # shorter cap when the BLOCKER IS ALSO
                                            # WAITING. A mutual standoff does not
                                            # clear by waiting, so the exemption
                                            # lapses fast. Symmetry is broken by
                                            # remaining graph distance: the fleet
                                            # further from its goal keeps the long
                                            # cap and holds, the closer one is
                                            # pushed to move first. Without that,
                                            # both lapse on the same step, both
                                            # move, and re-collide.
        "block_threshold": 2.0,            # graph cells. A mobile peer this close
                                            # means "blocked"; further means the fleet
                                            # is idling in open space and is penalised
                                            # exactly as before.
    },

    # ==========================================
    # 2a. PERCEPTION SCHEDULING
    # ==========================================
    # Both state-build loops in core_warehouse.step() iterate self.nodes, not
    # get_active_nodes(), so a PARKED fleet casts six rays, builds a 231-dim
    # affordance field and gets a Q-forward pass -- twice per step -- and then
    # active_mask zeroes its contribution to the loss. At step 200 of the
    # 2026-09-13 run that was 160 of 200 fleets: 80% of all perception computed
    # and thrown away.
    #
    # Episodically that is the last third of an episode. Under a CONTINUOUS
    # order stream, where most of the fleet idles between bursts, it is the
    # steady state -- which is why this is aimed at continuous mode rather than
    # at the benchmark.
    #
    # NOT A FREE SAVING. _build_adjacency() includes immobile fleets, so the GAT
    # attends over them and an ACTIVE fleet's embedding depends on its idle
    # neighbours' features. Reusing a stale vector changes what active fleets
    # see. Default stays "full"; the other modes are experiments to be measured
    # paired on a frozen checkpoint, not settings to flip.
    "perception": {
        "idle_mode": "full",               # "full" | "cached" | "skip"
        "idle_refresh_every": 10,          # steps between refreshes under "cached".
                                            # An idle fleet does not move, so what
                                            # goes stale is what is happening AROUND
                                            # it -- which is what its neighbours
                                            # attend to. Hence an interval, not a
                                            # permanent cache.
    },

    # ==========================================
    # 2b. PEER PROXIMITY METRIC
    # ==========================================
    # Replaces four independent copies of np.sum(np.abs(pos_a - pos_b)):
    # sf_peer_proximity, braking, the safety reward, and check_integrity. See
    # proximity_warehouse.py.
    #
    # EVERY PUBLISHED FLOWRRA NUMBER WAS MEASURED UNDER "manhattan". The flag is
    # here so those runs can be reproduced after this change, not as a fallback
    # for when "graph" misbehaves.
    "proximity": {
        "metric": "graph",                 # "graph" | "manhattan"

        # Metric for the GAT's ATTENTION GRAPH -- who attends to whom.
        # The fourth and last instance of the phantom-pair defect, and the worst
        # place for it: every other consumer of a bad distance produced a bad
        # number, this one produced a bad TOPOLOGY. Fleets in adjacent aisles,
        # graph-20 apart behind a solid rack, were exchanging messages at every
        # layer. Default stays "manhattan" until the retrain; "graph" is correct.
        # Selecting "graph" widens the proximity index to interaction_radius,
        # which also widens the censoring point of mean_peer_gap.
        "adjacency_metric": "manhattan",   # "graph" | "manhattan"
        "search_radius": 4.0,              # Cells. Must be >= warning_threshold or
                                            # decisions would be truncated; core raises
                                            # it to warning_threshold if set lower.
                                            # Also the censoring point for mean_peer_gap
                                            # (which previously summed float('inf')
                                            # whenever every peer was immobile, poisoning
                                            # the metric for the rest of the run).
        "phantom_audit_every": 25,         # Run the O(N^2) Manhattan-vs-graph comparison
                                            # every N steps and log how many pairs the
                                            # graph metric rejected. 0 disables.
                                            # TWO SPECIES get counted here: pairs behind
                                            # racks, and pairs both mid-edge on parallel
                                            # tracks. The second is unrelated to racks and
                                            # the earlier phantom_pct audit never saw it;
                                            # at base_speed 0.5 a fleet is mid-edge about
                                            # half the time.
    },

    # ==========================================
    # 2c. LESION (inference-time, no retraining)
    # ==========================================
    # Zero named blocks of the state vector on an ALREADY TRAINED checkpoint, to
    # measure how much the policy actually reads them.
    #
    # THIS IS A LESION, NOT AN ABLATION. It answers "does the trained policy use
    # this?" -- NOT "could a policy trained without it have done as well?".
    # A block can lesion badly and still be redundant (the network leaned on it
    # because it was there), or lesion harmlessly and still be necessary during
    # training. Do not report a lesion result as an ablation result.
    #
    # Block names come from FleetNode.state_layout(), plus "density":
    #   self_direction (3), self_displacement (3), ray_distances (6),
    #   peer_velocities (18), peer_displacements (18), goal_gradient (6),
    #   situation_features (6), heading (1, only if use_orientation),
    #   rays_all (42 = the three ray blocks together), density (231)
    "lesion": {
        "zero_blocks": [],                 # e.g. ["rays_all"] or ["goal_gradient"]
    },
    
    # ==========================================
    # 3. GRAPH NEURAL NETWORK (GAT)
    # ==========================================
    "gnn": {
        # "flat" -> Linear(input_dim, hidden). "conv" -> gated 3D convolution
        # over the density volume plus a SEPARATE base encoder and LayerNorm on
        # the fusion. Requires density.output_mode = "channels".
        # See encoder_warehouse.py.
        "encoder": "flat",                 # "flat" | "conv"

        "action_size": 7,                  # {-1, 0, 1} across X, Y, Z + Idle (0)
        "hidden_dim": 128,                 # Neural network width
        "num_layers": 3,                   # Graph Attention Layers
        "num_heads": 4,                    # Multi-head attention count
        "dropout": 0.1,                    # Prevents overfitting
        "learning_rate": 0.0003,
        "interaction_radius": 10.0,        # Manhattan distance within which two fleets are
                                            # linked in the GAT's adjacency matrix. Was
                                            # effectively INFINITE: core built the adjacency as
                                            # np.ones((N, N)), a complete graph, so every fleet
                                            # attended to all 24 others regardless of where they
                                            # were in a 9900-node warehouse. The attention softmax
                                            # then averaged each fleet's representation across the
                                            # whole fleet, cancelling the direction-specific goal
                                            # gradient (one fleet's "+X" against another's "-X").
                                            # That produced a policy picking the best move 42.6% of
                                            # the time and its exact opposite 42.1% -- a 50/50
                                            # oscillation that travelled 328 cells to cover 19.6
                                            # hops. Larger = more coordination but more smearing;
                                            # 10.0 is 5x the warning_threshold, so fleets link well
                                            # before they can conflict.
        "stability_coef": 0.5,             # Weight of the auxiliary integrity-prediction loss
                                            # against the main Q-loss (loss = q_loss + stability_coef*stability_loss).
                                            # Previously only settable via GNNAgent's hardcoded default --
                                            # never actually read from here by any caller.
    },
    
    # ==========================================
    # 4. REINFORCEMENT LEARNING & RECOVERY
    # ==========================================
    "training": {
        "gamma": 0.99,                     # Future reward discount factor.
                                            # RAISED from 0.95. At base_speed 0.5 a
                                            # 40-hop journey is ~80 simulator steps, and
                                            # 0.95^80 = 0.016 -- so the +100
                                            # mission_complete bonus was worth 1.6 at the
                                            # moment a fleet set off, i.e. about what 16
                                            # steps of baseline_reward pays for doing
                                            # nothing at all. The proportional movement
                                            # term was carrying essentially the entire
                                            # learning signal and the goal itself was
                                            # decorative. 0.99^80 = 0.45 puts the bonus at
                                            # 45 instead, a 28x difference in how much the
                                            # goal is worth at journey start.
                                            # This matters MORE with handover missions
                                            # (see "errors" below), which roughly double
                                            # the horizon: at 0.95 a rescuer's eventual
                                            # delivery reward is worth under 1.0 at the
                                            # moment it is asked to divert, so it would
                                            # never learn to divert at all.
        "buffer_capacity": 15000,          # Replay buffer size
        "batch_size": 64,                  # <--- Bumped for smoother gradient averaging  (# Experiences sampled per learn step)
        "total_episodes": 11,             # Total benchmark runs
        "max_steps_per_episode": 780,      # Timeout limit for a single run
    },
    
    "recovery": {
        "history_length": 200,              # How many steps the temporal rewind remembers
        "spatial_safe_threshold": 0.35,    # RECALIBRATED for the new affordance scale.
                                            # Under the old clip(1-R) transform this was a
                                            # SELECTION criterion on a scale where an
                                            # isolated peer already pushed a cell to 0.39
                                            # and a fresh crash pinned everything within 3
                                            # cells to exactly 0.0 -- so no graph neighbour
                                            # of a crashed fleet could ever clear 0.7, which
                                            # is why Tier 1 essentially never fired and
                                            # Tier 2 "succeeds almost every time" in the
                                            # logs. Under A = 1/(1+R) it is now a safety
                                            # FLOOR instead: Tier 1 ranks all neighbours and
                                            # takes the best one, rejecting it only if even
                                            # the best is below this. Reference values on
                                            # the new scale: clear cell 1.0, one peer two
                                            # cells away 0.81, one peer adjacent 0.68,
                                            # adjacent to a fresh crash ~0.40.
        "base_yield_steps": 5,             # How many steps a Tier-3 yield loser must hold
                                            # still, on its FIRST offense this episode.
        "livelock_patience": 40,           # Steps a fleet may go without beating its BEST
                                            # distance-to-goal before its action is handed
                                            # back to the BFS gradient (get_goal_gradient()).
                                            # Guards against the deterministic limit cycle a
                                            # greedy epsilon=0 policy falls into once all
                                            # peers have frozen: the environment goes
                                            # stationary, the policy becomes a deterministic
                                            # map position -> action, and any such map on a
                                            # finite position set must eventually cycle. The
                                            # 400-episode checkpoint deployed to 23/25 with
                                            # zero collisions and zero recovery events, one
                                            # fleet stranded 2 hops out, identical across
                                            # repeated runs. Raise it to intervene less and
                                            # trust the policy more; lower it to escape
                                            # faster. 40 steps is ~20 cells of travel at
                                            # base_speed 0.5.
        "livelock_escape_steps": 20,       # Once triggered, how many CONSECUTIVE steps the
                                            # gradient keeps control. A single-step override
                                            # is useless -- the cycling policy reclaims the
                                            # wheel immediately and the fleet waits another
                                            # full patience window for one more assisted
                                            # step. 20 steps is ~10 cells, enough to walk
                                            # clear of a local cycle in one intervention.
        "pair_escalation_threshold": 2,    # After a specific PAIR of fleets has fatally
                                            # collided this many times in one episode, stop
                                            # treating each collision as an independent
                                            # incident: whichever tier resolves the geometry
                                            # ALSO holds the losing fleet still afterwards.
                                            # Without this, Tier 1 kept reporting "Spatial
                                            # Escape Successful" while the same pair collided
                                            # again 5-40 steps later -- fleets 9 and 24 at
                                            # steps 107, 142, 177, 189, 203 within a single
                                            # episode -- and Tier 3, the mechanism built for
                                            # exactly this, recorded 0 across all 30 episodes
                                            # because Tiers 1/2 always resolved first. Tier 1
                                            # was not failing; it separates a pair by one cell
                                            # and they steer straight back. The missing
                                            # ingredient was TIME, not space.
        "yield_escalation_per_repeat": 5,  # Added per repeat: 1st offence holds for
                                            # base_yield_steps, 2nd for +5, 3rd for +10...
                                            # NOW ACTUALLY READ -- this was accepted by
                                            # WarehouseRecovery.__init__ and never used by
                                            # anything; the old code multiplied
                                            # base_yield_steps by a per-NODE count instead,
                                            # which cannot distinguish five separate conflicts
                                            # from the same conflict five times.  # Added per repeat offense: 1st=5 steps,
                                            # 2nd=10, 3rd=15, etc. -- a fleet that keeps
                                            # ending up back in a fatal collision gets
                                            # progressively longer forced separation
                                            # instead of an instant retry into the same
                                            # mistake.
    },

    # ==========================================
    # 4b. VDA 5050 ERROR STOPS & PACKAGE HANDOVER
    # ==========================================
    # Models an AGV that stops dead mid-mission (VDA 5050 reports an `error`
    # state; the vehicle holds position and the master must respond).
    #
    # THE MECHANIC, in full:
    #   1. A random active fleet errors. It stops moving immediately.
    #   2. After stop_confirm_steps still stopped, it is declared
    #      STOPPED_FUNCTIONING: the master gives up on it, cancels its order,
    #      and its goal returns to the pool.
    #   3. Its cell becomes a PICKUP. Another fleet is dispatched there.
    #   4. On arrival, the rescuer inherits the errored fleet's goal -- the
    #      package carries on to where it was always going.
    #
    # THE KEY DESIGN DECISION -- a stopped fleet leaves the COLLISION system
    # entirely and is represented only as a soft repulsion in the density field.
    # It is not a participant in a conflict, it is furniture. Everything falls
    # out of that one choice:
    #   * no fatal collision when the rescuer arrives at distance 0, so the
    #     rescuer can drive onto the cell and the package "passes through";
    #   * no integrity drop to 0.5 parked next to it, so no spurious WFC
    #     recovery events every time a handover happens;
    #   * no warning_splat accumulating around the pickup, which would slowly
    #     poison the exact cell the next rescuer needs to reach;
    #   * and it cannot pay the warning_zone bonus, so there is no "park next
    #     to a corpse and collect +0.5 every 10 steps" exploit to close.
    # It stays SOFT (density only, graph untouched) rather than being removed
    # from the routing graph, because removing a node invalidates every
    # precomputed BFS goal-distance map and would force an O(|goals| * |E|)
    # rebuild on every error -- precisely the full-graph replan cost the whole
    # timing argument says FLOWRRA does not pay.
    "errors": {
        "enabled": True,                   # Master switch. False = exactly the old
                                            # behaviour, no errors ever injected.
        "prob_per_step": 0.006,            # TRAINING RATE, deliberately unrealistic.
                                            # Was 0.0008, giving 0.28 errors per episode
                                            # -- about one rescue reward per 30,000
                                            # node-steps. loss_rescue sat at 0.001 flat
                                            # for 300 episodes because the head simply
                                            # had almost nothing to fit. 0.006 gives
                                            # ~2.3 per episode, roughly 8x the events.
                                            #
                                            # A real warehouse does not lose 3 of 40
                                            # AGVs every episode. This is a curriculum
                                            # rate: train where the signal is dense
                                            # enough to learn from, then EVALUATE at a
                                            # realistic rate. Reporting resilience
                                            # numbers measured at this rate would
                                            # overstate how often the mechanism is
                                            # exercised in deployment.           # Per-step probability that SOME active fleet
                                            # errors. Over a 780-step episode that is
                                            # ~46% chance of at least one error, so
                                            # roughly half of episodes contain one and
                                            # the network still sees plenty of clean
                                            # ones. Rate, not count, so longer episodes
                                            # get proportionally more.
        "max_per_episode": 5,              # Raised with prob_per_step: at 3 the cap, not
                                            # the rate, would decide the event count and
                                            # the increase above would be wasted.              # Hard cap regardless of the rate above.
        "max_step_fraction": 0.55,         # No errors after this fraction of the episode.
                                            # RESCUE CAPACITY SHRINKS MONOTONICALLY: a
                                            # fleet that reaches its goal freezes and can
                                            # never be recalled, so by late in an episode
                                            # there may be nobody left near enough to
                                            # reach the pickup at all. Observed: an error
                                            # at step 162 left 7 active fleets, the
                                            # nearest 118 hops away, and the episode ended
                                            # with the handover never completed. That is
                                            # not a policy failure the network can learn
                                            # from -- it is an unrecoverable instance, and
                                            # training on it just adds noise. Capping at
                                            # ~55% leaves both enough fleets to choose a
                                            # rescuer from and enough clock to complete
                                            # two legs.
        "min_step": 40,                    # No errors before this step. An error at
                                            # step 2 is indistinguishable from a bad
                                            # spawn and teaches nothing about recovery
                                            # from a mission already underway.
        "min_progress_fraction": 0.15,     # Only error a fleet that is genuinely en
                                            # route. Erroring one that has not left its
                                            # spawn yet makes the pickup and the
                                            # original start the same cell.
        "stop_confirm_steps": 15,          # Steps a fleet must be stopped before the
                                            # master declares it dead and reassigns.
                                            # This gates the MISSION response, not the
                                            # physics: nothing about a dead robot
                                            # changes at step 15, but the fleet
                                            # manager's decision to cancel the order
                                            # legitimately does. Maps to VDA 5050's
                                            # WARNING -> FATAL errorLevel escalation.
        "pickup_dwell_steps": 4,           # Steps a rescuer must remain on the pickup
                                            # cell to "transfer the load" before it
                                            # inherits the goal. Non-zero so the
                                            # handover is an event with a duration
                                            # rather than something that happens in the
                                            # same instant a fleet passes over a cell.
        "pickup_reward": 25.0,             # Leg-1 completion bonus, paid on finishing
                                            # the dwell. WITHOUT THIS leg 1 is entirely
                                            # unrewarded and, at any gamma, effectively
                                            # invisible: the rescuer would be asked to
                                            # abandon a goal it was making measurable
                                            # progress toward in exchange for a distant
                                            # discounted payoff. Deliberately well under
                                            # mission_complete (100) so a pickup is
                                            # never worth more than the delivery it
                                            # exists to enable.
        "recall_retired": True,            # Let a RETIRED fleet come back to service a
                                            # pickup. Rescue capacity otherwise shrinks
                                            # monotonically: every fleet that reaches its
                                            # goal freezes forever, so a late error can
                                            # find nobody in range. Observed: an error at
                                            # step 311 left ONE active fleet, 116 hops
                                            # away, while 31 idle fleets sat parked across
                                            # the warehouse -- several certainly closer.
                                            # A parked AGV that has finished its order is
                                            # an idle vehicle, and no real fleet manager
                                            # leaves it parked while interrupting one
                                            # mid-delivery. Recall costs nothing on the
                                            # network side: GNNAgent.unfreeze_node()
                                            # already exists and just drops the id from
                                            # the frozen set.
        "recall_distance_slack": 1.5,      # Prefer an idle fleet over diverting a working
                                            # one, as long as it is no more than this
                                            # multiple further from the pickup. Above 1.0
                                            # because the two are not equivalent: diverting
                                            # an active fleet delays an order already in
                                            # progress, while an idle one has nothing to
                                            # delay. Set to 1.0 for pure nearest-first, or
                                            # 0 to disable the preference entirely.
        "static_obstacle_severity": 3.0,   # Density stamp for an UNREGISTERED obstacle
                                            # -- a human, debris, a dropped pallet.
                                            # Harder than any fleet (1.0 parked, 1.4
                                            # stopped) and with NO near-goal discount:
                                            # that discount exists to make an obstacle
                                            # transparent to whoever needs to reach it,
                                            # which is right for a corpse being rescued
                                            # and wrong for a person. Repulsion is only
                                            # the soft half; the HARD veto is in
                                            # FleetNode.get_valid_action_mask().
        "stopped_obstacle_severity": 1.4,  # Density stamp for a stopped fleet, vs 1.0
                                            # for a parked one. Higher because a parked
                                            # fleet is where somebody WANTED to be
                                            # (goals are useful places) while a stopped
                                            # one is an unplanned hazard -- but still
                                            # finite, so a rescuer can push through it
                                            # to reach the pickup.
    },

    # ==========================================
    # 4c. ABLATION SWITCHES (research only)
    # ==========================================
    # Each of these puts ONE fixed bug back so its individual contribution can be
    # measured. All False = the corrected system. Never enable for a real run.
    "ablation": {
        # Control arm for the 2026-09-13 ray-origin fix. A fleet can sit half a
        # cell past a dead end: moving +X from x=59 proposes 59.5, whose two
        # bounding nodes are BOTH node 59, so is_structurally_valid allows it.
        # But round(59.5) is 60 under banker's rounding, 60 is not a node, and
        # before the fix every ray read zero -- 42 of 291 state dims blank, while
        # _structure_mask failed OPEN and let the density field report repulsion
        # straight through racks.
        #
        # Set False to reproduce that blindness. Needed because with the fix ON,
        # a fleet at a dead end can still SEE, so comparing collision rates
        # post-fix measures "at a dead end", not "blind". Only the two arms
        # together separate the two.
        "ray_origin_fallback": True,

        "brake_on_immobile": False,    # True restores the bug where affordance braking
                                        # measured distance to EVERY other fleet, including
                                        # parked ones that check_integrity already exempts
                                        # from collisions -- so a fleet could be throttled
                                        # to 0.05 cells/step by a neighbour it could not
                                        # physically crash into.
        "hardcoded_bounds": False,     # True restores the hardcoded (50,50,10)
                                        # normalization instead of per-map extents. On the
                                        # official maps that clipped 81%+ of the Z axis to
                                        # exactly 1.0 on 13 of 16 maps, reducing the
                                        # vertical goal-direction feature to a sign bit.
    },

    # ==========================================
    # 4d. DECOMPOSED REWARD HEADS
    # ==========================================
    # Which reward sources feed which Q-head. Each head does TD learning on its
    # OWN component, and Q_total = sum_k head_weights[k] * Q_k at action
    # selection. See agent_warehouse.py's reward_heads comment for why the
    # previous single summed scalar made the rare signals unlearnable.
    #
    # head_weights are applied ONLY at action selection, never in the loss, so
    # "how much do I care about this" and "how learnable is this" are separate
    # knobs. They were the same knob before: raising the collision penalty to
    # make collisions matter also made that head's TD targets larger and noisier.
    # Start every weight at 1.0 so the first run is attributable, then tune.
    "reward_decomposition": {
        "heads": ["goal", "safety", "integrity", "rescue", "time"],
        "weights": [1.0, 1.0, 1.0, 1.0, 1.0],
        # Which scalar terms land in which head. Purely documentation -- the
        # mapping is implemented in core_warehouse.py's _reward_vector().
        #   goal      : movement progress, baseline, final approach, arrival
        #   safety    : fatal collision, graded warning-zone proximity
        #   integrity : loop integrity level, recovery invocation cost
        #   rescue    : pickup bonus, handover completion
        #   time      : idle penalty, overtime penalty
    },

    # ==========================================
    # 4e. POLICY-INVOKED RECOVERY
    # ==========================================
    # Recovery used to fire only from inside the deadlocked branch -- i.e. AFTER
    # check_integrity returned 0.0, after a collision had already happened. It
    # was structurally incapable of acting first.
    #
    # Now the policy decides. A graph-level head over the pooled node embedding
    # emits Q-values for {none, spatial collapse, temporal collapse}, and the
    # orchestrator executes the choice. Tier 1 (spatial escape) and Tier 3
    # (yield) do not need this -- they are an ordinary move and an ordinary idle,
    # both already in the per-fleet action space. Tier 2 does, because a temporal
    # rewind restores MANY fleets at once and so cannot be a per-fleet decision.
    "recovery_policy": {
        "enabled": True,
        "invocation_cost": -4.0,           # Charged to the integrity head on every
                                            # invocation. WITHOUT A COST THE POLICY
                                            # LEARNS TO CRASH AND UNDO: a rewind erases
                                            # a collision, so if it were free the
                                            # optimal strategy would be to ignore
                                            # safety entirely and rewind out of every
                                            # consequence. A rewind also loses travel
                                            # progress, which the goal head penalises
                                            # on its own -- the two heads pull against
                                            # each other and the policy has to resolve
                                            # the tension, which is the decision we
                                            # actually want it to learn.
        "exploration_scale": 0.25,         # Recovery exploration as a FRACTION of the
                                            # fleet-action epsilon.
                                            #
                                            # choose_recovery() used the same epsilon as
                                            # the per-fleet moves. At eps 0.95 that is a
                                            # random 3-way draw EVERY step, so ~2/3 of
                                            # 780 steps became invocations -- measured at
                                            # 400 wasted per episode.
                                            #
                                            # That is not harmless noise. 400 x -4.0 x 40
                                            # fleets is -64,000 landing on the integrity
                                            # head against maybe +2,000 from successful
                                            # preemptions: the useful signal outnumbered
                                            # 30 to 1, which is very likely why
                                            # loss_integrity spikes to 1.8 during the
                                            # exploration phase and only converges after
                                            # epsilon decays.
                                            #
                                            # A once-per-step 3-way global choice does
                                            # not need the exploration budget of a
                                            # per-fleet 7-way movement choice -- it gets
                                            # 780 samples an episode either way. 0.25
                                            # keeps it exploring without drowning its own
                                            # reward signal. Set 0.0 for a purely greedy
                                            # recovery head.
        "forced_fallback_steps": 6,        # If integrity sits at 0.0 for this many
                                            # consecutive steps and the policy has NOT
                                            # invoked anything, the orchestrator forces
                                            # a collapse. Without this an untrained
                                            # policy never invokes and every early
                                            # episode deadlocks permanently, so there
                                            # would be no usable experience to learn
                                            # from. Forced invocations are logged
                                            # SEPARATELY: forced count falling toward
                                            # zero is the measure of whether the policy
                                            # has learned to call recovery itself.
        "preemptive_bonus": 9.0,           # Paid when a PREEMPTIVE separation actually
                                            # works -- the at-risk fleets it moved are
                                            # clear of the warning band on the next
                                            # step, with no collision.
                                            #
                                            # WITHOUT THIS PREEMPTION IS UNREWARDABLE.
                                            # resolution_bonus below is gated on
                                            # n_dead_before > 0, and a preemptive
                                            # invocation has nothing deadlocked yet BY
                                            # DEFINITION -- so the one behaviour we
                                            # asked for was, in the reward, identical
                                            # to invoking on an empty set: pure cost.
                                            # Measured over 300 episodes: the head
                                            # correctly learned to stop wasting
                                            # invocations (335 -> 12) and, having no
                                            # way to distinguish them, stopped
                                            # intervening early too. Collisions rose
                                            # 0.02 -> 0.54 per episode and forced
                                            # recoveries 0.00 -> 0.50 as a direct
                                            # result.
                                            #
                                            # Larger than resolution_bonus on purpose:
                                            # preventing a collision is worth more than
                                            # cleaning one up.
        "resolution_bonus": 6.0,           # Paid to the integrity head when an
                                            # invocation actually clears the deadlock
                                            # it was called for. Distinguishes a
                                            # well-timed collapse from a panicked one.
    },

    # ==========================================
    # 5. REWARD SHAPING
    # ==========================================
    # Every one of these used to be a bare numeric literal buried inside
    # core_warehouse.py's step() method -- not configurable from here at all.
    #
    # REWORKED (2026-08-25) around the philosophy from the FLOWRRA-GNN swarm
    # version (FLOWRRA_JOURNEY/Version_0__0_3), which never had a "frozen
    # fleet" problem: an unconditional baseline reward every step, movement
    # rewarded PROPORTIONALLY to how much it actually helped rather than a
    # flat bonus/penalty regardless of magnitude, and a much softer collision
    # penalty (their collapse_penalty is -3.0; ours was -15.0 even after the
    # first reduction from -50.0). The swarm task has no fixed per-agent goals
    # though, so mission_complete/final_approach_bonus/the new overtime
    # pressure below don't have a swarm analogue -- those are this task's own
    # goal-directed additions on top of the borrowed philosophy.
    "rewards": {
        "mission_complete": 100.0,         # One-time bonus for reaching the goal
        "baseline_reward": 0.1,            # Unconditional, every active fleet, every
                                            # step -- matches the swarm version exactly.
                                            # Makes plain idling only mildly positive
                                            # rather than zero, so there's no cliff to
                                            # fear -- the incentive to actually move
                                            # comes from movement paying MUCH more, not
                                            # from idling being punishing.
        "movement_reward_multiplier": 3.0, # Replaces the old flat moving_closer/
                                            # wandering_away with a PROPORTIONAL reward:
                                            # (old_dist - new_dist) * this. A perfect
                                            # full-speed step toward goal nets ~1.5; a
                                            # half-hearted/braked step earns proportionally
                                            # less instead of the same flat +1.0 a
                                            # full-speed step got. A forced detour that's
                                            # the only available progress now earns
                                            # something close to zero instead of a flat
                                            # -1.5, rather than being punished as hard as
                                            # a genuinely wrong move.
        "idle_penalty": -0.5,              # Down from -2.0. Applies ONLY when current_pos
                                            # genuinely didn't change at all this step
                                            # (idle action, or an invalid/blocked move) --
                                            # NOT "moved but zero net distance progress",
                                            # which the proportional term above already
                                            # handles correctly on its own (nets to ~0).
        "fatal_collision": -50.0,          # RAISED from -3.0. Charged to the two
                                            # colliding fleets, so a collision cost -6
                                            # total against a completed delivery worth
                                            # +100 -- roughly 6% of one delivery. In a
                                            # warehouse a collision is an incident:
                                            # stopped vehicles, manual intervention,
                                            # possible damage. It is not 6% of a
                                            # delivery.
                                            #
                                            # The policy was optimising this correctly.
                                            # Measured over 100 episodes it traded 8
                                            # preventive interventions per episode for
                                            # 0.3 collisions and got BETTER completion
                                            # doing it, because intervention cost
                                            # -4.0 x N (every fleet) while a collision
                                            # cost -6. At -50 a collision costs -100,
                                            # equal to losing a whole delivery, which
                                            # makes prevention worth paying for.           # Down from -15.0 (originally -50.0), matching
                                            # the swarm version's collapse_penalty. See the
                                            # breakeven-risk-tolerance math from earlier in
                                            # this project: this raises the perceived-risk
                                            # threshold at which idling starts looking
                                            # better than attempting a move.
        "warning_zone": -1.2,              # PER-STEP PENALTY for sitting in another
                                            # fleet's warning band (collision_threshold
                                            # < d <= warning_threshold), scaled by how
                                            # close to the collision boundary you are:
                                            # ~0 at d = warning_threshold, full value
                                            # just outside a crash.
                                            #
                                            # SIGN FLIP. This used to be +0.5 -- a
                                            # BONUS for entering the exact band that
                                            # precedes a collision. Nothing in the
                                            # reward discouraged approach and something
                                            # actively paid for it, which is the
                                            # mechanical reason Tier 1 only ever fired
                                            # AFTER impact: the policy had no gradient
                                            # pointing away from danger until the
                                            # danger had already happened.
                                            #
                                            # A graded penalty gives the safety head a
                                            # signal one step BEFORE the fatal one, so
                                            # Q-values for closing actions fall while
                                            # there is still room to act. That is
                                            # preemptive avoidance, learned rather than
                                            # overridden.
                                            #
                                            # No cooldown any more. The cooldown existed
                                            # only to stop two fleets farming a repeated
                                            # BONUS by loitering; a penalty cannot be
                                            # farmed, and every step spent near a crash
                                            # is genuinely a step of risk.               # Bonus for a close-encounter with another fleet
        "final_approach_bonus": 3.5,       # EXTRA reward (on top of the proportional
                                            # movement term) for genuine progress while in
                                            # the warning zone during final approach (see
                                            # warehouse.final_approach_threshold) --
                                            # reinforces pushing through instead of being
                                            # cautious right when a fleet is nearly home.
        "warning_zone_cooldown": 10,       # Steps a fleet must wait before earning warning_zone
                                            # again. THE FIX: previously this fired every single
                                            # step a fleet remained in another fleet's warning
                                            # zone, uncapped -- so two fleets that just sat near
                                            # each other collected +3.0/step indefinitely, often
                                            # outearning the reward for actually making
                                            # progress. Now it's a one-time bonus per distinct
                                            # encounter: earned once, then locked out for this
                                            # many steps regardless of whether the fleet stays in
                                            # or re-enters the zone, before it can be earned again.

        # Overtime pressure: escalating penalty for any fleet still not home once
        # the episode runs past overtime_threshold_steps, ramping linearly from 0
        # at that step up to overtime_max_penalty at max_steps_per_episode. Applies
        # every step until the fleet either finishes (no overtime penalty on the
        # exact step it completes -- see step()) or the episode ends. This is the
        # "escalating penalty, not abandon" interpretation discussed earlier: a
        # smooth urgency gradient in the final stretch rather than a sudden cliff.
        "overtime_threshold_steps": 750,
        "overtime_max_penalty": -10.0,
    },
}