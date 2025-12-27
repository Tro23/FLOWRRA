"""
core.py

Enhanced FLOWRRA Orchestrator with:
- Loop structure and spring forces
- Static and moving obstacles
- Loop break detection and recovery
- Enhanced metrics tracking
- Proper collision response with WFC micro-collapse
"""

from typing import Any, Dict, List

import numpy as np

from config import CONFIG

from .agent import GNNAgent, build_adjacency_matrix
from .density import DensityFunctionEstimatorND
from .exploration import ExplorationMap
from .loop import LoopStructure
from .node import NodePositionND
from .obstacles import ObstacleManager
from .recovery import Wave_Function_Collapse


class FLOWRRA_Orchestrator:
    """
    Enhanced FLOWRRA orchestrator with loop structure, obstacles,
    and Gaussian exploration schedule for organic swarm expansion.
    """

    def __init__(self, mode="training"):
        self.cfg = CONFIG
        self.dims = self.cfg["spatial"]["dimensions"]
        self.mode = mode  # 'training' or 'deployment'

        # Components
        self.map = ExplorationMap(
            self.cfg["spatial"]["world_bounds"],
            self.cfg["exploration"]["map_resolution"],
        )

        self.density = DensityFunctionEstimatorND(
            dimensions=self.dims,
            local_grid_size=self.cfg["repulsion"]["local_grid_size"],
            global_grid_shape=self.cfg["repulsion"]["global_grid_shape"],
        )

        self.wfc = Wave_Function_Collapse(
            history_length=self.cfg["wfc"]["history_length"],
            tail_length=self.cfg["wfc"]["tail_length"],
            collapse_threshold=self.cfg["wfc"]["collapse_threshold"],
            tau=self.cfg["wfc"]["tau"],
            # NEW: Pass grid parameters for spatial collapse
            global_grid_shape=self.cfg["repulsion"]["global_grid_shape"],
            local_grid_size=self.cfg["repulsion"]["local_grid_size"],
        )

        # NEW: Obstacle Manager
        self.obstacle_manager = ObstacleManager(dimensions=self.dims)
        self._initialize_obstacles()

        # NEW: Loop Structure initialized BEFORE nodes (so we can use its params)
        self.loop = LoopStructure(
            ideal_distance=self.cfg["loop"]["ideal_distance"],
            stiffness=self.cfg["loop"]["stiffness"],
            break_threshold=self.cfg["loop"]["break_threshold"],
            dimensions=self.dims,
        )

        # FIX 1: Initialize Nodes at Equilibrium
        self.nodes = self._initialize_nodes_in_loop()

        # Initialize topology
        self.loop.initialize_ring_topology(len(self.nodes))

        # FIX 2: Run Physics Warmup to stabilize energy before AI starts
        self._physics_warmup(steps=100)

        # Calculate GNN input dimension
        input_dim = self._calculate_input_dim()

        # GNN Setup
        action_size = 4 if self.dims == 2 else 6

        self.gnn = GNNAgent(
            node_feature_dim=input_dim,
            edge_feature_dim=0,
            action_size=action_size,
            hidden_dim=self.cfg["gnn"]["hidden_dim"],
            lr=self.cfg["gnn"].get("lr", 0.0003),
            gamma=self.cfg["gnn"].get("gamma", 0.95),
            stability_coef=self.cfg["gnn"].get("stability_coef", 0.5),
        )

        # State tracking
        self.history = []
        self.current_episode = 0
        self.step_count = 0
        self.last_state = None

        # Performance tracking
        self.total_reward = 0.0
        self.episode_rewards = []
        self.training_losses = []  # Track GNN training loss

        # NEW: Enhanced metrics
        self.metrics_history = []
        self.loop_break_events = []
        self.wfc_trigger_events = []
        self.collision_events = []  # Track collision-recovery events
        self.total_reconnections = 0  # Track healing behavior

    def _initialize_obstacles(self):
        """Initialize static and moving obstacles from config."""
        # Static obstacles from config
        for obs_cfg in self.cfg.get("obstacles", []):
            if len(obs_cfg) == 3:  # (x, y, radius)
                x, y, r = obs_cfg

                pos = np.array([x, y])

                self.obstacle_manager.add_static_obstacle(
                    pos, r / self.cfg["spatial"]["world_bounds"][0]
                )

        # Optional: Add moving obstacles
        if self.cfg.get("moving_obstacles"):
            for mov_obs in self.cfg["moving_obstacles"]:
                x, y, r, vx, vy = mov_obs
                pos = np.array([x, y])
                vel = np.array([vx, vy])  # Scale velocity
                self.obstacle_manager.add_moving_obstacle(
                    pos, r / self.cfg["spatial"]["world_bounds"][0], vel
                )

        print(
            f"[Obstacles] Initialized {len(self.obstacle_manager.obstacles)} obstacles"
        )

    def _initialize_nodes_in_loop(self) -> List[NodePositionND]:
        """
        Initialize nodes at the 'Relaxed State' radius.
        Calculates the exact radius needed for the spring loop to be at rest.
        """
        # FIX: Robustly determine node count for both Phase 1 and Federated modes
        num_nodes = self.cfg["node"].get("num_nodes")
        if num_nodes is None:
            num_nodes = self.cfg["node"].get("num_nodes_per_holon")

        if num_nodes is None:
            raise ValueError(
                "[Core] Configuration Error: Neither 'num_nodes' nor 'num_nodes_per_holon' found in CONFIG['node']"
            )

        ideal_dist = self.cfg["loop"]["ideal_distance"]

        # MATH FIX: Circumference = N * ideal_dist = 2 * pi * r
        # Therefore r = (N * ideal_dist) / (2 * pi)
        equilibrium_radius = (num_nodes * ideal_dist) / (2 * np.pi)

        center = np.array([0.5, 0.5] if self.dims == 2 else [0.5, 0.5, 0.5])

        # Safety clamp to ensure we don't spawn inside walls if ideal_dist is huge
        equilibrium_radius = min(equilibrium_radius, 0.3)

        nodes = []
        for i in range(num_nodes):
            angle = (i / num_nodes) * 2 * np.pi

            # Position on the equilibrium ring
            if self.dims == 2:
                offset = np.array([np.cos(angle), np.sin(angle)]) * equilibrium_radius
            else:
                offset = (
                    np.array([np.cos(angle), np.sin(angle), 0]) * equilibrium_radius
                )

            # Add TINY noise just to break perfect symmetry (prevents numerical singularity)
            noise = np.random.normal(0, 0.001, self.dims)

            pos = center + offset + noise

            node = NodePositionND(i, pos, self.dims)
            node.sensor_range = self.cfg["exploration"]["sensor_range"]
            node.move_speed = self.cfg["node"]["move_speed"]

            # Align orientation outward initially (optional, helps exploration)
            # node.azimuth_idx = ... (can leave default)

            nodes.append(node)

        print(
            f"[Nodes] Initialized {num_nodes} nodes at Equilibrium Radius: {equilibrium_radius:.4f} (World units: {equilibrium_radius:.2f})"
        )
        return nodes

    def _physics_warmup(self, steps=80):
        """
        Runs the physics engine without the GNN to let springs settle.
        Drains kinetic energy (damping) to reach a stable state.
        """
        print(f"[System] Running {steps} steps of physics warmup...")

        for _ in range(steps):
            # 1. Calculate Forces
            forces = self.loop.calculate_spring_forces(self.nodes)

            # 2. Apply Forces with Heavy Damping (simulating friction)
            for node in self.nodes:
                if node.id in forces:
                    # Apply force
                    node.pos = node.pos + forces[node.id] * 0.05

            # 3. Enforce Constraints (Reconnect immediately if broken during warmup)
            # We force repair here because we want a valid loop to start
            self.loop.repair_all_connections()

            # FIX: Record these stable states to WFC!
            # We simulate a "perfect" state: Coherence=1.0, Integrity=1.0
            if _ > steps - 50:  # Only record the last 50 stable steps
                # We create a dummy "high coherence" entry so WFC has a safe place to return to
                self.wfc.assess_loop_coherence(
                    coherence=1.0, nodes=self.nodes, loop_integrity=1.0
                )

        print(
            f"[System] Warmup complete. Loop is Complete.\n WFC Memory Seeded with {len(self.wfc.history)} stable states."
        )

    def _get_current_node_features(self):
        node_features = []
        for node in self.nodes:
            # (Copy logic from step 6 in your code)
            node_detections = node.sense_nodes(self.nodes)
            obstacle_detections = node.sense_obstacles(
                self.obstacle_manager.get_all_states()
            )
            repulsion_sources = node_detections + obstacle_detections
            local_grid = self.density.get_affordance_potential_for_node(
                node.pos, repulsion_sources
            )
            feats = node.get_state_vector(
                local_grid, node_detections, obstacle_detections
            )

            node_features.append(feats)

        return np.array(node_features, dtype=np.float32)

    def detect_and_unstick_nodes(self):
        """
        Detect nodes that haven't moved in a while and give them a kick.
        Add this method around line 150 in core.py
        """
        if not hasattr(self, "node_position_history"):
            self.node_position_history = {n.id: [] for n in self.nodes}
            self.node_stuck_counters = {n.id: 0 for n in self.nodes}

        for node in self.nodes:
            # Track last 10 positions
            history = self.node_position_history[node.id]
            history.append(node.pos.copy())
            if len(history) > 10:
                history.pop(0)

            # Check if stuck (moved less than 0.01 units in last 10 steps)
            if len(history) >= 10:
                total_movement = sum(
                    np.linalg.norm(history[i] - history[i - 1])
                    for i in range(1, len(history))
                )

                if total_movement < 0.01:  # Barely moved
                    self.node_stuck_counters[node.id] += 1

                    if self.node_stuck_counters[node.id] > 5:
                        # UNSTICK: Apply strong random displacement
                        kick_magnitude = 0.08
                        random_direction = np.random.uniform(-1, 1, self.dims)
                        random_direction = random_direction / np.linalg.norm(
                            random_direction
                        )

                        node.pos = node.pos + random_direction * kick_magnitude

                        print(f"[Unstick] Node {node.id} was stuck - applied kick")
                        self.node_stuck_counters[node.id] = 0
                        history.clear()
                else:
                    self.node_stuck_counters[node.id] = 0

    def _calculate_input_dim(self) -> int:
        """Calculate the dimension of the GNN input vector."""
        dummy_rep = np.zeros(np.prod(self.cfg["repulsion"]["local_grid_size"]))
        dummy_detections = []
        base_dim = len(
            self.nodes[0].get_state_vector(
                dummy_rep, dummy_detections, dummy_detections
            )
        )
        return base_dim  # for the pre-emptive features

    def calculate_coherence(self, rewards: np.ndarray, loop_integrity: float) -> float:
        """
        Revised Coherence:
        A single collision shouldn't zero out the coherence of the whole swarm immediately,
        unless the loop is also broken.
        """
        # 1. Structural Integrity is the baseline
        base_coherence = loop_integrity

        # 2. Collision Penalty
        # Count how many nodes are experiencing deep negative rewards (collisions)
        # r_collision is 25.0, so look for values < -10
        colliding_nodes = np.sum(rewards < -10.0)

        collision_penalty = 0.0
        if colliding_nodes > 0:
            # Penalize proportional to how many nodes are crashing
            # If 1 node crashes: -0.15. If all 10 crash: -1.0
            collision_penalty = (colliding_nodes / len(self.nodes)) * 1.5

        # 3. Calculate final
        coherence = base_coherence - collision_penalty

        # Smooth clamping
        return np.clip(coherence, 0.0, 1.0)

    def step(self, episode_step: int, total_episodes: int = 8000) -> float:
        """
        Execute one simulation step with loop dynamics and Gaussian exploration.
        """
        # --- 1. Update Obstacles ---
        self.obstacle_manager.update_all()

        # After obstacle updates, before loop breaks check stuck nodes
        self.detect_and_unstick_nodes()

        # --- 2. Check Loop Breaks ---
        broken = self.loop.check_breaks(
            self.nodes, self.obstacle_manager, self.step_count
        )
        if broken:
            self.loop_break_events.append(
                {
                    "timestep": self.step_count,
                    "broken_connections": [(c.node_a_id, c.node_b_id) for c in broken],
                }
            )

        # --- 3. Attempt Reconnections ---
        reconnected = self.loop.attempt_reconnection(
            self.nodes, self.obstacle_manager, self.step_count
        )

        # Log successful reconnections
        if reconnected:
            self.total_reconnections += len(reconnected)
            self.loop_break_events.append(
                {
                    "timestep": self.step_count,
                    "event_type": "reconnection",
                    "reconnected_connections": [
                        (c.node_a_id, c.node_b_id) for c in reconnected
                    ],
                }
            )
            print(
                f"[Loop] Step {self.step_count}: Reconnected {len(reconnected)} connection(s)"
            )

        # Calculate reconnection bonus for reward later
        reconnection_bonus = len(reconnected) * self.cfg["rewards"].get(
            "r_reconnection", 0.5
        )

        # --- 4. Calculate Spring Forces ---
        spring_forces = self.loop.calculate_spring_forces(self.nodes)

        # --- 5. Density Update ---
        self.density.update_from_sensor_data(
            all_nodes=self.nodes,
            all_obstacle_states=self.obstacle_manager.get_all_states(),
        )

        # --- 6. Build State Representations & Store Local Grids ---
        node_features = []
        local_grids = []  # NEW: Store grids for WFC spatial collapse
        adj_mat = build_adjacency_matrix(
            self.nodes, self.cfg["exploration"]["sensor_range"]
        )

        for node in self.nodes:
            # Sense environment
            node_detections = node.sense_nodes(self.nodes)
            obstacle_detections = node.sense_obstacles(
                self.obstacle_manager.get_all_states()
            )

            # Get affordance field
            repulsion_sources = node_detections + obstacle_detections
            local_grid = self.density.get_affordance_potential_for_node(
                node_pos=node.pos, repulsion_sources=repulsion_sources
            )

            # Store grid for spatial collapse
            local_grids.append(local_grid)

            # Construct state vector
            feats = node.get_state_vector(
                local_repulsion_grid=local_grid,
                node_detections=node_detections,
                obstacle_detections=obstacle_detections,
            )

            node_features.append(feats)

        node_features_array = np.array(node_features, dtype=np.float32)

        # --- 7. GNN Action Selection ---
        # GNN handles its own epsilon schedule internally
        actions = self.gnn.choose_actions(
            node_features=node_features_array,
            adj_matrix=adj_mat,
            episode_number=self.current_episode,
            total_episodes=total_episodes,
        )

        # Get current epsilon for tracking
        current_epsilon = self.gnn.epsilon_gaussian(
            self.current_episode, total_episodes
        )

        # --- 8. Physics & Reward Calculation with Collision Response ---
        step_rewards = []

        for i, node in enumerate(self.nodes):
            action_id = actions[i]
            old_pos = node.pos.copy()

            # Apply action + spring force
            node.apply_directional_action(action_id, dt=1.0)

            # Add spring force influence
            # Update: WITH obstacle-aware damping
            if node.id in spring_forces:
                force = spring_forces[node.id]

                # Check if node is near obstacles
                near_obstacle = False
                for obs in self.obstacle_manager.obstacles:
                    dist = np.linalg.norm(node.pos - obs.pos) - obs.radius
                    if dist < 0.08:  # Within danger zone
                        near_obstacle = True
                        break

                # Reduce spring force near obstacles to allow escape
                force_scale = 0.05 if near_obstacle else 0.1
                node.pos = node.pos + force * force_scale  # Scale Force Influence

            # NEW: Collision Response with Wave Function Collapse
            collides, obs_ids = self.obstacle_manager.check_collision(
                node.pos, safety_margin=0.05
            )

            if collides:
                # FLOWRRA Point 8: Micro-level WFC - collapse to nearby valid state
                # Instead of hard snapback, intelligently search for valid positions
                # IMPROVED: Momentum-based escape with larger search radius
                best_candidate = old_pos.copy()
                best_distance_from_obstacle = 0.0  # Track how far we get from obstacles

                # Calculate attempted movement direction (momentum)
                attempted_dir = node.pos - old_pos
                attempted_mag = np.linalg.norm(attempted_dir)

                if attempted_mag > 0.0001:
                    attempted_dir = attempted_dir / attempted_mag
                else:
                    # If no movement, use a random direction
                    angle = np.random.uniform(0, 2 * np.pi)
                    attempted_dir = np.array([np.cos(angle), np.sin(angle)])

                max_attempts = 16  # Increased from 8
                search_radius = 0.04  # DOUBLED from 0.02

                for attempt in range(max_attempts):
                    angle = (attempt / max_attempts) * 2 * np.pi

                    # Mix radial search with momentum direction
                    radial_offset = (
                        np.array([np.cos(angle), np.sin(angle)]) * search_radius
                    )
                    momentum_offset = attempted_dir * search_radius * 0.5

                    # Combine: 70% radial exploration, 30% momentum direction
                    offset = radial_offset * 0.7 + momentum_offset * 0.3

                    candidate = old_pos + offset

                    # Test candidate
                    coll_check, coll_obs_ids = self.obstacle_manager.check_collision(
                        candidate,
                        safety_margin=0.03,  # Reduced safety margin slightly
                    )

                    if not coll_check:
                        # Calculate distance to nearest obstacle to pick BEST candidate
                        min_dist = float("inf")
                        for obs in self.obstacle_manager.obstacles:
                            dist = np.linalg.norm(candidate - obs.pos) - obs.radius
                            min_dist = min(min_dist, dist)

                        if min_dist > best_distance_from_obstacle:
                            best_candidate = candidate
                            best_distance_from_obstacle = min_dist

                # Collapse to best candidate found
                node.pos = best_candidate

                # If we're STILL stuck (best_distance is very small), add extra random kick
                if best_distance_from_obstacle < 0.09:
                    # Emergency escape: large random displacement
                    random_kick = np.random.uniform(-0.09, 0.09, self.dims)
                    node.pos = node.pos + random_kick
                    print(f"[Emergency] Node {node.id} received random escape kick")

                # Heavy collision penalty - this teaches avoidance
                r_coll = -self.cfg["rewards"]["r_collision"]

                # Log collision event for WFC awareness
                self.collision_events.append(
                    {
                        "timestep": self.step_count,
                        "node_id": node.id,
                        "attempted_pos": old_pos.copy(),
                        "recovered_pos": best_candidate.copy(),
                        "obstacle_ids": obs_ids,
                    }
                )

                # CRITICAL: Add repulsion scar at collision site
                # Estimate velocity from movement attempt
                attempted_velocity = node.pos - old_pos  # The failed move direction
                impact_speed = np.linalg.norm(attempted_velocity)

                # This teaches "don't go here" via density field learning
                if impact_speed > 0.002:
                    collision_severity = 0.5  # Full severity for direct collision

                    # Splat repulsion at collision point with forward projection
                    self.density.splat_collision_event(
                        position=node.pos.copy(),
                        velocity=attempted_velocity,
                        severity=collision_severity,
                        node_id=node.id,
                    )
            else:
                r_coll = 0.0

            # Calculate movement (from old_pos to final pos after collision handling)
            move_mag = np.linalg.norm(node.pos - old_pos)

            # Movement reward - reward for valid movement
            r_flow = self.cfg["rewards"]["r_flow"] * move_mag if not collides else 0.0

            # Idle penalty
            r_idle = -self.cfg["rewards"]["r_idle"] if move_mag < 0.001 else 0.0

            # Loop integrity reward/penalty
            loop_integrity = self.loop.calculate_integrity()
            r_loop = self.cfg["rewards"]["r_loop_integrity"] * loop_integrity

            # Penalty for being in broken loop
            if not self.loop.is_loop_coherent(min_integrity=0.7):
                r_loop -= self.cfg["rewards"]["r_collapse_penalty"]

            # Total reward
            reward = r_flow + r_coll + r_idle + r_loop
            step_rewards.append(reward)

        step_rewards_array = np.array(step_rewards, dtype=np.float32)

        # Update map and calculate exploration reward
        """new_coverage = self.map.update(self.nodes)
        r_explore = new_coverage * self.cfg["rewards"]["r_explore"]
        step_rewards_array += r_explore"""
        # Updating Patrol mode: Once coverage > 85% maintain loop and move around.
        # 1. Update Map & Calculate Standard Exploration
        new_coverage_diff = self.map.update(self.nodes)
        loop_integrity = self.loop.calculate_integrity()
        r_loop = self.cfg["rewards"]["r_loop_integrity"] * loop_integrity
        r_explore = new_coverage_diff * self.cfg["rewards"]["r_explore"]

        # 2. Add to rewards array
        step_rewards_array += r_explore

        # 3. THE "STRICT PATROL" PROTOCOL

        # Only activate if we are in "End Game" (> 2/3 rd of total_episodes)
        if episode_step > (total_episodes * 2 / 3):
            # We already calculated 'loop_integrity' a few lines above in the standard physics block
            # Logic: We need to replace the missing "Exploration Dopamine" (usually ~15.0)
            # to keep the agents interested.

            if loop_integrity >= 0.70:
                # VICTORY LAP: High reward for perfect formation
                # This matches the intensity of finding new chunks
                # Only give the full 15.0 if they are actually MOVING while maintaining integrity
                avg_speed = np.mean(
                    [np.linalg.norm(node.velocity()) for node in self.nodes]
                )

                if avg_speed > 0.009:
                    r_patrol = 15.0
                else:
                    r_patrol = 2.0  # Lower reward for static "parking lot" behavior

                # Distribute the "Exploration Dopamine" replacement
                step_rewards_array += r_patrol
                step_rewards_array += r_explore * 2.5

            elif loop_integrity >= 0.50:
                # MEDIOCRE: Small maintenance reward
                r_patrol = 4.0
                step_rewards_array += r_patrol
                step_rewards_array += r_explore
            else:
                # SLACKING: Penalty for losing focus during patrol
                # This prevents the "boredom chaos"
                r_patrol = -5.0
                step_rewards_array += r_patrol

        # Add reconnection bonus - distributed across all nodes
        # Reconnecting the loop is a collective achievement!
        if reconnection_bonus > 0:
            step_rewards_array += reconnection_bonus

        # --- 9. Store Experience (Training Mode Only) ---
        training_loss = None  # Initialize loss tracking

        if self.mode == "training":
            if self.last_state is not None:
                last_features, last_adj, last_actions = self.last_state

                self.gnn.memory.push(
                    node_features=last_features,
                    adj_matrix=last_adj,
                    actions=last_actions,
                    rewards=np.clip(step_rewards_array, -50.0, 50.0) / 10.0,
                    next_node_features=node_features_array,
                    next_adj_matrix=adj_mat,
                    done=False,
                    integrity=float(loop_integrity),
                )

                if len(self.gnn.memory) >= self.gnn.batch_size:
                    training_loss = self.gnn.learn()
                    self.training_losses.append(
                        {
                            "timestep": self.step_count,
                            "episode": self.current_episode,
                            "loss": float(training_loss),
                        }
                    )

            self.last_state = (node_features_array, adj_mat, actions)

            # Update target network periodically
            if self.step_count % 100 == 0:
                self.gnn.update_target_network()

        # --- 10. Enhanced WFC with Loop Awareness & Spatial Collapse ---
        loop_integrity = self.loop.calculate_integrity()
        current_coherence = self.calculate_coherence(step_rewards_array, loop_integrity)

        self.wfc.assess_loop_coherence(current_coherence, self.nodes, loop_integrity)

        # NEW: Track recovery mode for differential rewards
        recovery_mode = None
        wfc_recovery_bonus = 0.0

        # Trigger recovery if needed
        if self.wfc.needs_recovery() or not self.loop.is_loop_coherent(
            min_integrity=0.5
        ):
            print(
                f"\n[!] Step {episode_step}: System unstable "
                f"(coherence={current_coherence:.2f}, integrity={loop_integrity:.2f})"
            )
            print("[WFC] Triggering recovery...")

            # CRITICAL: Store pre-collapse positions for retrocausal repulsion
            failed_positions = [node.pos.copy() for node in self.nodes]
            failed_velocities = [node.velocity() for node in self.nodes]

            # Call WFC with local grids for spatial collapse attempt
            recovery_info = self.wfc.collapse_and_reinitialize(
                nodes=self.nodes,
                local_grids=local_grids,
                ideal_dist=self.cfg["loop"]["ideal_distance"],
                config=self.cfg,  # NEW: Pass full config for tuning
            )

            # Repair all loop connections after recovery
            self.loop.repair_all_connections()

            print(f"[WFC] Recovery complete: {recovery_info}")

            # NEW: Determine recovery mode and calculate differential reward
            recovery_mode = recovery_info.get("reinit_from")

            if recovery_mode == "spatial_affordance":
                # SPATIAL RECOVERY: Big reward for forward-looking solution!
                wfc_recovery_bonus = self.cfg["rewards"]["r_reconnection_spatial"]
                print(
                    f"[WFC Reward] +{wfc_recovery_bonus:.1f} for SPATIAL recovery! 🎉"
                )

            elif recovery_mode == "coherent_tail":
                # TEMPORAL RECOVERY: Small reward for backward-looking fallback
                wfc_recovery_bonus = self.cfg["rewards"]["r_reconnection_temporal"]
                print(
                    f"[WFC Reward] +{wfc_recovery_bonus:.1f} for temporal recovery (fallback)"
                )

            else:
                # RANDOM JITTER: No reward (failure mode)
                wfc_recovery_bonus = 0.0
                print(f"[WFC Reward] No bonus - random jitter used")

            # RETROCAUSAL REPULSION SPLATTING
            collapse_severity = 0.0

            # Only splat if TEMPORAL recovery (spatial doesn't need it)
            if recovery_mode != "spatial_affordance":
                collapse_severity = (1.0 - current_coherence) * 0.15

                print(
                    f"[WFC] Splatting retrocausal repulsion "
                    f"(severity={collapse_severity:.2f}) along failed path..."
                )

                for i, node in enumerate(self.nodes):
                    self.density.splat_collision_event(
                        position=failed_positions[i],
                        velocity=failed_velocities[i],
                        severity=collapse_severity,
                        node_id=node.id,
                        is_wfc_event=True,
                    )

            self.wfc_trigger_events.append(
                {
                    "timestep": self.step_count,
                    "coherence": current_coherence,
                    "loop_integrity": loop_integrity,
                    "recovery_info": recovery_info,
                    "recovery_mode": recovery_mode,
                    "recovery_bonus": wfc_recovery_bonus,
                    "repulsion_severity": collapse_severity,
                }
            )

            # Force a learning update for the crash (only for temporal)
            if self.mode == "training" and self.last_state is not None:
                if recovery_mode != "spatial_affordance":
                    last_features, last_adj, last_actions = self.last_state

                    # NEW: Differential punishment
                    # Temporal recovery gets moderate penalty (you failed but recovered)
                    # Random jitter gets heavy penalty (you failed completely)
                    if recovery_mode == "coherent_tail":
                        collapse_penalty = np.full(
                            len(self.nodes), -2.0
                        )  # Moderate penalty
                    else:
                        collapse_penalty = np.full(
                            len(self.nodes), -8.0
                        )  # Heavy penalty

                    recovered_features = self._get_current_node_features()
                    recovered_adj = build_adjacency_matrix(
                        self.nodes, self.cfg["exploration"]["sensor_range"]
                    )

                    self.gnn.memory.push(
                        node_features=last_features,
                        adj_matrix=last_adj,
                        actions=last_actions,
                        rewards=collapse_penalty,
                        next_node_features=recovered_features,
                        next_adj_matrix=recovered_adj,
                        done=False,
                        integrity=float(loop_integrity),
                    )

            # Update last_state to recovered state
            if recovery_mode != "spatial_affordance":
                recovered_features = self._get_current_node_features()
                recovered_adj = build_adjacency_matrix(
                    self.nodes, self.cfg["exploration"]["sensor_range"]
                )
                self.last_state = (recovered_features, recovered_adj, actions)

        # NEW: Apply WFC recovery bonus to all nodes (collective achievement!)
        if wfc_recovery_bonus > 0:
            step_rewards_array += wfc_recovery_bonus
            print(
                f"[Reward] All nodes receive +{wfc_recovery_bonus:.1f} for {recovery_mode} recovery"
            )

        # --- 11. Record State & Metrics ---
        self.record_state(episode_step, current_coherence, loop_integrity)

        # Track metrics
        metrics = {
            "timestep": self.step_count,
            "episode": self.current_episode,
            "epsilon": current_epsilon,  # Track exploration rate
            "training_loss": float(training_loss)
            if training_loss is not None
            else None,
            "avg_reward": float(np.mean(step_rewards_array)),
            "coherence": current_coherence,
            "loop_integrity": loop_integrity,
            "coverage": self.map.get_coverage_percentage(),
            "broken_connections": len(self.loop.get_broken_connections()),
            "total_breaks": self.loop.total_breaks,
            "reconnections_this_step": len(reconnected) if reconnected else 0,
            "collision_recoveries": len(
                [e for e in self.collision_events if e["timestep"] == self.step_count]
            ),
        }
        self.metrics_history.append(metrics)

        # Update tracking
        self.step_count += 1
        avg_reward = float(np.mean(step_rewards_array))
        self.total_reward += avg_reward

        return avg_reward

    def record_state(self, t: int, coherence: float, loop_integrity: float):
        """Record current state for visualization."""
        snap = {
            "time": t,
            "coherence": coherence,
            "loop_integrity": loop_integrity,
            "nodes": [{"id": n.id, "pos": n.pos.tolist()} for n in self.nodes],
            "connections": [
                {"node_a": c.node_a_id, "node_b": c.node_b_id, "broken": c.is_broken}
                for c in self.loop.connections
            ],
            "obstacles": [
                {
                    "id": obs.id,
                    "pos": obs.pos.tolist(),
                    "radius": obs.radius,
                    "is_static": obs.is_static,
                }
                for obs in self.obstacle_manager.obstacles
            ],
        }
        self.history.append(snap)

    def get_statistics(self) -> dict:
        """Get comprehensive simulation statistics."""
        coverage = self.map.get_coverage_percentage()
        loop_stats = self.loop.get_statistics()
        density_stats = self.density.get_statistics()

        return {
            "step": self.step_count,
            "mode": self.mode,
            "coverage": coverage,
            "avg_reward": self.total_reward / max(1, self.step_count),
            "num_nodes": len(self.nodes),
            "buffer_size": len(self.gnn.memory) if self.mode == "training" else 0,
            "loop_integrity": loop_stats["current_integrity"],
            "total_loop_breaks": loop_stats["total_breaks_occurred"],
            "total_reconnections": self.total_reconnections,
            "wfc_triggers": len(self.wfc_trigger_events),
            "collision_events": len(self.collision_events),
            "num_obstacles": len(self.obstacle_manager.obstacles),
            "density_collision_splats": density_stats["total_collision_splats"],
            "density_wfc_splats": density_stats["total_wfc_splats"],
            "repulsion_field_coverage": density_stats[
                "repulsion_field_nonzero_fraction"
            ],
        }

    def save_metrics(self, filepath: str):
        """Save detailed metrics to JSON."""
        import json

        # Convert numpy types to native Python types for JSON serialization
        def convert_to_serializable(obj):
            """Recursively convert numpy types to Python native types."""
            if isinstance(obj, dict):
                return {
                    key: convert_to_serializable(value) for key, value in obj.items()
                }
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            else:
                return obj

        data = {
            "final_statistics": convert_to_serializable(self.get_statistics()),
            "loop_statistics": convert_to_serializable(self.loop.get_statistics()),
            "metrics_timeseries": convert_to_serializable(self.metrics_history),
            "training_losses": convert_to_serializable(self.training_losses),
            "loop_break_events": convert_to_serializable(self.loop_break_events),
            "wfc_trigger_events": convert_to_serializable(self.wfc_trigger_events),
            "collision_events": convert_to_serializable(self.collision_events),
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

        print(f"[Metrics] Saved to {filepath}")
