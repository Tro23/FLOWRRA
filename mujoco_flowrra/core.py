"""
core.py

MuJoCo-Powered FLOWRRA Orchestrator.
Implements a Brain/Brainstem architecture:
- GNN (Brain)      runs at ~10 Hz for trajectory planning.
- PID (Brainstem)  runs at 100 Hz for physical motor control.
"""

from typing import Set

import mujoco
import numpy as np
from agent import GNNAgent, build_adjacency_matrix
from config import CONFIG
from density import DensityFunctionEstimatorND
from loop import LoopStructure
from motor_mixer import QuadcopterMixer
from obstacles import ObstacleManager
from pid_controller import DronePID
from recovery import Wave_Function_Collapse
from scene_builder import generate_swarm_xml


class FLOWRRA_Orchestrator:
    def __init__(self, mode: str = "training"):
        self.cfg = CONFIG
        self.mode = mode
        self.dims = 3
        self.num_nodes = self.cfg["node"].get("num_nodes", 10)

        # ===================================================================
        # 1. BOOT MUJOCO
        # ===================================================================
        print("[Orchestrator] Booting MuJoCo Physics Engine...")
        self.scene_xml = generate_swarm_xml(
            num_nodes=self.num_nodes, num_static_obs=5, num_moving_obs=3
        )
        self.model = mujoco.MjModel.from_xml_string(self.scene_xml)
        self.data = mujoco.MjData(self.model)

        self.dt = self.model.opt.timestep  # 0.01 s
        self.physics_steps_per_gnn_step = 10  # GNN at 10 Hz

        # FIX #3 — Cache body IDs once at init to avoid O(N) mj_name2id
        # lookups inside the hot physics loop.
        self._body_ids: list[int] = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"drone_{i}")
            for i in range(self.num_nodes)
        ]

        # ===================================================================
        # 2. BRAINSTEM
        # ===================================================================
        self.pid_controllers = [DronePID() for _ in range(self.num_nodes)]
        self.mixer = QuadcopterMixer(motor_limit=15.0)

        # ===================================================================
        # 3. SYSTEM MANAGERS
        # ===================================================================
        self.obstacle_manager = ObstacleManager(self.model, self.data)

        self.loop = LoopStructure(
            ideal_distance=self.cfg["loop"]["ideal_distance"],
            break_threshold=self.cfg["loop"]["break_threshold"],
            dimensions=self.dims,
        )
        self.loop.initialize_ring_topology(self.num_nodes)

        self.density = DensityFunctionEstimatorND(dimensions=self.dims)

        # FIX #6 — Wire config values into WFC instead of using defaults.
        self.wfc = Wave_Function_Collapse(
            history_length=self.cfg["wfc"]["history_length"],
            collapse_threshold=self.cfg["wfc"]["collapse_threshold"],
            tau=self.cfg["wfc"]["tau"],
        )

        # ===================================================================
        # 4. GNN BRAIN
        # ===================================================================
        node_feature_dim = 3 + 3 + int(np.prod(self.density.local_grid_size))

        self.gnn = GNNAgent(
            node_feature_dim=node_feature_dim,
            action_size=6,
            hidden_dim=self.cfg["gnn"]["hidden_dim"],
        )

        self.step_count = 0
        self.last_state = None
        self.frozen_nodes: Set[int] = set()
        self.grace_period = 0
        self.stagnation_counter = 0
        self.smoothed_target_vels = np.zeros(
            (self.num_nodes, 3)
        )  # Acceleration Clutch.

        self.statistics = {
            "reward": 0.0,
            "integrity": 1.0,
            "coherence": 1.0,
            "actor_loss": 0.0,
            "critic_loss": 0.0,
            "wfc_collapses": 0,
            "wfc_recoveries": 0,
        }

    # ========================================================================
    # MUJOCO DATA BRIDGES  (all use cached _body_ids)
    # ========================================================================
    def _get_mujoco_positions(self) -> np.ndarray:
        return np.array([self.data.xpos[bid].copy() for bid in self._body_ids])

    def _get_mujoco_velocities(self) -> np.ndarray:
        # cvel layout: [rot_x, rot_y, rot_z, lin_x, lin_y, lin_z]
        return np.array([self.data.cvel[bid][3:6].copy() for bid in self._body_ids])

    def _get_mujoco_rpy_and_gyro(self) -> tuple[np.ndarray, np.ndarray]:
        rpys, gyros = [], []
        for bid in self._body_ids:
            w, x, y, z = self.data.xquat[bid]

            sinr_cosp = 2 * (w * x + y * z)
            cosr_cosp = 1 - 2 * (x**2 + y**2)
            roll = np.arctan2(sinr_cosp, cosr_cosp)

            sinp = 2 * (w * y - z * x)
            pitch = np.copysign(np.pi / 2, sinp) if abs(sinp) >= 1 else np.arcsin(sinp)

            siny_cosp = 2 * (w * z + x * y)
            cosy_cosp = 1 - 2 * (y**2 + z**2)
            yaw = np.arctan2(siny_cosp, cosy_cosp)

            rpys.append([roll, pitch, yaw])
            gyros.append(self.data.cvel[bid][0:3].copy())

        return np.array(rpys), np.array(gyros)

    def _build_gnn_inputs(
        self, positions: np.ndarray, velocities: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        obs_states = self.obstacle_manager.get_all_states()
        peer_states = [{"pos": p, "velocity": v} for p, v in zip(positions, velocities)]
        node_features = []

        for i in range(self.num_nodes):
            local_grid = self.density.get_affordance_potential_for_node(
                node_pos=positions[i],
                repulsion_sources=peer_states + obs_states,
            )
            feat = np.concatenate([positions[i], velocities[i], local_grid.flatten()])
            node_features.append(feat)

        adj_matrix = build_adjacency_matrix(
            positions, self.cfg["exploration"]["sensor_range"]
        )
        return np.array(node_features, dtype=np.float32), adj_matrix

    # ========================================================================
    # MAIN STEP
    # ========================================================================
    def step(self, episode_step: int, steps_per_episode: int) -> float:
        # NOTE: parameter renamed from total_episodes → steps_per_episode
        #       to match what main.py actually passes.

        # 1. Read ground truth
        current_positions = self._get_mujoco_positions()
        current_velocities = self._get_mujoco_velocities()

        # Build DummyNodes — lightweight wrappers the Loop/WFC systems need.
        # FIX #2 — Store .vel so WFC shape memory records real velocities.
        class DummyNode:
            __slots__ = ("id", "pos", "vel")

        dummy_nodes = []
        for i in range(self.num_nodes):
            n = DummyNode()
            n.id = i
            n.pos = current_positions[i]
            n.vel = current_velocities[i]
            dummy_nodes.append(n)

        # 2. Topological health
        current_integrity = self.loop.calculate_integrity()

        # 3. Exploration governor — Gaussian pulse throttled by swarm health
        progress = episode_step / max(1, steps_per_episode)
        gaussian_pulse = np.exp(-((progress - 0.5) ** 2) / 0.05) * 0.2
        dynamic_noise_scale = gaussian_pulse * current_integrity

        # 4. GNN forward pass (the Brain)
        node_features, adj_matrix = self._build_gnn_inputs(
            current_positions, current_velocities
        )
        waypoints, target_vels, raw_actions = self.gnn.choose_actions(
            node_features=node_features,
            adj_matrix=adj_matrix,
            current_positions=current_positions,
            current_integrity=current_integrity,
            noise_scale=dynamic_noise_scale,
        )

        # Safety overrides — clamp velocity, project waypoint, floor guard
        max_separation = 8.0
        current_centroid = np.mean(current_positions, axis=0)

        for i in range(self.num_nodes):
            raw_target_vel = np.clip(target_vels[i], -2.0, 2.0)

            # --- THE CLUTCH (EMA Smoothing) ---
            # Blend 80% of the old velocity with 20% of the GNN's new requested velocity.
            # This prevents instant whiplash and forces the drones to accelerate smoothly.
            self.smoothed_target_vels[i] = (0.8 * self.smoothed_target_vels[i]) + (
                0.2 * raw_target_vel
            )

            # Overwrite the target_vel so the PID gets the safe, smoothed version
            target_vels[i] = self.smoothed_target_vels[i]

            # Project carrot 1 s ahead along the SMOOTHED velocity
            waypoints[i] = current_positions[i] + (target_vels[i] * 1.0)

            # Hard floor: never command below Z = 1.5 m
            waypoints[i][2] = max(1.5, waypoints[i][2])

            # Soft-wall cohesion tether
            if np.linalg.norm(current_positions[i] - current_centroid) > max_separation:
                waypoints[i] = current_positions[i].copy()
                target_vels[i] = np.zeros(3)
                self.smoothed_target_vels[i] = np.zeros(
                    3
                )  # Reset clutch if tether pulls

        # Grace period: lock GNN out after a WFC rewind
        if self.grace_period > 0:
            for i in range(self.num_nodes):
                waypoints[i] = current_positions[i]
                target_vels[i] = np.zeros(3)
                self.smoothed_target_vels[i] = np.zeros(3)
            self.grace_period -= 1
            wfc_needs_recovery = False
        else:
            wfc_needs_recovery = self.wfc.needs_recovery()

        # 5. WFC recovery — pure temporal rewind
        wfc_penalty = 0.0
        if wfc_needs_recovery:
            print("[WFC] Swarm shattered! Executing Pure Temporal Rewind...")

            # Zap them for triggering a WFC!
            wfc_penalty = -self.cfg["rewards"]["r_collapse_penalty"]

            recovery_info = self.wfc.collapse_and_reinitialize(dummy_nodes)

            if recovery_info["success"]:
                safe_targets = recovery_info["target_positions"]

                for i in range(self.num_nodes):
                    # FIX #7 — Use cached body IDs instead of re-calling mj_name2id
                    bid = self._body_ids[i]
                    jnt_id = self.model.body_jntadr[bid]
                    qpos_idx = self.model.jnt_qposadr[jnt_id]
                    qvel_idx = self.model.jnt_dofadr[jnt_id]

                    self.data.qpos[qpos_idx : qpos_idx + 3] = safe_targets[i]
                    self.data.qvel[qvel_idx : qvel_idx + 6] = np.zeros(6)
                    self.pid_controllers[i].integral_error = np.zeros(3)

                mujoco.mj_forward(self.model, self.data)
                self.loop.repair_all_connections()

                for i in range(self.num_nodes):
                    dummy_nodes[i].pos = safe_targets[i]
                    dummy_nodes[i].vel = np.zeros(3)

                for _ in range(5):
                    self.wfc.assess_loop_coherence(1.0, dummy_nodes, 1.0)

                self.grace_period = 10.0
                current_positions = np.array(safe_targets)
                current_velocities = np.zeros((self.num_nodes, 3))
            # Wipe clutch on reset
            self.smoothed_target_vels = np.zeros((self.num_nodes, 3))

        # 6. Physics micro-loop (100 Hz)
        # FIX #3 + #4 — Read ALL sensor data once per sub-step, not once per
        # drone per sub-step.  This collapses 10 × N × N mj_name2id calls
        # down to 10 batch reads, and keeps the PID gyro loop seeing fresh
        # attitude data every physics tick instead of stale pre-loop values.
        for _ in range(self.physics_steps_per_gnn_step):
            step_positions = self._get_mujoco_positions()
            step_velocities = self._get_mujoco_velocities()
            step_rpys, step_gyros = self._get_mujoco_rpy_and_gyro()

            for i in range(self.num_nodes):
                if i in self.frozen_nodes:
                    continue

                thrust, roll, pitch, yaw = self.pid_controllers[i].compute(
                    current_pos=step_positions[i],
                    target_pos=waypoints[i],
                    current_vel=step_velocities[i],
                    target_vel=target_vels[i],
                    current_rpy=step_rpys[i],
                    current_gyro=step_gyros[i],
                    dt=self.dt,
                )

                m1, m2, m3, m4 = self.mixer.mix(thrust, roll, pitch, yaw)
                ctrl_idx = i * 4
                self.data.ctrl[ctrl_idx : ctrl_idx + 4] = [m1, m2, m3, m4]

            mujoco.mj_step(self.model, self.data)

        # 7. Post-physics reality check
        new_positions = self._get_mujoco_positions()
        new_velocities = self._get_mujoco_velocities()

        for i in range(self.num_nodes):
            dummy_nodes[i].pos = new_positions[i]
            dummy_nodes[i].vel = new_velocities[i]

        self.loop.check_breaks(dummy_nodes, self.obstacle_manager)
        self.loop.attempt_reconnection(dummy_nodes, self.obstacle_manager)
        new_integrity = self.loop.calculate_integrity()

        # 8. Physical Rewards & Penalties
        step_rewards = []
        max_separation = 6.0
        colliding_nodes = 0

        # Calculate the centroids
        current_centroid = np.mean(current_positions, axis=0)
        new_centroid = np.mean(new_positions, axis=0)

        # --- THE TIME-OUT FIX ---
        if self.grace_period > 0:
            # The environment is forcing them to be safe. They earn NO points.
            for i in range(self.num_nodes):
                step_rewards.append(0.0)
        else:
            # --- NORMAL REWARD CALCULATION ---
            # (Only run this if they are actually in control of their bodies)

            # The Macro Hive Mind Reward (X/Y only)
            centroid_dist_xy = np.linalg.norm(new_centroid[:2] - current_centroid[:2])
            global_flow_reward = self.cfg["rewards"]["r_flow"] * centroid_dist_xy * 8.0

            # The Z-Axis Ceiling Penalty
            r_altitude = 0.0
            if new_centroid[2] > 3.5:
                r_altitude = -20.0 * (new_centroid[2] - 3.5)

            # --- THE LOITERING TAX ---
            # --- THE 5-STEP STAGNATION TRACKER ---
            # We give them a 5cm buffer. If they move less than that, the timer ticks up.
            if centroid_dist_xy < 0.05:
                self.stagnation_counter += 1
            else:
                self.stagnation_counter = 0  # Reset the timer if they move!

            r_loiter = 0.0
            # If they sit still for 5 frames (0.5 seconds), hit them with the config penalty
            if self.stagnation_counter >= 5:
                r_loiter = -self.cfg["rewards"]["r_idle"]

            # --- SALARY FREEZE ---
            r_loop = 0.0
            if self.stagnation_counter < 5:
                # They only get their salary if they aren't officially stagnating
                r_loop = self.cfg["rewards"]["r_loop_integrity"] * new_integrity

            for i in range(self.num_nodes):
                # The Micro Internal Hustle
                indiv_dist_xy = np.linalg.norm(
                    new_positions[i][:2] - current_positions[i][:2]
                )
                r_indiv_flow = self.cfg["rewards"]["r_flow"] * indiv_dist_xy * 5.0

                # Bankruptcy Cohesion Tether
                dist_from_centroid = np.linalg.norm(new_positions[i] - new_centroid)
                r_cohesion = 0.0
                if dist_from_centroid > max_separation:
                    excess_dist = dist_from_centroid - max_separation
                    r_cohesion = -10.0 * excess_dist

                # Collision Detection
                r_coll = 0.0
                if (
                    np.linalg.norm(new_velocities[i]) < 0.05
                    and np.linalg.norm(target_vels[i]) > 0.5
                ):
                    r_coll = -self.cfg["rewards"]["r_collision"]
                    colliding_nodes += 1
                    self.density.splat_collision_event(
                        position=new_positions[i],
                        velocity=current_velocities[i],
                        severity=1.0,
                    )

                # Combine it all
                step_rewards.append(
                    global_flow_reward
                    + r_indiv_flow
                    + r_altitude
                    + r_coll
                    + r_loop
                    + r_cohesion
                    + r_loiter
                    + wfc_penalty
                )

        # 9. Obstacle pooling — the infinite treadmill
        # FIX #1 — data.geom_xpos is READ-ONLY (computed by MuJoCo forward).
        # Static geom positions live in model.geom_pos.  Write there, then
        # call mj_forward once after all teleports to flush into data.
        swarm_center = np.mean(new_positions, axis=0)
        teleported_any = False

        for geom_id in self.obstacle_manager.static_geom_ids:
            # Read world position from model (correct for worldbody-attached geoms)
            obs_pos = self.model.geom_pos[geom_id]

            if np.linalg.norm(obs_pos[:2] - swarm_center[:2]) > 10.0:
                avg_vel = np.mean(new_velocities, axis=0)[:2]

                if np.linalg.norm(avg_vel) < 0.1:
                    flight_dir = np.array([1.0, 0.0])
                else:
                    flight_dir = avg_vel / np.linalg.norm(avg_vel)

                spread = np.random.uniform(-8.0, 8.0)
                sideways_dir = np.array([-flight_dir[1], flight_dir[0]])
                new_xy = (
                    swarm_center[:2] + (flight_dir * 10.0) + (sideways_dir * spread)
                )

                # Write to model.geom_pos — the correct mutable location
                self.model.geom_pos[geom_id][:2] = new_xy
                teleported_any = True

        # Single mj_forward after ALL geom writes (not one per geom)
        if teleported_any:
            mujoco.mj_forward(self.model, self.data)

        # 10. WFC coherence assessment
        step_rewards_array = np.array(step_rewards, dtype=np.float32)
        collision_penalty = (colliding_nodes / self.num_nodes) * 1.5
        current_coherence = np.clip(new_integrity - collision_penalty, 0.0, 1.0)

        self.wfc.assess_loop_coherence(current_coherence, dummy_nodes, new_integrity)

        # 11. Store memory & learn
        if self.mode == "training" and self.last_state is not None:
            last_features, last_adj, last_actions = self.last_state
            new_features, new_adj = self._build_gnn_inputs(
                new_positions, new_velocities
            )

            self.gnn.memory.push(
                state=last_features,
                adj=last_adj,
                action=last_actions,
                reward=np.clip(step_rewards_array, -100.0, 100.0),
                next_state=new_features,
                next_adj=new_adj,
                done=False,
                integrity=float(new_integrity),
            )

            if len(self.gnn.memory) >= self.gnn.batch_size:
                actor_loss, critic_loss = self.gnn.learn()
                self.statistics["actor_loss"] = actor_loss
                self.statistics["critic_loss"] = critic_loss

        self.last_state = (node_features, adj_matrix, raw_actions)
        self.step_count += 1

        step_reward = float(np.mean(step_rewards_array))
        self.statistics.update(
            {
                "reward": step_reward,
                "integrity": float(new_integrity),
                "coherence": float(current_coherence),
                "wfc_collapses": self.wfc.total_collapses,
                "wfc_recoveries": self.wfc.successful_recoveries,
            }
        )
        return step_reward
