"""
core.py

MuJoCo-Powered FLOWRRA Orchestrator.
Implements a Brain/Brainstem architecture:
- GNN (Brain) runs at ~10Hz for trajectory planning.
- PID (Brainstem) runs at 100Hz for physical motor control.
"""

from typing import Any, Dict, List, Set

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
    def __init__(self, mode="training"):
        self.cfg = CONFIG
        self.mode = mode
        self.dims = 3  # MuJoCo is strictly 3D
        self.num_nodes = self.cfg["node"].get("num_nodes", 10)

        # =======================================================
        # 1. BOOT MUJOCO ENVIRONMENT
        # =======================================================
        print("[Orchestrator] Booting MuJoCo Physics Engine...")
        self.scene_xml = generate_swarm_xml(
            num_nodes=self.num_nodes, num_static_obs=5, num_moving_obs=3
        )
        self.model = mujoco.MjModel.from_xml_string(self.scene_xml)
        self.data = mujoco.MjData(self.model)

        # Timings: MuJoCo runs at 100Hz (dt=0.01)
        self.dt = self.model.opt.timestep
        self.physics_steps_per_gnn_step = 10  # GNN runs at 10Hz

        # =======================================================
        # 2. INITIALIZE LOW-LEVEL CONTROL STACK (The Brainstem)
        # =======================================================
        self.pid_controllers = [DronePID() for _ in range(self.num_nodes)]
        self.mixer = QuadcopterMixer(motor_limit=15.0)

        # =======================================================
        # 3. INITIALIZE SYSTEM MANAGERS
        # =======================================================
        self.obstacle_manager = ObstacleManager(self.model, self.data)

        self.loop = LoopStructure(
            ideal_distance=self.cfg["loop"]["ideal_distance"],
            break_threshold=self.cfg["loop"]["break_threshold"],
            dimensions=self.dims,
        )
        self.loop.initialize_ring_topology(self.num_nodes)

        self.density = DensityFunctionEstimatorND(dimensions=self.dims)
        self.wfc = Wave_Function_Collapse()

        # =======================================================
        # 4. INITIALIZE THE GNN (The Brain)
        # =======================================================
        # Calculate dummy input dim (Position + Velocity + Density + Sensor stuff)
        # Assuming a flat vector for now based on your old get_state_vector
        dummy_state_dim = 3 + 3 + np.prod(self.density.local_grid_size)

        self.gnn = GNNAgent(
            node_feature_dim=dummy_state_dim,
            action_size=6,  # 3D Waypoint + 3D Target Velocity
            hidden_dim=self.cfg["gnn"]["hidden_dim"],
        )

        # State tracking
        self.step_count = 0
        self.last_state = None
        self.frozen_nodes: Set[int] = set()

        self.grace_period = 0  # To get stable states...

        # Telemetry Dashboard
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
    # MUJOCO DATA BRIDGES
    # ========================================================================
    def _get_mujoco_positions(self) -> np.ndarray:
        """Extracts exact 3D world coordinates for all drones."""
        positions = []
        for i in range(self.num_nodes):
            body_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, f"drone_{i}"
            )
            positions.append(self.data.xpos[body_id].copy())
        return np.array(positions)

    def _get_mujoco_velocities(self) -> np.ndarray:
        """Extracts exact 3D linear velocities for all drones."""
        velocities = []
        for i in range(self.num_nodes):
            body_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, f"drone_{i}"
            )
            # cvel is [rot_x, rot_y, rot_z, lin_x, lin_y, lin_z]
            velocities.append(self.data.cvel[body_id][3:6].copy())
        return np.array(velocities)

    def _build_gnn_inputs(self, positions, velocities):
        """Constructs the state vectors for the neural network."""
        node_features = []

        # 1. Update Density Field with actual physical obstacle states
        obs_states = self.obstacle_manager.get_all_states()
        dummy_peer_states = [
            {"pos": p, "velocity": v} for p, v in zip(positions, velocities)
        ]

        for i in range(self.num_nodes):
            # Affordance Vision
            local_grid = self.density.get_affordance_potential_for_node(
                node_pos=positions[i], repulsion_sources=dummy_peer_states + obs_states
            )

            # Combine [Pos, Vel, Affordance]
            feat = np.concatenate([positions[i], velocities[i], local_grid.flatten()])
            node_features.append(feat)

        adj_matrix = build_adjacency_matrix(
            positions, self.cfg["exploration"]["sensor_range"]
        )
        return np.array(node_features, dtype=np.float32), adj_matrix

    def _get_mujoco_rpy_and_gyro(self) -> tuple[np.ndarray, np.ndarray]:
        rpys = []
        gyros = []
        for i in range(self.num_nodes):
            body_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, f"drone_{i}"
            )
            # MuJoCo quaternion format is [w, x, y, z]
            quat = self.data.xquat[body_id].copy()
            w, x, y, z = quat

            # Convert Quaternion to Roll, Pitch, Yaw mathematically
            sinr_cosp = 2 * (w * x + y * z)
            cosr_cosp = 1 - 2 * (x**2 + y**2)
            roll = np.arctan2(sinr_cosp, cosr_cosp)

            sinp = 2 * (w * y - z * x)
            if abs(sinp) >= 1:
                pitch = np.copysign(np.pi / 2, sinp)  # out of bounds fallback
            else:
                pitch = np.arcsin(sinp)

            siny_cosp = 2 * (w * z + x * y)
            cosy_cosp = 1 - 2 * (y**2 + z**2)
            yaw = np.arctan2(siny_cosp, cosy_cosp)

            rpy = np.array([roll, pitch, yaw])

            # cvel[0:3] is angular velocity (gyro)
            gyro = self.data.cvel[body_id][0:3].copy()
            rpys.append(rpy)
            gyros.append(gyro)

        return np.array(rpys), np.array(gyros)

    # ========================================================================
    # THE MAIN EXECUTION LOOP
    # ========================================================================
    def step(self, episode_step: int, total_episodes: int) -> float:

        # 1. Read Ground Truth from MuJoCo
        current_positions = self._get_mujoco_positions()
        current_velocities = self._get_mujoco_velocities()

        # We need pseudo-nodes for Loop Structure to check breaks
        class DummyNode:
            pass

        dummy_nodes = []
        for i in range(self.num_nodes):
            n = DummyNode()
            n.id = i
            n.pos = current_positions[i]
            dummy_nodes.append(n)

        # 2. Assess Topological Health
        current_integrity = self.loop.calculate_integrity()

        # 3. Calculate Cyclic Exploration (The Safety Governor)
        # Gaussian pulse that throbs over the episode
        progress = episode_step / max(1, total_episodes)
        gaussian_pulse = np.exp(-((progress - 0.5) ** 2) / 0.05) * 0.2

        # THE GOVERNOR: Throttle exploration if the swarm is dying!
        dynamic_noise_scale = gaussian_pulse * current_integrity

        # 4. GNN Forward Pass (The Brain)
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

        # 4.5. THE SWARM COHESION TETHER (The "Soft Wall" Brake)
        max_separation = 6.0
        current_centroid = np.mean(current_positions, axis=0)

        for i in range(self.num_nodes):
            dist_from_centroid = np.linalg.norm(current_positions[i] - current_centroid)
            if dist_from_centroid > max_separation:
                # OVERRIDE GNN: Hit the brakes.
                # We don't force them back, we just stop them dead in their tracks.
                waypoints[i] = current_positions[i].copy()
                target_vels[i] = np.zeros(3)

        # --- THE GRACE PERIOD OVERRIDE ---
        if self.grace_period > 0:
            # The system just rewound. Lock out the GNN and force a perfect hover.
            for i in range(self.num_nodes):
                waypoints[i] = current_positions[i]
                target_vels[i] = np.zeros(3)
            self.grace_period -= 1
            wfc_needs_recovery = False  # Blind the WFC while we stabilize
        else:
            wfc_needs_recovery = self.wfc.needs_recovery()

        # 5. TRIGGER RECOVERY: Pure Temporal Rewind + Safe Initialization
        if wfc_needs_recovery:
            print("[WFC] Swarm shattered! Executing Pure Temporal Rewind...")
            recovery_info = self.wfc.collapse_and_reinitialize(dummy_nodes)

            if recovery_info["success"]:
                safe_targets = recovery_info["target_positions"]

                for i in range(self.num_nodes):
                    # 1. Find exact MuJoCo memory addresses
                    body_id = mujoco.mj_name2id(
                        self.model, mujoco.mjtObj.mjOBJ_BODY, f"drone_{i}"
                    )
                    jnt_id = self.model.body_jntadr[body_id]
                    qpos_idx = self.model.jnt_qposadr[jnt_id]
                    qvel_idx = self.model.jnt_dofadr[jnt_id]

                    # 2. Teleport backward in time and wipe momentum
                    self.data.qpos[qpos_idx : qpos_idx + 3] = safe_targets[i]
                    self.data.qvel[qvel_idx : qvel_idx + 6] = np.zeros(6)
                    self.pid_controllers[i].integral_error = np.zeros(3)

                # Push the time-travel into the MuJoCo reality immediately
                mujoco.mj_forward(self.model, self.data)
                self.loop.repair_all_connections()

                # --- YOUR IDEA: PRE-LOAD STABLE STATES ---
                # Update dummy nodes to the new perfect shape
                for i in range(self.num_nodes):
                    dummy_nodes[i].pos = safe_targets[i]

                # Inject 5 frames of "Perfect Coherence" directly into the WFC memory
                for _ in range(5):
                    self.wfc.assess_loop_coherence(1.0, dummy_nodes, 1.0)

                # Give the PID 10 steps (1 full second) to establish a clean hover
                self.grace_period = 10

                # Update ground truth so the physics loop doesn't panic
                current_positions = np.array(safe_targets)
                current_velocities = np.zeros((self.num_nodes, 3))

        # 6. The Physics Micro-Loop
        current_rpys, current_gyros = self._get_mujoco_rpy_and_gyro()

        for _ in range(self.physics_steps_per_gnn_step):
            for i in range(self.num_nodes):
                if i in self.frozen_nodes:
                    continue

                # A. Cascaded PID Reflexes
                thrust, roll, pitch, yaw = self.pid_controllers[i].compute(
                    current_pos=self._get_mujoco_positions()[i],
                    target_pos=waypoints[i],
                    current_vel=self._get_mujoco_velocities()[i],
                    target_vel=target_vels[i],
                    current_rpy=current_rpys[i],
                    current_gyro=current_gyros[i],
                    dt=self.dt,
                )

                m1, m2, m3, m4 = self.mixer.mix(thrust, roll, pitch, yaw)
                ctrl_idx = i * 4
                self.data.ctrl[ctrl_idx : ctrl_idx + 4] = [m1, m2, m3, m4]

            mujoco.mj_step(self.model, self.data)

        # 7. Post-Physics Reality Check
        new_positions = self._get_mujoco_positions()
        new_velocities = self._get_mujoco_velocities()

        # Update pseudo-nodes for checking breaks
        for i in range(self.num_nodes):
            dummy_nodes[i].pos = new_positions[i]

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

        # --- THE HIVE MIND REWARD ---
        # Did the center of the entire swarm move forward along the +X axis?
        centroid_dx = new_centroid[0] - current_centroid[0]

        # Massive global payout for moving the whole family forward!
        global_flow_reward = self.cfg["rewards"]["r_flow"] * centroid_dx * 25.0

        for i in range(self.num_nodes):
            # --- THE COHESION TETHER (Dynamic Pain) ---
            dist_from_centroid = np.linalg.norm(new_positions[i] - new_centroid)
            r_cohesion = 0.0
            if dist_from_centroid > max_separation:
                # Dynamic Pain: -5 points for EVERY meter they stray too far
                excess_dist = dist_from_centroid - max_separation
                r_cohesion = -5.0 * excess_dist

            # [Collision Detection]
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

            # Loop penalty
            r_loop = self.cfg["rewards"]["r_loop_integrity"] * new_integrity

            # Combine Global Team Reward + Individual Penalties
            step_rewards.append(global_flow_reward + r_coll + r_loop + r_cohesion)

        # --- THE INFINITE TREADMILL (Object Pooling) ---
        swarm_center = np.mean(new_positions, axis=0)

        for geom_id in self.obstacle_manager.static_geom_ids:
            obs_pos = self.data.geom_xpos[geom_id]

            # If the obstacle is more than 15 meters behind the swarm's center...
            if np.linalg.norm(obs_pos[:2] - swarm_center[:2]) > 15.0:
                # ...Teleport it 15 meters IN FRONT of the swarm's flight path
                avg_vel = np.mean(new_velocities, axis=0)[:2]

                # Figure out which way the swarm is flying
                if np.linalg.norm(avg_vel) < 0.1:
                    flight_dir = np.array([1.0, 0.0])  # Default forward if hovering
                else:
                    flight_dir = avg_vel / np.linalg.norm(avg_vel)

                # New spawn point: 15m ahead + random sideways spread
                spread = np.random.uniform(-8.0, 8.0)
                sideways_dir = np.array(
                    [-flight_dir[1], flight_dir[0]]
                )  # Perpendicular

                new_x_y = (
                    swarm_center[:2] + (flight_dir * 15.0) + (sideways_dir * spread)
                )

                # Update MuJoCo state directly!
                self.data.geom_xpos[geom_id][:2] = new_x_y

        # 9. Calculate Coherence & Wave Function Collapse
        step_rewards_array = np.array(step_rewards, dtype=np.float32)

        # Coherence drops if nodes are physically crashing
        collision_penalty = (colliding_nodes / self.num_nodes) * 1.5
        current_coherence = np.clip(new_integrity - collision_penalty, 0.0, 1.0)

        # Record reality into the WFC
        self.wfc.assess_loop_coherence(current_coherence, dummy_nodes, new_integrity)

        # 10. Store Memory & Learn
        if self.mode == "training" and self.last_state is not None:
            last_features, last_adj, last_actions = self.last_state

            new_features, new_adj = self._build_gnn_inputs(
                new_positions, new_velocities
            )

            self.gnn.memory.push(
                state=last_features,
                adj=last_adj,
                action=last_actions,
                reward=np.clip(step_rewards_array, -50.0, 50.0),
                next_state=new_features,
                next_adj=new_adj,
                done=False,
                integrity=float(new_integrity),  # Critic Target!
            )

            if len(self.gnn.memory) >= self.gnn.batch_size:
                actor_loss, critic_loss = self.gnn.learn()
                # Update losses only when learning happens
                self.statistics["actor_loss"] = actor_loss
                self.statistics["critic_loss"] = critic_loss

        self.last_state = (node_features, adj_matrix, raw_actions)
        self.step_count += 1

        step_reward = float(np.mean(step_rewards_array))

        # Update the public telemetry state
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
