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

        # 5. The Physics Micro-Loop (The Brainstem)
        # Run 10 frames of rigid-body physics for every 1 GNN thought
        for _ in range(self.physics_steps_per_gnn_step):
            for i in range(self.num_nodes):
                if i in self.frozen_nodes:
                    # Cut motors for frozen nodes (they will drop, or we can command hover)
                    ctrl_idx = i * 4
                    self.data.ctrl[ctrl_idx : ctrl_idx + 4] = [0, 0, 0, 0]
                    continue

                # A. PID Reflexes
                thrust, roll, pitch, yaw = self.pid_controllers[i].compute(
                    current_pos=self._get_mujoco_positions()[i],
                    target_pos=waypoints[i],
                    current_vel=self._get_mujoco_velocities()[i],
                    target_vel=target_vels[i],
                    dt=self.dt,
                )

                # B. Motor Mixing
                m1, m2, m3, m4 = self.mixer.mix(thrust, roll, pitch, yaw)

                # C. Apply to MuJoCo
                ctrl_idx = i * 4
                self.data.ctrl[ctrl_idx : ctrl_idx + 4] = [m1, m2, m3, m4]

            # D. Step the Universe
            mujoco.mj_step(self.model, self.data)

        # 6. Post-Physics Reality Check
        new_positions = self._get_mujoco_positions()
        new_velocities = self._get_mujoco_velocities()

        # Update pseudo-nodes for checking breaks
        for i in range(self.num_nodes):
            dummy_nodes[i].pos = new_positions[i]

        # Check actual physical loop breaks (obstacles blocking sight, or drift)
        self.loop.check_breaks(dummy_nodes, self.obstacle_manager)
        new_integrity = self.loop.calculate_integrity()

        # 7. Physical Rewards & Penalties
        step_rewards = []
        colliding_nodes = 0

        for i in range(self.num_nodes):
            # Calculate distance actually traveled
            dist_moved = np.linalg.norm(new_positions[i] - current_positions[i])
            r_flow = self.cfg["rewards"]["r_flow"] * dist_moved

            # Detect Physical Collisions (Velocity drops abruptly)
            r_coll = 0.0
            if (
                np.linalg.norm(new_velocities[i]) < 0.05
                and np.linalg.norm(target_vels[i]) > 0.5
            ):
                # The GNN asked to move fast, but physical velocity is 0. That's a wall.
                r_coll = -self.cfg["rewards"]["r_collision"]
                colliding_nodes += 1

                # Splat the danger zone in the density field
                self.density.splat_collision_event(
                    position=new_positions[i],
                    velocity=current_velocities[i],  # The velocity *before* the crash
                    severity=1.0,
                )

            # Loop penalty
            r_loop = self.cfg["rewards"]["r_loop_integrity"] * new_integrity

            step_rewards.append(r_flow + r_coll + r_loop)

        # 8. Calculate Coherence & Wave Function Collapse
        step_rewards_array = np.array(step_rewards, dtype=np.float32)

        # Coherence drops if nodes are physically crashing
        collision_penalty = (colliding_nodes / self.num_nodes) * 1.5
        current_coherence = np.clip(new_integrity - collision_penalty, 0.0, 1.0)

        # Record reality into the WFC
        self.wfc.assess_loop_coherence(current_coherence, dummy_nodes, new_integrity)

        # TRIGGER RECOVERY
        if self.wfc.needs_recovery():
            print(f"[!] SYSTEM SHATTERED. Triggering WFC Recovery.")

            # Massive punishment to the GNN for causing a collapse
            step_rewards_array -= 15.0

            # Perform relative shape rewind
            recovery_info = self.wfc.collapse_and_reinitialize(dummy_nodes)

            if recovery_info["success"]:
                # Forcefully inject the safe coordinates back into MuJoCo's physics state
                for i in range(self.num_nodes):
                    body_id = mujoco.mj_name2id(
                        self.model, mujoco.mjtObj.mjOBJ_BODY, f"drone_{i}"
                    )
                    jnt_id = self.model.body_jntadr[body_id]
                    qpos_idx = self.model.jnt_qposadr[jnt_id]
                    qvel_idx = self.model.jnt_dofadr[jnt_id]

                    # Set 3D Position (keeping default quaternion for rotation)
                    self.data.qpos[qpos_idx : qpos_idx + 3] = dummy_nodes[i].pos
                    self.data.qpos[qpos_idx + 3 : qpos_idx + 7] = [1, 0, 0, 0]

                    # Zero out velocity to stop the crash momentum
                    self.data.qvel[qvel_idx : qvel_idx + 6] = np.zeros(6)

                mujoco.mj_forward(self.model, self.data)
                self.loop.repair_all_connections()

                # Re-pull safe state
                new_positions = self._get_mujoco_positions()
                new_velocities = self._get_mujoco_velocities()
                new_integrity = 1.0

        # 9. Store Memory & Learn
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
