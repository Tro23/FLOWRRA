"""
core.py

Enhanced FLOWRRA Orchestrator with:
- Loop structure and spring forces
- Static and moving obstacles
- Loop break detection and recovery
- Enhanced metrics tracking
"""

from typing import Any, Dict, List

import numpy as np

from .agent import GNNAgent, build_adjacency_matrix
from .config import CONFIG
from .density import DensityFunctionEstimatorND
from .exploration import ExplorationMap
from .loop import LoopStructure
from .node import NodePositionND
from .obstacles import ObstacleManager
from .recovery import Wave_Function_Collapse


class FLOWRRA_Orchestrator:
    """
    Enhanced FLOWRRA orchestrator with loop structure and obstacles.
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
        )

        # NEW: Obstacle Manager
        self.obstacle_manager = ObstacleManager(dimensions=self.dims)
        self._initialize_obstacles()

        # NEW: Loop Structure
        self.loop = LoopStructure(
            ideal_distance=self.cfg["loop"]["ideal_distance"],
            stiffness=self.cfg["loop"]["stiffness"],
            break_threshold=self.cfg["loop"]["break_threshold"],
            dimensions=self.dims,
        )

        # Init Nodes
        self.nodes = self._initialize_nodes_in_loop()

        # Initialize loop topology
        self.loop.initialize_ring_topology(len(self.nodes))

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
        )

        # State tracking
        self.history = []
        self.current_episode = 0
        self.step_count = 0
        self.last_state = None

        # Performance tracking
        self.total_reward = 0.0
        self.episode_rewards = []

        # NEW: Enhanced metrics
        self.metrics_history = []
        self.loop_break_events = []
        self.wfc_trigger_events = []

    def _initialize_obstacles(self):
        """Initialize static and moving obstacles from config."""
        # Static obstacles from config
        for obs_cfg in self.cfg.get("obstacles", []):
            if len(obs_cfg) == 3:  # (x, y, radius)
                x, y, r = obs_cfg
                pos = np.array(
                    [
                        x / self.cfg["spatial"]["world_bounds"][0],
                        y / self.cfg["spatial"]["world_bounds"][1],
                    ]
                )
                self.obstacle_manager.add_static_obstacle(
                    pos, r / self.cfg["spatial"]["world_bounds"][0]
                )

        # Optional: Add moving obstacles
        if self.cfg.get("moving_obstacles"):
            for mov_obs in self.cfg["moving_obstacles"]:
                x, y, r, vx, vy = mov_obs
                pos = np.array(
                    [
                        x / self.cfg["spatial"]["world_bounds"][0],
                        y / self.cfg["spatial"]["world_bounds"][1],
                    ]
                )
                vel = np.array([vx, vy]) * 0.01  # Scale velocity
                self.obstacle_manager.add_moving_obstacle(
                    pos, r / self.cfg["spatial"]["world_bounds"][0], vel
                )

        print(
            f"[Obstacles] Initialized {len(self.obstacle_manager.obstacles)} obstacles"
        )

    def _initialize_nodes_in_loop(self) -> List[NodePositionND]:
        """Initialize nodes in a circular formation."""
        num_nodes = self.cfg["node"]["num_nodes"]
        nodes = []

        # Place nodes in a circle
        center = np.array([0.5, 0.5] if self.dims == 2 else [0.5, 0.5, 0.5])
        radius = 0.3  # Initial loop radius

        for i in range(num_nodes):
            angle = (i / num_nodes) * 2 * np.pi

            if self.dims == 2:
                offset = np.array([np.cos(angle), np.sin(angle)]) * radius
            else:
                # For 3D, distribute on a ring in XY plane
                offset = np.array([np.cos(angle), np.sin(angle), 0]) * radius

            pos = center + offset
            node = NodePositionND(i, pos, self.dims)
            node.sensor_range = self.cfg["exploration"]["sensor_range"]
            node.move_speed = self.cfg["node"]["move_speed"]
            nodes.append(node)

        print(f"[Nodes] Initialized {num_nodes} nodes in loop formation")
        return nodes

    def _calculate_input_dim(self) -> int:
        """Calculate the dimension of the GNN input vector."""
        dummy_rep = np.zeros(np.prod(self.cfg["repulsion"]["local_grid_size"]))
        dummy_detections = []
        return len(
            self.nodes[0].get_state_vector(
                dummy_rep, dummy_detections, dummy_detections
            )
        )

    def calculate_coherence(self, rewards: np.ndarray, loop_integrity: float) -> float:
        """
        Enhanced coherence calculation incorporating loop integrity.

        Coherence = weighted combination of:
        - Reward performance
        - Loop integrity
        """
        reward_coherence = 1.0 / (1.0 + np.exp(-np.mean(rewards)))

        # Combine with loop integrity (weighted)
        coherence = 0.6 * reward_coherence + 0.4 * loop_integrity
        return coherence

    def step(self, episode_step: int, total_episodes: int = 8000) -> float:
        """
        Execute one simulation step with loop dynamics.
        """
        # --- 1. Update Obstacles ---
        self.obstacle_manager.update_all()

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

        # --- 4. Calculate Spring Forces ---
        spring_forces = self.loop.calculate_spring_forces(self.nodes)

        # --- 5. Density Update ---
        self.density.update_from_sensor_data(
            all_nodes=self.nodes,
            all_obstacle_states=self.obstacle_manager.get_all_states(),
        )

        # --- 6. Build State Representations ---
        node_features = []
        adj_mat = build_adjacency_matrix(
            self.nodes, self.cfg["exploration"]["sensor_range"]
        )

        for node in self.nodes:
            # Sense environment
            node_detections = node.sense_nodes(self.nodes)
            obstacle_detections = node.sense_obstacles(
                self.obstacle_manager.get_all_states()
            )

            # Get repulsion field
            repulsion_sources = node_detections + obstacle_detections
            local_grid = self.density.get_repulsion_potential_for_node(
                node_pos=node.pos, repulsion_sources=repulsion_sources
            )

            # Construct state vector
            feats = node.get_state_vector(
                local_repulsion_grid=local_grid,
                node_detections=node_detections,
                obstacle_detections=obstacle_detections,
            )
            node_features.append(feats)

        node_features_array = np.array(node_features, dtype=np.float32)

        # --- 7. GNN Action Selection ---
        actions = self.gnn.choose_actions(
            node_features=node_features_array,
            adj_matrix=adj_mat,
            episode_number=self.current_episode,
            total_episodes=total_episodes,
        )

        # --- 8. Physics & Reward Calculation ---
        step_rewards = []

        for i, node in enumerate(self.nodes):
            action_id = actions[i]
            old_pos = node.pos.copy()

            # Apply action + spring force
            node.apply_directional_action(action_id, dt=1.0)

            # Add spring force influence
            if node.id in spring_forces:
                force = spring_forces[node.id]
                # Apply force as velocity modification
                node.pos = np.mod(node.pos + force * 0.1, 1.0)  # Scale force

            # Calculate movement
            move_mag = np.linalg.norm(node.velocity())

            # Collision check with obstacles
            collides, obs_ids = self.obstacle_manager.check_collision(
                node.pos, safety_margin=0.05
            )
            r_coll = -self.cfg["rewards"]["r_collision"] if collides else 0.0

            # Movement reward
            r_flow = self.cfg["rewards"]["r_flow"] * move_mag

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
        new_coverage = self.map.update(self.nodes)
        r_explore = new_coverage * self.cfg["rewards"]["r_explore"]
        step_rewards_array += r_explore

        # --- 9. Store Experience (Training Mode Only) ---
        if self.mode == "training":
            if self.last_state is not None:
                last_features, last_adj, last_actions = self.last_state

                self.gnn.memory.push(
                    node_features=last_features,
                    adj_matrix=last_adj,
                    actions=last_actions,
                    rewards=step_rewards_array,
                    next_node_features=node_features_array,
                    next_adj_matrix=adj_mat,
                    done=False,
                )

                if len(self.gnn.memory) >= self.gnn.batch_size:
                    loss = self.gnn.learn()

            self.last_state = (node_features_array, adj_mat, actions)

            # Update target network periodically
            if self.step_count % 100 == 0:
                self.gnn.update_target_network()

        # --- 10. Enhanced WFC with Loop Awareness ---
        loop_integrity = self.loop.calculate_integrity()
        current_coherence = self.calculate_coherence(step_rewards_array, loop_integrity)

        # Log to WFC
        self.wfc.assess_loop_coherence(current_coherence, self.nodes)

        # Trigger recovery if needed
        if self.wfc.needs_recovery() or not self.loop.is_loop_coherent(
            min_integrity=0.5
        ):
            print(
                f"\n[!] Step {episode_step}: System unstable (coherence={current_coherence:.2f}, integrity={loop_integrity:.2f})"
            )
            print("[WFC] Triggering recovery...")

            recovery_info = self.wfc.collapse_and_reinitialize(self.nodes)

            # Repair all loop connections after recovery
            self.loop.repair_all_connections()

            print(f"[WFC] Recovery complete: {recovery_info}")

            self.wfc_trigger_events.append(
                {
                    "timestep": self.step_count,
                    "coherence": current_coherence,
                    "loop_integrity": loop_integrity,
                    "recovery_info": recovery_info,
                }
            )

            self.last_state = None

        # --- 11. Record State & Metrics ---
        self.record_state(episode_step, current_coherence, loop_integrity)

        # Track metrics
        metrics = {
            "timestep": self.step_count,
            "avg_reward": float(np.mean(step_rewards_array)),
            "coherence": current_coherence,
            "loop_integrity": loop_integrity,
            "coverage": self.map.get_coverage_percentage(),
            "broken_connections": len(self.loop.get_broken_connections()),
            "total_breaks": self.loop.total_breaks,
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

        return {
            "step": self.step_count,
            "mode": self.mode,
            "coverage": coverage,
            "avg_reward": self.total_reward / max(1, self.step_count),
            "num_nodes": len(self.nodes),
            "buffer_size": len(self.gnn.memory) if self.mode == "training" else 0,
            "loop_integrity": loop_stats["current_integrity"],
            "total_loop_breaks": loop_stats["total_breaks_occurred"],
            "wfc_triggers": len(self.wfc_trigger_events),
            "num_obstacles": len(self.obstacle_manager.obstacles),
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
            "loop_break_events": convert_to_serializable(self.loop_break_events),
            "wfc_trigger_events": convert_to_serializable(self.wfc_trigger_events),
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

        print(f"[Metrics] Saved to {filepath}")
