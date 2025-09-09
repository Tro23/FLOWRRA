# FLOWRRA_RL.py
import logging
import csv
import os
import numpy as np
import math
from typing import List, Dict, Any, Tuple, Optional
from scipy.stats import entropy

from NodePosition_RL import Node_Position
from EnvironmentA_RL import Environment_A
from EnvironmentB_RL import EnvironmentB
from DensityFunctionEstimator_RL import Density_Function_Estimator
from WaveFunctionCollapse_RL import Wave_Function_Collapse
from RLAgent import SharedRLAgent  # shared single agent

# Setup logger for this module
logger = logging.getLogger("FLOWRRA_RL")
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(h)
logger.setLevel(logging.INFO)

class Flowrra_RL:
    """
    Main orchestrator integrating EnvironmentA, EnvironmentB, Density Estimator,
    Wave Function Collapse (WFC), and a shared RL agent interface.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config

        # Environments
        self.env = Environment_A(
            num_nodes=config.get('num_nodes', 10),
            angle_steps=config.get('angle_steps', 24),
            seed=config.get('seed', None)
        )
        self.env_b = EnvironmentB(
            grid_size=config.get('env_b_grid_size', 60),
            num_fixed=config.get('env_b_num_fixed', 20),
            num_moving=config.get('env_b_num_moving', 6),
            seed=config.get('seed', None)
        )
        self.density_estimator = Density_Function_Estimator(
            grid_shape=config.get('grid_size', (60, 60))
        )
        self.wfc = Wave_Function_Collapse(
            history_length=config.get('wfc_history_length', 200),
            tail_length=config.get('wfc_tail_length', 15),
            collapse_threshold=config.get('wfc_collapse_threshold', 0.25),
            tau=config.get('wfc_tau', 5)
        )
        self.agent: Optional[SharedRLAgent] = None
        self.log_file = config.get('logfile', 'flowrra_rl_log.csv')
        self.visual_dir = config.get('visual_dir', 'flowrra_rl_visuals')
        
        self.log_header = ['step', 'episode', 'total_reward', 'avg_coherence', 'loss', 'wfc_reinit']
        
        # New state and action dimensions
        self.num_nodes = self.env.num_nodes
        # Per-node state size: pos(2) + vel(2) + sensor_data(variable) + repulsion_potential(16)
        # We will fix sensor data representation for a consistent state size.
        # Let's assume sensor data for 2 nearest neighbors and 2 nearest obstacles:
        # Neighbor 1: dist(1), bearing(1), vel_x(1), vel_y(1) = 4
        # Neighbor 2: ... = 4
        # Obstacle 1: ... = 4
        # Obstacle 2: ... = 4
        # Total sensor data per node = 16
        # Total per-node state size: 2 + 2 + 16 + 16 = 36
        self.per_node_state_size = 36
        self.state_size = self.num_nodes * self.per_node_state_size
        self.action_size = 4 # for position, e.g., move left, right, up, down
        self.angle_action_size = 4 # for angle, e.g., turn left, right, noop, noop
        self.combined_action_size = self.action_size * self.angle_action_size
        
        # Reset and get initial state
        self.reset()
        
    def reset(self):
        """
        Resets the entire system.
        """
        self.env.reset()
        self.env_b.reset()
        self.wfc.reset()

    def get_state(self) -> np.ndarray:
        """
        Generates the current state vector for the RL agent based on the new
        multi-slice state representation.
        
        Returns:
            np.ndarray: A flattened state vector (1D array).
        """
        state_vector = []
        obstacle_states = self.env_b.get_obstacle_states()

        for node in self.env.nodes:
            # Slices of the state for a single node
            # 1. Position and Velocity (2D arrays)
            node_state = np.concatenate([node.pos, node.velocity()])

            # 2. Sensor Data (variable array - now a fixed size)
            all_detections = node.sense_nodes(self.env.nodes) + node.sense_obstacles(obstacle_states)
            
            # Sort by distance and take the top N
            all_detections.sort(key=lambda d: d['distance'])
            
            # Fixed-size sensor data representation (padding with zeros)
            sensor_data = np.zeros(16) # 4 detections * 4 values each
            for i, detection in enumerate(all_detections[:4]):
                sensor_data[i*4 + 0] = detection['distance']
                sensor_data[i*4 + 1] = detection['bearing_rad']
                sensor_data[i*4 + 2] = detection['velocity'][0]
                sensor_data[i*4 + 3] = detection['velocity'][1]
            
            node_state = np.concatenate([node_state, sensor_data])
            
            # 3. Repulsion Potential Grid (4x4 array)
            # This is now calculated and managed by the DensityFunctionEstimator per-node
            local_repulsion_grid = self.density_estimator.get_repulsion_potential_for_node(
                node_pos=node.pos,
                all_node_positions=np.array([n.pos for n in self.env.nodes]),
                all_obstacle_states=obstacle_states
            )
            node_state = np.concatenate([node_state, local_repulsion_grid.flatten()])
            
            state_vector.append(node_state)
            
        return np.concatenate(state_vector)

    def calculate_coherence(self) -> float:
        """
        Calculates the coherence score for the entire state based on the entropy
        of each node's local repulsion potential grid.
        
        Coherence is 1 - H(repulsion_potential_grids), where H is entropy.
        
        Returns:
            float: A single coherence score between 0 and 1. Higher is better.
        """
        repulsion_grids = []
        obstacle_states = self.env_b.get_obstacle_states()
        node_positions = np.array([n.pos for n in self.env.nodes])
        
        for node in self.env.nodes:
            grid = self.density_estimator.get_repulsion_potential_for_node(
                node_pos=node.pos,
                all_node_positions=node_positions,
                all_obstacle_states=obstacle_states
            )
            repulsion_grids.append(grid.flatten())
        
        # Combine all grids and normalize for entropy calculation
        combined_repulsion = np.concatenate(repulsion_grids)
        combined_repulsion = np.abs(combined_repulsion)
        if np.sum(combined_repulsion) == 0:
            return 1.0 # Max coherence if no repulsion
            
        prob_dist = combined_repulsion / np.sum(combined_repulsion)
        
        # Calculate entropy (high entropy = high disorder = low coherence)
        # Adding a small epsilon for numerical stability
        entropy_val = entropy(prob_dist + 1e-9, base=2)
        
        # Normalize entropy to a [0,1] range and invert for coherence
        max_entropy = math.log2(len(prob_dist))
        if max_entropy == 0:
            return 1.0
            
        coherence = 1.0 - (entropy_val / max_entropy)
        
        return coherence

    def step(self, actions: List[int]) -> Tuple[np.ndarray, bool, Dict[str, Any]]:
        """
        Advances the simulation by one timestep using the new two-stage action process.

        Args:
            actions (List[int]): A list of integer actions, one for each node.

        Returns:
            Tuple[np.ndarray, bool, Dict[str, Any]]: (rewards, done, info)
        """
        info = {'wfc_reinit': 'none'}
        
        # Split actions into position and angle movements
        # Assuming action is a combined index: 0-3 for pos, 0-3 for angle
        pos_actions = [int(a / self.angle_action_size) for a in actions]
        angle_actions = [a % self.angle_action_size for a in actions]
        
        # STAGE 1: Apply Position Actions
        for node, pos_action in zip(self.env.nodes, pos_actions):
            node.apply_position_action(pos_action)
        
        # Update obstacle positions
        self.env_b.step()
        
        # Calculate repulsion based on new positions
        self.density_estimator.update_from_sensor_data(
            all_nodes=self.env.nodes,
            all_obstacle_states=self.env_b.get_obstacle_states()
        )
        
        # Check coherence after position move
        coherence_after_pos = self.calculate_coherence()
        if coherence_after_pos < self.wfc.collapse_threshold:
            reinit_info = self.wfc.collapse_and_reinitialize(self.env, None)
            info['wfc_reinit'] = reinit_info['reinit_from']
            # Return zero reward on collapse
            rewards = np.zeros(self.num_nodes)
            return rewards, False, info
        
        # STAGE 2: Apply Angle Actions (only if position move was coherent)
        for node, angle_action in zip(self.env.nodes, angle_actions):
            node.apply_angle_action(angle_action)
            
        # Recalculate everything for the final state
        self.density_estimator.update_from_sensor_data(
            all_nodes=self.env.nodes,
            all_obstacle_states=self.env_b.get_obstacle_states()
        )
        
        final_coherence = self.calculate_coherence()
        if final_coherence < self.wfc.collapse_threshold:
            reinit_info = self.wfc.collapse_and_reinitialize(self.env, None)
            info['wfc_reinit'] = reinit_info['reinit_from']
            # Return zero reward on collapse
            rewards = np.zeros(self.num_nodes)
            return rewards, False, info
            
        # Final state is good, calculate rewards and proceed
        rewards = np.full(self.num_nodes, final_coherence)
        
        done = False
        self.env.t += 1
        
        return rewards, done, info
        
    def attach_agent(self, agent: SharedRLAgent):
        """
        Attaches the RL agent to the Flowrra instance.
        """
        self.agent = agent

    def train(self, total_steps: int, episode_steps: int, visualize_every_n_steps: int, agent: SharedRLAgent):
        """
        Trains the RL agent.
        """
        self.attach_agent(agent)
        
        if not os.path.exists(self.visual_dir):
            os.makedirs(self.visual_dir)
            
        with open(self.log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(self.log_header)

        logger.info("--- Starting FLOWRRA RL Training ---")
        self.reset()
        
        for step in range(total_steps):
            episode_num = step // episode_steps
            
            # Reset environment if episode is over or a collapse occurs
            if step % episode_steps == 0:
                self.reset()
                
            state = self.get_state()
            actions = self.agent.choose_actions(state)
            rewards, done, info = self.step(list(actions))
            next_state = self.get_state()
            
            # Store transition in replay buffer
            self.agent.memory.push(state, actions, rewards, next_state, done)
            
            # Update agent
            loss = 0.0
            if len(self.agent.memory) >= self.agent.batch_size:
                loss = self.agent.learn()
                self.agent.update_target_network()
                
            # Log training data
            total_reward = np.sum(rewards)
            avg_coherence = self.calculate_coherence()
            
            with open(self.log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([step, episode_num, total_reward, avg_coherence, loss, info['wfc_reinit']])

            if step % 50 == 0:
                logger.info(f"Step {step} | Episode {episode_num} | Coherence: {avg_coherence:.4f} | Loss: {loss:.4f} | WFC: {info['wfc_reinit']}")

            if step % visualize_every_n_steps == 0:
                try:
                    from utils_rl import plot_system_state_rl
                    plot_system_state_rl(
                        density_field=self.density_estimator.get_full_repulsion_grid(),
                        nodes=self.env.nodes,
                        env_b=self.env_b,
                        out_path=os.path.join(self.visual_dir, f"train_t_{step:05d}.png"),
                        title=f"FLOWRRA RL: Training Step {step}"
                    )
                except Exception as e:
                    logger.debug(f"Training visualization failed at step {step}: {e}")

            if done:
                logger.info(f"Episode {episode_num} finished at step {step}")
                self.reset()

        logger.info("--- Training Complete ---")

    def deploy(self, total_steps: int, visualize_every_n_steps: int):
        """
        Greedy deployment using the attached shared agent (epsilon=0).
        """
        if self.agent is None:
            raise ValueError("Agent not attached. Call attach_agent() before deploy().")

        logger.info("--- Starting deployment ---")
        self.env.reset()
        self.env_b.reset()
        self.wfc.reset()

        for step in range(total_steps):
            state = self.get_state()
            actions = self.agent.choose_actions(state, epsilon=0.0)
            rewards, done, info = self.step(list(actions))

            if step % visualize_every_n_steps == 0:
                try:
                    from utils_rl import plot_system_state_rl
                    plot_system_state_rl(
                        density_field=self.density_estimator.get_full_repulsion_grid(),
                        nodes=self.env.nodes,
                        env_b=self.env_b,
                        out_path=os.path.join(self.visual_dir, f"deploy_t_{step:05d}.png"),
                        title=f"FLOWRRA RL: Deployment Step {step}"
                    )
                except Exception as e:
                    logger.debug(f"Deployment visualization failed at step {step}: {e}")

            if done:
                logger.info(f"Deployment ended at step {step}")
                break
