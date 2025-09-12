"""
FLOWRRA_RL.py - FIXED VERSION

Fixed reward structure to encourage exploration while maintaining coherence.
Added movement incentives and better reward balance.
"""

import logging
import csv
import os
import numpy as np
import math
import time
from typing import List, Dict, Any, Tuple, Optional
from scipy.stats import entropy

from NodePosition_RL import Node_Position
from EnvironmentA_RL import Environment_A
from EnvironmentB_RL import EnvironmentB
from DensityFunctionEstimator_RL import Density_Function_Estimator
from WaveFunctionCollapse_RL import Wave_Function_Collapse
from RLAgent import SharedRLAgent

logger = logging.getLogger("FLOWRRA_RL")
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(h)
logger.setLevel(logging.INFO)

class Flowrra_RL:
    """
    Main orchestrator with FIXED reward structure to encourage movement and exploration.
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
            collapse_threshold=config.get('wfc_collapse_threshold', 0.88),
            tau=config.get('wfc_tau', 2)
        )
        self.agent: Optional[SharedRLAgent] = None
        self.log_file = config.get('logfile', 'flowrra_rl_log.csv')
        self.visual_dir = config.get('visual_dir', 'flowrra_rl_visuals')
        
        self.log_header = ['step', 'episode', 'total_reward', 'avg_coherence', 'exploration_reward', 'loss', 'wfc_reinit']
        
        # State and action configuration
        self.num_nodes = self.env.num_nodes
        self.per_node_state_size = 36
        self.state_size = self.num_nodes * self.per_node_state_size
        self.action_size = 4
        self.angle_action_size = 4
        self.combined_action_size = self.action_size * self.angle_action_size
        
        # NEW: Track previous positions for movement rewards
        self.prev_positions = None
        
        self.reset()
        
    def reset(self):
        """Reset the entire system."""
        self.env.reset()
        self.env_b.reset()
        self.wfc.reset()
        # Reset position tracking
        self.prev_positions = np.array([node.pos.copy() for node in self.env.nodes])

    def get_state(self) -> np.ndarray:
        """Generate the full state vector for the shared RL agent."""
        all_node_states = []
        all_obstacle_states = self.env_b.get_obstacle_states()

        for node in self.env.nodes:
            node_detections = node.sense_nodes(self.env.nodes)
            obstacle_detections = node.sense_obstacles(all_obstacle_states)

            # Sort by distance and pad/truncate to fixed size
            node_detections_sorted = sorted(node_detections, key=lambda x: x['distance'])
            obstacle_detections_sorted = sorted(obstacle_detections, key=lambda x: x['distance'])

            num_neighbors = 2
            num_obstacles = 2

            # Flatten and pad node data
            node_data = []
            for i in range(num_neighbors):
                if i < len(node_detections_sorted):
                    d = node_detections_sorted[i]
                    node_data.extend([d['distance'], d['bearing_rad'], d['velocity'][0], d['velocity'][1]])
                else:
                    node_data.extend([1.0, 0.0, 0.0, 0.0])

            # Flatten and pad obstacle data
            obstacle_data = []
            for i in range(num_obstacles):
                if i < len(obstacle_detections_sorted):
                    d = obstacle_detections_sorted[i]
                    obstacle_data.extend([d['distance'], d['bearing_rad'], d['velocity'][0], d['velocity'][1]])
                else:
                    obstacle_data.extend([1.0, 0.0, 0.0, 0.0])

            # Get local repulsion potential
            all_detections = node_detections + obstacle_detections
            repulsion_potential = self.density_estimator.get_repulsion_potential_for_node(
                node_pos=node.pos,
                repulsion_sources=all_detections
            )
            
            repulsion_potential_flat = repulsion_potential.flatten()
            
            # Combine all parts
            node_state_vector = np.concatenate([
                node.pos,
                node.velocity(),
                np.array(node_data),
                np.array(obstacle_data),
                repulsion_potential_flat
            ])
            
            all_node_states.append(node_state_vector)

        return np.concatenate(all_node_states).astype(np.float32)

    def calculate_coherence(self) -> float:
        """Calculate system coherence based on repulsion potential."""
        all_node_coherences = []
        all_obstacle_states = self.env_b.get_obstacle_states()

        for node in self.env.nodes:
            all_detections = node.sense_nodes(self.env.nodes) + node.sense_obstacles(all_obstacle_states)
            
            repulsion_potential_local_grid = self.density_estimator.get_repulsion_potential_for_node(
                node_pos=node.pos,
                repulsion_sources=all_detections
            )

            x, y = int(np.clip((node.pos[0] + 0.5) * 4, 0, 3)), int(np.clip((node.pos[1] + 0.5) * 4, 0, 3))
            repulsion_at_node_pos = repulsion_potential_local_grid[y, x]
            
            coherence = 1.0 / (1.0 + repulsion_at_node_pos)
            all_node_coherences.append(coherence)
            
        return np.mean(all_node_coherences)

    def calculate_exploration_reward(self) -> float:
        """
        NEW: Calculate reward for exploration and movement.
        
        Returns:
            float: Exploration reward based on movement and coverage
        """
        if self.prev_positions is None:
            return 0.0
            
        # 1. Movement reward - encourage nodes to actually move
        current_positions = np.array([node.pos.copy() for node in self.env.nodes])
        movements = []
        
        for i in range(self.num_nodes):
            # Calculate toroidal distance moved
            delta = current_positions[i] - self.prev_positions[i]
            toroidal_delta = np.mod(delta + 0.5, 1.0) - 0.5
            movement_distance = np.linalg.norm(toroidal_delta)
            movements.append(movement_distance)
        
        # Reward movement but not too much (balanced exploration)
        movement_reward = np.mean(movements) * 57.0  # Scale up movement reward
        
        # 2. Spread reward - encourage nodes to explore different areas
        # Calculate pairwise distances between nodes
        pairwise_distances = []
        for i in range(self.num_nodes):
            for j in range(i+1, self.num_nodes):
                delta = current_positions[i] - current_positions[j]
                toroidal_delta = np.mod(delta + 0.5, 1.0) - 0.5
                distance = np.linalg.norm(toroidal_delta)
                pairwise_distances.append(distance)
        
        # Reward well-spread configurations (but not too spread)
        avg_distance = np.mean(pairwise_distances)
        optimal_distance = 0.3  # Tunable parameter
        spread_reward = 1.0 - abs(avg_distance - optimal_distance)
        spread_reward = max(0.0, spread_reward)
        
        # 3. Velocity alignment reward (swarm behavior)
        velocities = np.array([node.velocity() for node in self.env.nodes])
        if np.any(np.linalg.norm(velocities, axis=1) > 0.001):  # If there's any movement
            # Normalize velocities
            vel_norms = np.linalg.norm(velocities, axis=1)
            vel_norms[vel_norms == 0] = 1.0  # Avoid division by zero
            normalized_vels = velocities / vel_norms.reshape(-1, 1)
            
            # Calculate alignment (average dot product)
            alignment = 0.0
            count = 0
            for i in range(self.num_nodes):
                for j in range(i+1, self.num_nodes):
                    if vel_norms[i] > 0.001 and vel_norms[j] > 0.001:
                        alignment += np.dot(normalized_vels[i], normalized_vels[j])
                        count += 1
            
            if count > 0:
                alignment /= count
                alignment_reward = max(0.0, alignment) * 0.5
            else:
                alignment_reward = 0.0
        else:
            alignment_reward = 0.0
        
        total_exploration_reward = movement_reward + spread_reward * 1.5 + alignment_reward
        
        # Update previous positions
        self.prev_positions = current_positions.copy()
        
        return total_exploration_reward

    def step(self, actions: List[int]) -> Tuple[np.ndarray, bool, Dict[str, Any]]:
        """
        FIXED: Advanced simulation with improved reward structure.
        """
        info = {'wfc_reinit': 'none'}
        
        # Split actions
        pos_actions = [int(a / self.angle_action_size) for a in actions]
        angle_actions = [a % self.angle_action_size for a in actions]
        
        # STAGE 1: Apply Position Actions
        for node, pos_action in zip(self.env.nodes, pos_actions):
            node.apply_position_action(pos_action)
        
        self.env_b.step()
        
        self.density_estimator.update_from_sensor_data(
            all_nodes=self.env.nodes,
            all_obstacle_states=self.env_b.get_obstacle_states()
        )
        
        coherence_after_pos = self.calculate_coherence()
        exploration_reward = self.calculate_exploration_reward()
        
        # If collapse after position move
        if coherence_after_pos < self.wfc.collapse_threshold:
            reinit_info = self.wfc.collapse_and_reinitialize(self.env, None)
            info['wfc_reinit'] = reinit_info['reinit_from']
            rewards = np.full(self.num_nodes, -2.0)  # Heavy penalty for collapse
            info['exploration_reward'] = 0.0
            return rewards, False, info
        
        # STAGE 2: Apply Angle Actions
        for node, angle_action in zip(self.env.nodes, angle_actions):
            node.apply_angle_action(angle_action)
            
        self.density_estimator.update_from_sensor_data(
            all_nodes=self.env.nodes,
            all_obstacle_states=self.env_b.get_obstacle_states()
        )
        
        final_coherence = self.calculate_coherence()
        
        # FIXED REWARD CALCULATION
        if final_coherence < self.wfc.collapse_threshold:
            reinit_info = self.wfc.collapse_and_reinitialize(self.env, None)
            info['wfc_reinit'] = reinit_info['reinit_from']
            rewards = np.full(self.num_nodes, -3.0)  # Severe penalty
            info['exploration_reward'] = 0.0
        else:
            # Balanced reward: coherence + exploration + small baseline
            coherence_reward = (coherence_after_pos + final_coherence) * 0.5  # Scale down coherence
            base_reward = 0.1  # Small positive baseline to encourage any action
            
            # Combined reward encourages both coherence and exploration
            total_reward = coherence_reward + exploration_reward + base_reward
            rewards = np.full(self.num_nodes, total_reward)
            info['exploration_reward'] = exploration_reward
            
        done = False
        self.env.t += 1
        
        return rewards, done, info
        
    def attach_agent(self, agent: SharedRLAgent):
        """Attach the RL agent to the Flowrra instance."""
        self.agent = agent

    def train(self, total_steps: int, episode_steps: int, visualize_every_n_steps: int, agent: SharedRLAgent):
        """Train the RL agent with improved logging."""
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
            
            if step % episode_steps == 0:
                self.reset()
                
            state = self.get_state()
            actions = self.agent.choose_actions(state, episode_number=episode_num, total_episodes=total_steps // episode_steps)
            
            rewards, done, info = self.step(list(actions))
            
            if info['wfc_reinit'] != 'none':
                next_state = self.get_state()
                logger.info(f"WFC reinit at step {step} | Episode {episode_num}")
            else:
                next_state = self.get_state()
            
            # Store transition
            self.agent.memory.push(state, actions, rewards, next_state, done)
            
            # Update agent
            loss = 0.0
            if len(self.agent.memory) >= self.agent.batch_size:
                loss = self.agent.learn()
                self.agent.update_target_network()
                
            # Enhanced logging
            total_reward = np.sum(rewards)
            avg_coherence = self.calculate_coherence()
            exploration_reward = info.get('exploration_reward', 0.0)
            
            with open(self.log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([step, episode_num, total_reward, avg_coherence, exploration_reward, loss, info['wfc_reinit']])

            if step % 50 == 0:
                logger.info(f"Step {step} | Episode {episode_num} | Coherence: {avg_coherence:.4f} | Exploration: {exploration_reward:.4f} | Loss: {loss:.4f}")

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

    def deploy(self, total_steps: int, visualize_every_n_steps: int, num_episodes: int = 1):
        """Deployment with the attached agent."""
        if self.agent is None:
            raise ValueError("Agent not attached. Call attach_agent() before deploy().")

        logger.info("--- Starting deployment ---")
        
        for episode_num in range(num_episodes):
            logger.info(f"--- Starting Deployment Episode {episode_num + 1}/{num_episodes} ---")
            self.env.reset()
            self.env_b.reset()
            self.wfc.reset()
            
            for step in range(total_steps):
                state = self.get_state()
                # Increased exploration during deployment to see swarm behavior
                actions = self.agent.choose_actions(state, episode_number=episode_num, total_episodes=num_episodes, eps_peak=0.7)
                rewards, done, info = self.step(list(actions))

                if step % visualize_every_n_steps == 0:
                    try:
                        from utils_rl import plot_system_state_rl
                        plot_system_state_rl(
                            density_field=self.density_estimator.get_full_repulsion_grid(),
                            nodes=self.env.nodes,
                            env_b=self.env_b,
                            out_path=os.path.join(self.visual_dir, f"deploy_ep{episode_num:02d}_t_{step:05d}.png"),
                            title=f"FLOWRRA RL: Deployment Ep {episode_num+1} Step {step}"
                        )
                    except Exception as e:
                        logger.debug(f"Deployment visualization failed at step {step}: {e}")
                    
                    time.sleep(1)  # Reduced pause for faster visualization

                if done:
                    logger.info(f"Deployment episode {episode_num + 1} ended at step {step}")
                    break
        
        logger.info("--- Deployment Complete ---")