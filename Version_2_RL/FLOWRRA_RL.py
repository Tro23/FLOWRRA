import logging
import csv
import numpy as np
import os
from NodePosition_RL import Node_Position
from EnvironmentA_RL import Environment_A
from EnvironmentB_RL import EnvironmentB
from DensityFunctionEstimator_RL import Density_Function_Estimator
from WaveFunctionCollapse_RL import Wave_Function_Collapse
from RLAgent import RL_Agent
from utils_rl import plot_system_state_rl, create_gif

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('FLOWRRA_RL')

class Flowrra_RL:
    """
    The main class that integrates all FLOWRRA v2 components with a
    Reinforcement Learning agent.
    """
    def __init__(self, config: dict):
        self.config = config
        self.env = Environment_A(
            num_nodes=config.get('num_nodes', 10),
            seed=config.get('seed', None)
        )
        self.env_b = EnvironmentB(
            grid_size=config.get('env_b_grid_size', 60),
            num_fixed=config.get('env_b_num_fixed', 10),
            num_moving=config.get('env_b_num_moving', 4),
            seed=config.get('seed', None)
        )
        self.density_estimator = Density_Function_Estimator(
            grid_shape=config.get('grid_size', (60, 60)),
            eta=config.get('eta', 0.01),
            gamma_f=config.get('gamma_f', 0.4),
            k_f=config.get('k_f', 4),
            sigma_f=config.get('sigma_f', 2.0),
            decay_lambda=config.get('decay_lambda', 0.003),
            blur_delta=config.get('blur_delta', 0.1)
        )
        self.wfc = Wave_Function_Collapse(
            history_length=config.get('wfc_history_length', 200),
            tail_length=config.get('wfc_tail_length', 15),
            collapse_threshold=config.get('wfc_collapse_threshold', 0.25),
            tau=config.get('wfc_tau', 5)
        )
        
        # RL Specific Initialization
        self.state_size = config.get('state_size', 122)
        self.action_size = config.get('action_size', 3)
        self.rl_agent = RL_Agent(self.state_size, self.action_size)
        self.visual_dir = config.get('visual_dir', 'flowrra_rl_visuals')
        
        if not os.path.exists(self.visual_dir):
            os.makedirs(self.visual_dir)

        self.logfile = config.get('logfile', 'flowrra_rl_log.csv')
        self._setup_log_file()
        
    def _setup_log_file(self):
        """Initializes the CSV log file with headers."""
        with open(self.logfile, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestep', 'episode', 'coherence', 'reward', 'epsilon', 'collapse_event'])

    def _log_step(self, episode: int, coherence: float, reward: float, epsilon: float, collapse_event: bool):
        """Logs a single timestep's data to the CSV file."""
        with open(self.logfile, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([self.env.t, episode, coherence, reward, epsilon, collapse_event])

    '''def _calculate_state_size(self) -> int:
        """
        Calculates the fixed size of the state vector.
        
        State components:
        - Self-node state: pos (2) + velocity (2) = 4
        - Sensed nodes: max_sensed_nodes * (distance (1) + bearing (1) + velocity (2)) = max_sensed_nodes * 4
        - Sensed obstacles: max_sensed_obstacles * (distance (1) + bearing (1) + velocity (2)) = max_sensed_obstacles * 4
        """
        # State Vector Configuration
        max_sensed_nodes: int = 5
        max_sensed_obstacles: int = 5

        self_state_size = 4
        sensed_node_size = max_sensed_nodes * 4
        sensed_obstacle_size = max_sensed_obstacles * 4
        return self_state_size + sensed_node_size + sensed_obstacle_size'''
            
    def _get_state_from_node(self, node: Node_Position, env_b: EnvironmentB, env_a: Environment_A) -> np.ndarray:
        """
        Gathers and processes sensor data from a single node to form a state vector.
        """
        # Get sensor detections for other nodes and obstacles
        sensed_nodes = node.sense_other_nodes(env_a.nodes)
        sensed_obstacles = node.sense_obstacles(env_b.get_obstacle_states())
        
        all_detections = sensed_nodes + sensed_obstacles
        
        # Sort by distance for consistent feature order
        all_detections.sort(key=lambda x: x['distance'])
        
        state = []
        # Pad with zeros to maintain a fixed state size
        for i in range(self.state_size // 6 - 1):
            if i < len(all_detections):
                det = all_detections[i]
                state.extend([
                    det['pos'][0],
                    det['pos'][1],
                    det['distance'],
                    det['bearing_rad'],
                    det['velocity'][0],
                    det['velocity'][1]
                ])
            else:
                state.extend([0.0] * 6) # Pad with zeros

        # Add the node's own velocity as part of the state
        node_velocity = node.velocity()
        state.extend([node_velocity[0], node_velocity[1]])
        
        return np.array(state)

    def _compute_reward(self, node: Node_Position, new_pos: np.ndarray, collision: bool, step: int) -> float:
        """
        Computes the reward for a single node's action.
        """
        reward = -0.01
        if collision:
            reward -= 5.0
        distances_to_others = [np.linalg.norm(new_pos) - n.pos for n in self.env.nodes if n.id != node.id]
        if distances_to_others:
            avg_distance = np.mean(distances_to_others)
            
            # Penalize being too close (reinforces repulsive field)
            if avg_distance < 0.2:
                reward -= 0.5
            # Penalize being too far (encourages cohesion)
            elif avg_distance > 0.6:  # Updated cohesion threshold
                reward -= 0.5
            # Reward for being in the ideal range
            else:
                reward += 0.5
        return reward
        
    def train(self, total_steps: int, episode_steps: int, visualize_every_n_steps: int):
        """
        The main training loop for the RL model.
        """
        bell_curve_epsilon = lambda t, total_t: 0.1 + 0.9 * np.exp(-0.5 * ((t - total_t / 2) / (total_t / 6))**2)

        for episode in range(total_steps // episode_steps):
            logger.info(f"--- Starting Episode {episode+1} ---")
            
            self.env.reset()
            self.env_b.reset()
            self.density_estimator.reset()
            self.wfc.reset()
            
            for step in range(episode_steps):
                global_step = episode * episode_steps + step
                epsilon = bell_curve_epsilon(global_step, total_steps)
                
                # SENSE & PLAN
                current_states = [self._get_state_from_node(node, self.env_b, self.env) for node in self.env.nodes]
                actions = [self.rl_agent.choose_action(state, epsilon) for state in current_states]

                # ACT
                flowrra_actions = []
                for i, action_idx in enumerate(actions):
                    target_angle = self.env.nodes[i].angle_idx
                    if action_idx == 0: target_angle -= 1
                    elif action_idx == 1: target_angle += 1
                    flowrra_actions.append({'target_angle_idx': target_angle})

                self.env.step(flowrra_actions)
                self.env_b.step()
                self.density_estimator.step_dynamics()
                
                # Check for repulsion-based collapse
                collapse_event = False
                
                # Get the repulsion potential at each node's new position
                node_positions = np.array([node.pos for node in self.env.nodes])
                repulsion_potentials = self.density_estimator.get_potential_at_positions(node_positions)
                
                if np.any(repulsion_potentials > self.config['repulsion_collapse_threshold']):
                    collapse_event = True
                    logger.warning(f"Repulsion-based collapse triggered at t={self.env.t}! Re-initializing loop.")
                    meta = self.wfc.collapse_and_reinitialize(self.env)
                    logger.info(f"Re-initialization complete. Method: {meta.get('reinit_from')}")
                    
                    # For a clean restart in the RL context, we can end the episode here
                    break

                # EVALUATE & LEARN
                new_states = [self._get_state_from_node(node, self.env_b, self.env) for node in self.env.nodes]
                collision_detected = False
                for node in self.env.nodes:
                    node_pos_grid = np.floor(node.pos * self.env_b.grid_size).astype(int)
                    if (node_pos_grid[0], node_pos_grid[1]) in self.env_b.all_blocks:
                        collision_detected = True
                        break

                rewards = [self._compute_reward(node, new_pos, collision_detected, global_step) for node, new_pos in zip(self.env.nodes, new_states)]

                for i in range(len(self.env.nodes)):
                    done = collision_detected
                    self.rl_agent.replay_buffer.push(current_states[i], actions[i], rewards[i], new_states[i], done)
                    
                self.rl_agent.train(self.config.get('batch_size', 32))
                if global_step % self.config.get('target_update_freq', 100) == 0:
                    self.rl_agent.update_target_network()

                # LOGGING & VISUALIZATION
                coherence = self.wfc.compute_coherence(self.env.nodes)
                self.wfc.record_step(self.env.snapshot_nodes(), coherence, self.env.t)
                self._log_step(episode, coherence, np.sum(rewards), epsilon, collapse_event)
                
                if global_step % visualize_every_n_steps == 0:
                     plot_system_state_rl(
                        density_field=self.density_estimator.repulsion_field,
                        nodes=self.env.nodes,
                        env_b=self.env_b,
                        out_path=os.path.join(self.visual_dir, f"t_{global_step:05d}.png"),
                        title=f"FLOWRRA RL: Step {global_step} | Epsilon: {epsilon:.2f}"
                    )
            
            logger.info(f"Episode {episode+1} complete.")
            
        logger.info("Training complete. Saving model.")
        self.rl_agent.save_model("flowrra_rl_model.pth")
    
    def deploy(self, total_steps: int, visualize_every_n_steps: int):
        """
        Runs the simulation in a deployed (greedy) mode after training.
        """
        logger.info("--- Starting FLOWRRA RL Deployment ---")
        self.rl_agent.load_model("flowrra_rl_model.pth")
        
        self.env.reset()
        self.env_b.reset()
        self.density_estimator.reset()
        self.wfc.reset()
        
        epsilon = 0.0
        
        for step in range(total_steps):
            current_states = [self._get_state_from_node(node, self.env_b, self.env) for node in self.env.nodes]
            actions = [self.rl_agent.choose_action(state, epsilon) for state in current_states]
            
            flowrra_actions = []
            for i, action_idx in enumerate(actions):
                target_angle = self.env.nodes[i].angle_idx
                if action_idx == 0: target_angle -= 1
                elif action_idx == 1: target_angle += 1
                flowrra_actions.append({'target_angle_idx': target_angle})

            self.env.step(flowrra_actions)
            self.env_b.step()
            self.density_estimator.step_dynamics()
            
            # Note: No collapse logic in deployment, as the agent is expected to have learned to avoid bad areas.
            
            if step % visualize_every_n_steps == 0:
                plot_system_state_rl(
                    density_field=self.density_estimator.repulsion_field,
                    nodes=self.env.nodes,
                    env_b=self.env_b,
                    out_path=os.path.join(self.visual_dir, f"deploy_t_{step:05d}.png"),
                    title=f"FLOWRRA RL: Deployment Step {step}"
                )