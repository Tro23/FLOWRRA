"""
FLOWRRA.py

The main orchestrator class for the FLOWRRA system. It initializes all the
sub-modules (Environment, DensityEstimator, WFC) and runs the main simulation
loop. It connects the different components, for example, by feeding sensor data
from the Environment into the DensityEstimator and using the resulting coherence
to inform the WFC module.
"""
import logging
import csv
import numpy as np
import os
from NodePosition import Node_Position
from NodeSensor import Node_Sensor
from EnvironmentA import Environment_A
from EnvironmentB import EnvironmentB
from DensityFunctionEstimator import Density_Function_Estimator
from WaveFunctionCollapse import Wave_Function_Collapse
from utils import plot_system_state

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('FLOWRRA')

class Flowrra:
    """
    The main class that integrates all FLOWRRA v2 components.
    """
    def __init__(self, config: dict):
        self.config = config
        self.env = Environment_A(
            num_nodes=config.get('num_nodes', 10),
            seed=config.get('seed', None)
        )
        self.env_b = EnvironmentB(
            grid_size=config.get('env_b_grid_size', 40),
            num_fixed=config.get('env_b_num_fixed', 20),
            num_moving=config.get('env_b_num_moving', 8)
        )
        self.density_estimator = Density_Function_Estimator(
            grid_shape=config.get('grid_size', (64, 64)),
            eta=config.get('eta', 0.1),
            gamma_f=config.get('gamma_f', 0.8),
            k_f=config.get('k_f', 5),
            sigma_f=config.get('sigma_f', 2.5),
            decay_lambda=config.get('decay_lambda', 0.01)
        )
        self.wfc = Wave_Function_Collapse(
            history_length=config.get('history_length', 200),
            tail_length=config.get('tail_length', 15),
            collapse_threshold=config.get('collapse_threshold', 0.2),
            tau=config.get('tau', 5)
        )

        # --- Logging and Visualization Setup ---
        self.log_file = config.get('logfile', 'flowrra_log.csv')
        self.visual_dir = config.get('visual_dir', 'flowrra_visuals')
        os.makedirs(self.visual_dir, exist_ok=True)
        self._setup_csv_log()

    def _setup_csv_log(self):
        """Initializes the CSV log file with headers."""
        with open(self.log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            # Dynamic header generation based on the number of nodes
            node_headers = []
            for i in range(self.env.num_nodes):
                node_headers.extend([f'node_{i}_x', f'node_{i}_y'])
            
            headers = ['t', 'coherence', 'reward', 'collapse_event'] + node_headers
            writer.writerow(headers)

    def _log_step(self, coherence: float, reward: float, collapse_event: bool):
        """Writes a single row to the CSV log."""
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            # Retrieve flattened node positions
            node_positions = self.env.get_node_positions_flat()
            
            row = [self.env.t, coherence, reward, int(collapse_event)] + node_positions
            writer.writerow(row)

    def _calculate_angle_idx(self, p1: np.ndarray, p2: np.ndarray) -> int:
        """Helper to calculate the discrete angle from p1 to p2."""
        angle_steps = self.env.angle_steps
        dx, dy = p2 - p1
        
        # Adjust for toroidal boundaries
        if abs(dx) > 0.5: dx = dx - np.sign(dx)
        if abs(dy) > 0.5: dy = dy - np.sign(dy)
            
        angle_rad = np.arctan2(dy, dx)
        angle_degrees = np.degrees(angle_rad)
        
        # Convert to discrete index
        angle_idx = int((angle_degrees / 360) * angle_steps)
        return (angle_idx + angle_steps) % angle_steps

    def get_move_vector_from_eye_angle(self, eye_angle_idx: int) -> np.ndarray:
        """Converts the discrete eye angle to a continuous movement vector."""
        angle_rad = (eye_angle_idx / self.env.angle_steps) * 2 * np.pi
        return np.array([np.cos(angle_rad), np.sin(angle_rad)])

    def generate_actions(self) -> list[dict[str, any]]:
        """
        Generates actions (move, rotate) for each node based on the repulsion field
        and adds a small random walk to ensure constant movement.
        """
        actions = []
        for node in self.env.nodes:
            pos = node.pos
            
            # --- 1. Repulsion Force from Density Field (as before) ---
            epsilon = 1e-4
            current_pot = self.density_estimator.get_potential_at_positions(np.array([pos]))[0]
            pot_x = self.density_estimator.get_potential_at_positions(np.array([pos + np.array([epsilon, 0.0])]))[0]
            pot_y = self.density_estimator.get_potential_at_positions(np.array([pos + np.array([0.0, epsilon])]))[0]
            
            force_x = (pot_x - current_pot) / epsilon
            force_y = (pot_y - current_pot) / epsilon
            repulsion_force_vector = np.array([force_x, force_y])
            
            # Move *away* from the repulsion
            move_vector = -repulsion_force_vector
            
            # --- 2. Add the Random Walk (U_pos) ---
            # This is the new, simple movement component
            # A small random vector that ensures movement even when repulsion is zero.
            random_walk_vector = (np.random.rand(2) - 0.5) * 0.01 
            move_vector += random_walk_vector
            
            # --- 3. Add an Attraction Force to the Center (as before) ---
            center_of_mass = np.mean([n.pos for n in self.env.nodes], axis=0)
            attraction_vector = center_of_mass - pos
            move_vector += 0.05 * attraction_vector
            
            # Clamp the final movement vector to the node's maximum speed
            max_move_speed = np.mean(node.move_speed)
            move_magnitude = np.linalg.norm(move_vector)
            if move_magnitude > max_move_speed:
                move_vector = (move_vector / move_magnitude) * max_move_speed

            actions.append({
                'move': move_vector,
                'target_angle_idx': None
            })
        return actions

    def compute_coherence(self) -> float:
        """
        Computes the coherence metric based on the average density of the nodes.
        Higher coherence means nodes are in higher-density (lower potential) regions.
        """
        current_node_positions = np.array([node.pos for node in self.env.nodes])
        
        # Call the new method from the density estimator
        # This is where the fix is implemented
        densities = self.density_estimator.get_density_at_positions(current_node_positions)

        # The coherence metric is the average density of the nodes, as per the design doc
        coherence = np.mean(densities)
        return float(coherence)

    def compute_reward(self, coherence: float) -> float:
        """
        Computes a reward signal for the current step.
        Higher coherence is good. Being too close to other nodes is bad.
        """
        reward = coherence

        # Penalty for being too close to other nodes
        positions = np.array([n.pos for n in self.env.nodes])
        dists = np.linalg.norm(positions[:, np.newaxis, :] - positions[np.newaxis, :, :], axis=-1)
        np.fill_diagonal(dists, np.inf)
        min_dist = np.min(dists)

        if min_dist < 0.03: # Collision penalty
            reward -= 1.0

        # Penalty for collisions with external obstacles
        obstacle_states = self.env_b.get_obstacle_states()
        if obstacle_states:
            obstacle_positions = np.array([s['pos'] for s in obstacle_states])
            # Calculate distance from each node to each obstacle
            node_obs_dists = np.linalg.norm(positions[:, np.newaxis, :] - obstacle_positions[np.newaxis, :, :], axis=-1)
            min_node_obs_dist = np.min(node_obs_dists)
            if min_node_obs_dist < 0.03: # Obstacle collision penalty
                reward -= 1.5 # Make it more severe than inter-node collision

        return reward

    def STEP(self, visualize: bool = False) -> dict:
        """
        Performs a single simulation step.
        """
        # 1. SENSE: Gather all sensor data from the nodes & obstacles
        node_detections = self.env.snapshot_nodes()
        obstacle_detections = self.env_b.get_obstacle_states()
        all_detections = node_detections + obstacle_detections
        
        # 2. PERCEIVE: Update the density field
        self.density_estimator.update_from_detections(all_detections)
        self.density_estimator.step_dynamics()

        # 3. ACT: Generate and apply actions
        actions = self.generate_actions()
        self.env.step(actions)
        
        # 4. EVALUATE & LEARN: Compute coherence and reward, check for collapse
        coherence = self.compute_coherence()
        reward = self.compute_reward(coherence)
        self.wfc.record_step(self.env.snapshot_nodes(), coherence, self.env.t)

        collapse_event = False
        if self.wfc.check_for_collapse():
            collapse_event = True
            logger.warning(f"Collapse triggered at t={self.env.t}! Re-initializing loop.")
            meta = self.wfc.collapse_and_reinitialize(self.env)
            logger.info(f"Re-initialization complete. Method: {meta.get('reinit_from')}")

        # 5. LOGGING
        self._log_step(coherence, reward, collapse_event)
        if visualize:
            plot_system_state(
                self.density_estimator.repulsion_field,
                self.env.nodes,
                self.env_b,
                out_path=os.path.join(self.visual_dir, f"t_{self.env.t:05d}.png"),
                title=f"t={self.env.t}, Coherence={coherence:.2f}"
            )
        
        return {
            'coherence': coherence,
            'reward': reward,
            'collapse_event': collapse_event
        }