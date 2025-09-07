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
            grid_size=config.get('env_b_grid_size', 60),
            num_fixed=config.get('env_b_num_fixed', 10),
            num_moving=config.get('env_b_num_moving', 4),
            seed=config.get('seed', None)
        )
        # --- MODIFIED: Pass angle_steps and beta from config to the estimator ---
        # This ensures that parameters are consistently used across modules.
        self.density_estimator = Density_Function_Estimator(
            grid_shape=config.get('grid_size', (60, 60)),
            eta=config.get('eta', 0.1),
            gamma_f=config.get('gamma_f', 4),
            k_f=config.get('k_f', 5),
            sigma_f=config.get('sigma_f', 2.5),
            decay_lambda=config.get('decay_lambda', 0.01),
            angle_steps=self.env.angle_steps,
            beta=config.get('beta', 0.8)
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
        """Helper to calculate the discrete angle from p1 to p2, handling toroidal space."""
        angle_steps = self.env.angle_steps
        delta = p2 - p1
        
        # Adjust for toroidal boundaries (shortest path wrap-around)
        delta[delta > 0.5] -= 1.0
        delta[delta < -0.5] += 1.0
            
        angle_rad = np.arctan2(delta[1], delta[0])
        
        # Convert rad to discrete index
        angle_idx = int(np.round((angle_rad / (2 * np.pi)) * angle_steps))
        return angle_idx % angle_steps

    # --- REWRITTEN: generate_actions method ---
    # This is a major change to address the core issues of movement and rotation.
    def generate_actions(self) -> list[dict[str, any]]:
        """
        Generates actions (move, rotate) for each node based on various forces.
        - Repulsion Force: Pushes nodes away from high-density areas (obstacles, other nodes).
        - Attraction Force: A gentle pull towards the center of the loop to maintain cohesion.
        - Random Walk: Ensures continuous micro-movements to prevent stagnation.
        - Loop Maintenance Rotation: Orients nodes to face their next neighbor in the sequence.
        """
        actions = []
        num_nodes = len(self.env.nodes)
        if num_nodes == 0:
            return []
    
        # Calculate center of mass with toroidal wrapping consideration
        positions = np.array([n.pos for n in self.env.nodes])
    
        # Use toroidal mean calculation
        angles_x = positions[:, 0] * 2 * np.pi
        angles_y = positions[:, 1] * 2 * np.pi
        mean_x = np.arctan2(np.mean(np.sin(angles_x)), np.mean(np.cos(angles_x))) / (2 * np.pi)
        mean_y = np.arctan2(np.mean(np.sin(angles_y)), np.mean(np.cos(angles_y))) / (2 * np.pi)
        center_of_mass = np.array([mean_x % 1.0, mean_y % 1.0])

        for i, node in enumerate(self.env.nodes):
            pos = node.pos
        
            # --- 1. Repulsion Force from Density Field ---
            # Use a larger epsilon for more stable gradient calculation
            epsilon = 0.002
            current_pot = self.density_estimator.get_potential_at_positions(np.array([pos]))[0]
        
            # Calculate gradient with toroidal wrapping
            pos_x_plus = np.array([(pos[0] + epsilon) % 1.0, pos[1]])
            pos_y_plus = np.array([pos[0], (pos[1] + epsilon) % 1.0])
        
            pot_x_plus = self.density_estimator.get_potential_at_positions(np.array([pos_x_plus]))[0]
            pot_y_plus = self.density_estimator.get_potential_at_positions(np.array([pos_y_plus]))[0]
        
            grad_x = (pot_x_plus - current_pot) / epsilon
            grad_y = (pot_y_plus - current_pot) / epsilon
            repulsion_force = -np.array([grad_x, grad_y])
        
        
            # --- 3. Separation Force (prevent nodes from getting too close) ---
            separation_force = np.zeros(2)
            for j, other_node in enumerate(self.env.nodes):
                if i == j:
                    continue
                diff = other_node.pos - pos
                # Handle toroidal distance
                if diff[0] > 0.5:
                    diff[0] -= 1.0
                elif diff[0] < -0.5:
                    diff[0] += 1.0
                if diff[1] > 0.5:
                    diff[1] -= 1.0
                elif diff[1] < -0.5:
                    diff[1] += 1.0
            
                dist = np.linalg.norm(diff)
                if dist < 0.05 and dist > 0:  # Too close!
                    # Push away with inverse square law
                    separation_force -= (diff / dist) * (0.01 / (dist ** 2))
        
            # --- 4. Random Walk ---
            random_walk_vector = (np.random.rand(2) - 0.5) * 0.9
        
            # --- 5. Combine Forces with adjusted coefficients ---
            # Reduced attraction, increased repulsion influence
            move_vector = (
                0.25 * repulsion_force +      # Increased repulsion weight
                0.25 * separation_force +      # Strong separation to prevent collapse
                0.75 * random_walk_vector
            )
        
            # --- 6. Rotation to Maintain Loop Structure ---
            next_node = self.env.nodes[(i + 1) % num_nodes]
            target_angle_idx = self._calculate_angle_idx(node.pos, next_node.pos)

            # --- 7. Clamp Final Movement to Node's Maximum Speed ---
            max_move_speed = np.mean(node.move_speed)
            move_magnitude = np.linalg.norm(move_vector)
            if move_magnitude > max_move_speed:
                move_vector = (move_vector / move_magnitude) * max_move_speed

            actions.append({
                'move': move_vector,
                'target_angle_idx': target_angle_idx
            })
        return actions

    def compute_coherence(self) -> float:
        """
        Computes coherence as the average density at node positions.
        High coherence = nodes are in low-repulsion (good) areas.
        """
        if len(self.env.nodes) == 0:
            return 0.0
            
        current_node_positions = np.array([node.pos for node in self.env.nodes])
        
        # Get densities (high density = good areas)
        densities = self.density_estimator.get_density_at_positions(current_node_positions)
        
        # Coherence is the mean density
        coherence = float(np.mean(densities))
        
        # Add a penalty if nodes are too clustered
        distances = []
        for i in range(len(self.env.nodes)):
            for j in range(i+1, len(self.env.nodes)):
                diff = self.env.nodes[j].pos - self.env.nodes[i].pos
                # Handle toroidal wrapping
                diff[diff > 0.5] -= 1.0
                diff[diff < -0.5] += 1.0
                distances.append(np.linalg.norm(diff))
        
        if distances:
            min_dist = min(distances)
            if min_dist < 0.02:  # Too close!
                coherence *= 0.1  # Severe penalty
            elif min_dist < 0.05:
                coherence *= 0.5  # Moderate penalty
        
        return coherence

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
        if dists.size > 0:
            min_dist = np.min(dists)
            if min_dist < 0.03: # Collision penalty
                reward -= 1.0

        # Penalty for collisions with external obstacles
        obstacle_states = self.env_b.get_obstacle_states()
        if obstacle_states:
            obstacle_positions = np.array([s['pos'] for s in obstacle_states])
            node_obs_dists = np.linalg.norm(positions[:, np.newaxis, :] - obstacle_positions[np.newaxis, :, :], axis=-1)
            if node_obs_dists.size > 0:
                min_node_obs_dist = np.min(node_obs_dists)
                if min_node_obs_dist < 0.03: # Obstacle collision penalty
                    reward -= 1.5

        return reward

    def STEP(self, visualize: bool = False) -> dict:
        """
        Performs a single simulation step.
        """
        # --- MODIFIED: Proper SENSE step using Node_Sensor ---
        # 1. SENSE: Gather all sensor data from the nodes & obstacles.
        # This now correctly uses the Node_Sensor class to simulate imperfect sensing.
        all_node_detections = []
        for node in self.env.nodes:
            detections = self.env.node_sensor.sense(node, self.env.nodes)
            all_node_detections.extend(detections)

        # Remove duplicate detections (e.g., node A sees B, and B sees A)
        unique_node_detections = {det['id']: det for det in all_node_detections if det.get('id', -1) != -1}
        phantom_detections = [det for det in all_node_detections if det.get('id', -1) == -1]
        node_detections = list(unique_node_detections.values()) + phantom_detections
        
        obstacle_detections = self.env_b.get_obstacle_states()
        all_detections = node_detections + obstacle_detections

        # Adaptive feedback to prevent collapse
        current_coherence = self.compute_coherence()
        # If coherence drops below a certain point, increase repulsion
        if current_coherence < 0.5 and self.density_estimator.beta < 0.5:
            self.density_estimator.beta += 0.05
        elif self.density_estimator.beta > 0.8:
            # Decay the beta back to its base value after stability is restored
            self.density_estimator.beta -= 0.01
        
        # 2. PERCEIVE: Update the density field
        self.density_estimator.update_from_detections(all_detections)
        self.density_estimator.step_dynamics()

        # 3. ACT: Generate and apply actions
        # The generate_actions method is now much more sophisticated.
        actions = self.generate_actions()
        self.env.step(actions)
        self.env_b.step() # Also step the external environment
        
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
