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
            writer.writerow(['t', 'coherence', 'reward', 'collapse_event'])

    def _log_step(self, coherence: float, reward: float, collapse_event: bool):
        """Writes a single row to the CSV log."""
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([self.env.t, coherence, reward, int(collapse_event)])

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

    def generate_actions(self, rotation_substeps: int = 4) -> list[dict[str, any]]:
        """
        Generates actions for each node to maintain the loop structure,
        using sub-steps to ensure fluid rotation.
        """
        
        # This will contain the final actions after all substeps
        final_actions = []

        for _ in range(rotation_substeps):
            num_nodes = len(self.env.nodes)
            
            for i in range(num_nodes):
                curr_node = self.env.nodes[i]
                next_node = self.env.nodes[(i + 1) % num_nodes]
                
                # 1. Calculate the target angle towards the next node
                target_angle_idx = self._calculate_angle_idx(curr_node.pos, next_node.pos)

                # 2. Update the node's eye to rotate towards the target
                # This logic is what was in your original Environment_A.step method
                diff = (target_angle_idx - curr_node.angle_idx) % self.env.angle_steps
                if diff > self.env.angle_steps // 2:
                    diff -= self.env.angle_steps
                
                if abs(diff) > curr_node.rotation_speed:
                    if diff > 0:
                        curr_node.angle_idx = (curr_node.angle_idx + curr_node.rotation_speed) % self.env.angle_steps
                    else:
                        curr_node.angle_idx = (curr_node.angle_idx - curr_node.rotation_speed) % self.env.angle_steps
                else:
                    curr_node.angle_idx = target_angle_idx
            
            # Since the position and velocity are not updated within this substep loop,
            # we simply let the rotation happen. The move action will be determined
            # after these sub-rotations.

        # 3. Finally, generate the move action based on the rotated eye direction
        for i in range(num_nodes):
            final_actions.append({
                'move_vector': self.get_move_vector_from_eye_angle(self.env.nodes[i].angle_idx)
            })

        return final_actions

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

    def STEP(self, visualize: bool = False):
        """
        Executes a single, complete timestep of the FLOWRRA simulation.
        """
        # Step the external environment to move obstacles
        self.env_b.step()

        # 1. SENSE: Each node senses other nodes
        all_detections = []
        for node in self.env.nodes:
            detections = self.env.node_sensor.sense(node, self.env.nodes)
            all_detections.extend(detections)

        # Sense external obstacles from EnvironmentB
        obstacle_states = self.env_b.get_obstacle_states()
        sensor_range = self.env.node_sensor.max_range
        for node in self.env.nodes:
            for i, obs in enumerate(obstacle_states):
                dist = np.linalg.norm(node.pos - obs['pos'])
                if dist < sensor_range:
                    # Create a detection that mimics the NodeSensor output
                    all_detections.append({
                        'id': f"obs_{i}", # Unique obstacle ID
                        'pos': obs['pos'],
                        'velocity': obs['velocity'],
                        'signal': 2.5 / (dist + 1e-6), # Stronger signal for obstacles
                    })

        # 2. UPDATE DENSITY: Update the repulsion field based on new sensory data
        self.density_estimator.update_from_detections(all_detections)
        self.density_estimator.step_dynamics() # Apply decay and blur

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
            # Optionally, you could add a "repulsion scar" here from the failing state

        # 5. LOGGING
        self._log_step(coherence, reward, collapse_event)
        if visualize:
            plot_system_state(
                self.density_estimator.repulsion_field,
                self.env.nodes,
                self.env_b,
                out_path=os.path.join(self.visual_dir, f"t_{self.env.t:05d}.png"),
                title=f"t={self.env.t}, Coherence={coherence:.3f}"
            )

        return {'coherence': coherence, 'reward': reward, 'collapse': collapse_event}

