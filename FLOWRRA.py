import numpy as np
import random
import math
from sklearn.neighbors import KernelDensity
from scipy.stats import entropy as ShannonEntropy

# --- Moved NodePosition and NodeSensor outside the Flowrra class ---
class NodePosition:
    """Data container for a single node's state."""
    def __init__(self, position, eye_angles, rotation_speed, move_speed):
        self.pos = position
        self.eye_angle_idx = eye_angles
        self.rotation_speed = rotation_speed
        self.move_speed = move_speed
        self.target_angle_idx = eye_angles

class NodeSensor:
    """
    Sensor for each node, using Bresenham's raycasting to detect obstacles.
    """
    def __init__(self, max_range, noise_std, false_negative_prob, false_positive_prob):
        self.max_range = max_range
        self.noise_std = noise_std
        self.false_negative_prob = false_negative_prob
        self.false_positive_prob = false_positive_prob

    def _bresenham_line(self, x0, y0, x1, y1):
        """
        Bresenham's line algorithm to get all points on a line segment.
        Yields (x, y) integer tuples.
        """
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        
        while True:
            yield int(x0), int(y0)
            if int(x0) == int(x1) and int(y0) == int(y1):
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy

    def sense(self, node_position, env_b, grid_size):
        """
        Casts a ray from the node's position and returns the distance to the nearest obstacle.
        Adds noise and probabilistic errors.
        """
        x0, y0 = node_position.pos
        eye_angle_rad = node_position.eye_angle_idx * (2 * math.pi / 36)
        
        x1 = x0 + self.max_range * math.cos(eye_angle_rad)
        y1 = y0 + self.max_range * math.sin(eye_angle_rad)

        ray_points = self._bresenham_line(x0, y0, x1, y1)
        
        min_dist = self.max_range
        detected_obstacle = False
        
        # The first point is the node's own position, so skip it.
        next(ray_points)

        for x, y in ray_points:
            if not (0 <= x < grid_size and 0 <= y < grid_size):
                # Ray hit the boundary
                detected_obstacle = True
                dist = math.hypot(x - x0, y - y0)
                min_dist = min(min_dist, dist)
                break
            
            if (x, y) in env_b.all_blocks:
                # Ray hit an obstacle in Environment B
                detected_obstacle = True
                dist = math.hypot(x - x0, y - y0)
                min_dist = min(min_dist, dist)
                break

        # Add noise to the measurement
        measurement = min_dist + random.gauss(0, self.noise_std)
        
        # Simulate probabilistic errors
        if detected_obstacle:
            if random.random() < self.false_negative_prob:
                # False negative: don't report the obstacle
                return self.max_range + random.gauss(0, self.noise_std)
            return measurement
        else:
            if random.random() < self.false_positive_prob:
                # False positive: report a phantom obstacle
                return random.uniform(0, self.max_range) + random.gauss(0, self.noise_std)
            return self.max_range + random.gauss(0, self.noise_std)


class Flowrra:
    """
    The centralized FLOWRRA system that manages multi-agent exploration
    (Environment_B) while maintaining loop coherence (Environment_A).
    """

    class Environment_A:
        """
        The internal loop of agents.
        """
        def __init__(self, node_params, loop_data, grid_size=30, angle_steps=36, NodePositionClass=NodePosition):
            self.grid_size = grid_size
            self.angle_steps = angle_steps
            self.nodes = []
            self.num_nodes = len(loop_data)
            self.NodePositionClass = NodePositionClass # Store a reference to the class
            self._initialize_nodes(loop_data, node_params)

        def _initialize_nodes(self, loop_data, node_params):
            """Initializes node positions and properties from loop data."""
            self.nodes = []
            for i, point in enumerate(loop_data):
                # Now we use the stored class reference
                node_pos = self.NodePositionClass(
                    position=(point[0], point[1]),
                    eye_angles=point[2],
                    rotation_speed=node_params.rotation_speed,
                    move_speed=node_params.move_speed
                )
                self.nodes.append(node_pos)

        def _calculate_angle_idx(self, pos1, pos2):
            """Calculates the angle from pos1 to pos2."""
            dx, dy = pos2[0] - pos1[0], pos2[1] - pos1[1]
            angle = math.degrees(math.atan2(dy, dx))
            if angle < 0:
                angle += 360
            return int(angle * self.angle_steps / 360)

        def step(self):
            """Updates the state of the nodes in the loop."""
            for i in range(self.num_nodes):
                curr_node = self.nodes[i]
                next_node = self.nodes[(i + 1) % self.num_nodes]
                
                # Update target angle to next node in the loop
                curr_node.target_angle_idx = self._calculate_angle_idx(curr_node.pos, next_node.pos)
                
                # Rotate eye towards target angle
                diff = (curr_node.target_angle_idx - curr_node.eye_angle_idx) % self.angle_steps
                if diff > self.angle_steps // 2:
                    diff -= self.angle_steps
                
                if abs(diff) > curr_node.rotation_speed:
                    if diff > 0:
                        curr_node.eye_angle_idx = (curr_node.eye_angle_idx + curr_node.rotation_speed) % self.angle_steps
                    else:
                        curr_node.eye_angle_idx = (curr_node.eye_angle_idx - curr_node.rotation_speed) % self.angle_steps
                else:
                    curr_node.eye_angle_idx = curr_node.target_angle_idx

            return [(node.pos, node.eye_angle_idx) for node in self.nodes]

    class DensityFunctionEstimator:
        """
        Estimates the probability density of coherent states.
        """
        def __init__(self, bandwidth):
            self.kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth)
            self.trained = False

        def fit(self, data):
            self.kde.fit(np.array(data))
            self.trained = True

        def sample(self, num_samples):
            if not self.trained:
                raise ValueError("Density estimator not fitted yet.")
            return self.kde.sample(num_samples)

        def score_samples(self, data_points):
            return self.kde.score_samples(np.array(data_points))



    class WaveFunctionCollapse:
        def __init__(self, estimator, grid_size=30, angle_steps=36):
            self.estimator = estimator
            self.grid_size = grid_size
            self.angle_steps = angle_steps

        def generate_loop(self, positions):
            """Generates a loop by connecting nearest nodes iteratively."""
            if len(positions) == 0:
                return []

            loop = [positions[0]]
            current = positions[0]
            remaining = positions[1:].tolist()

            while remaining:
                distances = [(math.hypot(x - current[0], y - current[1]), [x, y, a]) for x, y, a in remaining]
                if not distances:
                    break
                _, next_point = min(distances)
                loop.append(next_point)
                remaining.remove(next_point)
                current = next_point

            loop.append(loop[0])  # Close the loop
            return np.array(loop)

        def collapse(self, num_nodes=12, last_state_data=None):
            """
            Performs a more intelligent wave function collapse by iteratively building
            a coherent loop from the density estimator.
            """
            # --- Stage 1: Initialize the loop with a starting point ---
            if last_state_data is not None and len(last_state_data) > 0:
                # Use the most recent positions of the last state as a starting point
                start_pos_x, start_pos_y = last_state_data[0][0][0], last_state_data[0][0][1]
                start_angle = last_state_data[0][1]
                loop_nodes = [np.array([start_pos_x, start_pos_y, start_angle])]
            else:
                # Fallback: Sample a random point from the estimator
                samples = self.estimator.sample(1)
                loop_nodes = [samples[0]]
            
            # --- Stage 2: Iteratively add nodes to build the coherent loop ---
            for _ in range(num_nodes - 1):
                last_node = loop_nodes[-1]
            
                # Sample a batch of candidates from the KDE
                candidate_samples = self.estimator.sample(50) 
            
                # Score each candidate
                best_score = -float('inf')
                best_candidate = None
            
                for candidate in candidate_samples:
                    # Calculate distance to the last node
                    distance = np.linalg.norm(candidate[:2] - last_node[:2])
                
                    # Get the KDE score for the candidate
                    kde_score = self.estimator.score_samples([candidate])[0]
                
                    # Combine scores: reward high KDE density and penalize large distances
                    # The weighting (e.g., 5.0) can be tuned to balance coherence and proximity
                    combined_score = kde_score - 5.0 * distance
                
                    if combined_score > best_score:
                        best_score = combined_score
                        best_candidate = candidate
            
                # If a best candidate was found, add it to the loop
                if best_candidate is not None:
                    loop_nodes.append(best_candidate)
                else:
                    # Fallback: If no good candidates found, just sample randomly
                    loop_nodes.append(self.estimator.sample(1)[0])

            # --- Stage 3: Post-processing and finalizing the loop ---
            # Ensure all coordinates and angles are within valid ranges
            final_nodes = np.array(loop_nodes)
            final_nodes[:, 0] = np.clip(final_nodes[:, 0], 0, self.grid_size - 1)
            final_nodes[:, 1] = np.clip(final_nodes[:, 1], 0, self.grid_size - 1)
            final_nodes[:, 2] = np.clip(final_nodes[:, 2], 0, self.angle_steps - 1)
            final_nodes_int = np.round(final_nodes).astype(int)

            return self.generate_loop(final_nodes_int)

    def __init__(self, node_position_config, node_sensor_config, loop_data_config, coherence_score_config, alpha=0.1, gamma=0.95):
        """
        Initializes the FLOWRRA system.
        """
        self.node_pos_config = node_position_config
        self.node_sensor_config = node_sensor_config
        self.loop_data_config = loop_data_config
        self.coherence_score_config = coherence_score_config
        
        self.alpha = alpha
        self.gamma = gamma
        self.num_nodes = len(loop_data_config)

        # Initialize inner components
        self.env_a = self.Environment_A(
            NodePositionClass=NodePosition,
            node_params=self.node_pos_config,
            loop_data=self.loop_data_config,
            grid_size=30,
            angle_steps=36
        )
        self.density_estimator = self.DensityFunctionEstimator(
            bandwidth=1.5
        )
        self.wfc = self.WaveFunctionCollapse(
            estimator=self.density_estimator
        )
        self.node_sensor = NodeSensor(
            max_range=self.node_sensor_config.max_range,
            noise_std=self.node_sensor_config.noise_std,
            false_negative_prob=self.node_sensor_config.false_negative_prob,
            false_positive_prob=self.node_sensor_config.false_positive_prob
        )

        self.q_tables = {i: {} for i in range(self.num_nodes)}
        self.action_space = self._generate_actions()
        
        self.visited_cells_b = set()
        
        self.last_collapse_state = None

    # --- CHANGED ---: The agent's action space is now simplified.
    # It only controls movement (dx, dy), not rotation. This makes the learning
    # problem easier and enforces that eye angles are determined by the loop's structure.
    def _generate_actions(self):
        """Generates the action space, now only for movement."""
        actions = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                actions.append((dx, dy))
        return actions

    def _get_q_state(self, env_a_state_data, env_b_sensor_readings):
        """Maps the complex environment state to a simplified Q-learning state."""
        state = tuple(env_a_state_data) + tuple(env_b_sensor_readings)
        return state

    # --- CHANGED ---: The apply action method no longer handles rotation.
    # The `Environment_A.step()` method is now the sole controller of eye angles,
    # ensuring they always point to the next node in the loop.
    def _apply_action(self, node_index, action):
        """Applies a chosen MOVEMENT action to a single node."""
        node = self.env_a.nodes[node_index]
        dx, dy = action # Action is now just (dx, dy)
        
        # Apply movement
        new_x = int(node.pos[0] + dx)
        new_y = int(node.pos[1] + dy)
        
        # Bounds checking for movement
        if 0 <= new_x < self.env_a.grid_size and 0 <= new_y < self.env_a.grid_size:
            node.pos = (new_x, new_y)

        # Rotation is no longer part of the action.

    def _compute_entropy(self, state_data_list):
        """Computes the Shannon entropy of the loop's eye angles."""
        angles = [angle for (pos, angle) in state_data_list]
        if not angles:
            return 0.0
        
        counts = np.bincount(angles, minlength=self.env_a.angle_steps)
        total_count = np.sum(counts)
        if total_count == 0:
            return 0.0
            
        probs = counts / total_count
        probs = probs[probs > 0]
        
        if len(probs) == 0:
            return 0.0

        return ShannonEntropy(probs, base=2)

    def _score_environment(self, state_data_list):
        """Scores the current environment A loop against the coherent density model."""
        if not self.density_estimator.trained:
            return -np.inf
        data_for_scoring = np.array([[x, y, angle] for (x, y), angle in state_data_list])
        log_likelihoods = self.density_estimator.score_samples(data_for_scoring)
        return np.sum(log_likelihoods)

    def _compute_reward(self, next_state_data, new_cells_visited_count, entropy_threshold):
        """
        Calculates reward based on exploration, coherence, and memory of past failures.
        """
        # 1. Reward from Environment B: Exploration
        exploration_reward = new_cells_visited_count * 0.5

        # 2. Reward/Penalty from Environment A: Coherence and Failure State Avoidance
        next_entropy = self._compute_entropy(next_state_data)
        coherence_reward = 0

        # Priority 1: Check if the agent has returned to the state that previously caused a collapse.
        if self.last_collapse_state is not None and tuple(next_state_data) == self.last_collapse_state:
            coherence_reward = -100.0
            print("!!! Agent returned to a previous collapse state. Applying large penalty. !!!")
            self.last_collapse_state = None 
        
        # Priority 2: Check if the new state breaks coherence (crosses entropy threshold).
        elif next_entropy > entropy_threshold:
            coherence_reward = -20.0
        
        # Priority 3: If coherent, reward the agent for maintaining low entropy.
        else:
            coherence_reward = (entropy_threshold - next_entropy) * 2.0

        return exploration_reward + coherence_reward
    
    def record_collapse_state(self, state_data):
        """Records the state that led to a wave function collapse."""
        self.last_collapse_state = tuple(state_data)
        print(f"Recorded a collapse state. Will now penalize returning to it.")

    def reinitialize_loop(self, new_loop_data):
        """Re-initializes Environment_A with a new coherent loop."""
        self.env_a._initialize_nodes(new_loop_data, self.node_pos_config)
        print("Flowrra internal loop has been re-initialized.")

    def step(self, env_b, epsilon):
        """
        Executes one step of the RL process for the multi-agent system.
        """
        current_env_a_state_data = self.env_a.step()

        env_b_sensor_readings = []
        for node in self.env_a.nodes:
            reading = self.node_sensor.sense(node, env_b, self.env_a.grid_size)
            env_b_sensor_readings.append(reading)

        q_state = self._get_q_state(current_env_a_state_data, env_b_sensor_readings)

        chosen_actions = []
        for node_index in range(self.num_nodes):
            if random.random() < epsilon:
                action = random.choice(self.action_space)
            else:
                if q_state in self.q_tables[node_index] and self.q_tables[node_index][q_state]:
                    q_values = self.q_tables[node_index][q_state]
                    action = max(q_values, key=q_values.get)
                else:
                    action = random.choice(self.action_space)
            chosen_actions.append(action)

        for node_index, action in enumerate(chosen_actions):
            self._apply_action(node_index, action)
            
        next_env_a_state_data = self.env_a.step()

        previous_visited_count = len(self.visited_cells_b)
        for node in self.env_a.nodes:
            self.visited_cells_b.add(node.pos)
        new_cells_visited_count = len(self.visited_cells_b) - previous_visited_count

        reward = self._compute_reward(
            next_env_a_state_data, 
            new_cells_visited_count, 
            self.coherence_score_config.entropy_threshold
        )

        next_q_state = self._get_q_state(next_env_a_state_data, env_b_sensor_readings)
        
        for node_index, action in enumerate(chosen_actions):
            if q_state not in self.q_tables[node_index]:
                self.q_tables[node_index][q_state] = {a: 0.0 for a in self.action_space}
            if next_q_state not in self.q_tables[node_index]:
                self.q_tables[node_index][next_q_state] = {a: 0.0 for a in self.action_space}
                
            old_q_value = self.q_tables[node_index][q_state][action]
            next_max_q = max(self.q_tables[node_index][next_q_state].values())
            
            new_q_value = old_q_value + self.alpha * (reward + self.gamma * next_max_q - old_q_value)
            self.q_tables[node_index][q_state][action] = new_q_value

        return next_env_a_state_data, reward