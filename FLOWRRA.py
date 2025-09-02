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
        """
        Collapses the 'wave function' of the loop to a coherent state.
        Now collapses to a state in the neighborhood of the previous state.
        """
        def __init__(self, density_estimator, env_a):
            self.estimator = density_estimator
            self.env_a = env_a
            self.grid_size = self.env_a.grid_size
            self.angle_steps = self.env_a.angle_steps
            
        def generate_loop(self, positions):
            """Generates a loop by connecting nearest nodes iteratively."""
            if len(positions) == 0:
                return []
            
            # Sort positions to form a coherent loop
            positions_list = positions.tolist()
            loop = [positions_list.pop(0)]
            
            while positions_list:
                current_point = loop[-1]
                distances = [(math.hypot(x - current_point[0], y - current_point[1]), point) 
                             for point in positions_list for x, y, _ in [point]]
                
                min_dist_point = min(distances)[1]
                loop.append(min_dist_point)
                positions_list.remove(min_dist_point)

            return np.array(loop)

        def collapse(self, num_nodes, last_state_data):
            """
            Performs wave function collapse by sampling and looping.
            The collapse is biased towards the neighborhood of the last state.
            """
            # Step 1: Sample a large pool of potential new states
            num_samples = 1000 # Sample a larger pool to find a good candidate
            samples = self.estimator.sample(num_samples)
            
            # Step 2: Calculate the "neighborhood distance" for each sample
            last_state_array = np.array([p[0] for p in last_state_data])
            
            candidate_distances = []
            for i in range(num_samples):
                candidate_pos = samples[i, :2].reshape(1, -1)
                
                # Check for the closest node in the previous state to the current sample
                distances_to_last_state = np.linalg.norm(last_state_array - candidate_pos, axis=1)
                min_dist = np.min(distances_to_last_state)
                candidate_distances.append(min_dist)

            # Step 3: Select the best sample based on neighborhood proximity
            best_sample_idx = np.argmin(candidate_distances)
            best_sample = samples[best_sample_idx]
            
            # Step 4: Generate a new coherent loop from the best sample
            # The loop generation logic remains the same
            best_sample[0] = np.clip(best_sample[0], 0, self.grid_size - 1)
            best_sample[1] = np.clip(best_sample[1], 0, self.grid_size - 1)
            best_sample[2] = np.clip(best_sample[2], 0, self.angle_steps - 1)

            # Generate a full loop based on the chosen sample
            # This is a simplification. A more complex approach would involve
            # a multi-point collapse. For now, we take one "anchor" point
            # and re-generate the loop from a new set of samples.
            new_samples = self.estimator.sample(num_nodes)
            new_samples[:, 0] = np.clip(new_samples[:, 0], 0, self.grid_size - 1)
            new_samples[:, 1] = np.clip(new_samples[:, 1], 0, self.grid_size - 1)
            new_samples[:, 2] = np.clip(new_samples[:, 2], 0, self.angle_steps - 1)
            
            return self.generate_loop(np.round(new_samples).astype(int))

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
            NodePositionClass=NodePosition,  # Pass the top-level class
            node_params=self.node_pos_config,
            loop_data=self.loop_data_config,
            grid_size=30,
            angle_steps=36
        )
        self.density_estimator = self.DensityFunctionEstimator(
            bandwidth=1.5
        )
        self.wfc = self.WaveFunctionCollapse(
            density_estimator=self.density_estimator,
            env_a=self.env_a
        )
        # Use the top-level NodeSensor class
        self.node_sensor = NodeSensor(
            max_range=self.node_sensor_config.max_range,
            noise_std=self.node_sensor_config.noise_std,
            false_negative_prob=self.node_sensor_config.false_negative_prob,
            false_positive_prob=self.node_sensor_config.false_positive_prob
        )

        # Q-learning tables for each node
        self.q_tables = {i: {} for i in range(self.num_nodes)}
        self.action_space = self._generate_actions()

    def _generate_actions(self):
        # A tuple of (move_x, move_y, rotate_steps)
        # Move actions: 0, -1, 1 for x and y
        # Rotate actions: -1, 0, 1 for rotation steps
        # This is a simplified action space.
        actions = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dr in [-1, 0, 1]:
                    actions.append((dx, dy, dr))
        return actions

    def _get_q_state(self, env_a_state_data, env_b_sensor_readings):
        """
        Maps the complex environment state to a simplified Q-learning state.
        This is a crucial design choice for RL.
        """
        # A simple state representation could be the positions of all agents,
        # plus the sensor readings for all agents.
        # This will create a very large state space, which is a known challenge.
        # For simplicity, we'll use a tuple of tuples.
        state = tuple(env_a_state_data) + tuple(env_b_sensor_readings)
        return state

    def _apply_action(self, node_index, action):
        """Applies a chosen action to a single node."""
        node = self.env_a.nodes[node_index]
        dx, dy, dr = action
        
        # Apply movement
        new_x = int(node.pos[0] + dx)
        new_y = int(node.pos[1] + dy)
        
        # Bounds checking for movement
        if 0 <= new_x < self.env_a.grid_size and 0 <= new_y < self.env_a.grid_size:
            node.pos = (new_x, new_y)

        # Apply rotation
        node.eye_angle_idx = (node.eye_angle_idx + dr) % self.env_a.angle_steps

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
        """
        Scores the current environment A loop against the coherent density model.
        Returns the log-likelihood sum.
        """
        data_for_scoring = np.array([[x, y, angle] for (x, y), angle in state_data_list])
        log_likelihoods = self.density_estimator.score_samples(data_for_scoring)
        return np.sum(log_likelihoods)

    def _compute_reward(self, current_state_data, next_state_data, visited_cells_b, entropy_threshold):
        """
        Calculates the reward based on exploration of Environment B and
        coherence of Environment A.
        """
        # Reward from Environment B: Exploration
        new_visited_cells = len(visited_cells_b)
        exploration_reward = new_visited_cells * 0.1 # Small reward for each new cell visited

        # Reward/Penalty from Environment A: Coherence
        current_entropy = self._compute_entropy(current_state_data)
        next_entropy = self._compute_entropy(next_state_data)
        
        coherence_reward = 0
        if next_entropy > entropy_threshold:
            coherence_reward = -10 # Large penalty for high entropy
        elif next_entropy < current_entropy:
            coherence_reward = 1 # Small positive reward for decreasing entropy
        else:
            coherence_reward = -0.1 # Small penalty for increasing or stable entropy

        return exploration_reward + coherence_reward

    def step(self, env_b, visited_cells_b, epsilon):
        """
        Executes one step of the RL process for the multi-agent system.
        """
        # Get current state from Environment A
        current_env_a_state_data = self.env_a.step()

        # Each node performs a sensor reading from Environment B
        env_b_sensor_readings = []
        for node in self.env_a.nodes:
            reading = self.node_sensor.sense(node, env_b, self.env_a.grid_size)
            env_b_sensor_readings.append(reading)

        # Combine states for Q-learning
        q_state = self._get_q_state(current_env_a_state_data, env_b_sensor_readings)

        # Epsilon-greedy action selection for each node
        chosen_actions = []
        for node_index in range(self.num_nodes):
            if random.random() < epsilon:
                action = random.choice(self.action_space)  # Explore
            else:
                # Exploit
                if q_state in self.q_tables[node_index]:
                    q_values = self.q_tables[node_index][q_state]
                    action = max(q_values, key=q_values.get)
                else:
                    action = random.choice(self.action_space) # No Q-value yet, explore
            chosen_actions.append(action)

        # Apply chosen actions to Environment A
        for node_index, action in enumerate(chosen_actions):
            self._apply_action(node_index, action)
            
        # Update Environment A's state and get new data for reward calculation
        next_env_a_state_data = self.env_a.step()

        # Update visited cells in Environment B for reward
        for node in self.env_a.nodes:
            visited_cells_b.add(node.pos)

        # Compute reward
        reward = self._compute_reward(current_env_a_state_data, next_env_a_state_data, visited_cells_b, self.coherence_score_config.entropy_threshold)

        # Get next state and update Q-table
        next_q_state = self._get_q_state(next_env_a_state_data, env_b_sensor_readings)
        
        for node_index, action in enumerate(chosen_actions):
            # Q-table update using the centralized reward
            if q_state not in self.q_tables[node_index]:
                self.q_tables[node_index][q_state] = {a: 0 for a in self.action_space}
            if next_q_state not in self.q_tables[node_index]:
                self.q_tables[node_index][next_q_state] = {a: 0 for a in self.action_space}
                
            old_q_value = self.q_tables[node_index][q_state][action]
            next_max_q = max(self.q_tables[node_index][next_q_state].values())
            
            new_q_value = old_q_value + self.alpha * (reward + self.gamma * next_max_q - old_q_value)
            self.q_tables[node_index][q_state][action] = new_q_value

        return next_env_a_state_data, reward, visited_cells_b

    def warm_up(self, env_b, wfc, num_episodes, steps_per_episode):
        """
        Performs a Q-table warm-up session to pre-populate the tables.
        """
        print("Warming up Q-tables with a short training session...")
        epsilon = 0.6 # Start with pure exploration
        epsilon_decay = 0.6 / (num_episodes * steps_per_episode) # Decay over the session
        
        for episode in range(num_episodes):
            # Re-initialize with a new coherent loop from WFC at the start of each warmup episode
            new_loop = wfc.collapse(num_nodes=self.env_a.num_nodes, last_state_data=self.env_a.nodes)
            self.env_a._initialize_nodes(new_loop.tolist(), self.node_pos_config)

            visited_cells_b = set()
            for step in range(steps_per_episode):
                self.step(env_b, visited_cells_b, epsilon)
                epsilon = max(0.05, epsilon - epsilon_decay) # Decay epsilon

        print(f"Warm-up finished. Q-tables populated with {sum(len(t) for t in self.q_tables.values())} state-action pairs.")
