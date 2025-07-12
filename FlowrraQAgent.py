import numpy as np
import random
from sklearn.neighbors import KernelDensity # Only needed if FlowrraRLAgent itself manages KDE fitting, which it should not for the main estimator.

class FlowrraRLAgent:
    def __init__(self, env_a, env_b, estimator, wfc,
                 angle_steps=36, alpha=0.1, gamma=0.95, entropy_threshold=3.0):
        # Removed 'bandwidth' from init as KDE is handled by the 'estimator' object passed in
        self.env_a = env_a
        self.env_b = env_b
        self.estimator = estimator # This estimator should be pre-fitted on desired coherent data
        self.wfc = wfc

        self.angle_steps = angle_steps
        self.alpha = alpha
        self.gamma = gamma
        self.entropy_threshold = entropy_threshold

        # self.q_tables: key = node_index (int), value = dict of Q[state_tuple][action_tuple] = Q_value
        self.q_tables = {}
        self.action_space = self._generate_actions()

        # Initialize Q-tables for each node using its index (0 to num_nodes-1)
        for i in range(self.env_a.num_nodes):
            self.q_tables[i] = {}

    def _generate_actions(self):
        # Movement actions: (dx, dy)
        moves = [(0, 0), (0, 1), (0, -1), (-1, 0), (1, 0)] # Stay, Up, Down, Left, Right
        # Rotation actions: (d_theta)
        rots = [0, -1, 1] # Stay, Rotate Left, Rotate Right
        # Combine into (dx, dy, d_theta) tuples
        return [(dx, dy, d_theta) for dx, dy in moves for d_theta in rots]

    def _encode_state(self, pos, eye_angle_idx):
        # State representation for Q-table: (x_coordinate, y_coordinate, eye_angle_index)
        return (int(pos[0]), int(pos[1]), int(eye_angle_idx))

    def _choose_action(self, node_idx, state_encoded, entropy_for_epsilon):
        q_table = self.q_tables[node_idx]
        
        # Adaptive epsilon: higher system entropy -> more exploration (to find coherent states)
        # Max possible entropy is log2(angle_steps) for a uniform distribution
        max_entropy = np.log2(self.angle_steps) if self.angle_steps > 0 else 1.0 # Protect against angle_steps = 0
        epsilon = min(1.0, entropy_for_epsilon / max_entropy) if max_entropy > 0 else 0.0

        # Initialize Q-values for this state if it's new
        if state_encoded not in q_table:
            q_table[state_encoded] = {a: 0.0 for a in self.action_space}

        # Epsilon-greedy action selection
        if random.random() < epsilon:
            # Explore: choose a random action
            return random.choice(self.action_space)
        else:
            # Exploit: choose the action with the highest Q-value for this state
            # BUG FIX: Use key=q_table[state_encoded].get to get the action (key) with max value
            return max(q_table[state_encoded], key=q_table[state_encoded].get)

    def step(self):
        # --- 1. Capture previous state of all nodes ---
        prev_node_states = [] # List of (pos, eye_angle_idx) for each node
        for node in self.env_a.nodes:
            prev_node_states.append((node['pos'], node['eye_angle_idx']))
        
        # --- 2. Calculate overall system entropy before actions (for epsilon calculation) ---
        # This entropy reflects the overall system's coherence
        current_system_entropy = self._compute_entropy(prev_node_states)

        # --- 3. Determine actions for all nodes based on their individual Q-tables and current overall entropy ---
        chosen_actions_per_node = [] # Store the action taken by each node
        for i, node in enumerate(self.env_a.nodes):
            state_encoded = self._encode_state(node['pos'], node['eye_angle_idx'])
            action = self._choose_action(i, state_encoded, current_system_entropy)
            chosen_actions_per_node.append(action)

        # --- 4. Calculate previous overall coherence score for reward calculation ---
        prev_overall_coherence_score = self._score_environment(prev_node_states)

        # --- 5. Update Environment B (obstacles) ---
        external_obstacles = self.env_b.step()

        # --- 6. Apply chosen actions to Environment A nodes ---
        # This is where the RL agent directly influences EnvironmentA's nodes
        for i, node in enumerate(self.env_a.nodes):
            dx, dy, d_theta = chosen_actions_per_node[i]
            
            new_x = node['pos'][0] + dx
            new_y = node['pos'][1] + dy

            # Check for valid movement (within bounds, no self-collision, no external obstacles)
            if (0 <= new_x < self.env_a.grid_size and
                0 <= new_y < self.env_a.grid_size and
                (new_x, new_y) not in external_obstacles and
                all((new_x, new_y) != other_node['pos'] for j, other_node in enumerate(self.env_a.nodes) if j != i)):
                node['pos'] = (new_x, new_y) # Update position
            # Note: If move is invalid, the node stays in its current position. No negative reward for invalid moves.

            # Update eye direction (always possible unless d_theta is too large to handle by modulo)
            node['eye_angle_idx'] = (node['eye_angle_idx'] + d_theta) % self.angle_steps
            if node['eye_angle_idx'] < 0: # Ensure angle_idx stays positive
                node['eye_angle_idx'] += self.angle_steps

            # IMPORTANT: The current design assumes FlowrraRLAgent fully controls node movement.
            # If EnvironmentA.step() (as called in main_rl.py) still performs random movements
            # *after* agent.step(), it might interfere with the learned policy.
            # You might need to modify EnvironmentA.step() to only update target angles/eye movement
            # or remove its random movement logic if RL is meant to be the sole movement controller.

        # --- 7. Capture current state of all nodes after actions have been applied ---
        curr_node_states = []
        for node in self.env_a.nodes:
            curr_node_states.append((node['pos'], node['eye_angle_idx']))

        # --- 8. Calculate current overall coherence score and reward ---
        curr_overall_coherence_score = self._score_environment(curr_node_states)
        reward = curr_overall_coherence_score - prev_overall_coherence_score # Reward is the change in overall coherence

        # --- 9. Q-table update for each node ---
        for i in range(self.env_a.num_nodes):
            # Previous state (s), action taken (a), and next state (s') for this specific node
            s_encoded = self._encode_state(prev_node_states[i][0], prev_node_states[i][1])
            action_taken = chosen_actions_per_node[i]
            s_prime_encoded = self._encode_state(curr_node_states[i][0], curr_node_states[i][1])

            q_table = self.q_tables[i] # Get the Q-table for the current node

            # Ensure states exist in the Q-table (initialize with 0.0 for all actions if new)
            if s_encoded not in q_table:
                q_table[s_encoded] = {a_val: 0.0 for a_val in self.action_space}
            if s_prime_encoded not in q_table:
                q_table[s_prime_encoded] = {a_val: 0.0 for a_val in self.action_space}

            # Get the maximum Q-value for the next state (s') to compute the Q-target
            # BUG FIX: Use .values() to get values, then find max
            max_q_s_prime = max(q_table[s_prime_encoded].values()) if q_table[s_prime_encoded] else 0.0 

            # Q-learning update rule:
            # Q(s,a) = Q(s,a) + alpha * (reward + gamma * max(Q(s',a')) - Q(s,a))
            q_table[s_encoded][action_taken] += self.alpha * (
                reward + self.gamma * max_q_s_prime - q_table[s_encoded][action_taken]
            )

        # --- 10. Evaluate collapse condition based on current overall system entropy ---
        system_entropy_after_step = self._compute_entropy(curr_node_states)
        collapse_needed = system_entropy_after_step > self.entropy_threshold

        return collapse_needed, self.env_a # Return the modified EnvironmentA instance

    def _score_environment(self, state_data_list):
        # BUG FIX: Removed self.estimator.fit(data). The estimator should be fitted once
        # before the RL loop starts, on a representative dataset of coherent states.
        # This method should only *score* against that already fitted density.
        
        data_for_scoring = np.array([[x, y, angle] for (x, y), angle in state_data_list])
        
        # score_samples returns log-likelihoods for *each* point. Summing them up for an overall score.
        # This acts as the "fitness" or coherence score for the entire system state.
        log_likelihoods = self.estimator.score_samples(data_for_scoring)
        return np.sum(log_likelihoods) 

    def _compute_entropy(self, state_data_list):
        angles = [angle for (pos, angle) in state_data_list]
        if not angles: # Handle case of empty state list (e.g., if no nodes or invalid data)
            return 0.0
        
        counts = np.bincount(angles, minlength=self.angle_steps)
        total_count = np.sum(counts)
        if total_count == 0: # Avoid division by zero if no angles were present
            return 0.0
            
        probs = counts / total_count
        probs = probs[probs > 0] # Filter out zero probabilities to avoid log(0) issues

        if len(probs) == 0: # If all probabilities become zero after filtering (unlikely with valid input)
            return 0.0
            
        # Calculate Shannon entropy in bits (base 2)
        return -np.sum(probs * np.log2(probs))