from FLOWRRA import Flowrra
from Environment_B import EnvironmentB
from EnvironmentAB_Visualizer import EnvironmentABVisualizer
import numpy as np
import random # We need random for the initial loop data generation

# --- 1. Define the system configuration based on your requested structure ---
# Configuration for Node_Position
class NodePositionConfig:
    def __init__(self, position, eye_angles, rotation_speed, move_speed):
        self.position = position
        self.eye_angles = eye_angles
        self.rotation_speed = rotation_speed
        self.move_speed = move_speed

# Configuration for Node_Sensor
class NodeSensorConfig:
    def __init__(self, max_range, noise_std, false_negative_prob, false_positive_prob):
        self.max_range = max_range
        self.noise_std = noise_std
        self.false_negative_prob = false_negative_prob
        self.false_positive_prob = false_positive_prob

# Configuration for Coherence_Score
class CoherenceScoreConfig:
    def __init__(self, entropy_threshold):
        self.entropy_threshold = entropy_threshold

# Generate some initial loop data for the first coherent state
# This would typically come from an an initial WFC collapse or a pre-defined pattern
num_nodes = 12
initial_loop_data = []
for i in range(num_nodes):
    x = random.randint(0, 29)
    y = random.randint(0, 29)
    angle = random.randint(0, 35)
    initial_loop_data.append([x, y, angle])

# Initialize the main FLOWRRA class with the new structured configuration
flowrra = Flowrra(
    node_position_config=NodePositionConfig(position=(0,0), eye_angles=0, rotation_speed=2, move_speed=1),
    node_sensor_config=NodeSensorConfig(max_range=10, noise_std=0.5, false_negative_prob=0.05, false_positive_prob=0.05),
    loop_data_config=initial_loop_data,
    coherence_score_config=CoherenceScoreConfig(entropy_threshold=3.4),
    alpha=0.1,
    gamma=0.95
)

# Initialize the external Environment B
env_b = EnvironmentB()

# Initialize the visualizer with the correct environment objects
visualizer = EnvironmentABVisualizer(env_b)

# --- 2. Train the KDE (the 'Stochastic Wave Function') ---
print("Collecting initial data for KDE training...")
initial_kde_training_data = []
# Simulate a few hundred steps of the initial loop to collect data
for _ in range(500):
    state = flowrra.env_a.step()
    initial_kde_training_data.extend([[p[0][0], p[0][1], p[1]] for p in state])
    
flowrra.density_estimator.fit(initial_kde_training_data)
print("KDE has been successfully trained on initial coherent data.")

# --- 3. Warm-up Phase: Random Exploration to Populate Experience ---
# This is a crucial step for many RL algorithms. We let the agent interact with
# the environment randomly for a set number of steps to fill its internal state
# with experiences before it starts policy-based learning.
print("Starting warm-up phase with random actions...")
warm_up_episodes = 50
for episode in range(warm_up_episodes):
    # Reset Environment B for a new episode
    env_b = EnvironmentB()
    visited_cells_b = set()
    
    for step in range(50): # Steps within a warm-up episode
        # The agent steps with an epsilon of 1.0, ensuring purely random actions
        next_env_a_state_data, reward, visited_cells_b = flowrra.step(env_b, visited_cells_b, epsilon=1.0)
        
        # Step Environment B (moving obstacles)
        env_b.step()

print(f"Warm-up complete after {warm_up_episodes} episodes.")

# --- 4. Run the Reinforcement Learning Loop ---
num_episodes = 2000
epsilon = 0.05  # Start with a very low epsilon for pure exploitation
epsilon_increase = 0.0005 # A small, gradual increase rate
max_epsilon = 0.2 # The steady-state epsilon threshold

coherence_scores_per_episode = []
entropies_per_episode = []
visited_cells_b = set()

for episode in range(num_episodes):
    # Reset Environment B for a new episode
    env_b = EnvironmentB()
    
    # Get initial state of Environment A for the episode
    current_env_a_state_data = flowrra.env_a.step()
    
    for step in range(500): # Steps within an episode
        # Step the FLOWRRA agent
        next_env_a_state_data, reward, visited_cells_b = flowrra.step(env_b, visited_cells_b, epsilon)
        
        # Step Environment B (moving obstacles)
        env_b.step()

        # Check for coherence and potentially trigger WFC
        current_coherence_score = flowrra._score_environment(current_env_a_state_data)
        current_entropy = flowrra._compute_entropy(current_env_a_state_data)

        if current_entropy > flowrra.coherence_score_config.entropy_threshold:
            print(f"Collapse triggered at episode {episode}, step {step} due to high entropy ({current_entropy:.2f}).")
            
            # Use WFC to generate a new coherent loop from the neighborhood of the previous state
            new_loop = flowrra.wfc.collapse(num_nodes=flowrra.num_nodes, last_state_data=current_env_a_state_data)
            
            # Re-initialize Environment A with the new coherent loop
            flowrra.env_a = flowrra.Environment_A(
                node_params=flowrra.node_pos_config,
                loop_data=new_loop.tolist(),
                grid_size=flowrra.env_a.grid_size,
                angle_steps=flowrra.env_a.angle_steps
            )
            print("Environment A re-initialized with a new coherent loop.")
            break # End the current episode
        
        # Update state for next step
        current_env_a_state_data = next_env_a_state_data

    # Log episode stats at the end of the episode
    coherence_scores_per_episode.append(current_coherence_score)
    entropies_per_episode.append(current_entropy)
    
    # Gradually increase epsilon towards the max_epsilon threshold
    epsilon = min(max_epsilon, epsilon + epsilon_increase)

    # Optional: visualize every N episodes and log progress
    '''if episode % 50 == 0:
        print(f"Episode {episode}: Coherence Score = {coherence_scores_per_episode[-1]:.2f}, Entropy = {entropies_per_episode[-1]:.2f}, Epsilon = {epsilon:.2f}")
        visualizer.render(nodes_to_render=flowrra.node_pos_config.position, 
                          grid_size=flowrra.env_a.grid_size,
                          angle_steps=flowrra.env_a.angle_steps, 
                          visited_cells_b=visited_cells_b, 
                          save=True, 
                          filename=f"Version_1_Outputs/FLowrra_frame_{episode}.png")'''

print("Training finished!")
