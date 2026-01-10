from FLOWRRA import Flowrra
from Environment_B import EnvironmentB
from EnvironmentAB_Visualizer import EnvironmentABVisualizer
import numpy as np
import random 
import matplotlib.pyplot as plt

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

#Define breathing epsilon 
import math

def epsilon_gaussian(t, total_episodes, eps_min=0.05, eps_peak=0.6, mu=None, sigma=None):
    """
    Gaussian-shaped epsilon schedule.
    - t: current episode (int)
    - total_episodes: total training episodes
    - eps_min: baseline minimum epsilon
    - eps_peak: maximum exploration value
    - mu: center of the bell (defaults to mid-training)
    - sigma: width of the bell (defaults to total_episodes/4)
    """
    if mu is None:
        mu = total_episodes / 2.0
    if sigma is None:
        sigma = total_episodes / 6.0  # covers most of training: 99.7%
    
    return eps_min + (eps_peak - eps_min) * math.exp(-((t - mu) ** 2) / (2 * sigma ** 2))


# Generate some initial loop data for the first coherent state
num_nodes = 12
initial_loop_data = []
for i in range(num_nodes):
    x = random.randint(5, 24)
    y = random.randint(5, 24)
    angle = random.randint(0, 35)
    initial_loop_data.append([x, y, angle])

# Initialize the main FLOWRRA class with the new structured configuration
flowrra = Flowrra(
    node_position_config=NodePositionConfig(position=(0,0), eye_angles=0, rotation_speed=2, move_speed=1),
    node_sensor_config=NodeSensorConfig(max_range=10, noise_std=0.5, false_negative_prob=0.05, false_positive_prob=0.05),
    loop_data_config=initial_loop_data,
    coherence_score_config=CoherenceScoreConfig(entropy_threshold=2.8),
    alpha=0.09,
    gamma=0.85
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
print("Starting warm-up phase with random actions...")
warm_up_episodes = 20
steps_per_warmup = 100
for episode in range(warm_up_episodes):
    env_b = EnvironmentB()
    flowrra.visited_cells_b = set()
    
    for step in range(steps_per_warmup):
        # The agent steps with an epsilon of 1.0, ensuring purely random actions
        flowrra.step(env_b, epsilon=1.0)
        env_b.step()

print(f"Warm-up complete after {warm_up_episodes} episodes.")

# --- 4. Run the Reinforcement Learning Loop ---
num_episodes = 500
steps_per_episode = 1000
eps_min=0.05
eps_peak=0.8
epsilon = eps_min

rewards_per_episode = []
entropies_per_episode = []


for episode in range(num_episodes):
    # Reset Environment B and visited cells for a new episode
    env_b = EnvironmentB()
    flowrra.visited_cells_b = set()
    
    # Get initial state of Environment A for the episode
    current_env_a_state_data = flowrra.env_a.step()
    episode_reward = 0
    
    for step in range(steps_per_episode):
        # Step the FLOWRRA agent
        next_env_a_state_data, reward = flowrra.step(env_b, epsilon)
        episode_reward += reward
        
        # Step Environment B (moving obstacles)
        env_b.step()

        # Check for coherence and potentially trigger WFC
        current_entropy = flowrra._compute_entropy(next_env_a_state_data)

        if current_entropy > flowrra.coherence_score_config.entropy_threshold:
            print(f"Collapse triggered at episode {episode}, step {step} due to high entropy ({current_entropy:.2f}).")
            
            # --- NEW ---: As per your request, record the state that caused the collapse.
            # This allows the reward function to penalize returning to this state.
            flowrra.record_collapse_state(next_env_a_state_data)
            
            # 1. Use WFC to generate a new coherent loop
            new_loop = flowrra.wfc.collapse(num_nodes=flowrra.num_nodes, last_state_data=current_env_a_state_data)
            
            # 2. Re-initialize Environment A with the new loop
            flowrra.reinitialize_loop(new_loop.tolist())
            
            # 3. CRITICAL: Immediately fetch the new state after re-initialization.
            next_env_a_state_data = flowrra.env_a.step()
        
        # Update state for next step
        current_env_a_state_data = next_env_a_state_data

    # Log episode stats at the end of the episode
    rewards_per_episode.append(episode_reward)
    final_entropy = flowrra._compute_entropy(current_env_a_state_data)
    entropies_per_episode.append(final_entropy)
    
    # Decay epsilon
    epsilon = epsilon_gaussian(episode, total_episodes=num_episodes,
                           eps_min=0.05, eps_peak=0.6,
                           mu=num_episodes/2, sigma=num_episodes/6)

    # Optional: visualize every N episodes and log progress
    if episode % 20 == 0 or episode == num_episodes - 1:
        print(f"Episode {episode}: Total Reward = {episode_reward:.2f}, Final Entropy = {final_entropy:.2f}, Epsilon = {epsilon:.3f}")
        visualizer.render(nodes_to_render=flowrra.env_a.nodes, 
                          grid_size=flowrra.env_a.grid_size, 
                          angle_steps=flowrra.env_a.angle_steps, 
                          visited_cells_b=flowrra.visited_cells_b, 
                          title=f"State at Episode {episode}",
                          save=True, 
                          filename=f"Version_1_Outputs/flowrra_frame_{episode}.png")

print("Training finished!")

# --- 5. Final Visualization and Results ---
print("Rendering final state...")
visualizer.render(nodes_to_render=flowrra.env_a.nodes, 
                  grid_size=flowrra.env_a.grid_size, 
                  angle_steps=flowrra.env_a.angle_steps, 
                  visited_cells_b=flowrra.visited_cells_b, 
                  save=True,
                  filename="Version_1_Outputs/flowrra_final_state.png")

# Plot results
fig, ax1 = plt.subplots(figsize=(12, 7))

ax1.set_xlabel('Episode')
ax1.set_ylabel('Total Reward per Episode', color='blue')
ax1.plot(rewards_per_episode, label='Total Reward', color='blue', alpha=0.8)
ax1.tick_params(axis='y', labelcolor='blue')
ax1.set_title("Training Metrics Over Time")
ax1.grid(True, linestyle='--', alpha=0.6)

ax2 = ax1.twinx()
ax2.set_ylabel('Final Entropy per Episode', color='red')
ax2.plot(entropies_per_episode, label='Entropy', color='red', linestyle='--')
ax2.axhline(y=flowrra.coherence_score_config.entropy_threshold, color='darkred', linestyle=':', label='Collapse Threshold')
ax2.tick_params(axis='y', labelcolor='red')
ax2.legend(loc='upper right')

fig.tight_layout()
plt.savefig("Flowrra_Training_Metrics.png")
plt.show()

# --- 6. Save the entire trained FLOWRRA instance ---
import pickle

print("Saving the entire trained FLOWRRA instance...")
flowrra_filename = "flowrra_q_tables.pkl"
with open(flowrra_filename, 'wb') as f:
    pickle.dump(flowrra, f)
print(f"FLOWRRA instance saved to {flowrra_filename}")

