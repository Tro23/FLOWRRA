from EnvironmentA import EnvironmentA
from EnvironmentB import EnvironmentB
from density_function_estimator import DensityFunctionEstimator
from wave_function_collapse import WaveFunctionCollapse
from environmentABVisualizer import EnvironmentABVisualizer # Assuming this correctly handles both env_a and env_b
from FlowrraQAgent import FlowrraRLAgent

import numpy as np
import matplotlib.pyplot as plt

# --- 1. Initialize core environments ---
# Initialize with default parameters; these might be updated later if a WFC collapse re-initializes.
env_a = EnvironmentA()
env_b = EnvironmentB() # Environment B for obstacles

# --- 2. Collect initial data to train the KDE (the 'Stochastic Wave Function') ---
# This is CRUCIAL. The KDE needs to learn the density of *desired* coherent states.
# Simulate EnvironmentA following some ideal/desired coherent patterns or collect data from pre-defined loops.
# For now, let's simulate EnvironmentA in a way that generates somewhat coherent (or representative) data.
print("Collecting initial data for KDE training...")
initial_kde_training_data = []
# It's better to get data from a coherent, stable phase of EnvA, or pre-defined loops.
# For example, run EnvA initialized with a WFC loop for a few steps to generate data.
# Or, if you have historical data of 'good' flow, use that.
# For demonstration, let's just use EnvA's random movement but for enough steps to get varied data.
temp_env_a_for_data_collection = EnvironmentA(grid_size=env_a.grid_size, num_nodes=env_a.num_nodes, angle_steps=env_a.angle_steps)
for _ in range(1000): # Collect more data for better KDE
    state_data = temp_env_a_for_data_collection.step()
    initial_kde_training_data.extend([[x, y, angle] for (x, y), angle in state_data])
initial_kde_training_data = np.array(initial_kde_training_data)
print(f"Collected {len(initial_kde_training_data)} data points for KDE training.")


# --- 3. Fit the Density Function Estimator (KDE) ---
# This fitted estimator is the 'Stochastic Wave Function' against which coherence will be measured throughout RL training.
estimator = DensityFunctionEstimator(bandwidth=1.5) # You can tune bandwidth
estimator.fit(initial_kde_training_data)
print("Kernel Density Estimator (KDE) fitted on initial data.")

# --- 4. Instantiate Wave Function Collapse (WFC) ---
# WFC uses the fitted estimator to generate coherent loops.
wfc = WaveFunctionCollapse(estimator, grid_size=env_a.grid_size, angle_steps=env_a.angle_steps)
print("WaveFunctionCollapse instantiated.")

# --- 5. Initialize RL Agent ---
# The agent needs the pre-fitted estimator.
agent = FlowrraRLAgent(env_a, env_b, estimator, wfc)
print("FlowrraRLAgent instantiated.")

# --- 6. Perform initial 'Collapse' to set up EnvironmentA for the first time with a coherent loop ---
# This ensures the RL training starts from a relatively coherent state.
print("Performing initial WFC collapse to set EnvironmentA state for RL training...")
initial_loop = wfc.collapse(num_nodes=env_a.num_nodes) # Generate a coherent loop
# Re-initialize agent's env_a with this coherent loop. Update num_nodes based on actual loop length.
agent.env_a = EnvironmentA(grid_size=env_a.grid_size, num_nodes=len(initial_loop)-1,
                           angle_steps=env_a.angle_steps, initial_loop_data=initial_loop.tolist())
env_a = agent.env_a # Make sure the main script's env_a reference is updated
print("EnvironmentA initialized with a collapsed loop for RL training.")

# --- 7. Run the RL training loop ---
max_episodes = 2000 # Increased episodes to give RL more time to learn
coherence_scores_per_episode = []
entropies_per_episode = []

print("Starting RL training loop...")
for episode in range(max_episodes):
    # agent.step() controls env_a's nodes, updates Q-tables, and returns if collapse is needed
    # The external_obstacles from env_b.step() are handled *inside* agent.step().
    collapse_triggered, current_env_a = agent.step()

    # Get current coherence score and entropy for logging and visualization
    current_state_data = [(n['pos'], n['eye_angle_idx']) for n in current_env_a.nodes]
    current_coherence_score = agent._score_environment(current_state_data)
    current_entropy = agent._compute_entropy(current_state_data)

    coherence_scores_per_episode.append(current_coherence_score)
    entropies_per_episode.append(current_entropy)

    if collapse_triggered:
        print(f"Collapse triggered at episode {episode} due to high entropy ({current_entropy:.2f}).")
        # Generate a new coherent loop from WFC
        new_loop = wfc.collapse(num_nodes=current_env_a.num_nodes)
        
        # Re-initialize EnvironmentA with the new loop to restore coherence
        agent.env_a = EnvironmentA(grid_size=current_env_a.grid_size, num_nodes=len(new_loop)-1,
                                   angle_steps=current_env_a.angle_steps, initial_loop_data=new_loop.tolist())
        env_a = agent.env_a # Update reference in main script
        print(f"EnvironmentA re-initialized with a new coherent loop.")

    # Optional: visualize every N episodes and log progress
    if episode % 100 == 0: # Visualize less frequently to save time/resources
        print(f"Episode {episode}: Coherence Score = {current_coherence_score:.2f}, Entropy = {current_entropy:.2f}")
        visualizer = EnvironmentABVisualizer(current_env_a, env_b)
        visualizer.render(save=True, filename=f"Loop_Outputs/flowrra_rl_frame_{episode}.png")

print("RL training complete.")

