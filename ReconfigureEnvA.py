import numpy as np
import matplotlib.pyplot as plt
import math
from EnvironmentA import EnvironmentA
from density_function_estimator import DensityFunctionEstimator
from wave_function_collapse import WaveFunctionCollapse
import matplotlib.patches as patches

# Step 1: Simulate data using EnvironmentA
env_a = EnvironmentA()
simulated_data = []
for _ in range(100):
    state = env_a.step()
    simulated_data.extend([[x, y, angle] for (x, y), angle in state])
simulated_data = np.array(simulated_data)

# Step 2: Fit KDE
estimator = DensityFunctionEstimator(bandwidth=1.5)
estimator.fit(simulated_data)

# Step 3: Instantiate Wave Function Collapse
wfc = WaveFunctionCollapse(estimator)

#Step 4: Run WFC
collapsed_loop = wfc.collapse(num_nodes=12)

# Step 5: Initialize new EnvironmentA from collapsed loop
env_a_reinit = EnvironmentA(initial_loop_data=collapsed_loop.tolist())

# Step 5: Run for N steps and compute feedback scores
log_scores = []
steps = 100
for _ in range(steps):
    state = env_a_reinit.step()
    state_arr = np.array([[x, y, angle] for (x, y), angle in state])
    log_score = estimator.kde.score(state_arr)
    log_scores.append(log_score)

# Step 6: Plot feedback score over time
plt.figure(figsize=(8, 4))
plt.plot(log_scores, color='purple', linewidth=2)
plt.title("FLOWRRA Feedback Score Over Time (Log-Likelihood)")
plt.xlabel("Step")
plt.ylabel("Log-Likelihood")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# Step 7: Visualizing the Reconfigured Environment A

# --- Matplotlib Setup ---
fig, ax = plt.subplots(figsize=(8, 8)) # Slightly larger figure for clarity
ax.set_xlim(0, 30)
ax.set_ylim(0, 30)
ax.set_title("Environment-A: Moving Disks with Dynamic Eye Direction & Loop")
ax.set_xticks([])
ax.set_yticks([])

# Draw nodes (disks) and initial arrows
all_patches = [] # To collect all patches for animation blitting
num_nodes = 12
nodes = env_a_reinit._initialize_nodes()
for i in range(len(nodes)):
    node = nodes[i]
    x_center, y_center = node['pos'][0] + 0.5, node['pos'][1] + 0.5 # Center of the grid cell

    # Draw Disk
    disk_patch = patches.Circle((x_center, y_center), 0.4, fc='skyblue', ec='blue', lw=0.8, alpha=0.9)
    ax.add_patch(disk_patch)
    node['patch_disk'] = disk_patch
    all_patches.append(disk_patch)

    # Draw Arrow
    angle_idx = node['eye_angle_idx']
    angle_deg = angle_idx * (360 / env_a_reinit.angle_steps)
    rad = np.deg2rad(angle_deg)
    dx, dy = 0.3 * np.cos(rad), 0.3 * np.sin(rad)

    arrow_patch = patches.FancyArrow(x_center, y_center, dx, dy,
                                     width=0.05, head_width=0.3, head_length=0.3,
                                     fc='darkorange', ec='red', lw=1.0, zorder=2)
    ax.add_patch(arrow_patch)
    node['patch_arrow'] = arrow_patch
    all_patches.append(arrow_patch)



# Initialize Dotted Lines for the loop
# We need to store line objects so we can update their data in the animation
for i in range(num_nodes):
    current_node = nodes[i]
    next_node = nodes[(i + 1) % num_nodes]

    x1_center, y1_center = current_node['pos'][0] + 0.5, current_node['pos'][1] + 0.5
    x2_center, y2_center = next_node['pos'][0] + 0.5, next_node['pos'][1] + 0.5

    line, = ax.plot([x1_center, x2_center], [y1_center, y2_center], 
                    'k:', alpha=0.4, lw=1.0, zorder=1) # 'k:' for black dotted line
    current_node['patch_line'] = line # Store the line patch for the 'current_node'
    all_patches.append(line)

plt.gca().set_aspect('equal', adjustable='box')
plt.tight_layout()
plt.show()


