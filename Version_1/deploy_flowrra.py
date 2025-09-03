import pickle
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from FLOWRRA import Flowrra
from Environment_B import EnvironmentB
from EnvironmentAB_Visualizer import EnvironmentABVisualizer
import shutil

# --- Ensure pickle finds these configs ---
class NodePositionConfig:
    def __init__(self, position, eye_angles, rotation_speed, move_speed):
        self.position = position
        self.eye_angles = eye_angles
        self.rotation_speed = rotation_speed
        self.move_speed = move_speed

class NodeSensorConfig:
    def __init__(self, max_range, noise_std, false_negative_prob, false_positive_prob):
        self.max_range = max_range
        self.noise_std = noise_std
        self.false_negative_prob = false_negative_prob
        self.false_positive_prob = false_positive_prob

class CoherenceScoreConfig:
    def __init__(self, entropy_threshold):
        self.entropy_threshold = entropy_threshold


# --- SETTINGS ---
mode = "side_by_side"   # "single" or "side_by_side"
steps = 200             # number of frames
epsilon = 0.0           # exploitation only
window_size = 20        # sliding window for metrics
output_file = "flowrra_deployment_with_metrics.gif"


# --- 1. Load trained FLOWRRA ---
with open("flowrra_q_tables.pkl", "rb") as f:
    flowrra = pickle.load(f)

# --- 2. Setup environment ---
env_b = EnvironmentB()
visualizer = EnvironmentABVisualizer(env_b)

# --- 3. Setup figure(s) ---
if mode == "single":
    fig, ax = plt.subplots(figsize=(8, 8))
else:
    fig = plt.figure(figsize=(14, 7))
    gs = fig.add_gridspec(2, 2, width_ratios=[2, 1], height_ratios=[1, 1])

    ax_env = fig.add_subplot(gs[:, 0])     # full left column
    ax_entropy = fig.add_subplot(gs[0, 1]) # top-right
    ax_reward = fig.add_subplot(gs[1, 1])  # bottom-right

# --- Metrics tracking ---
rewards, entropies = [], []
cumulative_reward = 0

# --- 4. Update function ---
def update(frame):
    global cumulative_reward

    # FLOWRRA + environment step
    state_data, reward = flowrra.step(env_b, epsilon)
    env_b.step()
    cumulative_reward += reward
    rewards.append(cumulative_reward)
    entropies.append(flowrra._compute_entropy(state_data))

    if mode == "single":
        ax.clear()
        visualizer.render(
            nodes_to_render=flowrra.env_a.nodes,
            grid_size=flowrra.env_a.grid_size,
            angle_steps=flowrra.env_a.angle_steps,
            visited_cells_b=flowrra.visited_cells_b,
            title=f"FLOWRRA Deployment Step {frame}",
            save=False,
            ax=ax
        )

    else:  # side_by_side
        # Left panel: environment
        visualizer.render(
            nodes_to_render=flowrra.env_a.nodes,
            grid_size=flowrra.env_a.grid_size,
            angle_steps=flowrra.env_a.angle_steps,
            visited_cells_b=flowrra.visited_cells_b,
            title=f"FLOWRRA Deployment Step {frame}",
            save=False,
            ax=ax_env
        )

        # Top-right: entropy
        ax_entropy.clear()
        ax_entropy.set_title("Entropy vs Threshold")
        ax_entropy.set_xlabel("Step")
        ax_entropy.set_ylabel("Entropy")

        start = max(0, frame - window_size)
        x_vals = list(range(start, frame + 1))
        y_entropies = entropies[start: start + len(x_vals)]

        ax_entropy.plot(x_vals, y_entropies, color="red", label="Entropy")
        ax_entropy.axhline(
            y=flowrra.coherence_score_config.entropy_threshold,
            color="darkred", linestyle=":", label="Threshold"
        )
        ax_entropy.legend()
        ax_entropy.grid(True, linestyle="--", alpha=0.6)

        # Bottom-right: reward
        ax_reward.clear()
        ax_reward.set_title("Cumulative Reward")
        ax_reward.set_xlabel("Step")
        ax_reward.set_ylabel("Reward")

        y_rewards = rewards[start: start + len(x_vals)]
        ax_reward.plot(x_vals, y_rewards, color="blue", label="Cumulative Reward")
        ax_reward.legend()
        ax_reward.grid(True, linestyle="--", alpha=0.6)

# --- 5. Create animation ---
ani = animation.FuncAnimation(fig, update, frames=steps, interval=200, repeat=False)

# --- 6. Save animation ---
# Save as MP4
ani.save("Flowrra_deployment_with_metrics.gif", writer="pillow", fps=5)

print("Animation saved as Flowrra_deployment_with_metrics.gif")
