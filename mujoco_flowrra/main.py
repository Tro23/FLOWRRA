"""
main.py

Ignition Switch for FLOWRRA.
Boots the MuJoCo passive viewer, runs the Orchestrator loop,
and generates telemetry graphs at the end.
"""

import time

import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer
import numpy as np
from core import FLOWRRA_Orchestrator


def plot_training_history(history):
    """Generates a 4-panel dashboard of the training run."""
    print("\n[System] Generating training telemetry graphs...")
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("FLOWRRA Swarm Training Telemetry", fontsize=16, fontweight="bold")

    # 1. Reward History
    axs[0, 0].plot(history["step"], history["reward"], color="#2ca02c", alpha=0.8)
    axs[0, 0].set_title("Average Step Reward")
    axs[0, 0].set_ylabel("Reward")
    axs[0, 0].grid(True, alpha=0.3)

    # 2. Topological Health
    axs[0, 1].plot(
        history["step"], history["integrity"], label="Integrity", color="#1f77b4"
    )
    axs[0, 1].plot(
        history["step"],
        history["coherence"],
        label="Coherence",
        color="#17becf",
        alpha=0.6,
    )
    axs[0, 1].set_title("Topological Health (1.0 = Perfect)")
    axs[0, 1].set_ylim(0, 1.1)
    axs[0, 1].legend()
    axs[0, 1].grid(True, alpha=0.3)

    # 3. Neural Network Losses
    axs[1, 0].plot(
        history["step"],
        history["actor_loss"],
        label="Actor Loss",
        color="#9467bd",
        alpha=0.8,
    )
    axs[1, 0].plot(
        history["step"],
        history["critic_loss"],
        label="Critic Loss",
        color="#ff7f0e",
        alpha=0.8,
    )
    axs[1, 0].set_title("GNN Loss")
    axs[1, 0].set_xlabel("Global Step")
    axs[1, 0].legend()
    axs[1, 0].grid(True, alpha=0.3)

    # 4. Wave Function Collapses
    axs[1, 1].plot(
        history["step"], history["wfc_collapses"], color="#d62728", linewidth=2
    )
    axs[1, 1].set_title("Cumulative WFC Collapses (Crashes)")
    axs[1, 1].set_xlabel("Global Step")
    axs[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("flowrra_training_stats.png", dpi=300)
    print("[System] Graph saved as 'flowrra_training_stats.png'.")
    plt.show()


def main():
    print("==================================================")
    print("🚀 INITIATING FLOWRRA MUJOCO SIMULATION 🚀")
    print("==================================================")

    orchestrator = FLOWRRA_Orchestrator(mode="training")

    total_episodes = 10
    steps_per_episode = 1000

    # Initialize history tracker
    history = {
        "step": [],
        "reward": [],
        "integrity": [],
        "coherence": [],
        "actor_loss": [],
        "critic_loss": [],
        "wfc_collapses": [],
    }
    global_step = 0

    print("[System] Launching MuJoCo Viewer...")
    with mujoco.viewer.launch_passive(orchestrator.model, orchestrator.data) as viewer:
        time.sleep(1.0)

        for episode in range(total_episodes):
            print(f"\n--- Starting Episode {episode + 1}/{total_episodes} ---")

            for step in range(steps_per_episode):
                step_start_time = time.time()

                # Execute one Brain/Brainstem cycle
                orchestrator.step(episode_step=step, total_episodes=steps_per_episode)
                viewer.sync()

                stats = orchestrator.statistics

                # Record telemetry every 10 steps (keeps the graph high-res but saves memory)
                if global_step % 10 == 0:
                    history["step"].append(global_step)
                    history["reward"].append(stats["reward"])
                    history["integrity"].append(stats["integrity"])
                    history["coherence"].append(stats["coherence"])
                    history["actor_loss"].append(stats["actor_loss"])
                    history["critic_loss"].append(stats["critic_loss"])
                    history["wfc_collapses"].append(stats["wfc_collapses"])

                # Print a tiny heartbeat every 100 steps just so you know it's alive
                if step % 100 == 0:
                    print(
                        f"  > Ep {episode + 1} | Step {step:04d} | Rew: {stats['reward']:+06.2f} | Integrity: {stats['integrity']:.2f} | WFCs: {stats['wfc_collapses']}"
                    )

                global_step += 1

                elapsed = time.time() - step_start_time
                target_loop_time = 0.05
                if elapsed < target_loop_time:
                    time.sleep(target_loop_time - elapsed)

                if not viewer.is_running():
                    print("\n[System] Viewer closed by user. Terminating early...")
                    break

            if not viewer.is_running():
                break

    print("\n==================================================")
    print("🏁 SIMULATION TERMINATED 🏁")
    print("==================================================")

    # Generate the graphs when the viewer closes or finishes!
    if len(history["step"]) > 0:
        plot_training_history(history)


if __name__ == "__main__":
    main()
