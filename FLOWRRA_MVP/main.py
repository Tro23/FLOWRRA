"""
main.py - FLOWRRA Training Script

Train the GNN agent to navigate with loop structure preservation.
"""

import matplotlib.pyplot as plt

# import numpy as np
from flowrra.config import CONFIG
from flowrra.core import FLOWRRA_Orchestrator


def plot_training_results(metrics_history):
    """Plot training metrics."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("FLOWRRA Training Results", fontsize=16)

    timesteps = [m["timestep"] for m in metrics_history]
    rewards = [m["avg_reward"] for m in metrics_history]
    coherence = [m["coherence"] for m in metrics_history]
    integrity = [m["loop_integrity"] for m in metrics_history]
    coverage = [m["coverage"] for m in metrics_history]
    breaks = [m["broken_connections"] for m in metrics_history]

    # Average Reward
    axes[0, 0].plot(timesteps, rewards, alpha=0.7)
    axes[0, 0].set_title("Average Reward")
    axes[0, 0].set_xlabel("Step")
    axes[0, 0].set_ylabel("Reward")
    axes[0, 0].grid(True, alpha=0.3)

    # Coherence
    axes[0, 1].plot(timesteps, coherence, color="green", alpha=0.7)
    axes[0, 1].set_title("System Coherence")
    axes[0, 1].set_xlabel("Step")
    axes[0, 1].set_ylabel("Coherence")
    axes[0, 1].axhline(
        y=CONFIG["wfc"]["collapse_threshold"],
        color="r",
        linestyle="--",
        label="Threshold",
    )
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Loop Integrity
    axes[0, 2].plot(timesteps, integrity, color="purple", alpha=0.7)
    axes[0, 2].set_title("Loop Integrity")
    axes[0, 2].set_xlabel("Step")
    axes[0, 2].set_ylabel("Integrity (0-1)")
    axes[0, 2].grid(True, alpha=0.3)

    # Coverage
    axes[1, 0].plot(timesteps, coverage, color="orange", alpha=0.7)
    axes[1, 0].set_title("Map Coverage")
    axes[1, 0].set_xlabel("Step")
    axes[1, 0].set_ylabel("Coverage %")
    axes[1, 0].grid(True, alpha=0.3)

    # Broken Connections
    axes[1, 1].plot(timesteps, breaks, color="red", alpha=0.7)
    axes[1, 1].set_title("Broken Connections")
    axes[1, 1].set_xlabel("Step")
    axes[1, 1].set_ylabel("Count")
    axes[1, 1].grid(True, alpha=0.3)

    # Cumulative Total Breaks
    total_breaks = [m["total_breaks"] for m in metrics_history]
    axes[1, 2].plot(timesteps, total_breaks, color="darkred", alpha=0.7)
    axes[1, 2].set_title("Cumulative Loop Breaks")
    axes[1, 2].set_xlabel("Step")
    axes[1, 2].set_ylabel("Total Breaks")
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("training_results.png", dpi=150)
    print("📊 Training plots saved to training_results.png")
    plt.show()


def main():
    print("=" * 60)
    print("FLOWRRA TRAINING MODE")
    print("=" * 60)
    print(f"Target: Explore {CONFIG['spatial']['world_bounds']} area")
    print(f"Nodes: {CONFIG['node']['num_nodes']}")
    print(f"Obstacles: {len(CONFIG['obstacles'])}")
    print("=" * 60)

    # Initialize orchestrator in training mode
    sim = FLOWRRA_Orchestrator(mode="training")

    training_steps = 3000
    print_interval = 50

    try:
        for t in range(training_steps):
            avg_reward = sim.step(t, total_episodes=training_steps)

            if t % print_interval == 0:
                stats = sim.get_statistics()
                print(f"\n[Step {t}/{training_steps}]")
                print(f"  Coverage: {stats['coverage']:.2f}%")
                print(f"  Avg Reward: {stats['avg_reward']:.4f}", avg_reward)
                print(f"  Loop Integrity: {stats['loop_integrity']:.2f}")
                print(f"  Total Breaks: {stats['total_loop_breaks']}")
                print(f"  WFC Triggers: {stats['wfc_triggers']}")
                print(f"  Buffer Size: {stats['buffer_size']}")

            # Early termination on high coverage
            if sim.map.get_coverage_percentage() > 90.0:
                print(f"\n✅ Mission Complete: >90% coverage at step {t}")
                break

    except KeyboardInterrupt:
        print("\n⏸️  Training stopped by user.")

    # Final Statistics
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    final_stats = sim.get_statistics()
    for key, value in final_stats.items():
        print(f"  {key}: {value}")
    print("=" * 60)

    # Save Model
    model_path = "flowrra_trained_model.pth"
    sim.gnn.save(model_path)
    print(f"💾 Model saved to {model_path}")

    # Save Metrics
    sim.save_metrics("training_metrics.json")

    # Plot Results
    if sim.metrics_history:
        plot_training_results(sim.metrics_history)

    print("\n✅ Training session complete!")


if __name__ == "__main__":
    main()
