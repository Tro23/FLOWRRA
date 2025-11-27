"""
deploy.py - FLOWRRA Deployment Script

Run trained agent and export visualization data for Blender.
No training occurs - pure inference mode.
"""

import json

import matplotlib.pyplot as plt
import numpy as np
from flowrra.config import CONFIG
from flowrra.core import FLOWRRA_Orchestrator


def save_deployment_for_blender(history, filepath="deployment_viz.json"):
    """
    Save deployment history in Blender-friendly format.

    Includes:
    - Node positions and trails
    - Loop connections
    - Obstacles
    - Coherence/integrity metrics
    """
    print(f"\n💾 Saving visualization data to {filepath}...")

    # Convert numpy types to native Python types for JSON serialization
    def convert_to_serializable(obj):
        """Recursively convert numpy types to Python native types."""
        if isinstance(obj, dict):
            return {key: convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        else:
            return obj

    viz_data = {
        "config": {
            "world_bounds": list(CONFIG["spatial"]["world_bounds"]),
            "num_nodes": CONFIG["node"]["num_nodes"],
            "dimensions": CONFIG["spatial"]["dimensions"],
            "loop_ideal_distance": CONFIG["loop"]["ideal_distance"],
            "obstacles": CONFIG["obstacles"],
        },
        "frames": convert_to_serializable(history),
        "metadata": {
            "total_frames": len(history),
            "has_loop_data": True,
            "has_obstacles": True,
            "mode": "deployment",
        },
    }

    with open(filepath, "w") as f:
        json.dump(viz_data, f, indent=2)

    print(f"✅ Saved {len(history)} frames for visualization")
    print(f"   Use this file in Blender: {filepath}")


def plot_deployment_metrics(metrics_history):
    """Plot deployment performance metrics."""
    if not metrics_history:
        print("⚠️  No metrics to plot")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("FLOWRRA Deployment Performance", fontsize=16, fontweight="bold")

    timesteps = [m["timestep"] for m in metrics_history]
    coherence = [m["coherence"] for m in metrics_history]
    integrity = [m["loop_integrity"] for m in metrics_history]
    coverage = [m["coverage"] for m in metrics_history]
    breaks = [m["broken_connections"] for m in metrics_history]

    # Coherence
    axes[0, 0].plot(timesteps, coherence, color="green", linewidth=2, alpha=0.8)
    axes[0, 0].set_title("System Coherence", fontweight="bold", fontsize=12)
    axes[0, 0].set_xlabel("Step")
    axes[0, 0].set_ylabel("Coherence")
    axes[0, 0].axhline(
        y=CONFIG["wfc"]["collapse_threshold"],
        color="red",
        linestyle="--",
        alpha=0.5,
        label="Collapse Threshold",
    )
    axes[0, 0].fill_between(timesteps, 0, coherence, alpha=0.3, color="green")
    axes[0, 0].legend(loc="lower right")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim([0, 1.05])

    # Loop Integrity
    axes[0, 1].plot(timesteps, integrity, color="purple", linewidth=2, alpha=0.8)
    axes[0, 1].set_title("Loop Integrity", fontweight="bold", fontsize=12)
    axes[0, 1].set_xlabel("Step")
    axes[0, 1].set_ylabel("Integrity (0-1)")
    axes[0, 1].fill_between(timesteps, 0, integrity, alpha=0.3, color="purple")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim([0, 1.05])

    # Coverage
    axes[1, 0].plot(timesteps, coverage, color="orange", linewidth=2, alpha=0.8)
    axes[1, 0].set_title("Map Coverage", fontweight="bold", fontsize=12)
    axes[1, 0].set_xlabel("Step")
    axes[1, 0].set_ylabel("Coverage %")
    axes[1, 0].fill_between(timesteps, 0, coverage, alpha=0.3, color="orange")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim([0, 105])

    # Broken Connections
    axes[1, 1].plot(timesteps, breaks, color="red", linewidth=2, alpha=0.8)
    axes[1, 1].set_title("Active Break Count", fontweight="bold", fontsize=12)
    axes[1, 1].set_xlabel("Step")
    axes[1, 1].set_ylabel("Broken Connections")
    axes[1, 1].fill_between(timesteps, 0, breaks, alpha=0.3, color="red")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("deployment_metrics.png", dpi=150, bbox_inches="tight")
    print("📊 Deployment metrics plot saved to deployment_metrics.png")
    plt.show()


def print_statistics_table(stats, metrics_history):
    """Print a nice formatted statistics table."""
    print("\n" + "=" * 70)
    print("  DEPLOYMENT STATISTICS")
    print("=" * 70)

    # Calculate additional stats
    avg_coherence = np.mean([m["coherence"] for m in metrics_history])
    min_coherence = np.min([m["coherence"] for m in metrics_history])
    avg_integrity = np.mean([m["loop_integrity"] for m in metrics_history])
    min_integrity = np.min([m["loop_integrity"] for m in metrics_history])

    print(f"\n  Coverage:")
    print(f"    Final:              {stats['coverage']:6.2f}%")

    print(f"\n  Loop Integrity:")
    print(f"    Final:              {stats['loop_integrity']:6.2f}")
    print(f"    Average:            {avg_integrity:6.2f}")
    print(f"    Minimum:            {min_integrity:6.2f}")

    print(f"\n  System Coherence:")
    print(f"    Final:              {metrics_history[-1]['coherence']:6.2f}")
    print(f"    Average:            {avg_coherence:6.2f}")
    print(f"    Minimum:            {min_coherence:6.2f}")
    print(f"    Threshold:          {CONFIG['wfc']['collapse_threshold']:6.2f}")

    print(f"\n  Loop Breaks:")
    print(f"    Total Occurred:     {stats['total_loop_breaks']:6d}")
    print(
        f"    Currently Active:   {stats['broken_connections'] if 'broken_connections' in stats else 'N/A':6}"
    )

    print(f"\n  Recovery:")
    print(f"    WFC Triggers:       {stats['wfc_triggers']:6d}")

    print(f"\n  Environment:")
    print(f"    Total Nodes:        {stats['num_nodes']:6d}")
    print(f"    Total Obstacles:    {stats['num_obstacles']:6d}")
    print(f"    Simulation Steps:   {stats['step']:6d}")

    print("\n" + "=" * 70)


def main():
    print("=" * 70)
    print("  FLOWRRA DEPLOYMENT MODE")
    print("=" * 70)
    print(f"  World Size:    {CONFIG['spatial']['world_bounds']}")
    print(f"  Nodes:         {CONFIG['node']['num_nodes']}")
    print(
        f"  Obstacles:     {len(CONFIG['obstacles'])} static, "
        f"{len(CONFIG.get('moving_obstacles', []))} moving"
    )
    print("=" * 70)

    # Initialize orchestrator in deployment mode
    sim = FLOWRRA_Orchestrator(mode="deployment")

    # Load trained model
    model_path = "flowrra_trained_model.pth"
    try:
        sim.gnn.load(model_path)
        print(f"\n✅ Loaded trained model from {model_path}")
        print("   Agent will use learned policy")
    except FileNotFoundError:
        print(f"\n⚠️  Warning: No trained model found at {model_path}")
        print("   Running with UNTRAINED agent (random policy)")
        print("   Tip: Run 'python main.py' first to train the agent")
        response = input("\n   Continue anyway? (y/n): ")
        if response.lower() != "y":
            print("   Exiting. Train the agent first with: python main.py")
            return

    deployment_steps = 3000
    print_interval = 20

    print(f"\n🚀 Starting deployment for {deployment_steps} steps...")
    print("   (Press Ctrl+C to stop early)\n")
    print("-" * 70)

    try:
        for t in range(deployment_steps):
            # Run step without training
            avg_reward = sim.step(t, total_episodes=deployment_steps)

            if t % print_interval == 0 or t == deployment_steps - 1:
                stats = sim.get_statistics()
                print(
                    f"[Step {t:4d}/{deployment_steps}] "
                    f"Coverage: {stats['coverage']:5.1f}% | "
                    f"Integrity: {stats['loop_integrity']:.2f} | "
                    f"Breaks: {stats['total_loop_breaks']:3d} | "
                    f"WFC: {stats['wfc_triggers']:2d}"
                )
                print(f"Average Reward: {avg_reward}")

            # Early termination on high coverage
            if sim.map.get_coverage_percentage() > 99.0 and t >= int(
                deployment_steps * 2 / 3
            ):
                print("\n" + "=" * 70)
                print(f"  🎉 MISSION COMPLETE: >95% coverage achieved at step {t}!")
                print(f"Average Reward: {avg_reward}")
                print("=" * 70)
                break

    except KeyboardInterrupt:
        print("\n\n⏸️  Deployment stopped by user.")
        print(f"   Completed {sim.step_count} steps")

    # Final Statistics
    final_stats = sim.get_statistics()
    print_statistics_table(final_stats, sim.metrics_history)

    # Performance assessment
    print("\n📈 Performance Assessment:")
    success_metrics = 0
    total_metrics = 4

    if final_stats["coverage"] >= 80:
        print("   ✅ Excellent coverage (≥80%)")
        success_metrics += 1
    elif final_stats["coverage"] >= 60:
        print("   ✓  Good coverage (≥60%)")
        success_metrics += 0.5
    else:
        print("   ⚠️  Low coverage (<60%)")

    if final_stats["loop_integrity"] >= 0.85:
        print("   ✅ Excellent loop integrity (≥0.85)")
        success_metrics += 1
    elif final_stats["loop_integrity"] >= 0.70:
        print("   ✓  Good loop integrity (≥0.70)")
        success_metrics += 0.5
    else:
        print("   ⚠️  Low loop integrity (<0.70)")

    if final_stats["wfc_triggers"] <= 30:
        print("   ✅ Minimal WFC triggers (≤30)")
        success_metrics += 1
    elif final_stats["wfc_triggers"] <= 60:
        print("   ✓  Acceptable WFC triggers (≤60)")
        success_metrics += 0.5
    else:
        print("   ⚠️  Many WFC triggers (>5)")

    if final_stats["total_loop_breaks"] <= 15:
        print("   ✅ Minimal loop breaks (≤15)")
        success_metrics += 1
    elif final_stats["total_loop_breaks"] <= 30:
        print("   ✓  Acceptable loop breaks (≤30)")
        success_metrics += 0.5
    else:
        print("   ⚠️  Many loop breaks (>30)")

    score = (success_metrics / total_metrics) * 100
    print(f"\n   Overall Score: {score:.0f}/100")

    if score >= 80:
        print("   🌟 EXCELLENT performance!")
    elif score >= 60:
        print("   👍 GOOD performance!")
    else:
        print("   💡 Consider retraining with adjusted parameters")

    # Save deployment data for Blender
    print("\n" + "=" * 70)
    print("  SAVING OUTPUTS")
    print("=" * 70)

    save_deployment_for_blender(sim.history, "deployment_viz.json")

    # Save metrics
    sim.save_metrics("deployment_metrics.json")

    # Plot results
    if sim.metrics_history:
        print("\n📊 Generating plots...")
        plot_deployment_metrics(sim.metrics_history)

    print("\n" + "=" * 70)
    print("  ✅ DEPLOYMENT COMPLETE!")
    print("=" * 70)
    print("\n  Next Steps:")
    print("    1. Open Blender (3.0 or newer)")
    print("    2. Load blender_visualizer.py in Scripting workspace")
    print("    3. Update JSON_PATH to 'deployment_viz.json'")
    print("    4. Run the script")
    print("    5. Press SPACEBAR to watch the visualization!")
    print("\n  Files created:")
    print("    • deployment_viz.json       (for Blender)")
    print("    • deployment_metrics.json   (detailed metrics)")
    print("    • deployment_metrics.png    (performance plots)")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
