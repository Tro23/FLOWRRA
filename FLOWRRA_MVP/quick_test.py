"""
quicktest.py - Fast FLOWRRA Testing

Quick simulation to verify everything works without full training.
Useful for:
- Testing new configurations
- Debugging changes
- Validating visualization pipeline
"""

import json

import matplotlib.pyplot as plt
from flowrra.config import CONFIG
from flowrra.core import FLOWRRA_Orchestrator


def plot_quick_results(metrics):
    """Quick visualization of test results."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("FLOWRRA Quick Test Results", fontsize=14, fontweight="bold")

    steps = [m["timestep"] for m in metrics]

    # Coverage
    coverage = [m["coverage"] for m in metrics]
    axes[0].plot(steps, coverage, "o-", color="orange", linewidth=2)
    axes[0].set_title("Coverage", fontweight="bold")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Coverage %")
    axes[0].grid(True, alpha=0.3)
    axes[0].fill_between(steps, 0, coverage, alpha=0.3, color="orange")

    # Coherence vs Integrity
    coherence = [m["coherence"] for m in metrics]
    integrity = [m["loop_integrity"] for m in metrics]
    axes[1].plot(steps, coherence, "o-", label="Coherence", linewidth=2)
    axes[1].plot(steps, integrity, "s-", label="Loop Integrity", linewidth=2)
    axes[1].axhline(
        y=CONFIG["wfc"]["collapse_threshold"],
        color="r",
        linestyle="--",
        alpha=0.5,
        label="Threshold",
    )
    axes[1].set_title("System Health", fontweight="bold")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Score (0-1)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Breaks
    breaks = [m["broken_connections"] for m in metrics]
    axes[2].plot(steps, breaks, "o-", color="red", linewidth=2)
    axes[2].set_title("Loop Breaks", fontweight="bold")
    axes[2].set_xlabel("Step")
    axes[2].set_ylabel("Active Breaks")
    axes[2].grid(True, alpha=0.3)
    axes[2].fill_between(steps, 0, breaks, alpha=0.3, color="red")

    plt.tight_layout()
    plt.savefig("quicktest_results.png", dpi=150)
    print("\n📊 Test plots saved to quicktest_results.png")
    plt.show()


def main():
    print("=" * 60)
    print("FLOWRRA QUICK TEST")
    print("=" * 60)
    print("Running 200-step simulation for validation...")
    print("=" * 60)

    # Initialize in deployment mode (no training overhead)
    sim = FLOWRRA_Orchestrator(mode="deployment")

    test_steps = 200
    print_interval = 25

    print("\n🚀 Starting test simulation...\n")

    try:
        for t in range(test_steps):
            avg_reward = sim.step(t, total_episodes=test_steps)

            if t % print_interval == 0:
                stats = sim.get_statistics()
                print(
                    f"[Step {t:3d}/{test_steps}] "
                    f"Cov: {stats['coverage']:5.1f}% | "
                    f"Coh: {sim.metrics_history[-1]['coherence']:.2f} | "
                    f"Int: {stats['loop_integrity']:.2f} | "
                    f"Breaks: {stats['total_loop_breaks']:2d}"
                )

    except KeyboardInterrupt:
        print("\n⏸️  Test stopped by user.")

    # Final report
    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)

    final_stats = sim.get_statistics()

    print("\n📊 Final Statistics:")
    print(f"  Coverage:         {final_stats['coverage']:.2f}%")
    print(f"  Loop Integrity:   {final_stats['loop_integrity']:.2f}")
    print(f"  Total Breaks:     {final_stats['total_loop_breaks']}")
    print(f"  WFC Triggers:     {final_stats['wfc_triggers']}")
    print(f"  Total Nodes:      {final_stats['num_nodes']}")
    print(f"  Total Obstacles:  {final_stats['num_obstacles']}")

    # Check for issues
    print("\n🔍 Health Check:")
    issues = []

    if final_stats["coverage"] < 15:
        issues.append("⚠️  Very low coverage - nodes may not be exploring")

    if final_stats["loop_integrity"] < 0.5:
        issues.append("⚠️  Low loop integrity - too many obstacles or weak springs")

    if final_stats["total_loop_breaks"] > 50:
        issues.append("⚠️  Excessive loop breaks - adjust break_threshold or obstacles")

    if final_stats["wfc_triggers"] > 5:
        issues.append("⚠️  Too many WFC triggers - system unstable")

    if sim.metrics_history[-1]["coherence"] < CONFIG["wfc"]["collapse_threshold"]:
        issues.append("⚠️  Ending with low coherence - system struggling")

    if issues:
        print("\n".join(issues))
        print("\n💡 Tip: Check METRICS_GUIDE.md for troubleshooting")
    else:
        print("✅ All checks passed! System looks healthy.")

    # Visualization outputs
    print("\n📁 Outputs:")

    # Save visualization data
    viz_path = "quicktest_viz.json"
    viz_data = {
        "config": {
            "world_bounds": CONFIG["spatial"]["world_bounds"],
            "num_nodes": CONFIG["node"]["num_nodes"],
            "dimensions": CONFIG["spatial"]["dimensions"],
        },
        "frames": sim.history,
        "metadata": {"total_frames": len(sim.history), "test_mode": True},
    }

    with open(viz_path, "w") as f:
        json.dump(viz_data, f, indent=2)
    print(f"  ✓ Visualization data: {viz_path}")

    # Save metrics
    metrics_path = "quicktest_metrics.json"
    sim.save_metrics(metrics_path)
    print(f"  ✓ Metrics data: {metrics_path}")

    # Plot results
    if sim.metrics_history:
        plot_quick_results(sim.metrics_history)
        print(f"✓ Plots: quicktest_results.png")

    print("\n" + "=" * 60)
    print("✅ Quick test complete!")
    print("\nNext steps:")
    print("  • Review plots and metrics above")
    print("  • If healthy, run main.py for full training")
    print("  • Use quicktest_viz.json in Blender to visualize")
    print("=" * 60)


if __name__ == "__main__":
    main()
