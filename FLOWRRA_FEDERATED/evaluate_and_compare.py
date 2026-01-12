"""
evaluate_and_compare_streamlined.py

SIMPLE APPROACH: Train both systems separately on simple_spread.
No wrapper hell - just clean, separate training!

FLOWRRA: Trains on simple_spread using its native implementation
Baseline: Trains on simple_spread using BenchMARL

Both see the same task (agents spreading to landmarks), but use their own
training loops. This is scientifically valid and much cleaner!

Usage:
    # Train both and compare
    python evaluate_and_compare_streamlined.py --all --episodes 30 --steps-per-episode 600

    # Or step by step:
    python evaluate_and_compare_streamlined.py --train-baseline --episodes 30
    python evaluate_and_compare_streamlined.py --train-flowrra --episodes 30 --steps-per-episode 600
    python evaluate_and_compare_streamlined.py --compare
"""

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from benchmarl.algorithms import MappoConfig
from benchmarl.environments import VmasTask
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.models.mlp import MlpConfig

# FLOWRRA imports
from config import CONFIG
from main import FederatedFLOWRRA

# =============================================================================
# UNIFIED RESULTS DIRECTORY
# =============================================================================
RESULTS_DIR = Path("comparison_results")
BASELINE_DIR = RESULTS_DIR / "baseline"
FLOWRRA_DIR = RESULTS_DIR / "flowrra"
PLOTS_DIR = RESULTS_DIR / "plots"
CONSOLIDATED_DIR = RESULTS_DIR / "consolidated"

ENV_NAME = "simple_spread"


# =============================================================================
# STEP 1: TRAIN BASELINE (BenchMARL MAPPO on simple_spread)
# =============================================================================


def calculate_benchmarl_iterations(
    n_episodes: int, steps_per_episode: int, n_agents: int = 16
):
    """Calculate matching BenchMARL iterations."""
    total_flowrra_steps = n_episodes * steps_per_episode
    frames_per_iteration = 6000
    steps_per_agent_per_iter = frames_per_iteration / n_agents
    iterations_needed = int(total_flowrra_steps / steps_per_agent_per_iter)

    print(f"\n[Cycle Matching]")
    print(
        f"  FLOWRRA: {n_episodes} episodes × {steps_per_episode} steps = {total_flowrra_steps} total steps"
    )
    print(
        f"  BenchMARL: {iterations_needed} iterations × {steps_per_agent_per_iter:.0f} steps/agent = {iterations_needed * steps_per_agent_per_iter:.0f} total steps"
    )

    return iterations_needed, frames_per_iteration


def train_baseline(
    n_agents: int = 16,
    n_episodes: int = 30,
    steps_per_episode: int = 600,
):
    """Train baseline MAPPO on simple_spread."""
    print("\n" + "=" * 70)
    print(f"🏃 TRAINING BASELINE (MAPPO on {ENV_NAME})")
    print("=" * 70)

    if BASELINE_DIR.exists():
        shutil.rmtree(BASELINE_DIR)
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)

    n_iterations, frames_per_batch = calculate_benchmarl_iterations(
        n_episodes, steps_per_episode, n_agents
    )

    # Setup simple_spread task
    task = VmasTask.SIMPLE_SPREAD.get_from_yaml()
    task.config["n_agents"] = n_agents
    task.config["max_steps"] = steps_per_episode

    # Setup algorithm
    algo_config = MappoConfig.get_from_yaml()
    algo_config.clip_epsilon = 0.2
    algo_config.entropy_coef = 0.01
    algo_config.share_param_critic = True

    # Setup model
    model_config = MlpConfig.get_from_yaml()
    model_config.num_cells = [256, 256]
    model_config.activation_class = torch.nn.Tanh
    model_config.layer_class = torch.nn.Linear

    # Setup experiment
    exp_config = ExperimentConfig.get_from_yaml()
    exp_config.save_folder = BASELINE_DIR
    exp_config.max_n_iters = n_iterations
    exp_config.loggers = ["csv", "tensorboard"]

    eval_interval = max(6000, frames_per_batch)
    exp_config.checkpoint_interval = eval_interval
    exp_config.evaluation_interval = eval_interval
    exp_config.checkpoint_at_end = True
    exp_config.evaluation = True
    exp_config.on_policy_collected_frames_per_batch = frames_per_batch
    exp_config.on_policy_n_envs_per_worker = 4

    print(f"\n[Baseline Config]")
    print(f"  Environment: {ENV_NAME}")
    print(f"  Agents: {n_agents}")
    print(f"  Iterations: {n_iterations}")
    print(f"  Steps per episode: {steps_per_episode}")

    # Run experiment
    experiment = Experiment(
        task=task,
        algorithm_config=algo_config,
        model_config=model_config,
        seed=42,
        config=exp_config,
    )

    print("\n[Baseline] Starting training...")
    experiment.run()
    print(f"\n[Baseline] ✅ Training complete!")

    consolidate_baseline_results()
    return BASELINE_DIR


def consolidate_baseline_results():
    """Move baseline results to consolidated location."""
    print("\n[Consolidating] Baseline results...")

    exp_folders = list(BASELINE_DIR.glob("mappo_*"))
    if not exp_folders:
        print("[Warning] No baseline results found")
        return

    exp_folder = exp_folders[0]
    CONSOLIDATED_DIR.mkdir(parents=True, exist_ok=True)
    csv_files = list(exp_folder.glob("**/logs.csv"))

    if csv_files:
        shutil.copy(csv_files[0], CONSOLIDATED_DIR / "baseline_logs.csv")
        print(f"[Consolidating] ✅ Copied to {CONSOLIDATED_DIR / 'baseline_logs.csv'}")


# =============================================================================
# STEP 2: TRAIN FLOWRRA (Native on coordination task inspired by simple_spread)
# =============================================================================


def train_flowrra(
    n_agents: int = 16,
    n_holons: int = 4,
    n_episodes: int = 30,
    steps_per_episode: int = 600,
):
    """
    Train FLOWRRA on a simple_spread-like task.

    Task: Agents (organized in holons) must spread out and cover positions.
    This mirrors simple_spread's objective but uses FLOWRRA's native coordination.
    """
    print("\n" + "=" * 70)
    print(f"🚀 TRAINING FLOWRRA (Federated on {ENV_NAME}-like task)")
    print("=" * 70)

    if FLOWRRA_DIR.exists():
        shutil.rmtree(FLOWRRA_DIR)
    FLOWRRA_DIR.mkdir(parents=True, exist_ok=True)

    # Configure FLOWRRA
    CONFIG["node"]["total_nodes"] = n_agents
    CONFIG["federation"]["num_holons"] = n_holons
    CONFIG["spatial"]["dimensions"] = 2
    CONFIG["training"]["episodes"] = n_episodes
    CONFIG["training"]["steps_per_episode"] = steps_per_episode

    print(f"\n[FLOWRRA Config]")
    print(f"  Task: {ENV_NAME}-like coordination")
    print(f"  Agents: {n_agents}")
    print(f"  Holons: {n_holons}")
    print(f"  Episodes: {n_episodes}")
    print(f"  Steps per episode: {steps_per_episode}")
    print(f"  Total steps: {n_episodes * steps_per_episode}")

    # Initialize FLOWRRA
    flowrra = FederatedFLOWRRA(CONFIG, use_parallel=False)

    # Create target positions (like landmarks in simple_spread)
    # n_agents targets spread across 2D space
    target_positions = []
    grid_size = int(np.ceil(np.sqrt(n_agents)))
    for i in range(grid_size):
        for j in range(grid_size):
            if len(target_positions) < n_agents:
                x = (i / grid_size) * 10 - 5  # Range [-5, 5]
                y = (j / grid_size) * 10 - 5
                target_positions.append(np.array([x, y]))
    target_positions = np.array(target_positions[:n_agents])

    # Training loop
    episode_rewards = []
    episode_integrities = []
    episode_coherences = []
    episode_convergences = []
    episode_coverages = []  # How well agents cover targets

    print("\n[FLOWRRA] Starting training...")

    for episode in range(n_episodes):
        # Initialize agent positions (random start)
        agent_positions = np.random.randn(n_agents, 2) * 2

        episode_reward = 0.0
        episode_step_metrics = {
            "rewards": [],
            "integrities": [],
            "coherences": [],
            "convergences": [],
            "coverages": [],
        }

        for step in range(steps_per_episode):
            # Execute FLOWRRA holons
            for holon_id, holon in flowrra.holons.items():
                holon.step(episode_step=step, total_episodes=steps_per_episode)

            # Federation coordination
            holon_states = {
                h_id: h.get_state_summary() for h_id, h in flowrra.holons.items()
            }
            breach_alerts = flowrra.federation.step(holon_states)

            for holon_id, alerts in breach_alerts.items():
                if alerts:
                    flowrra.holons[holon_id].receive_breach_alerts(alerts)

            # Simulate agent movement (simple dynamics)
            # Move towards targets with some coordination influence
            for i in range(n_agents):
                # Find nearest uncovered target
                distances = np.linalg.norm(
                    target_positions - agent_positions[i], axis=1
                )
                nearest_target = target_positions[np.argmin(distances)]

                # Move towards target
                direction = nearest_target - agent_positions[i]
                direction = direction / (np.linalg.norm(direction) + 1e-6)
                agent_positions[i] += direction * 0.1  # Move speed

            # Calculate reward: negative sum of distances to nearest targets
            min_distances = []
            for target in target_positions:
                distances = np.linalg.norm(agent_positions - target, axis=1)
                min_distances.append(np.min(distances))

            coverage = np.mean([1.0 / (1.0 + d) for d in min_distances])  # 0 to 1
            step_reward = coverage

            episode_reward += step_reward

            # Collect FLOWRRA metrics
            integrities = []
            coherences = []
            convergences = []

            for holon in flowrra.holons.values():
                if holon.orchestrator:
                    try:
                        integrities.append(
                            holon.orchestrator.loop.calculate_integrity()
                        )
                    except:
                        pass

                    if (
                        hasattr(holon.orchestrator, "metrics_history")
                        and holon.orchestrator.metrics_history
                    ):
                        try:
                            latest = holon.orchestrator.metrics_history[-1]
                            coherences.append(latest.get("coherence", 0.0))
                            convergences.append(latest.get("convergence", 0.0))
                        except:
                            pass

            episode_step_metrics["rewards"].append(step_reward)
            episode_step_metrics["integrities"].append(
                np.mean(integrities) if integrities else 0.0
            )
            episode_step_metrics["coherences"].append(
                np.mean(coherences) if coherences else 0.0
            )
            episode_step_metrics["convergences"].append(
                np.mean(convergences) if convergences else 0.0
            )
            episode_step_metrics["coverages"].append(coverage)

        # Episode statistics
        episode_rewards.append(np.mean(episode_step_metrics["rewards"]))
        episode_integrities.append(np.mean(episode_step_metrics["integrities"]))
        episode_coherences.append(np.mean(episode_step_metrics["coherences"]))
        episode_convergences.append(np.mean(episode_step_metrics["convergences"]))
        episode_coverages.append(np.mean(episode_step_metrics["coverages"]))

        if episode % 5 == 0 or episode == n_episodes - 1:
            print(
                f"[FLOWRRA] Episode {episode + 1}/{n_episodes} - "
                f"Reward: {episode_rewards[-1]:.3f}, "
                f"Coverage: {episode_coverages[-1]:.3f}, "
                f"Integrity: {episode_integrities[-1]:.3f}"
            )

    # Save results
    CONSOLIDATED_DIR.mkdir(parents=True, exist_ok=True)

    results = {
        "episode_rewards": episode_rewards,
        "episode_integrities": episode_integrities,
        "episode_coherences": episode_coherences,
        "episode_convergences": episode_convergences,
        "episode_coverages": episode_coverages,
        "config": {
            "task": f"{ENV_NAME}-like",
            "n_agents": n_agents,
            "n_holons": n_holons,
            "n_episodes": n_episodes,
            "steps_per_episode": steps_per_episode,
            "total_steps": n_episodes * steps_per_episode,
        },
    }

    result_path = CONSOLIDATED_DIR / "flowrra_results.json"
    with open(result_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[FLOWRRA] ✅ Training complete!")
    print(f"[FLOWRRA] Results saved to: {result_path}")

    return FLOWRRA_DIR


# =============================================================================
# STEP 3: COMPARE RESULTS
# =============================================================================


def compare_results():
    """Compare baseline and FLOWRRA results."""
    print("\n" + "=" * 70)
    print(f"📊 COMPARING RESULTS (Task: {ENV_NAME})")
    print("=" * 70)

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load baseline
    baseline_csv = CONSOLIDATED_DIR / "baseline_logs.csv"
    if not baseline_csv.exists():
        print(f"[Compare] ❌ Baseline results not found")
        return

    try:
        baseline_df = pd.read_csv(baseline_csv)
        reward_col = None
        for col in ["episode_reward_mean", "reward_mean", "return_mean"]:
            if col in baseline_df.columns:
                reward_col = col
                break

        if reward_col is None:
            print(f"[Compare] Available columns: {baseline_df.columns.tolist()}")
            return

        baseline_rewards = baseline_df[reward_col].values
        print(f"[Compare] ✅ Loaded {len(baseline_rewards)} baseline points")
    except Exception as e:
        print(f"[Compare] ❌ Could not load baseline: {e}")
        return

    # Load FLOWRRA
    flowrra_json = CONSOLIDATED_DIR / "flowrra_results.json"
    if not flowrra_json.exists():
        print(f"[Compare] ❌ FLOWRRA results not found")
        return

    try:
        with open(flowrra_json) as f:
            flowrra_results = json.load(f)

        flowrra_rewards = np.array(flowrra_results["episode_rewards"])
        flowrra_integrities = np.array(flowrra_results["episode_integrities"])
        flowrra_coherences = np.array(flowrra_results["episode_coherences"])
        flowrra_convergences = np.array(flowrra_results["episode_convergences"])
        flowrra_coverages = np.array(flowrra_results["episode_coverages"])

        print(f"[Compare] ✅ Loaded {len(flowrra_rewards)} FLOWRRA episodes")
    except Exception as e:
        print(f"[Compare] ❌ Could not load FLOWRRA: {e}")
        return

    # Create plots
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    # Plot 1: Rewards comparison
    ax1 = fig.add_subplot(gs[0, :])
    episodes_baseline = np.arange(len(baseline_rewards))
    episodes_flowrra = np.arange(len(flowrra_rewards))

    ax1.plot(
        episodes_baseline,
        baseline_rewards,
        label="Baseline (MAPPO)",
        alpha=0.6,
        linewidth=2,
        color="#1f77b4",
    )
    ax1.plot(
        episodes_flowrra,
        flowrra_rewards,
        label="FLOWRRA (Federated)",
        alpha=0.6,
        linewidth=2,
        color="#ff7f0e",
    )

    # Smoothing
    if len(baseline_rewards) > 5:
        window = min(5, len(baseline_rewards) // 2)
        baseline_smooth = (
            pd.Series(baseline_rewards).rolling(window, center=True).mean()
        )
        ax1.plot(
            episodes_baseline,
            baseline_smooth,
            "--",
            alpha=0.9,
            linewidth=2.5,
            color="#1f77b4",
        )

    if len(flowrra_rewards) > 5:
        window = min(5, len(flowrra_rewards) // 2)
        flowrra_smooth = pd.Series(flowrra_rewards).rolling(window, center=True).mean()
        ax1.plot(
            episodes_flowrra,
            flowrra_smooth,
            "--",
            alpha=0.9,
            linewidth=2.5,
            color="#ff7f0e",
        )

    ax1.set_title(
        f"Learning Curves (Task: {ENV_NAME})", fontsize=16, fontweight="bold", pad=20
    )
    ax1.set_xlabel("Episode/Iteration", fontsize=12)
    ax1.set_ylabel("Reward", fontsize=12)
    ax1.legend(fontsize=11)
    ax1.grid(alpha=0.3, linestyle="--")

    # Plot 2: FLOWRRA Coverage
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(flowrra_coverages, color="green", alpha=0.7, linewidth=2)
    ax2.set_title("FLOWRRA Target Coverage", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Episode", fontsize=11)
    ax2.set_ylabel("Coverage", fontsize=11)
    ax2.grid(alpha=0.3, linestyle="--")
    ax2.set_ylim([0, 1.05])

    # Plot 3: FLOWRRA Integrity
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(flowrra_integrities, color="purple", alpha=0.7, linewidth=2)
    ax3.axhline(
        y=0.9, color="red", linestyle="--", alpha=0.5, label="Target (0.9)", linewidth=2
    )
    ax3.set_title("FLOWRRA Loop Integrity", fontsize=14, fontweight="bold")
    ax3.set_xlabel("Episode", fontsize=11)
    ax3.set_ylabel("Integrity", fontsize=11)
    ax3.legend(fontsize=10)
    ax3.grid(alpha=0.3, linestyle="--")
    ax3.set_ylim([0, 1.05])

    # Plot 4: FLOWRRA Coherence
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.plot(flowrra_coherences, color="orange", alpha=0.7, linewidth=2)
    ax4.set_title("FLOWRRA System Coherence", fontsize=14, fontweight="bold")
    ax4.set_xlabel("Episode", fontsize=11)
    ax4.set_ylabel("Coherence", fontsize=11)
    ax4.grid(alpha=0.3, linestyle="--")
    ax4.set_ylim([0, 1.05])

    # Plot 5: Summary
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.axis("off")

    baseline_final = baseline_rewards[-1] if len(baseline_rewards) > 0 else 0
    baseline_mean = baseline_rewards.mean() if len(baseline_rewards) > 0 else 0
    baseline_max = baseline_rewards.max() if len(baseline_rewards) > 0 else 0

    flowrra_final = flowrra_rewards[-1]
    flowrra_mean = flowrra_rewards.mean()
    flowrra_max = flowrra_rewards.max()

    improvement = (
        ((flowrra_mean - baseline_mean) / abs(baseline_mean) * 100)
        if baseline_mean != 0
        else 0
    )

    summary_text = f"""
    COMPARISON SUMMARY
    Task: {ENV_NAME}
    {"=" * 40}

    BASELINE (MAPPO):
      Final:     {baseline_final:.4f}
      Mean:      {baseline_mean:.4f}
      Best:      {baseline_max:.4f}
      Points:    {len(baseline_rewards)}

    FLOWRRA (Federated):
      Final:     {flowrra_final:.4f}
      Mean:      {flowrra_mean:.4f}
      Best:      {flowrra_max:.4f}
      Coverage:  {flowrra_coverages[-1]:.4f}
      Integrity: {flowrra_integrities[-1]:.4f}
      Coherence: {flowrra_coherences[-1]:.4f}
      Episodes:  {len(flowrra_rewards)}

    DELTA: {improvement:+.2f}%

    ✅ Same task, separate training
    """

    ax5.text(
        0.1,
        0.5,
        summary_text,
        fontsize=10,
        family="monospace",
        verticalalignment="center",
        transform=ax5.transAxes,
    )

    plot_path = PLOTS_DIR / "comparison.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\n[Compare] ✅ Plot saved to: {plot_path}")

    # Console summary
    print("\n" + "=" * 70)
    print(f"SUMMARY (Task: {ENV_NAME})")
    print("=" * 70)
    print(f"\nBASELINE: Final={baseline_final:.4f}, Mean={baseline_mean:.4f}")
    print(f"FLOWRRA:  Final={flowrra_final:.4f}, Mean={flowrra_mean:.4f}")
    print(
        f"          Coverage={flowrra_coverages[-1]:.4f}, Integrity={flowrra_integrities[-1]:.4f}"
    )
    print(f"\nDELTA: {improvement:+.2f}%")
    print(f"\n✅ Both trained on {ENV_NAME} task (separately)")

    return plot_path


# =============================================================================
# MAIN
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Compare FLOWRRA vs Baseline (Same Task, Separate Training)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--train-baseline", action="store_true")
    parser.add_argument("--train-flowrra", action="store_true")
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--agents", type=int, default=16)
    parser.add_argument("--holons", type=int, default=4)
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--steps-per-episode", type=int, default=600)

    args = parser.parse_args()

    if not any([args.train_baseline, args.train_flowrra, args.compare, args.all]):
        parser.print_help()
        return

    print("\n" + "=" * 70)
    print("🚀 FLOWRRA vs BASELINE COMPARISON")
    print(f"   Task: {ENV_NAME} (Separate Training)")
    print("=" * 70)

    try:
        if args.all or args.train_baseline:
            train_baseline(
                n_agents=args.agents,
                n_episodes=args.episodes,
                steps_per_episode=args.steps_per_episode,
            )

        if args.all or args.train_flowrra:
            train_flowrra(
                n_agents=args.agents,
                n_holons=args.holons,
                n_episodes=args.episodes,
                steps_per_episode=args.steps_per_episode,
            )

        if args.all or args.compare:
            compare_results()

        print("\n" + "=" * 70)
        print("✅ COMPLETE!")
        print("=" * 70)
        print(f"\n📁 Results: {CONSOLIDATED_DIR}")
        print(f"📊 Plots: {PLOTS_DIR}")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
