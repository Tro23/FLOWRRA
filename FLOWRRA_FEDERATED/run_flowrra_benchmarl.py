"""
run_flowrra_benchmarl.py - FINAL WORKING VERSION

This is the complete, working BenchMARL integration.

Features:
- Proper model registration
- FLOWRRA vs Baseline comparison
- Automatic metric logging
- Video generation
- WandB integration

Usage:
    python run_flowrra_benchmarl.py --mode both
    python run_flowrra_benchmarl.py --mode baseline
    python run_flowrra_benchmarl.py --mode flowrra
"""

import argparse
import os
from pathlib import Path

import torch
from benchmarl.algorithms import MappoConfig
from benchmarl.environments import PettingZooTask
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.models.mlp import MlpConfig

# Import FLOWRRA model and register it
from flowrra_model import (
    FlowrraModelConfig,
    create_flowrra_model_config,
    register_flowrra_model,
)


def setup_experiment_config(name: str = "flowrra_experiment") -> ExperimentConfig:
    """
    Setup experiment configuration.

    Args:
        name: Experiment name for logging
    """
    config = ExperimentConfig.get_from_yaml()

    # Paths
    current_dir = Path(__file__).parent
    config.save_folder = current_dir / "benchmarl_results" / name

    # Logging
    config.loggers = ["csv", "tensorboard"]  # CSV + TensorBoard (WandB optional)

    # Training parameters
    config.max_n_iters = 500  # Number of training iterations
    config.on_policy_collected_frames_per_batch = 1600  # 8 agents * 200 steps
    config.on_policy_n_minibatch_iters = 10

    # Evaluation
    config.evaluation = True
    config.evaluation_interval = 8000  # Evaluate every 5 iteration
    config.evaluation_episodes = 5

    # Checkpointing
    config.checkpoint_interval = 6400
    config.checkpoint_at_end = True

    # Video generation (disable for faster training)
    config.create_json = True
    config.render = False  # Set True to generate videos

    print(f"\n[Config] Experiment: {name}")
    print(f"[Config] Save folder: {config.save_folder}")
    print(f"[Config] Max iterations: {config.max_n_iters}")
    print(f"[Config] Batch size: {config.on_policy_collected_frames_per_batch}")

    return config


def setup_task(N: int = 8, max_steps: int = 800) -> PettingZooTask:
    """
    Setup PettingZoo task.

    Args:
        N: Number of agents
        max_steps: Episode length
    """
    # Use Simple Spread (cooperative navigation)
    task = PettingZooTask.SIMPLE_SPREAD.get_from_yaml()

    # Configure
    task.config["N"] = N
    task.config["max_cycles"] = max_steps
    task.config["continuous_actions"] = False  # Discrete for FLOWRRA

    print(f"\n[Task] PettingZoo Simple Spread")
    print(f"[Task] Agents: {N}")
    print(f"[Task] Episode length: {max_steps}")
    print(f"[Task] Action space: Discrete")

    return task


def setup_algorithm() -> MappoConfig:
    """Setup MAPPO algorithm."""
    algo_config = MappoConfig.get_from_yaml()

    # MAPPO hyperparameters
    algo_config.share_param_critic = True
    # algo_config.gae_lambda = 0.95
    algo_config.clip_epsilon = 0.2
    algo_config.critic_coef = 1.0
    algo_config.entropy_coef = 0.01

    print(f"\n[Algorithm] MAPPO")
    # print(f"[Algorithm] GAE lambda: {algo_config.gae_lambda}")
    print(f"[Algorithm] Clip epsilon: {algo_config.clip_epsilon}")

    return algo_config


def run_baseline_experiment(
    task: PettingZooTask, algo_config: MappoConfig, exp_config: ExperimentConfig
):
    """
    Run BASELINE experiment (standard MLP policy).

    This is the SOTA multi-agent RL approach without FLOWRRA.
    """
    print("\n" + "=" * 70)
    print("🏃 RUNNING: BASELINE (MLP Policy)")
    print("=" * 70)

    # Standard MLP model
    model_config = MlpConfig.get_from_yaml()
    model_config.num_cells = [256, 256]  # 2-layer MLP
    model_config.activation_class = torch.nn.Tanh
    model_config.layer_class = torch.nn.Linear

    print(f"[Baseline] Model: MLP")
    print(f"[Baseline] Architecture: {model_config.num_cells}")

    # Create experiment
    experiment = Experiment(
        task=task,
        algorithm_config=algo_config,
        model_config=model_config,
        seed=42,
        config=exp_config,
    )

    # Run
    print(f"[Baseline] Starting training...")
    experiment.run()

    print(f"[Baseline] ✅ Training complete!")
    print(f"[Baseline] Results saved to: {exp_config.save_folder}")


def run_flowrra_experiment(
    task: PettingZooTask,
    algo_config: MappoConfig,
    exp_config: ExperimentConfig,
    n_holons: int = 2,
    use_sensor_fusion: bool = True,
):
    """
    Run FLOWRRA experiment (federated system with sensor fusion).

    This uses FLOWRRA's GNN + physics + coordination.
    """
    print("\n" + "=" * 70)
    print("🚀 RUNNING: FLOWRRA (Federated + Sensor Fusion)")
    print("=" * 70)

    # Create FLOWRRA model config
    model_config = create_flowrra_model_config(
        n_agents=task.config["N"],
        n_holons=n_holons,
        use_sensor_fusion=use_sensor_fusion,
        gnn_hidden_dim=128,
        gnn_num_layers=3,
        gnn_n_heads=4,
        enable_flowrra_learning=True,
    )

    print(f"[FLOWRRA] Model: Federated GNN")
    print(f"[FLOWRRA] Holons: {n_holons}")
    print(f"[FLOWRRA] Sensor fusion: {use_sensor_fusion}")
    print(f"[FLOWRRA] GNN hidden dim: {model_config.gnn_hidden_dim}")
    print(f"[FLOWRRA] GNN layers: {model_config.gnn_num_layers}")

    # Create experiment
    experiment = Experiment(
        task=task,
        algorithm_config=algo_config,
        model_config=model_config,
        seed=42,
        config=exp_config,
    )

    # Run
    print(f"[FLOWRRA] Starting training...")
    experiment.run()

    print(f"[FLOWRRA] ✅ Training complete!")
    print(f"[FLOWRRA] Results saved to: {exp_config.save_folder}")


def compare_results(baseline_folder: Path, flowrra_folder: Path):
    """
    Compare results between baseline and FLOWRRA.

    This loads CSV logs and generates comparison plots.
    """
    print("\n" + "=" * 70)
    print("📊 COMPARING RESULTS")
    print("=" * 70)

    import matplotlib.pyplot as plt
    import pandas as pd

    # Load CSV logs
    try:
        baseline_csv = list(baseline_folder.glob("**/logs.csv"))[0]
        flowrra_csv = list(flowrra_folder.glob("**/logs.csv"))[0]

        baseline_df = pd.read_csv(baseline_csv)
        flowrra_df = pd.read_csv(flowrra_csv)

        # Plot rewards
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Episode rewards
        axes[0, 0].plot(baseline_df["episode_reward_mean"], label="Baseline", alpha=0.7)
        axes[0, 0].plot(flowrra_df["episode_reward_mean"], label="FLOWRRA", alpha=0.7)
        axes[0, 0].set_title("Episode Reward")
        axes[0, 0].set_xlabel("Iteration")
        axes[0, 0].set_ylabel("Reward")
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)

        # Episode length
        axes[0, 1].plot(baseline_df["episode_len_mean"], label="Baseline", alpha=0.7)
        axes[0, 1].plot(flowrra_df["episode_len_mean"], label="FLOWRRA", alpha=0.7)
        axes[0, 1].set_title("Episode Length")
        axes[0, 1].set_xlabel("Iteration")
        axes[0, 1].set_ylabel("Steps")
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)

        # FLOWRRA-specific metrics (if available)
        if "system/avg_integrity" in flowrra_df.columns:
            axes[1, 0].plot(
                flowrra_df["system/avg_integrity"], color="green", alpha=0.7
            )
            axes[1, 0].set_title("FLOWRRA Loop Integrity")
            axes[1, 0].set_xlabel("Iteration")
            axes[1, 0].set_ylabel("Integrity")
            axes[1, 0].grid(alpha=0.3)

        if "system/avg_coherence" in flowrra_df.columns:
            axes[1, 1].plot(
                flowrra_df["system/avg_coherence"], color="purple", alpha=0.7
            )
            axes[1, 1].set_title("FLOWRRA System Coherence")
            axes[1, 1].set_xlabel("Iteration")
            axes[1, 1].set_ylabel("Coherence")
            axes[1, 1].grid(alpha=0.3)

        plt.tight_layout()

        # Save comparison plot
        comparison_path = baseline_folder.parent / "comparison.png"
        plt.savefig(comparison_path, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"[Compare] ✅ Comparison plot saved to: {comparison_path}")

        # Print summary statistics
        print(f"\n[Compare] BASELINE:")
        print(f"  Final reward: {baseline_df['episode_reward_mean'].iloc[-1]:.2f}")
        print(f"  Best reward: {baseline_df['episode_reward_mean'].max():.2f}")

        print(f"\n[Compare] FLOWRRA:")
        print(f"  Final reward: {flowrra_df['episode_reward_mean'].iloc[-1]:.2f}")
        print(f"  Best reward: {flowrra_df['episode_reward_mean'].max():.2f}")

        if "system/avg_integrity" in flowrra_df.columns:
            print(
                f"  Final integrity: {flowrra_df['system/avg_integrity'].iloc[-1]:.3f}"
            )

    except Exception as e:
        print(f"[Compare] ⚠️  Could not compare results: {e}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="FLOWRRA BenchMARL Experiments")
    parser.add_argument(
        "--mode",
        type=str,
        default="both",
        choices=["baseline", "flowrra", "both"],
        help="Which experiment(s) to run",
    )
    parser.add_argument("--agents", type=int, default=16, help="Number of agents")
    parser.add_argument(
        "--holons", type=int, default=4, help="Number of holons (FLOWRRA only)"
    )
    parser.add_argument("--iters", type=int, default=400, help="Training iterations")
    parser.add_argument(
        "--no-sensor-fusion", action="store_true", help="Disable sensor fusion"
    )

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("🎯 FLOWRRA vs BASELINE COMPARISON")
    print("=" * 70)
    print(f"Mode: {args.mode}")
    print(f"Agents: {args.agents}")
    if args.mode in ["flowrra", "both"]:
        print(f"Holons: {args.holons}")
        print(f"Sensor fusion: {not args.no_sensor_fusion}")

    # Setup shared components
    task = setup_task(N=args.agents, max_steps=400)
    algo_config = setup_algorithm()

    baseline_folder = None
    flowrra_folder = None

    # Run experiments
    if args.mode in ["flowrra", "both"]:
        exp_config = setup_experiment_config(name="flowrra")
        exp_config.max_n_iters = args.iters
        flowrra_folder = exp_config.save_folder

        run_flowrra_experiment(
            task,
            algo_config,
            exp_config,
            n_holons=args.holons,
            use_sensor_fusion=not args.no_sensor_fusion,
        )

    if args.mode in ["baseline", "both"]:
        exp_config = setup_experiment_config(name="baseline")
        exp_config.max_n_iters = args.iters
        baseline_folder = exp_config.save_folder

        run_baseline_experiment(task, algo_config, exp_config)

    # Compare if both were run
    if args.mode == "both" and baseline_folder and flowrra_folder:
        compare_results(baseline_folder, flowrra_folder)

    print("\n" + "=" * 70)
    print("✅ ALL EXPERIMENTS COMPLETE")
    print("=" * 70)
    print(f"\nResults saved to: benchmarl_results/")
    print(f"View logs with: tensorboard --logdir benchmarl_results/")


if __name__ == "__main__":
    # Register FLOWRRA model with BenchMARL
    register_flowrra_model()

    # Run experiments
    main()
