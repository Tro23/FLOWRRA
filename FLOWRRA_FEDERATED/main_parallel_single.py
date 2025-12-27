"""
main.py - FIXED VERSION

Federated FLOWRRA Entry Point

FIXES:
- Proper NodePositionND creation
- Correct holon initialization sequence
- Better error handling
- Phase 2 R-GNN support hooks
- Fixed visualization method signature

Usage:
    python main.py --episodes 2000 --holons 4 --nodes 40
"""

import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

from config import CONFIG
from federation.manager import FederationManager
from holon.holon_core import Holon
from holon.node import NodePositionND


class FederatedFLOWRRA:
    """
    Main orchestrator for federated multi-holon system.
    """

    def __init__(self, config: Dict, use_parallel: bool = True):
        self.cfg = config
        self.use_parallel = use_parallel

        # Thread lock for metrics collection
        self.metrics_lock = threading.Lock()

        print("\n" + "=" * 70)
        print("FEDERATED FLOWRRA INITIALIZATION")
        if use_parallel:
            print("MODE: PARALLEL EXECUTION (Single GPU)")
        else:
            print("MODE: SEQUENTIAL EXECUTION")
        print("=" * 70)

        # Initialize Federation Manager
        self.federation = FederationManager(
            num_holons=config["federation"]["num_holons"],
            world_bounds=config["federation"]["world_bounds"],
            breach_threshold=config["federation"]["breach_threshold"],
            coordination_mode=config["federation"]["coordination_mode"],
        )

        # Initialize Holons (without nodes yet)
        self.holons: Dict[int, Holon] = {}
        self._initialize_holons()

        # Initialize Nodes and distribute to holons
        self._initialize_and_distribute_nodes()

        # Metrics tracking
        self.episode_rewards: List[float] = []
        self.federation_metrics: List[Dict] = []
        self.holon_metrics: Dict[int, List[Dict]] = {
            holon_id: [] for holon_id in self.holons.keys()
        }

        print("\n" + "=" * 70)
        print("INITIALIZATION COMPLETE")
        print("=" * 70 + "\n")

    def _initialize_holons(self):
        """Create holon instances for each spatial partition."""
        partition_assignments = self.federation.get_partition_assignments()

        for partition_id, partition in partition_assignments.items():
            holon = Holon(
                holon_id=partition_id,
                partition_id=partition_id,
                spatial_bounds={"x": partition.bounds_x, "y": partition.bounds_y},
                config=self.cfg,
                mode=self.cfg["holon"]["mode"],
            )

            self.holons[partition_id] = holon

        print(f"[Main] Created {len(self.holons)} holons")

    def _initialize_and_distribute_nodes(self):
        """Create REAL NodePositionND nodes and distribute them across holons."""
        total_nodes = self.cfg["node"]["total_nodes"]
        nodes_per_holon = self.cfg["node"]["num_nodes_per_holon"]
        dimensions = self.cfg["spatial"]["dimensions"]

        print(
            f"\n[Main] Distributing {total_nodes} nodes across {len(self.holons)} holons"
        )
        print(f"[Main] {nodes_per_holon} nodes per holon")

        node_id = 0

        for holon_id, holon in self.holons.items():
            # Get holon's spatial bounds
            x_min, x_max = holon.spatial_bounds["x"]
            y_min, y_max = holon.spatial_bounds["y"]

            # Calculate equilibrium radius for this holon's loop
            ideal_dist = self.cfg["loop"]["ideal_distance"]
            equilibrium_radius = (nodes_per_holon * ideal_dist) / (2 * np.pi)

            # Clamp to holon bounds
            max_radius = min((x_max - x_min) / 2, (y_max - y_min) / 2) * 0.8

            equilibrium_radius = min(equilibrium_radius, max_radius)

            # Holon center in global coordinates
            holon_center = holon.center

            # Create nodes for this holon in equilibrium ring
            holon_nodes = []

            for i in range(nodes_per_holon):
                angle = (i / nodes_per_holon) * 2 * np.pi

                # Position on equilibrium ring
                if dimensions == 2:
                    offset = (
                        np.array([np.cos(angle), np.sin(angle)]) * equilibrium_radius
                    )
                else:  # 3D
                    offset = (
                        np.array([np.cos(angle), np.sin(angle), 0.0])
                        * equilibrium_radius
                    )

                # Add tiny noise to break symmetry
                noise = np.random.normal(0, 0.001, dimensions)

                # Global position
                pos = holon_center + offset + noise

                # Clamp to holon bounds with small margin
                pos[0] = np.clip(pos[0], x_min + 0.02, x_max - 0.02)
                pos[1] = np.clip(pos[1], y_min + 0.02, y_max - 0.02)

                # Create REAL NodePositionND object
                node = NodePositionND(id=node_id, pos=pos, dimensions=dimensions)

                # Set node parameters
                node.sensor_range = self.cfg["node"]["sensor_range"]
                node.move_speed = self.cfg["node"]["move_speed"]

                holon_nodes.append(node)
                node_id += 1

            # Initialize orchestrator WITH the nodes
            holon.initialize_orchestrator_with_nodes(holon_nodes)

            print(
                f"[Main] Holon {holon_id}: {len(holon_nodes)} nodes at equilibrium (r={equilibrium_radius:.3f})"
            )

        print(f"[Main] Node distribution complete\n")

    def _execute_holon_step(
        self, holon: Holon, step: int, total_episodes: int
    ) -> tuple:
        """Execute a single holon step (for parallel execution)."""
        try:
            reward = holon.step(step, total_episodes)
            return (holon.holon_id, reward, None)
        except Exception as e:
            return (holon.holon_id, 0.0, e)

    def plot_federation_training_results(self):
        """Generate federation-wide training visualization."""

        print("[Viz] Creating federation training overview...")

        fig, axes = plt.subplots(3, 3, figsize=(18, 12))

        # ROW 1: Per-Holon Rewards
        for holon_id, holon in self.holons.items():
            if holon.orchestrator and hasattr(holon.orchestrator, "metrics_history"):
                if len(holon.orchestrator.metrics_history) > 0:
                    rewards = [
                        m.get("avg_reward", 0)
                        for m in holon.orchestrator.metrics_history
                    ]
                    axes[0, 0].plot(rewards, label=f"Holon {holon_id}", alpha=0.7)
        axes[0, 0].set_title("Per-Holon Average Reward")
        axes[0, 0].legend()
        axes[0, 0].set_xlabel("Step")
        axes[0, 0].set_ylabel("Reward")
        axes[0, 0].grid(alpha=0.3)

        # Federation Average Reward
        if len(self.episode_rewards) > 0:
            axes[0, 1].plot(self.episode_rewards, color="darkblue", linewidth=2)
        axes[0, 1].set_title("Federation Average Reward")
        axes[0, 1].set_xlabel("Episode")
        axes[0, 1].set_ylabel("Reward")
        axes[0, 1].grid(alpha=0.3)

        # Total Breaches Over Time
        if len(self.federation_metrics) > 0:
            breach_counts = [
                m.get("total_breaches", 0) for m in self.federation_metrics
            ]
            if breach_counts:
                axes[0, 2].plot(breach_counts, color="red", linewidth=2)
        axes[0, 2].set_title("Total Boundary Breaches")
        axes[0, 2].set_xlabel("Metric Save Point")
        axes[0, 2].set_ylabel("Breaches")
        axes[0, 2].grid(alpha=0.3)

        # ROW 2: Coherence & Integrity per Holon
        for holon_id, holon in self.holons.items():
            if holon.orchestrator and hasattr(holon.orchestrator, "metrics_history"):
                if len(holon.orchestrator.metrics_history) > 0:
                    coherence = [
                        m.get("coherence", 0)
                        for m in holon.orchestrator.metrics_history
                    ]
                    axes[1, 0].plot(coherence, label=f"Holon {holon_id}", alpha=0.7)
        axes[1, 0].set_title("System Coherence (All Holons)")
        axes[1, 0].axhline(
            y=0.4, linestyle="--", color="red", alpha=0.5, label="Threshold"
        )
        axes[1, 0].legend()
        axes[1, 0].set_xlabel("Step")
        axes[1, 0].set_ylabel("Coherence")
        axes[1, 0].grid(alpha=0.3)

        # Loop Integrity
        for holon_id, holon in self.holons.items():
            if holon.orchestrator and hasattr(holon.orchestrator, "metrics_history"):
                if len(holon.orchestrator.metrics_history) > 0:
                    integrity = [
                        m.get("loop_integrity", 0)
                        for m in holon.orchestrator.metrics_history
                    ]
                    axes[1, 1].plot(integrity, label=f"Holon {holon_id}", alpha=0.7)
        axes[1, 1].set_title("Loop Integrity (All Holons)")
        axes[1, 1].legend()
        axes[1, 1].set_xlabel("Step")
        axes[1, 1].set_ylabel("Integrity [0-1]")
        axes[1, 1].grid(alpha=0.3)

        # Coverage (combined across all holons)
        for holon_id, holon in self.holons.items():
            if holon.orchestrator and hasattr(holon.orchestrator, "metrics_history"):
                if len(holon.orchestrator.metrics_history) > 0:
                    coverage = [
                        m.get("coverage", 0) for m in holon.orchestrator.metrics_history
                    ]
                    axes[1, 2].plot(coverage, label=f"Holon {holon_id}", alpha=0.7)
        axes[1, 2].set_title("Map Coverage (Per Holon)")
        axes[1, 2].legend()
        axes[1, 2].set_xlabel("Step")
        axes[1, 2].set_ylabel("Coverage %")
        axes[1, 2].grid(alpha=0.3)

        # ROW 3: Training Dynamics
        # Training Loss per Holon
        has_loss_data = False
        for holon_id, holon in self.holons.items():
            if holon.orchestrator and hasattr(holon.orchestrator, "metrics_history"):
                losses = [
                    m.get("training_loss")
                    for m in holon.orchestrator.metrics_history
                    if m.get("training_loss") is not None
                ]
                if losses:
                    axes[2, 0].plot(losses, label=f"Holon {holon_id}", alpha=0.7)
                    has_loss_data = True
        if has_loss_data:
            axes[2, 0].set_title("GNN Training Loss")
            axes[2, 0].legend()
        else:
            axes[2, 0].set_title("GNN Training Loss (No Data Yet)")
        axes[2, 0].set_xlabel("Training Step")
        axes[2, 0].set_ylabel("Loss")
        axes[2, 0].grid(alpha=0.3)

        # WFC Recovery Mode Distribution
        wfc_spatial = []
        wfc_temporal = []
        holon_ids_wfc = []

        for holon_id, holon in self.holons.items():
            if holon.orchestrator and hasattr(holon.orchestrator, "wfc"):
                wfc_stats = holon.orchestrator.wfc.get_statistics()
                wfc_spatial.append(wfc_stats.get("spatial_recoveries", 0))
                wfc_temporal.append(wfc_stats.get("temporal_recoveries", 0))
                holon_ids_wfc.append(holon_id)

        if wfc_spatial or wfc_temporal:
            x = np.arange(len(holon_ids_wfc))
            width = 0.35

            axes[2, 1].bar(
                x - width / 2,
                wfc_spatial,
                width,
                label="Spatial (Forward)",
                color="green",
                alpha=0.7,
            )
            axes[2, 1].bar(
                x + width / 2,
                wfc_temporal,
                width,
                label="Temporal (Backward)",
                color="purple",
                alpha=0.7,
            )

            axes[2, 1].set_xticks(x)
            axes[2, 1].set_xticklabels([f"H{hid}" for hid in holon_ids_wfc])
            axes[2, 1].set_title("WFC Recovery Modes")
            axes[2, 1].set_xlabel("Holon ID")
            axes[2, 1].set_ylabel("Count")
            axes[2, 1].legend()
            axes[2, 1].grid(alpha=0.3, axis="y")
        else:
            axes[2, 1].set_title("WFC Recovery Modes (No Data Yet)")
            axes[2, 1].set_xlabel("Holon ID")
            axes[2, 1].set_ylabel("Count")
            axes[2, 1].grid(alpha=0.3)

        # Total WFC Triggers Across Federation
        wfc_counts = []
        holon_ids = []
        for holon_id, holon in self.holons.items():
            if holon.orchestrator and hasattr(holon.orchestrator, "wfc_trigger_events"):
                wfc_counts.append(len(holon.orchestrator.wfc_trigger_events))
                holon_ids.append(holon_id)

        if wfc_counts:
            axes[2, 2].bar(
                range(len(wfc_counts)), wfc_counts, color="orange", alpha=0.7
            )
            axes[2, 2].set_xticks(range(len(wfc_counts)))
            axes[2, 2].set_xticklabels([f"H{hid}" for hid in holon_ids])
        axes[2, 2].set_title("WFC Triggers Per Holon")
        axes[2, 2].set_xlabel("Holon ID")
        axes[2, 2].set_ylabel("Count")
        axes[2, 2].grid(alpha=0.3, axis="y")

        plt.suptitle(
            "FLOWRRA Federation Training Overview", fontsize=16, fontweight="bold"
        )
        plt.tight_layout()
        plt.savefig(
            "results/federation_training_overview.png", dpi=150, bbox_inches="tight"
        )
        plt.close()
        print(
            "[Viz] ✅ Saved federation training overview to results/federation_training_overview.png"
        )

    def train_episode(self, episode_num: int, total_episodes: int) -> float:
        """
        Execute one training episode with optional parallel execution.

        Returns:
            Average reward across all holons
        """
        steps_per_episode = self.cfg["training"]["steps_per_episode"]
        episode_start_time = time.time()

        total_reward = 0.0

        for step in range(steps_per_episode):
            # === PHASE A: HOLONS EXECUTE STEP ===
            holon_rewards = []

            if self.use_parallel:
                # PARALLEL EXECUTION (Single GPU with threading)
                with ThreadPoolExecutor(max_workers=len(self.holons)) as executor:
                    # Submit all holon steps concurrently
                    futures = {
                        executor.submit(
                            self._execute_holon_step, holon, step, total_episodes
                        ): holon.holon_id
                        for holon in self.holons.values()
                    }

                    # Collect results as they complete
                    for future in as_completed(futures):
                        holon_id, reward, error = future.result()

                        if error:
                            print(f"[Main] ERROR in Holon {holon_id} step: {error}")
                            holon_rewards.append(0.0)
                        else:
                            holon_rewards.append(reward)
                            total_reward += reward
            else:
                # SEQUENTIAL EXECUTION (Original behavior)
                for holon in self.holons.values():
                    try:
                        reward = holon.step(step, total_episodes)
                        holon_rewards.append(reward)
                        total_reward += reward
                    except Exception as e:
                        print(f"[Main] ERROR in Holon {holon.holon_id} step: {e}")
                        holon_rewards.append(0.0)

            # === PHASE B: FEDERATION CYCLE ===
            # Collect state summaries from holons
            holon_states = {
                holon_id: holon.get_state_summary()
                for holon_id, holon in self.holons.items()
            }

            # Federation detects breaches (thread-safe)
            breach_alerts = self.federation.step(holon_states)

            # === PHASE C: SEND BREACH ALERTS TO HOLONS ===
            for holon_id, alerts in breach_alerts.items():
                if alerts:
                    self.holons[holon_id].receive_breach_alerts(alerts)

        # Episode statistics
        episode_duration = time.time() - episode_start_time
        avg_episode_reward = total_reward / (len(self.holons) * steps_per_episode)

        self.episode_rewards.append(avg_episode_reward)

        # Print progress
        if episode_num % 10 == 0:
            fed_stats = self.federation.get_statistics()
            print(f"\n{'=' * 70}")
            print(
                f"Episode {episode_num}/{total_episodes} | Time: {episode_duration:.1f}s"
            )
            if self.use_parallel:
                print(f"Speedup: ~{1000.0 / episode_duration:.1f} steps/sec")
            print(
                f"Avg Reward: {avg_episode_reward:.3f} | Total Breaches: {fed_stats['total_breaches']}"
            )

            for holon in self.holons.values():
                stats = holon.get_statistics()
                print(
                    f"  Holon {stats['holon_id']}: reward={stats['avg_reward']:.3f}, breaches={stats['total_breaches']}"
                )
            print(f"{'=' * 70}\n")

        return avg_episode_reward

    def create_deployment_file(self, episode: int):
        """Create a JSON file containing all necessary data for deployment."""
        output_dir = Path("deployment")
        output_dir.mkdir(exist_ok=True)

        # 1. Collect Node Data
        all_nodes_data = []
        for holon_id, holon in self.holons.items():
            for node in holon.nodes:
                all_nodes_data.append(
                    {
                        "id": node.id,
                        "holon_id": holon_id,
                        "pos": node.pos.tolist(),
                        "last_pos": node.last_pos.tolist(),
                        "dimensions": node.dimensions,
                        "sensor_range": node.sensor_range,
                        "move_speed": node.move_speed,
                    }
                )

        # 2. Collect Holon Data (for bounds visualization in deployment)
        holon_data = [
            {
                "holon_id": h_id,
                "x_min": h.x_min,
                "x_max": h.x_max,
                "y_min": h.y_min,
                "y_max": h.y_max,
                "center": h.center.tolist(),
            }
            for h_id, h in self.holons.items()
        ]

        # 3. Compile Deployment Data
        deployment_data = {
            "metadata": {
                "episode": episode,
                "timestamp": time.time(),
                "dimensions": self.cfg["spatial"]["dimensions"],
                "total_nodes": self.cfg["node"]["total_nodes"],
            },
            "config": self.cfg,
            "nodes": all_nodes_data,
            "holons": holon_data,
        }

        # Determine filename based on the final episode
        filepath = output_dir / f"deployment_ep{episode}.json"

        with open(filepath, "w") as f:
            json.dump(deployment_data, f, indent=2)

        print(f"\n[Main] ✅ Deployment file created at {filepath}")
        return filepath

    def visualize_federated_map(self, episode: int):
        """
        Stitches all holons together into a single global view.
        FIX: Re-projects localized obstacles back to global space for the render.
        """
        fig, ax = plt.subplots(figsize=(12, 12))
        bounds = self.cfg["federation"]["world_bounds"]
        ax.set_xlim(0, bounds[0])
        ax.set_ylim(0, bounds[1])
        ax.set_title(f"FLOWRRA Federated Global Map - Episode {episode}", fontsize=15)
        save_path = Path("federated_maps")
        save_path.mkdir(exist_ok=True)

        for h_id, holon in self.holons.items():
            # 1. Draw Holon Partition Boundary
            rect = patches.Rectangle(
                (holon.x_min, holon.y_min),
                holon.x_max - holon.x_min,
                holon.y_max - holon.y_min,
                linewidth=2,
                edgecolor="black",
                facecolor="none",
                linestyle="--",
                alpha=0.3,
            )
            ax.add_patch(rect)

            # 2. Draw Obstacles (Translated back to Global Space)
            for obs in holon.orchestrator.obstacle_manager.obstacles:
                # Re-denormalize the local position back to the global quadrant
                global_obs_pos = holon._to_global(obs.pos).tolist()
                # Re-scale radius: local_r * holon_width = global_r
                global_radius = obs.radius * (holon.x_max - holon.x_min)

                circle = patches.Circle(
                    global_obs_pos, global_radius, color="red", alpha=0.25, zorder=2
                )
                ax.add_patch(circle)

            # 3. Draw Nodes and Connections
            node_positions = {node.id: node.pos for node in holon.nodes}
            for conn in holon.orchestrator.loop.connections:
                if (
                    conn.node_a_id in node_positions
                    and conn.node_b_id in node_positions
                ):
                    p1, p2 = (
                        node_positions[conn.node_a_id],
                        node_positions[conn.node_b_id],
                    )
                    ax.plot(
                        [p1[0], p2[0]],
                        [p1[1], p2[1]],
                        color="red" if conn.is_broken else "cyan",
                        alpha=0.8,
                        linewidth=1.5,
                        zorder=3,
                    )

            all_pos = np.array([n.pos for n in holon.nodes])
            ax.scatter(
                all_pos[:, 0],
                all_pos[:, 1],
                s=40,
                c="blue",
                edgecolors="white",
                zorder=4,
            )

        plt.grid(True, linestyle=":", alpha=0.5)
        save_path = save_path / f"Visualization_episode_{episode}.png"
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

    def train(self, num_episodes: int):
        """Run full training loop."""
        print(f"\n[Main] Starting training for {num_episodes} episodes")
        print(f"[Main] Steps per episode: {self.cfg['training']['steps_per_episode']}")

        for episode in range(1, num_episodes + 1):
            self.train_episode(episode, num_episodes)

            # Save checkpoints
            if episode % self.cfg["training"]["save_frequency"] == 0:
                self.save_checkpoint(episode)

            # Save metrics
            if episode % self.cfg["training"]["metrics_save_frequency"] == 0:
                self.save_metrics()

            # Save Visualization of Federated Map
            if episode % 10 == 0:
                self.visualize_federated_map(episode=episode)

        print(f"\n[Main] Training complete!")

        self.save_checkpoint(num_episodes)
        self.save_final_results()

        # NEW: Generate visualizations
        print("\n[Main] Generating training visualizations...")

        # Per-holon detailed metrics
        for holon_id, holon in self.holons.items():
            if holon.orchestrator:
                holon.orchestrator.save_metrics(
                    f"metrics/holon_{holon_id}_detailed.json"
                )

        # Federation overview
        self.plot_federation_training_results()

        # Create Deployment File
        self.create_deployment_file(num_episodes)

    def save_checkpoint(self, episode: int):
        """Save holon models and federation state."""
        checkpoint_dir = Path("checkpoints")
        checkpoint_dir.mkdir(exist_ok=True)

        for holon_id, holon in self.holons.items():
            filepath = checkpoint_dir / f"holon_{holon_id}_ep{episode}.pt"
            try:
                holon.save(str(filepath))
            except Exception as e:
                print(f"[Main] Warning: Could not save Holon {holon_id}: {e}")

        print(f"[Main] Saved checkpoint at episode {episode}")

    def save_metrics(self):
        """Save training metrics."""
        metrics_dir = Path("metrics")
        metrics_dir.mkdir(exist_ok=True)

        # Federation metrics
        fed_stats = self.federation.get_statistics()
        self.federation_metrics.append(fed_stats)

        # Holon metrics
        for holon_id, holon in self.holons.items():
            stats = holon.get_statistics()
            self.holon_metrics[holon_id].append(stats)

        # Save to JSON
        metrics_data = {
            "episode_rewards": self.episode_rewards,
            "federation_metrics": self.federation_metrics,
            "holon_metrics": {str(k): v for k, v in self.holon_metrics.items()},
            "config": self.cfg,
        }

        filepath = metrics_dir / "training_metrics.json"
        with open(filepath, "w") as f:
            json.dump(metrics_data, f, indent=2, default=str)

        print(f"[Main] Saved metrics to {filepath}")

    def save_final_results(self):
        """Save final training results and statistics."""
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)

        # Final statistics
        final_stats = {
            "total_episodes": len(self.episode_rewards),
            "final_avg_reward": self.episode_rewards[-1]
            if self.episode_rewards
            else 0.0,
            "best_reward": max(self.episode_rewards) if self.episode_rewards else 0.0,
            "federation": self.federation.get_statistics(),
            "holons": {
                holon_id: holon.get_statistics()
                for holon_id, holon in self.holons.items()
            },
        }

        filepath = results_dir / "final_results.json"
        with open(filepath, "w") as f:
            json.dump(final_stats, f, indent=2, default=str)

        print(f"\n[Main] Final results saved to {filepath}")
        print("\n" + "=" * 70)
        print("TRAINING SUMMARY")
        print("=" * 70)
        print(f"Total Episodes: {final_stats['total_episodes']}")
        print(f"Final Avg Reward: {final_stats['final_avg_reward']:.3f}")
        print(f"Best Reward: {final_stats['best_reward']:.3f}")
        print(f"Total Breaches: {final_stats['federation']['total_breaches']}")
        print("=" * 70 + "\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Federated FLOWRRA Training")
    parser.add_argument(
        "--episodes",
        type=int,
        default=CONFIG["training"]["num_episodes"],
        help="Number of training episodes",
    )
    parser.add_argument(
        "--holons",
        type=int,
        default=CONFIG["federation"]["num_holons"],
        help="Number of holons (must be perfect square)",
    )
    parser.add_argument(
        "--nodes",
        type=int,
        default=CONFIG["node"]["total_nodes"],
        help="Total number of nodes",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="training",
        choices=["training", "deployment"],
        help="Operation mode",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Enable parallel holon execution (single GPU)",
    )

    args = parser.parse_args()

    # Update config with arguments
    CONFIG["training"]["num_episodes"] = args.episodes
    CONFIG["federation"]["num_holons"] = args.holons
    CONFIG["node"]["total_nodes"] = args.nodes
    CONFIG["holon"]["mode"] = args.mode

    # Re-validate config
    from config import validate_config

    validated_config = validate_config(CONFIG)

    # Create and run federated system
    try:
        system = FederatedFLOWRRA(validated_config)

        if args.mode == "training":
            system.train(args.episodes)
        else:
            print("[Main] Deployment mode not yet implemented")

    except Exception as e:
        print(f"\n[Main] FATAL ERROR: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
