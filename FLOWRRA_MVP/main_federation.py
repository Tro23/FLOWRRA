"""
main_federation.py

Federated FLOWRRA Entry Point

Orchestrates multiple holons coordinated by Federation Manager.

Usage:
    python main.py --episodes 2000 --holons 4
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

from config_federation import CONFIG
from federation.manager import FederationManager
from holon_core import Holon

# NOTE: You'll need to import your actual node implementation
# from node import NodePositionND


class FederatedFLOWRRA:
    """
    Main orchestrator for federated multi-holon system.
    """

    def __init__(self, config: Dict):
        self.cfg = config

        print("\n" + "=" * 70)
        print("FEDERATED FLOWRRA INITIALIZATION")
        print("=" * 70)

        # Initialize Federation Manager
        self.federation = FederationManager(
            num_holons=config["federation"]["num_holons"],
            world_bounds=config["federation"]["world_bounds"],
            breach_threshold=config["federation"]["breach_threshold"],
            coordination_mode=config["federation"]["coordination_mode"],
        )

        # Initialize Holons
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
        """
        Create nodes and distribute them across holons.

        Strategy: Place nodes at equilibrium positions within each holon's bounds.
        """
        total_nodes = self.cfg["node"]["total_nodes"]
        nodes_per_holon = self.cfg["node"]["num_nodes_per_holon"]

        print(
            f"\n[Main] Distributing {total_nodes} nodes across {len(self.holons)} holons"
        )
        print(f"[Main] {nodes_per_holon} nodes per holon")

        # NOTE: This is placeholder code. In actual implementation:
        # 1. Import: from node import NodePositionND
        # 2. Create nodes with proper initialization
        # 3. Assign to holons based on spatial partition

        # For now, we'll create dummy node objects
        node_id = 0

        for holon_id, holon in self.holons.items():
            # Get holon's spatial bounds
            x_min, x_max = holon.spatial_bounds["x"]
            y_min, y_max = holon.spatial_bounds["y"]

            # Create nodes for this holon
            holon_nodes = []

            for i in range(nodes_per_holon):
                # Random position within holon bounds
                pos = np.array(
                    [
                        np.random.uniform(x_min + 0.05, x_max - 0.05),
                        np.random.uniform(y_min + 0.05, y_max - 0.05),
                    ]
                )

                # Create node (PLACEHOLDER - replace with actual NodePositionND)
                # node = NodePositionND(
                #     id=node_id,
                #     pos=pos,
                #     dimensions=self.cfg["spatial"]["dimensions"]
                # )

                # Dummy node object for demonstration
                class DummyNode:
                    def __init__(self, id, pos, dims):
                        self.id = id
                        self.pos = pos
                        self.dimensions = dims
                        self.last_pos = pos.copy()
                        self.sensor_range = 0.15

                node = DummyNode(node_id, pos, self.cfg["spatial"]["dimensions"])

                holon_nodes.append(node)
                node_id += 1

            # Assign nodes to holon
            holon.initialize_nodes_in_bounds(holon_nodes)

        print(f"[Main] Node distribution complete\n")

    def train_episode(self, episode_num: int, total_episodes: int) -> float:
        """
        Execute one training episode.

        Returns:
            Average reward across all holons
        """
        steps_per_episode = self.cfg["training"]["steps_per_episode"]
        episode_start_time = time.time()

        total_reward = 0.0

        for step in range(steps_per_episode):
            # === PHASE A: HOLONS EXECUTE STEP ===
            holon_rewards = []

            for holon in self.holons.values():
                reward = holon.step(step, total_episodes)
                holon_rewards.append(reward)
                total_reward += reward

            # === PHASE B: FEDERATION CYCLE ===
            # Collect state summaries from holons
            holon_states = {
                holon_id: holon.get_state_summary()
                for holon_id, holon in self.holons.items()
            }

            # Federation detects breaches
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

        print(f"\n[Main] Training complete!")
        self.save_final_results()

    def save_checkpoint(self, episode: int):
        """Save holon models and federation state."""
        checkpoint_dir = Path("checkpoints")
        checkpoint_dir.mkdir(exist_ok=True)

        for holon_id, holon in self.holons.items():
            filepath = checkpoint_dir / f"holon_{holon_id}_ep{episode}.pt"
            holon.save(str(filepath))

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
            json.dump(final_stats, f, indent=2)

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

    args = parser.parse_args()

    # Update config with arguments
    CONFIG["training"]["num_episodes"] = args.episodes
    CONFIG["federation"]["num_holons"] = args.holons
    CONFIG["node"]["total_nodes"] = args.nodes
    CONFIG["holon"]["mode"] = args.mode

    # Re-validate config
    from config_federation import validate_config

    validated_config = validate_config(CONFIG)

    # Create and run federated system
    system = FederatedFLOWRRA(validated_config)

    if args.mode == "training":
        system.train(args.episodes)
    else:
        print("[Main] Deployment mode not yet implemented")


if __name__ == "__main__":
    main()
