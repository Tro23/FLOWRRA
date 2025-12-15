"""
main.py - FIXED VERSION

Federated FLOWRRA Entry Point

FIXES:
- Proper NodePositionND creation
- Correct holon initialization sequence
- Better error handling
- Phase 2 R-GNN support hooks

Usage:
    python main.py --episodes 2000 --holons 4 --nodes 40
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

from config import CONFIG
from federation.manager import FederationManager
from holon.holon_core import Holon
from holon.node import NodePositionND


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
        """
        Create REAL NodePositionND nodes and distribute them across holons.

        FIX: Uses actual NodePositionND class, not dummy objects.
        """
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
            # Using the same logic as core.py but scaled to holon bounds
            ideal_dist = self.cfg["loop"]["ideal_distance"]
            equilibrium_radius = (nodes_per_holon * ideal_dist) / (2 * np.pi)

            # Clamp to holon bounds
            max_radius = (
                min((x_max - x_min) / 2, (y_max - y_min) / 2) * 0.8
            )  # Leave 20% margin from boundaries

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

                # FIX: Create REAL NodePositionND object
                node = NodePositionND(id=node_id, pos=pos, dimensions=dimensions)

                # Set node parameters
                node.sensor_range = self.cfg["node"]["sensor_range"]
                node.move_speed = self.cfg["node"]["move_speed"]

                holon_nodes.append(node)
                node_id += 1

            # FIX: Initialize orchestrator WITH the nodes
            holon.initialize_orchestrator_with_nodes(holon_nodes)

            print(
                f"[Main] Holon {holon_id}: {len(holon_nodes)} nodes at equilibrium (r={equilibrium_radius:.3f})"
            )

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
        self.save_final_state(num_episodes)

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

    def save_final_state(self, episode: int):
        """Save the final positions and states of all nodes for deployment."""
        state_dir = Path("deployment_states")
        state_dir.mkdir(exist_ok=True)

        all_node_states = []

        for holon_id, holon in self.holons.items():
            # Nodes are stored in the holon.nodes list
            for node in holon.nodes:
                # We save the essential state needed to reconstruct the node
                node_state = {
                    "id": node.id,
                    "pos": node.pos.tolist(),
                    "last_pos": node.last_pos.tolist(),  # Important for velocity calculation
                    "dimensions": node.dimensions,
                    "holon_id": holon_id,
                    # Add other necessary config/state data if needed,
                    # e.g., sensor_range, move_speed, etc., but they are usually global config.
                }
                all_node_states.append(node_state)

        filepath = state_dir / f"initial_state_ep{episode}.json"

        def convert_numpy(obj):
            """Helper for JSON dumping numpy arrays."""
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            raise TypeError(
                f"Object of type {type(obj).__name__} is not JSON serializable"
            )

        with open(filepath, "w") as f:
            json.dump(
                {"nodes": all_node_states, "config": self.cfg},
                f,
                indent=2,
                default=convert_numpy,
            )

        print(f"\n[Main] Saved final node state to {filepath}")


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
