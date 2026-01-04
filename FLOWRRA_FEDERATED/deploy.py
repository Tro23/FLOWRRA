"""
deploy.py - ENHANCED VERSION

Deployment script with frozen node removal event tracking.

Loads trained models and runs in deployment mode with visualization support,
including tracking of node freeze/unfreeze events.

Usage:
    python deploy.py --deployment-file deployment/deployment_ep1000.json
    python deploy.py --deployment-file deployment/deployment_ep1000.json --steps 500
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


class DeploymentRunner:
    """
    Runs trained federated FLOWRRA system in deployment mode with frozen node tracking.
    """

    def __init__(self, deployment_file: str):
        """Initialize deployment runner with frozen node event tracking."""
        # Load deployment data
        print(f"\n[Deploy] Loading deployment data from {deployment_file}")
        with open(deployment_file, "r") as f:
            self.deployment_data = json.load(f)

        self.cfg = self.deployment_data["config"]

        # Override mode to deployment
        self.cfg["holon"]["mode"] = "deployment"

        # NEW: Initialize frozen node event tracking
        self.frozen_events = []  # Track freeze/unfreeze events
        self.frozen_metrics = {}  # Per-holon frozen node statistics

        print(f"[Deploy] Loaded {len(self.deployment_data['nodes'])} nodes")
        print(f"[Deploy] Dimensions: {self.deployment_data['metadata']['dimensions']}D")

        # Initialize Federation
        self.federation = FederationManager(
            num_holons=self.cfg["federation"]["num_holons"],
            world_bounds=self.cfg["federation"]["world_bounds"],
            breach_threshold=self.cfg["federation"]["breach_threshold"],
            coordination_mode=self.cfg["federation"]["coordination_mode"],
        )

        # Initialize Holons
        self.holons: Dict[int, Holon] = {}
        self._initialize_holons_from_deployment()

        # Trajectory tracking
        self.trajectory_history = []
        self.step_count = 0

        # NEW: Track initial frozen node states
        self._initialize_frozen_tracking()

        print("[Deploy] System ready for deployment with frozen node tracking\n")

    def _initialize_frozen_tracking(self):
        """Initialize tracking of frozen nodes from deployment state."""
        for holon_id, holon in self.holons.items():
            if holon.orchestrator:
                frozen_nodes = holon.orchestrator.get_frozen_nodes()

                self.frozen_metrics[holon_id] = {
                    "total_frozen": len(frozen_nodes),
                    "freeze_events": 0,
                    "unfreeze_events": 0,
                    "frozen_node_ids": [n.id for n in frozen_nodes],
                }

                print(
                    f"[Deploy] Holon {holon_id}: {len(frozen_nodes)} nodes initially frozen"
                )

    def _initialize_holons_from_deployment(self):
        """Recreate holons from deployment data."""
        partition_assignments = self.federation.get_partition_assignments()

        # Group nodes by holon_id
        nodes_by_holon = {}
        for node_data in self.deployment_data["nodes"]:
            holon_id = node_data["holon_id"]
            if holon_id not in nodes_by_holon:
                nodes_by_holon[holon_id] = []
            nodes_by_holon[holon_id].append(node_data)

        # Create holons
        for partition_id, partition in partition_assignments.items():
            holon = Holon(
                holon_id=partition_id,
                partition_id=partition_id,
                spatial_bounds={"x": partition.bounds_x, "y": partition.bounds_y},
                config=self.cfg,
                mode="deployment",
            )

            # Recreate nodes
            node_list = []
            for node_data in nodes_by_holon[partition_id]:
                node = NodePositionND(
                    id=node_data["id"],
                    pos=np.array(node_data["pos"]),
                    dimensions=node_data["dimensions"],
                )
                node.last_pos = np.array(node_data["last_pos"])
                node.sensor_range = node_data.get(
                    "sensor_range", self.cfg["node"]["sensor_range"]
                )
                node.move_speed = node_data.get(
                    "move_speed", self.cfg["node"]["move_speed"]
                )

                node_list.append(node)

            # Initialize orchestrator with nodes
            holon.initialize_orchestrator_with_nodes(node_list)

            # Load trained model if available
            episode_num = self.deployment_data["metadata"]["episode"]
            checkpoint_path = Path(
                f"checkpoints/holon_{partition_id}_ep{episode_num}.pt"
            )
            if checkpoint_path.exists():
                holon.load(str(checkpoint_path))
                print(
                    f"[Deploy] Loaded trained model for Holon {partition_id} from {checkpoint_path}"
                )
            else:
                print(
                    f"[Deploy] ⚠️ Warning: No trained model found for Holon {partition_id} at {checkpoint_path}. Running with initial model."
                )

            self.holons[partition_id] = holon

        print(f"[Deploy] Initialized {len(self.holons)} holons from deployment data")

    def _track_frozen_events(self, step: int):
        """
        Track changes in frozen node status between steps.

        This captures:
        - New nodes being frozen
        - Frozen nodes being unfrozen (reintegrated)
        """
        for holon_id, holon in self.holons.items():
            if not holon.orchestrator:
                continue

            # Get current frozen nodes
            current_frozen_nodes = holon.orchestrator.get_frozen_nodes()
            current_frozen_ids = set(n.id for n in current_frozen_nodes)

            # Get previously tracked frozen nodes
            previous_frozen_ids = set(self.frozen_metrics[holon_id]["frozen_node_ids"])

            # Detect NEW freezes
            newly_frozen = current_frozen_ids - previous_frozen_ids
            for node_id in newly_frozen:
                event = {
                    "timestep": step,
                    "event_type": "node_frozen",
                    "node_id": node_id,
                    "holon_id": holon_id,
                    "reason": "deployment_step",
                }
                self.frozen_events.append(event)
                self.frozen_metrics[holon_id]["freeze_events"] += 1

                print(
                    f"[Deploy] 🔴 Step {step}: Node {node_id} frozen in Holon {holon_id}"
                )

            # Detect UNFREEZES (reintegrations)
            newly_unfrozen = previous_frozen_ids - current_frozen_ids
            for node_id in newly_unfrozen:
                event = {
                    "timestep": step,
                    "event_type": "node_unfrozen",
                    "node_id": node_id,
                    "holon_id": holon_id,
                    "reason": "reintegrated",
                }
                self.frozen_events.append(event)
                self.frozen_metrics[holon_id]["unfreeze_events"] += 1

                print(
                    f"[Deploy] 🟢 Step {step}: Node {node_id} unfrozen in Holon {holon_id}"
                )

            # Update tracked frozen nodes
            self.frozen_metrics[holon_id]["frozen_node_ids"] = list(current_frozen_ids)
            self.frozen_metrics[holon_id]["total_frozen"] = len(current_frozen_ids)

    def run_step(self, step: int) -> Dict:
        """Execute one deployment step and collect data."""
        # Track frozen node changes BEFORE step
        self._track_frozen_events(step)

        # Holons execute
        holon_metrics = []
        for holon in self.holons.values():
            reward = holon.step(step, total_episodes=1)

            # Accessing the orchestrator's current metrics
            current_coherence = 0.0
            current_integrity = 0.0

            if holon.orchestrator and hasattr(holon.orchestrator, "metrics_history"):
                history = holon.orchestrator.metrics_history
                if history:
                    latest = history[-1]
                    current_coherence = latest.get("coherence", 0.0)
                    current_integrity = latest.get("loop_integrity", 0.0)

            holon_metrics.append(
                {"coherence": current_coherence, "integrity": current_integrity}
            )

        # Federation cycle
        holon_states = {
            holon_id: holon.get_state_summary()
            for holon_id, holon in self.holons.items()
        }

        breach_alerts = self.federation.step(holon_states)

        # Send breach alerts
        for holon_id, alerts in breach_alerts.items():
            if alerts:
                self.holons[holon_id].receive_breach_alerts(alerts)

        # Calculate global averages
        avg_coherence = (
            float(sum(m["coherence"] for m in holon_metrics) / len(holon_metrics))
            if holon_metrics
            else 0
        )
        avg_integrity = (
            float(sum(m["integrity"] for m in holon_metrics) / len(holon_metrics))
            if holon_metrics
            else 0
        )

        # Collect snapshot for visualization
        snapshot = self._collect_snapshot(step, avg_coherence, avg_integrity)

        self.step_count += 1

        return snapshot

    def _collect_snapshot(self, step: int, coherence: float, integrity: float) -> Dict:
        """Collect current state snapshot for visualization."""
        all_nodes = []
        all_connections = []
        all_frozen_nodes = []

        for holon_id, holon in self.holons.items():
            # Get frozen nodes for this holon
            frozen_nodes = (
                holon.orchestrator.get_frozen_nodes() if holon.orchestrator else []
            )
            frozen_ids = set(n.id for n in frozen_nodes)

            # Collect node positions with frozen status
            for node in holon.nodes:
                node_data = {
                    "id": node.id,
                    "pos": node.pos.tolist(),
                    "holon_id": holon_id,
                    "is_frozen": node.id in frozen_ids,
                }
                all_nodes.append(node_data)

                if node.id in frozen_ids:
                    all_frozen_nodes.append(node.id)

            # Collect loop connections
            if holon.orchestrator:
                for conn in holon.orchestrator.loop.connections:
                    all_connections.append(
                        {
                            "node_a": conn.node_a_id,
                            "node_b": conn.node_b_id,
                            "broken": conn.is_broken,
                        }
                    )

        snapshot = {
            "time": step,
            "nodes": all_nodes,
            "connections": all_connections,
            "frozen_nodes": all_frozen_nodes,  # NEW: List of frozen node IDs
            "coherence": float(coherence),
            "loop_integrity": float(integrity),
            "holons": [
                {
                    "holon_id": h_id,
                    "bounds": {
                        "x_min": h.x_min,
                        "x_max": h.x_max,
                        "y_min": h.y_min,
                        "y_max": h.y_max,
                    },
                    "num_frozen": self.frozen_metrics[h_id]["total_frozen"],
                }
                for h_id, h in self.holons.items()
            ],
        }

        # Check if this timestep has freeze/unfreeze events
        matching_events = [e for e in self.frozen_events if e["timestep"] == step]
        if matching_events:
            snapshot["events"] = matching_events

        return snapshot

    def run(self, num_steps: int, save_interval: int = 25):
        """Run deployment for specified steps."""
        print(f"\n[Deploy] Starting deployment run for {num_steps} steps")

        start_time = time.time()

        for step in range(num_steps):
            snapshot = self.run_step(step)
            self.trajectory_history.append(snapshot)

            # Progress update
            if (step + 1) % save_interval == 0:
                elapsed = time.time() - start_time
                total_frozen = sum(
                    m["total_frozen"] for m in self.frozen_metrics.values()
                )
                print(
                    f"[Deploy] Step {step + 1}/{num_steps} ({elapsed:.1f}s) | Frozen: {total_frozen}"
                )

        elapsed_total = time.time() - start_time
        print(f"\n[Deploy] Completed {num_steps} steps in {elapsed_total:.1f}s")

        # Save trajectory
        self.save_trajectory()

    def save_trajectory(self):
        """Save full trajectory for visualization with frozen node data."""
        output_dir = Path("deployment")
        output_dir.mkdir(exist_ok=True)

        trajectory_data = {
            "metadata": self.deployment_data["metadata"],
            "config": self.cfg,
            "trajectory": self.trajectory_history,
            "holons": self.deployment_data["holons"],
            # NEW: Include frozen node events and metrics
            "frozen_events": self.frozen_events,
            "frozen_metrics": self.frozen_metrics,
        }

        filepath = output_dir / f"trajectory_{self.step_count}_steps.json"

        with open(filepath, "w") as f:
            json.dump(trajectory_data, f, indent=2)

        print(f"\n[Deploy] ✅ Saved trajectory to {filepath}")
        print(
            f"[Deploy] 📊 Trajectory contains {len(self.trajectory_history)} snapshots"
        )

        # Print frozen node event statistics
        if self.frozen_events:
            freeze_count = sum(
                1 for e in self.frozen_events if e["event_type"] == "node_frozen"
            )
            unfreeze_count = sum(
                1 for e in self.frozen_events if e["event_type"] == "node_unfrozen"
            )
            print(
                f"[Deploy] 🔄 Events: {freeze_count} freezes, {unfreeze_count} unfreezes"
            )

        if self.frozen_metrics:
            print(f"[Deploy] Frozen node metrics by holon:")
            for holon_id, metrics in self.frozen_metrics.items():
                print(
                    f"  Holon {holon_id}: {metrics['total_frozen']} frozen | "
                    f"{metrics['freeze_events']} freeze events | "
                    f"{metrics['unfreeze_events']} unfreeze events"
                )

        print(f"[Deploy] Ready for web visualization!")


def main():
    parser = argparse.ArgumentParser(description="FLOWRRA Deployment Runner")
    parser.add_argument(
        "--deployment-file",
        type=str,
        required=True,
        help="Path to deployment JSON file",
    )
    parser.add_argument(
        "--steps", type=int, default=500, help="Number of deployment steps to run"
    )
    parser.add_argument(
        "--save-interval", type=int, default=25, help="Progress update interval"
    )

    args = parser.parse_args()

    # Check if deployment file exists
    if not Path(args.deployment_file).exists():
        print(f"[Deploy] ERROR: Deployment file not found: {args.deployment_file}")
        return 1

    try:
        # Create deployment runner
        runner = DeploymentRunner(args.deployment_file)

        # Run deployment
        runner.run(args.steps, args.save_interval)

        print("\n[Deploy] Deployment complete! ✨")

    except Exception as e:
        print(f"\n[Deploy] ERROR: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
