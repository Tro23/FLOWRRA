"""
holon/holon_core.py - DIMENSION-SAFE VERSION

Critical Fix: Ensures all vector operations respect node dimensions
"""

from typing import Any, Dict, List, Optional

import numpy as np

from holon.core import FLOWRRA_Orchestrator
from holon.density import DensityFunctionEstimatorND
from holon.node import NodePositionND


class Holon:
    """Autonomous sub-swarm agent with dimension-safe operations."""

    def __init__(
        self,
        holon_id: int,
        partition_id: int,
        spatial_bounds: Dict[str, tuple],
        config: Dict[str, Any],
        mode: str = "training",
    ):
        self.holon_id = holon_id
        self.partition_id = partition_id
        self.spatial_bounds = spatial_bounds
        self.mode = mode

        # Extract bounds
        self.x_min, self.x_max = spatial_bounds["x"]
        self.y_min, self.y_max = spatial_bounds["y"]
        self.center = np.array(
            [(self.x_min + self.x_max) / 2, (self.y_min + self.y_max) / 2]
        )

        # Store config
        self.cfg = config
        self.dimensions = config["spatial"]["dimensions"]

        # Initialize orchestrator placeholder
        self.orchestrator = None
        self.nodes: List[NodePositionND] = []

        # Initialize Density Function Estimator
        self.density = DensityFunctionEstimatorND(
            dimensions=self.dimensions,
            local_grid_size=self.cfg["repulsion"]["local_grid_size"],
            global_grid_shape=self.cfg["repulsion"]["global_grid_shape"],
        )

        # Breach tracking
        self.current_breaches: List[Dict] = []
        self.total_breaches_received = 0
        self.boundary_repulsion_active = False

        # Performance metrics
        self.steps_taken = 0
        self.avg_reward = 0.0

        print(
            f"[Holon {holon_id}] Initialized at partition {partition_id} ({self.dimensions}D)"
        )
        print(
            f"  Bounds: x=[{self.x_min:.2f}, {self.x_max:.2f}], y=[{self.y_min:.2f}, {self.y_max:.2f}]"
        )

    def initialize_orchestrator_with_nodes(self, initial_nodes: List[NodePositionND]):
        """Initialize FLOWRRA orchestrator with pre-created nodes."""
        # Validate all nodes have correct dimensions
        for node in initial_nodes:
            if node.dimensions != self.dimensions:
                raise ValueError(
                    f"Node {node.id} has dimension {node.dimensions}, expected {self.dimensions}"
                )
            if node.pos.shape != (self.dimensions,):
                raise ValueError(
                    f"Node {node.id} position shape {node.pos.shape}, expected ({self.dimensions},)"
                )

        self.nodes = initial_nodes

        # Create orchestrator
        self.orchestrator = FLOWRRA_Orchestrator(mode=self.mode)
        self.orchestrator.cfg.update(self.cfg)

        # Override nodes
        self.orchestrator.nodes = self.nodes

        # Initialize loop
        self.orchestrator.loop.initialize_ring_topology(len(self.nodes))

        # Warmup
        self.orchestrator._physics_warmup(steps=50)

        print(
            f"[Holon {self.holon_id}] Orchestrator initialized with {len(self.nodes)} nodes"
        )

    def receive_breach_alerts(self, breach_alerts: List[Dict[str, Any]]):
        """Receive breach alerts from Federation."""
        self.current_breaches = breach_alerts
        self.total_breaches_received += len(breach_alerts)

        if breach_alerts:
            self.boundary_repulsion_active = True

            if len(breach_alerts) > 0:
                print(
                    f"[Holon {self.holon_id}] Received {len(breach_alerts)} breach alerts"
                )
                for alert in breach_alerts[:3]:
                    print(
                        f"  Node {alert['node_id']} breached {alert['boundary_edge']} edge (severity: {alert['severity']:.2f})"
                    )

    def _apply_boundary_constraints(self):
        """Apply boundary constraint responses with dimension safety."""
        if not self.current_breaches:
            self.boundary_repulsion_active = False
            return

        for alert in self.current_breaches:
            node_id = alert["node_id"]

            # Find node
            node = next((n for n in self.nodes if n.id == node_id), None)
            if node is None:
                continue

            # CRITICAL FIX: Ensure correction vector matches node dimensions
            correction = alert["suggested_correction"]

            # Validate correction dimension
            if len(correction) != self.dimensions:
                print(
                    f"[Holon {self.holon_id}] WARNING: Correction vector dimension mismatch"
                )
                # Truncate or pad to match
                if len(correction) > self.dimensions:
                    correction = correction[: self.dimensions]
                else:
                    correction = np.pad(
                        correction, (0, self.dimensions - len(correction))
                    )

            correction_magnitude = 0.008

            # Apply correction with dimension-safe modulo
            new_pos = node.pos + correction * correction_magnitude
            node.pos = np.mod(new_pos, 1.0)

            # Splat repulsion
            try:
                self.density.splat_collision_event(
                    position=node.pos.copy(),
                    velocity=correction,
                    severity=alert["severity"] * 0.4,
                    node_id=node.id,
                    is_wfc_event=False,
                )
            except Exception as e:
                print(
                    f"[Holon {self.holon_id}] WARNING: Could not splat repulsion: {e}"
                )

    def step(self, episode_step: int, total_episodes: int = 8000) -> float:
        """Execute one simulation step."""
        if self.orchestrator is None:
            raise RuntimeError(f"Holon {self.holon_id}: orchestrator not initialized!")

        self.steps_taken += 1

        # Apply boundary constraints
        self._apply_boundary_constraints()

        # Run orchestrator step
        avg_reward = self.orchestrator.step(episode_step, total_episodes)

        # Breach penalty
        if self.current_breaches:
            breach_penalty = (
                -len(self.current_breaches) * self.cfg["rewards"]["r_boundary_breach"]
            )
            avg_reward += breach_penalty

        # Clear breaches
        self.current_breaches = []

        # Track performance
        self.avg_reward = (
            self.avg_reward * (self.steps_taken - 1) + avg_reward
        ) / self.steps_taken

        return avg_reward

    def get_state_summary(self) -> Dict[str, Any]:
        """Generate state summary for Federation."""
        return {
            "holon_id": self.holon_id,
            "partition_id": self.partition_id,
            "nodes": self.nodes,
            "num_nodes": len(self.nodes),
            "center": self.center,
            "avg_reward": self.avg_reward,
            "total_breaches": self.total_breaches_received,
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Get holon performance metrics."""
        if self.orchestrator:
            orch_stats = self.orchestrator.get_statistics()
        else:
            orch_stats = {}

        return {
            "holon_id": self.holon_id,
            "partition_id": self.partition_id,
            "steps": self.steps_taken,
            "num_nodes": len(self.nodes),
            "avg_reward": self.avg_reward,
            "total_breaches": self.total_breaches_received,
            "boundary_repulsion_active": self.boundary_repulsion_active,
            "orchestrator_stats": orch_stats,
        }

    def save(self, filepath: str):
        """Save holon state."""
        if self.orchestrator and self.orchestrator.gnn:
            self.orchestrator.gnn.save(filepath)
            print(f"[Holon {self.holon_id}] Saved to {filepath}")

    def load(self, filepath: str):
        """Load holon state."""
        if self.orchestrator and self.orchestrator.gnn:
            self.orchestrator.gnn.load(filepath)
            print(f"[Holon {self.holon_id}] Loaded from {filepath}")
