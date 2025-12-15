"""
holon/holon_core.py - FIXED VERSION

Sovereign Holon Agent - Wraps FLOWRRA_Orchestrator with boundary awareness.

FIXES:
- Proper orchestrator initialization
- Correct node handling
- Boundary repulsion integration
- Phase 2 R-GNN support ready
"""

from typing import Any, Dict, List, Optional

import numpy as np

from holon.agent import GNNAgent

# Import original core components
from holon.core import FLOWRRA_Orchestrator
from holon.density import DensityFunctionEstimatorND
from holon.node import NodePositionND


class Holon:
    """
    Autonomous sub-swarm agent with sovereign control over assigned nodes.

    Each holon maintains:
    - Its own FLOWRRA_Orchestrator instance
    - Independent GNN/R-GNN brain
    - Boundary awareness and constraint response
    """

    def __init__(
        self,
        holon_id: int,
        partition_id: int,
        spatial_bounds: Dict[str, tuple],
        config: Dict[str, Any],
        mode: str = "training",
    ):
        """
        Args:
            holon_id: Unique identifier
            partition_id: Assigned spatial partition ID
            spatial_bounds: {'x': (min, max), 'y': (min, max)}
            config: Configuration dict (holon-specific subset)
            mode: 'training' or 'deployment'
        """
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
        self.density = DensityFunctionEstimatorND()

        # FIX: Initialize orchestrator AFTER config is set
        self.orchestrator = None

        self.nodes: List[NodePositionND] = []

        # Breach tracking
        self.current_breaches: List[Dict] = []
        self.total_breaches_received = 0
        self.boundary_repulsion_active = False

        # Performance metrics
        self.steps_taken = 0
        self.avg_reward = 0.0

        print(f"[Holon {holon_id}] Initialized at partition {partition_id}")
        print(
            f"  Bounds: x=[{self.x_min:.2f}, {self.x_max:.2f}], y=[{self.y_min:.2f}, {self.y_max:.2f}]"
        )

    def initialize_orchestrator_with_nodes(self, initial_nodes: List[NodePositionND]):
        """
        Initialize the FLOWRRA orchestrator with pre-created nodes.

        This must be called AFTER node creation in main.py.

        Args:
            initial_nodes: List of NodePositionND objects assigned to this holon
        """
        # Store nodes
        self.nodes = initial_nodes

        # Create orchestrator instance
        self.orchestrator = FLOWRRA_Orchestrator(mode=self.mode)

        # Override orchestrator's config with holon-specific settings
        self.orchestrator.cfg.update(self.cfg)

        # CRITICAL FIX: Override the orchestrator's node list
        # Don't let it create its own nodes - use ours!
        self.orchestrator.nodes = self.nodes

        # Initialize loop structure with our nodes
        self.orchestrator.loop.initialize_ring_topology(len(self.nodes))

        # Run physics warmup to stabilize
        self.orchestrator._physics_warmup(steps=50)

        print(
            f"[Holon {self.holon_id}] Orchestrator initialized with {len(self.nodes)} nodes"
        )

    def receive_breach_alerts(self, breach_alerts: List[Dict[str, Any]]):
        """
        Receive breach alerts from Federation.

        Holon decides how to respond autonomously.
        """
        self.current_breaches = breach_alerts
        self.total_breaches_received += len(breach_alerts)

        if breach_alerts:
            self.boundary_repulsion_active = True

            # Log breach response
            if len(breach_alerts) > 0:
                print(
                    f"[Holon {self.holon_id}] Received {len(breach_alerts)} breach alerts"
                )
                for alert in breach_alerts[:3]:  # Show first 3
                    print(
                        f"  Node {alert['node_id']} breached {alert['boundary_edge']} edge (severity: {alert['severity']:.2f})"
                    )

    def _apply_boundary_constraints(self):
        """
        Apply boundary constraint responses.

        Strategy:
        1. Splat repulsion at boundaries (via density field)
        2. Apply gentle corrective nudges
        3. Let spring forces do the heavy lifting
        """
        if not self.current_breaches:
            self.boundary_repulsion_active = False
            return

        # For each breached node, apply suggested correction
        for alert in self.current_breaches:
            node_id = alert["node_id"]

            # FIX: Find node in our node list
            node = next((n for n in self.nodes if n.id == node_id), None)
            if node is None:
                continue

            # Apply gentle corrective nudge
            # Don't force it - let the springs and repulsion guide it back
            correction = alert["suggested_correction"]
            correction_magnitude = 0.008  # Very gentle - trust the physics

            node.pos = np.mod(node.pos + correction * correction_magnitude, 1.0)

            # FIX: Splat repulsion at boundary in density field
            # This teaches the GNN "don't go here" via the repulsion grid
            self.density.splat_collision_event(
                position=node.pos.copy(),
                velocity=correction,
                severity=alert["severity"] * 0.4,  # Moderate severity
                node_id=node.id,
                is_wfc_event=False,
            )

    def step(self, episode_step: int, total_episodes: int = 8000) -> float:
        """
        Execute one simulation step for this holon.

        Args:
            episode_step: Current step in episode
            total_episodes: Total episodes (for epsilon schedule)

        Returns:
            Average reward for this step
        """
        if self.orchestrator is None:
            raise RuntimeError(
                f"Holon {self.holon_id}: orchestrator not initialized! Call initialize_orchestrator_with_nodes() first."
            )

        self.steps_taken += 1

        # 1. Apply boundary constraints BEFORE orchestrator step
        # This allows the physics engine to work with corrected positions
        self._apply_boundary_constraints()

        # 2. Run standard FLOWRRA step
        avg_reward = self.orchestrator.step(episode_step, total_episodes)

        # 3. Add breach penalty to reward signal
        # This teaches the GNN to avoid boundaries
        if self.current_breaches:
            breach_penalty = (
                -len(self.current_breaches) * self.cfg["rewards"]["r_boundary_breach"]
            )
            avg_reward += breach_penalty

        # 4. Clear breaches for next cycle
        self.current_breaches = []

        # Track performance
        self.avg_reward = (
            self.avg_reward * (self.steps_taken - 1) + avg_reward
        ) / self.steps_taken

        return avg_reward

    def get_state_summary(self) -> Dict[str, Any]:
        """
        Generate state summary for Federation.

        Returns minimal info needed for breach detection.
        """
        return {
            "holon_id": self.holon_id,
            "partition_id": self.partition_id,
            "nodes": self.nodes,  # Pass actual node objects
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
        """Save holon state and neural network."""
        if self.orchestrator and self.orchestrator.gnn:
            self.orchestrator.gnn.save(filepath)
            print(f"[Holon {self.holon_id}] Saved to {filepath}")

    def load(self, filepath: str):
        """Load holon state and neural network."""
        if self.orchestrator and self.orchestrator.gnn:
            self.orchestrator.gnn.load(filepath)
            print(f"[Holon {self.holon_id}] Loaded from {filepath}")
