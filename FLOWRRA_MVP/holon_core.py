"""
holon/holon_core.py

Sovereign Holon Agent - Wraps the original FLOWRRA_Orchestrator
with boundary awareness.

Each Holon:
- Has its own R-GNN brain (Phase 2)
- Manages its own nodes within assigned spatial partition
- Receives breach alerts from Federation
- Decides autonomously how to respond to constraints
"""

from typing import Any, Dict, List, Optional

import numpy as np

# Import original core components
# (These imports assume the original files are in parent directory or installed)
# You may need to adjust paths based on your project structure


class Holon:
    """
    Autonomous sub-swarm agent with sovereign control over assigned nodes.

    Wraps FLOWRRA_Orchestrator with:
    - Boundary constraint awareness
    - Breach response behavior
    - Independent training
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
            config: Configuration dict (subset of global config for this holon)
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

        # Initialize FLOWRRA components
        # NOTE: We'll create these dynamically to avoid import issues
        # In actual implementation, import your modules here
        self._initialize_orchestrator()

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
        print(f"  Managing {self.cfg['node']['num_nodes_per_holon']} nodes")

    def _initialize_orchestrator(self):
        """
        Initialize the FLOWRRA orchestrator for this holon.

        NOTE: This is a placeholder. In actual implementation, you would:
        1. Import: from core import FLOWRRA_Orchestrator
        2. Create instance with holon-specific config
        3. Initialize nodes within assigned spatial bounds
        """
        # Placeholder for demonstration
        # In actual code, uncomment and use:
        # from core import FLOWRRA_Orchestrator
        # self.orchestrator = FLOWRRA_Orchestrator(mode=self.mode)
        # self.orchestrator.cfg.update(self.cfg)

        # For now, we'll store a minimal state
        self.nodes = []  # Will be populated with NodePositionND objects
        self.step_count = 0

        print(f"[Holon {self.holon_id}] Orchestrator initialized (placeholder)")

    def initialize_nodes_in_bounds(self, nodes_to_assign: List[Any]):
        """
        Assign nodes to this holon and ensure they start within bounds.

        Args:
            nodes_to_assign: List of NodePositionND objects
        """
        self.nodes = nodes_to_assign

        # Ensure all nodes are within bounds
        for node in self.nodes:
            # Clamp position to holon bounds
            node.pos[0] = np.clip(node.pos[0], self.x_min, self.x_max - 0.001)
            node.pos[1] = np.clip(node.pos[1], self.y_min, self.y_max - 0.001)

            # Update last_pos to prevent velocity spikes
            node.last_pos = node.pos.copy()

        print(f"[Holon {self.holon_id}] Assigned {len(self.nodes)} nodes")

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
            print(
                f"[Holon {self.holon_id}] Received {len(breach_alerts)} breach alerts"
            )
            for alert in breach_alerts[:3]:  # Show first 3
                print(
                    f"  Node {alert['node_id']} breached {alert['boundary_edge']} edge"
                )

    def _apply_boundary_constraints(self):
        """
        Apply boundary constraint responses.

        Strategy:
        1. Add repulsion at boundaries (via density field)
        2. Increase spring stiffness for nodes near borders
        3. Apply corrective forces suggested by Federation
        """
        if not self.current_breaches:
            self.boundary_repulsion_active = False
            return

        # For each breached node, apply suggested correction
        for alert in self.current_breaches:
            node_id = alert["node_id"]

            # Find node
            node = next((n for n in self.nodes if n.id == node_id), None)
            if node is None:
                continue

            # Apply corrective force (conservative, let spring forces help)
            correction = alert["suggested_correction"]
            correction_magnitude = 0.01  # Gentle nudge

            node.pos = np.mod(node.pos + correction * correction_magnitude, 1.0)

            # Additional: Splat repulsion at boundary in density field
            # (This would integrate with your density.py)
            # self.orchestrator.density.splat_collision_event(
            #     position=node.pos,
            #     velocity=correction,
            #     severity=alert['severity'] * 0.5,
            #     node_id=node.id,
            #     is_wfc_event=False
            # )

    def step(self, episode_step: int, total_episodes: int = 8000) -> float:
        """
        Execute one simulation step for this holon.

        Args:
            episode_step: Current step in episode
            total_episodes: Total episodes (for epsilon schedule)

        Returns:
            Average reward for this step
        """
        self.steps_taken += 1

        # 1. Apply boundary constraints (if any breaches)
        self._apply_boundary_constraints()

        # 2. Run standard FLOWRRA step
        # In actual implementation:
        # avg_reward = self.orchestrator.step(episode_step, total_episodes)

        # Placeholder reward calculation
        avg_reward = 0.0

        # Penalty for breaches (teach avoidance)
        if self.current_breaches:
            breach_penalty = -len(self.current_breaches) * 2.0
            avg_reward += breach_penalty

        # 3. Clear breaches for next cycle
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
            "nodes": self.nodes,
            "num_nodes": len(self.nodes),
            "center": self.center,
            "avg_reward": self.avg_reward,
            "total_breaches": self.total_breaches_received,
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Get holon performance metrics."""
        return {
            "holon_id": self.holon_id,
            "partition_id": self.partition_id,
            "steps": self.steps_taken,
            "num_nodes": len(self.nodes),
            "avg_reward": self.avg_reward,
            "total_breaches": self.total_breaches_received,
            "boundary_repulsion_active": self.boundary_repulsion_active,
        }

    def save(self, filepath: str):
        """Save holon state and neural network."""
        # In actual implementation:
        # self.orchestrator.gnn.save(filepath)
        pass

    def load(self, filepath: str):
        """Load holon state and neural network."""
        # In actual implementation:
        # self.orchestrator.gnn.load(filepath)
        pass
