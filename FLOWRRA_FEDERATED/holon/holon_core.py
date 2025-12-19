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

    def _to_local(self, global_pos: np.ndarray) -> np.ndarray:
        """Translate global coordinates to local [0,1] scale."""
        lx = (global_pos[0] - self.x_min) / (self.x_max - self.x_min)
        ly = (global_pos[1] - self.y_min) / (self.y_max - self.y_min)
        return np.array([lx, ly])

    def _to_global(self, local_pos: np.ndarray) -> np.ndarray:
        """Translate local [0,1] coordinates back to global scale."""
        gx = local_pos[0] * (self.x_max - self.x_min) + self.x_min
        gy = local_pos[1] * (self.y_max - self.y_min) + self.y_min
        return np.array([gx, gy])

    def _localize_internal_obstacles(self):
        """
        Consolidated: Filters global obstacles and translates them into the
        local 1.0x1.0 space of the orchestrator.
        """
        if self.orchestrator is None:
            return

        self.orchestrator.obstacle_manager.obstacles = []

        # --- FIXED: Localize Obstacles for "The Delusion" ---
        self.orchestrator.obstacle_manager.obstacles = []
        global_obstacles = self.cfg.get("obstacles", [])

        # Calculate Holon width for radius scaling
        h_width = self.x_max - self.x_min

        for obs_cfg in global_obstacles:
            x, y, r = obs_cfg
            # 1. Check if obstacle overlaps with this Holon's bounds
            if (self.x_min - r <= x <= self.x_max + r) and (
                self.y_min - r <= y <= self.y_max + r
            ):
                # 2. FIX: Translate Global Pos to Local [0,1]
                local_pos = self._to_local(np.array([x, y]))

                # 3. FIX: Scale Radius relative to Holon size, not World size
                # If the holon is 0.5 wide, a 0.05 radius obstacle
                # is actually 0.1 wide in local [0,1] space.
                local_radius = r / h_width

                self.orchestrator.obstacle_manager.add_static_obstacle(
                    local_pos, local_radius
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

        # --- NEW: Localize Obstacles ---
        # Clear default obstacles from the internal manager
        self._localize_internal_obstacles()

        print(
            f"[Holon {self.holon_id}] Localized {len(self.orchestrator.obstacle_manager.obstacles)} obstacles."
        )

        # Override nodes
        self.orchestrator.nodes = self.nodes

        # Initialize loop
        # --- FIXED: Manual Ring Initialization with Offset IDs ---
        self.orchestrator.loop.connections.clear()
        num_local_nodes = len(self.nodes)

        for i in range(num_local_nodes):
            # Get the ACTUAL global IDs of the neighbors
            node_a_id = self.nodes[i].id
            node_b_id = self.nodes[(i + 1) % num_local_nodes].id

            # Create the connection using the real IDs
            from holon.loop import Connection  # Ensure import exists

            conn = Connection(
                node_a_id=node_a_id,
                node_b_id=node_b_id,
                ideal_distance=self.orchestrator.loop.ideal_distance,
            )
            self.orchestrator.loop.connections.append(conn)

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
        """Apply boundary constraints that keep nodes inside THIS holon."""
        # Note: This is called in Phase 4 (Global Space)
        for node in self.nodes:
            # Instead of np.mod(1.0) which wraps the whole world,
            # we CLIP to this Holon's specific bounds.
            node.pos[0] = np.clip(node.pos[0], self.x_min, self.x_max)
            node.pos[1] = np.clip(node.pos[1], self.y_min, self.y_max)

        # Apply Splat repulsion for breaches if alerts exist
        if self.current_breaches:
            for alert in self.current_breaches:
                node_id = alert["node_id"]
                node = next((n for n in self.nodes if n.id == node_id), None)
                if node:
                    # Splat using LOCAL coordinates so the local Density Map
                    # inside core.py understands where the 'pain' is coming from.
                    local_pos = self._to_local(node.pos)
                    self.density.splat_collision_event(
                        position=local_pos,
                        velocity=alert["suggested_correction"],
                        severity=alert["severity"] * 0.4,
                        node_id=node.id,
                    )

    def step(self, episode_step: int, total_episodes: int = 8000) -> float:
        """Execute one simulation step with Coordinate Normalization."""
        if self.orchestrator is None:
            raise RuntimeError(f"Holon {self.holon_id}: orchestrator not initialized!")

        self.steps_taken += 1

        # --- PHASE 1: NORMALIZE (Global -> Local) ---
        # We tell the nodes they are in a 1.0 x 1.0 world before the Orchestrator sees them
        for node in self.nodes:
            node.pos = self._to_local(node.pos)

        # Capture WFC state in local space
        pre_step_wfc_count = len(self.orchestrator.wfc_trigger_events)

        # --- PHASE 2: ORCHESTRATE (The Delusion) ---
        # Core.py runs its logic thinking it has a full 1.0x1.0 world.
        # This includes GNN inference and Spring Physics.
        avg_reward = self.orchestrator.step(episode_step, total_episodes)

        # --- PHASE 3: DENORMALIZE (Local -> Global) ---
        # Map positions back to the true Federation world coordinates
        for node in self.nodes:
            node.pos = self._to_global(node.pos)

        # --- PHASE 4: ENFORCE (The Cage) ---
        # Now that we are in global space, apply the 'Hard Wall' constraints
        # we wrote earlier to prevent leaking into other holons.
        self._apply_boundary_constraints()

        # WFC Reporting Logic (Kept from your current code)
        if len(self.orchestrator.wfc_trigger_events) > pre_step_wfc_count:
            latest_event = self.orchestrator.wfc_trigger_events[-1]
            integrity = latest_event.get("loop_integrity", 0.0)
            print(
                f"Holon-{self.holon_id} loop integrity failed ({integrity:.2f}), WFC initiated!"
            )

        # Breach penalty (Federation layer punishment)
        if self.current_breaches:
            breach_penalty = (
                -len(self.current_breaches) * self.cfg["rewards"]["r_boundary_breach"]
            )
            avg_reward += breach_penalty
            self.current_breaches = []  # Clear for next step

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
