"""
holon/holon_core.py - DIMENSION-SAFE VERSION

Critical Fix: Ensures all vector operations respect node dimensions
"""

from typing import Any, Dict, List, Optional, Set

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
        self._nodes_backing = []

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

        # NEW: Strategic freezing parameters
        self.enable_strategic_freezing = config.get("holon", {}).get(
            "enable_strategic_freezing", False
        )
        self.freeze_at_coverage = config.get("holon", {}).get(
            "freeze_at_coverage", 0.55
        )
        self.freeze_edge_nodes = config.get("holon", {}).get("freeze_edge_nodes", True)

        print(
            f"[Holon {holon_id}] Initialized at partition {partition_id} ({self.dimensions}D)"
        )
        print(
            f"  Bounds: x=[{self.x_min:.2f}, {self.x_max:.2f}], y=[{self.y_min:.2f}, {self.y_max:.2f}]"
        )
        if self.enable_strategic_freezing:
            print(
                f"  Strategic freezing ENABLED (trigger at {self.freeze_at_coverage * 100:.0f}% coverage)"
            )

    @property
    def nodes(self):
        """Always return the orchestrator's current node list."""
        if self.orchestrator is None:
            return self._nodes_backing
        return self.orchestrator.nodes

    @nodes.setter
    def nodes(self, value):
        """Update the orchestrator's node list."""
        if self.orchestrator is not None:
            self.orchestrator.nodes = value
        else:
            # Before orchestrator exists, store in backing field
            self._nodes_backing = value

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
        self.orchestrator.nodes = initial_nodes

        # Initialize loop
        # --- FIXED: Manual Ring Initialization with Offset IDs ---
        self.orchestrator.loop.connections.clear()
        num_local_nodes = len(initial_nodes)

        for i in range(num_local_nodes):
            # Get the ACTUAL global IDs of the neighbors
            node_a_id = self.orchestrator.nodes[i].id
            node_b_id = self.orchestrator.nodes[(i + 1) % num_local_nodes].id

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

        # CRITICAL: Sync the holon's node reference
        self.nodes = self.orchestrator.nodes

        print(
            f"[Holon {self.holon_id}] Orchestrator initialized with {len(self.nodes)} nodes"
        )

    def receive_breach_alerts(self, breach_alerts: List[Dict[str, Any]]):
        """
        Receive breach alerts from Federation.

        NEW: Ignore breaches for frozen nodes (they shouldn't move anyway).
        """

        # Filter out alerts for frozen nodes
        if self.orchestrator:
            active_breaches = [
                alert
                for alert in breach_alerts
                if alert["node_id"] not in self.orchestrator.frozen_nodes
            ]
        else:
            active_breaches = breach_alerts

        self.current_breaches = active_breaches
        self.total_breaches_received += len(active_breaches)

        if active_breaches:
            self.boundary_repulsion_active = True

            if len(active_breaches) > 0:
                print(
                    f"[Holon {self.holon_id}] Received {len(active_breaches)} breach alerts (filtered {len(breach_alerts) - len(active_breaches)} frozen)"
                )
                for alert in active_breaches[:3]:
                    print(
                        f"  Node {alert['node_id']} breached {alert['boundary_edge']} edge (severity: {alert['severity']:.2f})"
                    )

    def _apply_boundary_constraints(self):
        """
        Apply boundary constraints that keep nodes inside THIS holon.

        NEW: Only apply to active nodes (frozen nodes already fixed).
        """

        if self.orchestrator is None:
            return

        active_nodes = self.orchestrator.get_active_nodes()

        for node in active_nodes:
            if node is None or node.pos is None:
                continue
            # Clip to this holon's bounds
            node.pos[0] = np.clip(node.pos[0], self.x_min, self.x_max)
            node.pos[1] = np.clip(node.pos[1], self.y_min, self.y_max)

        # Apply splat repulsion for breaches
        if self.current_breaches:
            for alert in self.current_breaches:
                node_id = alert["node_id"]

                # Skip frozen nodes
                if node_id in self.orchestrator.frozen_nodes:
                    continue

                node = next((n for n in active_nodes if n.id == node_id), None)
                if node:
                    local_pos = self._to_local(node.pos)
                    self.density.splat_collision_event(
                        position=local_pos,
                        velocity=alert["suggested_correction"],
                        severity=alert["severity"] * 0.4,
                        node_id=node.id,
                    )

    # ========================================================================
    # STRATEGIC FREEZING LOGIC
    # ========================================================================

    def _check_strategic_freezing(self):
        """
        Decide if we should freeze any nodes based on strategic criteria.

        Triggers:
        1. High coverage reached (>85%)
        2. Node is at edge of explored region
        3. Node has been stable for N steps
        4. System integrity is good (>0.7)
        """
        if not self.enable_strategic_freezing:
            return

        if self.orchestrator is None:
            return

        # Check if we've reached freeze threshold
        coverage = self.orchestrator.map.get_coverage_percentage()
        if coverage < self.freeze_at_coverage:
            return

        # Check system health
        loop_integrity = self.orchestrator.loop.calculate_integrity()
        if loop_integrity < 0.7:
            return  # Don't freeze during instability

        active_nodes = self.orchestrator.get_active_nodes()

        # Don't freeze if we're down to minimum active nodes
        min_active_nodes = 4
        if len(active_nodes) <= min_active_nodes:
            return

        # Strategy 1: Freeze edge nodes (perimeter guards)
        if self.freeze_edge_nodes:
            edge_candidates = self._find_edge_nodes(active_nodes)

            for node_id in edge_candidates:
                # Check if this node has been stable
                if self._is_node_stable(node_id, stability_steps=5):
                    print(f"\n[Holon {self.holon_id}] 🎯 Strategic Freeze Decision:")
                    print(f"  Coverage: {coverage:.1f}%")
                    print(f"  Loop Integrity: {loop_integrity:.2f}")
                    print(f"  Active Nodes: {len(active_nodes)}")

                    self.orchestrator.freeze_node(
                        node_id, reason=f"edge_guard_coverage_{coverage:.0f}%"
                    )
                    return  # Only freeze one per check

    def _find_edge_nodes(self, nodes: List[NodePositionND]) -> List[int]:
        """
        Find nodes that are at the edge of explored territory.
        These make good candidates for static guards.
        """
        edge_candidates = []

        for node in nodes:
            # Check if node is near boundary
            near_x_edge = (node.pos[0] < 0.3) or (node.pos[0] > 0.7)
            near_y_edge = (node.pos[1] < 0.3) or (node.pos[1] > 0.7)

            if near_x_edge or near_y_edge:
                edge_candidates.append(node.id)

        return edge_candidates

    def _is_node_stable(self, node_id: int, stability_steps: int = 5) -> bool:
        """
        Check if a node has been relatively stationary for N steps.
        Stable nodes are good candidates for freezing.
        """
        if not hasattr(self.orchestrator, "node_position_history"):
            return False

        history = self.orchestrator.node_position_history.get(node_id, [])

        if len(history) < stability_steps:
            return False

        # Check recent movement
        recent_history = history[-stability_steps:]
        total_movement = sum(
            np.linalg.norm(recent_history[i] - recent_history[i - 1])
            for i in range(1, len(recent_history))
        )

        # Stable if moved less than 0.1 units in last N steps
        return True if total_movement < 0.4 else False

    def step(self, episode_step: int, total_episodes: int = 1000) -> float:
        """
        Execute one simulation step with Coordinate Normalization.

        Key changes:
            - Coordinate transforms apply to ALL nodes (frozen + active)
            - Boundary constraints only apply to active nodes
            - Strategic freezing checks happen here
        """

        if self.orchestrator is None:
            raise RuntimeError(f"Holon {self.holon_id}: orchestrator not initialized!")

        self.steps_taken += 1

        # --- PHASE 1: NORMALIZE (Global -> Local) ---
        # We tell the nodes they are in a 1.0 x 1.0 world before the Orchestrator sees them
        # This ensures we're operating on the correct nodes after removal/reintegration
        self.nodes = self.orchestrator.nodes

        for node in self.nodes:
            if node is None or node.pos is None:
                continue
            node.pos = self._to_local(node.pos)

        # Capture WFC state in local space
        pre_step_wfc_count = len(self.orchestrator.wfc_trigger_events)

        # --- PHASE 2: ORCHESTRATE (The Delusion) ---
        # Core.py runs its logic thinking it has a full 1.0x1.0 world.
        # This includes GNN inference and Spring Physics.
        raw_orch_reward = self.orchestrator.step(episode_step, total_episodes)
        avg_reward = raw_orch_reward if raw_orch_reward is not None else 0.0

        # --- PHASE 3: DENORMALIZE (Local -> Global) ---
        # Map positions back to the true Federation world coordinates
        # We MUST sync before denormalization!
        self.nodes = self.orchestrator.nodes

        for node in self.nodes:
            if node is None or node.pos is None:
                continue
            node.pos = self._to_global(node.pos)

        # --- PHASE 4: ENFORCE (The Cage) ---
        # Now that we are in global space, apply the 'Hard Wall' constraints
        # we wrote earlier to prevent leaking into other holons.
        self._apply_boundary_constraints()

        # --- PHASE 5: STRATEGIC FREEZING ---
        # Check if conditions are right to freeze a node
        self._check_strategic_freezing()

        # WFC Reporting Logic (Kept from your current code)
        if len(self.orchestrator.wfc_trigger_events) > pre_step_wfc_count:
            latest_event = self.orchestrator.wfc_trigger_events[-1]
            if latest_event:
                integrity = latest_event.get("loop_integrity", 0.0)
                print(
                    f"Holon-{self.holon_id} loop integrity failed ({integrity:.2f}), WFC initiated!"
                )

        # Breach penalty (Federation layer punishment)
        if self.current_breaches:
            r_penalty = self.cfg["rewards"]["r_boundary_breach"]
            breach_penalty = -len(self.current_breaches) * r_penalty
            avg_reward += breach_penalty
            self.current_breaches = []  # Clear for next step

        # Track performance
        self.avg_reward = (
            self.avg_reward * (self.steps_taken - 1) + avg_reward
        ) / self.steps_taken

        return avg_reward

    def get_state_summary(self) -> Dict[str, Any]:
        """
        Generate state summary for Federation.

        NEW: Include frozen node information.
        """
        frozen_nodes = []
        if self.orchestrator:
            frozen_nodes = [
                {"id": n.id, "pos": n.pos.tolist()}
                for n in self.orchestrator.get_frozen_nodes()
            ]

        return {
            "holon_id": self.holon_id,
            "partition_id": self.partition_id,
            "nodes": self.nodes,
            "num_nodes": len(self.nodes),
            "num_active_nodes": len(self.orchestrator.get_active_nodes())
            if self.orchestrator
            else len(self.nodes),
            "num_frozen_nodes": len(self.orchestrator.frozen_nodes)
            if self.orchestrator
            else 0,
            "frozen_nodes": frozen_nodes,
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
            "num_total_nodes": len(self.nodes),
            "num_active_nodes": len(self.orchestrator.get_active_nodes())
            if self.orchestrator
            else len(self.nodes),
            "num_frozen_nodes": len(self.orchestrator.frozen_nodes)
            if self.orchestrator
            else 0,
            "avg_reward": self.avg_reward,
            "total_breaches": self.total_breaches_received,
            "boundary_repulsion_active": self.boundary_repulsion_active,
            "orchestrator_stats": orch_stats,
            "strategic_freezing_enabled": self.enable_strategic_freezing,
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
