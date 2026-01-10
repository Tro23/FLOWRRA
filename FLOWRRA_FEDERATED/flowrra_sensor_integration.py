"""
flowrra_sensor_integration.py

Integration layer connecting sensor processors to FLOWRRA components.

Modifications needed for:
1. Orchestrator (core.py) - Per-node sensor processing
2. Holon (holon_core.py) - Coordinate transform-aware filtering
3. Federation Manager - Cross-holon consensus coordination

Usage:
    # In core.py __init__:
    from flowrra_sensor_integration import attach_sensor_processors
    attach_sensor_processors(self.nodes, self.dims)

    # In step():
    filtered_positions = process_sensor_readings(self.nodes, raw_measurements)
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
from sensor_fusion import SensorProcessor, ConsensusKalmanFilter


# =============================================================================
# ORCHESTRATOR INTEGRATION
# =============================================================================

def attach_sensor_processors(
    nodes: List,
    dimensions: int,
    use_consensus: bool = True,
    config: Optional[Dict] = None
) -> Dict[int, SensorProcessor]:
    """
    Attach sensor processors to all nodes in orchestrator.

    Call this in FLOWRRA_Orchestrator.__init__() after nodes are created.

    Args:
        nodes: List of NodePositionND objects
        dimensions: 2 or 3
        use_consensus: Enable distributed consensus
        config: Optional configuration overrides

    Returns:
        Dict mapping node_id -> SensorProcessor
    """
    processors = {}

    for node in nodes:
        processor = SensorProcessor(
            node_id=node.id,
            dimensions=dimensions,
            use_consensus=use_consensus,
            filter_mode="auto"  # Auto-select based on noise level
        )

        # Initialize with current node position
        processor.reset(node.pos.copy())

        # Store reference
        processors[node.id] = processor

        # Attach to node as attribute
        node.sensor_processor = processor

    print(f"[Sensor Fusion] Attached processors to {len(nodes)} nodes")
    return processors


def process_sensor_readings(
    nodes: List,
    raw_measurements: Optional[Dict[int, np.ndarray]] = None,
    enable_consensus: bool = True,
    frozen_node_ids: Optional[set] = None
) -> Dict[int, np.ndarray]:
    """
    Process sensor readings for all active nodes.

    Call this in FLOWRRA_Orchestrator.step() BEFORE physics calculations.

    Args:
        nodes: List of NodePositionND objects
        raw_measurements: Dict of node_id -> noisy position (if simulating sensors)
                         If None, assumes node.pos contains raw measurement
        enable_consensus: Perform consensus fusion across neighbors
        frozen_node_ids: Set of frozen node IDs (skip processing for these)

    Returns:
        Dict mapping node_id -> filtered position
    """
    if frozen_node_ids is None:
        frozen_node_ids = set()

    filtered_positions = {}

    # PHASE 1: Individual filtering
    for node in nodes:
        # Skip frozen nodes (their positions are fixed)
        if node.id in frozen_node_ids:
            filtered_positions[node.id] = node.pos.copy()
            continue

        if not hasattr(node, 'sensor_processor'):
            # No processor attached, use raw position
            filtered_positions[node.id] = node.pos.copy()
            continue

        # Get raw measurement
        if raw_measurements is not None and node.id in raw_measurements:
            raw_pos = raw_measurements[node.id]
        else:
            raw_pos = node.pos.copy()

        # Process through filter
        filtered_pos = node.sensor_processor.process_measurement(raw_pos)
        filtered_positions[node.id] = filtered_pos

        # Update node position with filtered value
        node.pos = filtered_pos

    # PHASE 2: Consensus fusion (if enabled)
    if enable_consensus:
        consensus_positions = consensus_fusion_step(
            nodes,
            filtered_positions,
            frozen_node_ids=frozen_node_ids
        )

        # Update with consensus values
        for node_id, consensus_pos in consensus_positions.items():
            filtered_positions[node_id] = consensus_pos

            # Update node
            node = next((n for n in nodes if n.id == node_id), None)
            if node:
                node.pos = consensus_pos

    return filtered_positions


def consensus_fusion_step(
    nodes: List,
    current_estimates: Dict[int, np.ndarray],
    sensor_range: float = 0.25,
    frozen_node_ids: Optional[set] = None
) -> Dict[int, np.ndarray]:
    """
    Perform one round of consensus fusion among neighbors.

    Args:
        nodes: List of nodes
        current_estimates: Current position estimates
        sensor_range: Communication range for consensus
        frozen_node_ids: Frozen nodes (included as landmarks)

    Returns:
        Updated consensus estimates
    """
    if frozen_node_ids is None:
        frozen_node_ids = set()

    consensus_positions = {}

    for node in nodes:
        # Skip frozen nodes
        if node.id in frozen_node_ids:
            consensus_positions[node.id] = node.pos.copy()
            continue

        if not hasattr(node, 'sensor_processor') or node.sensor_processor.consensus is None:
            consensus_positions[node.id] = current_estimates[node.id]
            continue

        # Find neighbors within sensor range
        neighbor_estimates = {}

        for other in nodes:
            if other.id == node.id:
                continue

            # Toroidal distance
            delta = other.pos - node.pos
            toroidal_delta = np.mod(delta + 0.5, 1.0) - 0.5
            distance = np.linalg.norm(toroidal_delta)

            if distance < sensor_range:
                neighbor_estimates[other.id] = current_estimates[other.id]

        # Perform consensus fusion
        consensus_pos = node.sensor_processor.exchange_with_neighbors(neighbor_estimates)
        consensus_positions[node.id] = consensus_pos

    return consensus_positions


# =============================================================================
# HOLON INTEGRATION (Coordinate Transform-Aware)
# =============================================================================

class HolonSensorManager:
    """
    Manages sensor processing for a holon with coordinate transforms.

    Handles the "delusion" - filtering happens in GLOBAL coordinates,
    then transforms to LOCAL for orchestrator.
    """

    def __init__(
        self,
        holon_id: int,
        spatial_bounds: Dict[str, Tuple],
        dimensions: int
    ):
        """
        Args:
            holon_id: Holon identifier
            spatial_bounds: Holon's spatial region
            dimensions: 2 or 3
        """
        self.holon_id = holon_id
        self.dims = dimensions

        # Coordinate transform parameters
        self.x_min, self.x_max = spatial_bounds["x"]
        self.y_min, self.y_max = spatial_bounds["y"]

        # Sensor processors (created when nodes attached)
        self.processors: Dict[int, SensorProcessor] = {}

    def initialize_processors(self, nodes: List):
        """Initialize sensor processors for nodes."""
        self.processors = attach_sensor_processors(
            nodes,
            self.dims,
            use_consensus=True
        )

    def _to_local(self, global_pos: np.ndarray) -> np.ndarray:
        """Transform global [0,1] to local [0,1]."""
        lx = (global_pos[0] - self.x_min) / (self.x_max - self.x_min)
        ly = (global_pos[1] - self.y_min) / (self.y_max - self.y_min)
        return np.array([lx, ly])

    def _to_global(self, local_pos: np.ndarray) -> np.ndarray:
        """Transform local [0,1] to global coordinates."""
        gx = local_pos[0] * (self.x_max - self.x_min) + self.x_min
        gy = local_pos[1] * (self.y_max - self.y_min) + self.y_min
        return np.array([gx, gy])

    def process_step(
        self,
        nodes: List,
        raw_measurements: Optional[Dict[int, np.ndarray]] = None,
        frozen_node_ids: Optional[set] = None
    ) -> Dict[int, np.ndarray]:
        """
        Process sensor readings with coordinate transform awareness.

        Pipeline:
        1. Raw measurements arrive in GLOBAL coordinates
        2. Filter in GLOBAL space (maintains physical consistency)
        3. Transform filtered positions to LOCAL space for orchestrator

        Args:
            nodes: Holon's nodes (currently in GLOBAL coords)
            raw_measurements: Raw sensor readings (GLOBAL coords)
            frozen_node_ids: Frozen node IDs

        Returns:
            Filtered positions in LOCAL coordinates
        """
        if frozen_node_ids is None:
            frozen_node_ids = set()

        # Process in GLOBAL coordinates
        global_filtered = {}

        for node in nodes:
            if node.id in frozen_node_ids:
                global_filtered[node.id] = node.pos.copy()
                continue

            if node.id not in self.processors:
                global_filtered[node.id] = node.pos.copy()
                continue

            # Get raw measurement (already in global coords)
            if raw_measurements and node.id in raw_measurements:
                raw_global = raw_measurements[node.id]
            else:
                raw_global = node.pos.copy()

            # Filter in global space
            filtered_global = self.processors[node.id].process_measurement(raw_global)
            global_filtered[node.id] = filtered_global

        # Transform to LOCAL coordinates
        local_filtered = {}
        for node_id, global_pos in global_filtered.items():
            local_filtered[node_id] = self._to_local(global_pos)

        return local_filtered


# =============================================================================
# FEDERATION INTEGRATION (Cross-Holon Consensus)
# =============================================================================

class FederationSensorCoordinator:
    """
    Coordinates sensor fusion across holons.

    Handles:
    - Cross-holon consensus for boundary nodes
    - Breach detection with filtered positions
    - Communication bandwidth management
    """

    def __init__(self, num_holons: int):
        """
        Args:
            num_holons: Number of holons in federation
        """
        self.num_holons = num_holons

        # Track communication bandwidth usage
        self.messages_per_holon = {i: 0 for i in range(num_holons)}
        self.bandwidth_limit = 100  # Max messages per step per holon

    def cross_holon_consensus(
        self,
        holon_states: Dict[int, Dict],
        breach_threshold: float = 0.1
    ) -> Dict[int, List[Dict]]:
        """
        Perform consensus for nodes near holon boundaries.

        Args:
            holon_states: State summaries from all holons
            breach_threshold: Distance from boundary to trigger consensus

        Returns:
            Dict mapping holon_id -> consensus updates
        """
        consensus_updates = {i: [] for i in range(self.num_holons)}

        # Find nodes near boundaries
        boundary_nodes = self._identify_boundary_nodes(
            holon_states,
            breach_threshold
        )

        # For each boundary node, fuse estimates from adjacent holons
        for node_info in boundary_nodes:
            node_id = node_info["node_id"]
            holon_id = node_info["holon_id"]
            adjacent_holons = node_info["adjacent_holons"]

            # Gather estimates from all relevant holons
            estimates = []
            for adj_holon_id in [holon_id] + adjacent_holons:
                holon_nodes = holon_states[adj_holon_id]["nodes"]
                node = next((n for n in holon_nodes if n.id == node_id), None)
                if node:
                    estimates.append(node.pos)

            if len(estimates) > 1:
                # Average estimates (simple consensus)
                consensus_pos = np.mean(estimates, axis=0)

                consensus_updates[holon_id].append({
                    "node_id": node_id,
                    "consensus_position": consensus_pos,
                    "contributing_holons": len(estimates)
                })

                self.messages_per_holon[holon_id] += len(adjacent_holons)

        return consensus_updates

    def _identify_boundary_nodes(
        self,
        holon_states: Dict[int, Dict],
        threshold: float
    ) -> List[Dict]:
        """
        Identify nodes near holon boundaries.

        Returns:
            List of dicts with node_id, holon_id, position, adjacent_holons
        """
        boundary_nodes = []

        for holon_id, state in holon_states.items():
            # Get holon bounds (assumed in state summary)
            # You'll need to add this to get_state_summary() in holon_core.py
            if "bounds" not in state:
                continue

            x_min, x_max = state["bounds"]["x"]
            y_min, y_max = state["bounds"]["y"]

            for node in state["nodes"]:
                # Check distance from each boundary
                dist_to_left = node.pos[0] - x_min
                dist_to_right = x_max - node.pos[0]
                dist_to_bottom = node.pos[1] - y_min
                dist_to_top = y_max - node.pos[1]

                min_dist = min(dist_to_left, dist_to_right, dist_to_bottom, dist_to_top)

                if min_dist < threshold:
                    # This node is near a boundary
                    adjacent = self._get_adjacent_holons(
                        holon_id,
                        dist_to_left < threshold,
                        dist_to_right < threshold,
                        dist_to_bottom < threshold,
                        dist_to_top < threshold
                    )

                    boundary_nodes.append({
                        "node_id": node.id,
                        "holon_id": holon_id,
                        "position": node.pos,
                        "adjacent_holons": adjacent
                    })

        return boundary_nodes

    def _get_adjacent_holons(
        self,
        holon_id: int,
        near_left: bool,
        near_right: bool,
        near_bottom: bool,
        near_top: bool
    ) -> List[int]:
        """
        Get IDs of holons adjacent to boundaries.

        Assumes grid layout of holons.
        """
        # Calculate grid position
        grid_size = int(np.sqrt(self.num_holons))
        row = holon_id // grid_size
        col = holon_id % grid_size

        adjacent = []

        if near_left and col > 0:
            adjacent.append(holon_id - 1)
        if near_right and col < grid_size - 1:
            adjacent.append(holon_id + 1)
        if near_bottom and row > 0:
            adjacent.append(holon_id - grid_size)
        if near_top and row < grid_size - 1:
            adjacent.append(holon_id + grid_size)

        return adjacent

    def get_bandwidth_usage(self) -> Dict[int, float]:
        """
        Get communication bandwidth usage per holon.

        Returns:
            Dict mapping holon_id -> usage fraction (0-1)
        """
        return {
            holon_id: msg_count / self.bandwidth_limit
            for holon_id, msg_count in self.messages_per_holon.items()
        }

    def reset_bandwidth_counters(self):
        """Reset bandwidth tracking (call each step)."""
        self.messages_per_holon = {i: 0 for i in range(self.num_holons)}


# =============================================================================
# INTEGRATION INSTRUCTIONS
# =============================================================================

"""
INTEGRATION GUIDE

1. ORCHESTRATOR (holon/core.py):

   In __init__():
   ```python
   from flowrra_sensor_integration import attach_sensor_processors

   # After creating self.nodes
   self.sensor_processors = attach_sensor_processors(
       self.nodes,
       self.dims,
       use_consensus=True
   )
   ```

   In step() method, BEFORE physics:
   ```python
   from flowrra_sensor_integration import process_sensor_readings

   # Filter sensor readings (replaces raw node.pos with filtered)
   filtered_positions = process_sensor_readings(
       self.nodes,
       raw_measurements=None,  # Or provide dict of noisy readings
       enable_consensus=True,
       frozen_node_ids=self.frozen_nodes
   )

   # Now node.pos contains filtered positions
   # Continue with normal physics...
   ```

2. HOLON (holon/holon_core.py):

   In __init__():
   ```python
   from flowrra_sensor_integration import HolonSensorManager

   self.sensor_manager = HolonSensorManager(
       holon_id=self.holon_id,
       spatial_bounds=self.spatial_bounds,
       dimensions=self.dimensions
   )
   ```

   In initialize_orchestrator_with_nodes():
   ```python
   # After orchestrator initialization
   self.sensor_manager.initialize_processors(self.nodes)
   ```

   In step() method, in PHASE 1 (before normalization):
   ```python
   # Process sensors in GLOBAL space
   global_filtered = self.sensor_manager.process_step(
       self.nodes,
       raw_measurements=None,
       frozen_node_ids=self.orchestrator.frozen_nodes if self.orchestrator else set()
   )

   # Apply filtered positions
   for node in self.nodes:
       if node.id in global_filtered:
           node.pos = global_filtered[node.id]

   # Now continue with coordinate normalization...
   ```

3. FEDERATION (federation/manager.py):

   In __init__():
   ```python
   from flowrra_sensor_integration import FederationSensorCoordinator

   self.sensor_coordinator = FederationSensorCoordinator(
       num_holons=self.num_holons
   )
   ```

   In step() method:
   ```python
   # After collecting holon states
   consensus_updates = self.sensor_coordinator.cross_holon_consensus(
       holon_states,
       breach_threshold=self.breach_threshold
   )

   # Send consensus updates to holons
   for holon_id, updates in consensus_updates.items():
       # You'll need to add a method to apply these
       # self.holons[holon_id].apply_consensus_updates(updates)

   # Check bandwidth usage
   bandwidth = self.sensor_coordinator.get_bandwidth_usage()

   # Reset for next step
   self.sensor_coordinator.reset_bandwidth_counters()
   ```

4. MODIFY get_state_summary() in holon_core.py:

   Add bounds to state summary:
   ```python
   return {
       "holon_id": self.holon_id,
       "nodes": self.nodes,
       "bounds": {  # ADD THIS
           "x": (self.x_min, self.x_max),
           "y": (self.y_min, self.y_max)
       },
       # ... rest of summary
   }
   ```
"""
