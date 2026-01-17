"""
dataset_adapter.py

Domain-Agnostic Dataset Adapter for FLOWRRA.

This adapter converts ANY domain dataset into FLOWRRA's standardized format:
- State representation
- Action space
- Obstacle/constraint definitions
- Reward functions

Supported Domains (Initial):
1. Warehouse Robotics
2. Smart Cities (Traffic)
3. Satellite Constellations
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class FlowrraState:
    """Standardized state representation for any domain."""
    
    # Spatial information
    positions: np.ndarray  # [num_agents, dimensions]
    velocities: np.ndarray  # [num_agents, dimensions]
    
    # Domain-specific features (as dictionary for O(1) hashing)
    features: Dict[str, np.ndarray]
    
    # Constraints (obstacles, boundaries, etc.)
    constraints: Dict[str, Any]
    
    # Metadata
    timestep: int
    domain: str


@dataclass
class FlowrraAction:
    """Standardized action representation."""
    
    agent_id: int
    action_type: str  # 'discrete' or 'continuous'
    action_value: np.ndarray  # Actual action (index or vector)
    
    # Domain-specific action metadata
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class FlowrraConfig:
    """Domain-agnostic configuration."""
    
    # Spatial parameters
    dimensions: int  # 2 or 3
    world_bounds: Tuple[float, ...]
    
    # Agent parameters
    num_agents: int
    agent_speed: float
    sensor_range: float
    
    # Action space
    action_space_type: str  # 'discrete' or 'continuous'
    action_space_size: int  # For discrete
    action_space_bounds: Optional[Tuple[float, ...]] = None  # For continuous
    
    # Domain-specific parameters
    domain_params: Dict[str, Any] = None
    
    # Federation parameters
    num_holons: int = 4
    enable_wfc: bool = True
    enable_frozen_nodes: bool = True


class DatasetAdapter(ABC):
    """
    Abstract base class for domain-specific dataset adapters.
    
    Each domain (warehouse, traffic, satellites) implements this interface.
    """
    
    def __init__(self, dataset_path: str, config: FlowrraConfig):
        self.dataset_path = dataset_path
        self.config = config
        self.current_episode = 0
        self.current_step = 0
        
    @abstractmethod
    def load_dataset(self) -> Dict[str, Any]:
        """
        Load and preprocess the domain-specific dataset.
        
        Returns:
            Dictionary containing:
            - 'episodes': List of episode data
            - 'metadata': Dataset metadata
            - 'statistics': Dataset statistics
        """
        pass
    
    @abstractmethod
    def get_initial_state(self, episode_idx: int = 0) -> FlowrraState:
        """
        Get initial state for an episode.
        
        Returns:
            FlowrraState with initial positions, velocities, constraints
        """
        pass
    
    @abstractmethod
    def convert_action(self, flowrra_action: np.ndarray) -> Any:
        """
        Convert FLOWRRA action (GNN output) to domain-specific action.
        
        Args:
            flowrra_action: [num_agents] action indices or vectors
            
        Returns:
            Domain-specific action format
        """
        pass
    
    @abstractmethod
    def compute_reward(
        self, 
        state: FlowrraState, 
        action: np.ndarray, 
        next_state: FlowrraState
    ) -> np.ndarray:
        """
        Compute domain-specific rewards.
        
        Returns:
            [num_agents] reward values
        """
        pass
    
    @abstractmethod
    def extract_constraints(self, state: FlowrraState) -> Dict[str, Any]:
        """
        Extract obstacles, boundaries, and other constraints from state.
        
        Returns:
            Dictionary with:
            - 'static_obstacles': List of (pos, radius)
            - 'dynamic_obstacles': List of (pos, radius, velocity)
            - 'boundaries': World boundaries
            - 'no_fly_zones': Domain-specific constraints
        """
        pass
    
    @abstractmethod
    def is_terminal(self, state: FlowrraState) -> bool:
        """Check if episode should terminate."""
        pass
    
    def step(
        self, 
        action: np.ndarray
    ) -> Tuple[FlowrraState, np.ndarray, bool, Dict[str, Any]]:
        """
        Execute one environment step.
        
        Returns:
            (next_state, rewards, done, info)
        """
        # This will be implemented by FLOWRRA's orchestrator
        # The adapter just provides the interface
        raise NotImplementedError("Use orchestrator.step() instead")


# =============================================================================
# DOMAIN-SPECIFIC ADAPTERS
# =============================================================================

class WarehouseRoboticsAdapter(DatasetAdapter):
    """
    Adapter for warehouse robotics datasets.
    
    Expected format:
    - Positions: (x, y) in meters
    - Obstacles: Shelves, walls, loading zones
    - Actions: Linear/angular velocity commands
    - Goals: Pick/place locations
    """
    
    def load_dataset(self) -> Dict[str, Any]:
        """Load warehouse simulation data or real-world logs."""
        # TODO: Implement dataset loading
        # For now, generate synthetic data
        
        return {
            'episodes': [],
            'metadata': {
                'domain': 'warehouse_robotics',
                'num_robots': self.config.num_agents,
                'warehouse_size': self.config.world_bounds,
            },
            'statistics': {}
        }
    
    def get_initial_state(self, episode_idx: int = 0) -> FlowrraState:
        """Initialize robots in loading zone."""
        num_robots = self.config.num_agents
        
        # Spawn in a grid at one corner
        positions = []
        for i in range(num_robots):
            x = 0.1 + (i % 4) * 0.05
            y = 0.1 + (i // 4) * 0.05
            positions.append([x, y])
        
        positions = np.array(positions)
        velocities = np.zeros_like(positions)
        
        # Domain features: battery, load status, task queue
        features = {
            'battery_level': np.ones(num_robots) * 100.0,
            'carrying_load': np.zeros(num_robots, dtype=bool),
            'task_id': np.arange(num_robots),
        }
        
        # Static obstacles: shelves
        constraints = self.extract_constraints(None)
        
        return FlowrraState(
            positions=positions,
            velocities=velocities,
            features=features,
            constraints=constraints,
            timestep=0,
            domain='warehouse_robotics'
        )
    
    def convert_action(self, flowrra_action: np.ndarray) -> Any:
        """
        Convert discrete actions to velocity commands.
        
        Actions: 0=left, 1=right, 2=up, 3=down (2D)
        """
        num_agents = len(flowrra_action)
        velocities = np.zeros((num_agents, 2))
        
        speed = self.config.agent_speed
        
        for i, action in enumerate(flowrra_action):
            if action == 0:  # Left
                velocities[i] = [-speed, 0]
            elif action == 1:  # Right
                velocities[i] = [speed, 0]
            elif action == 2:  # Up
                velocities[i] = [0, speed]
            elif action == 3:  # Down
                velocities[i] = [0, -speed]
        
        return velocities
    
    def compute_reward(
        self, 
        state: FlowrraState, 
        action: np.ndarray, 
        next_state: FlowrraState
    ) -> np.ndarray:
        """
        Warehouse reward:
        - Movement toward goal: +1
        - Collision with shelf: -10
        - Task completion: +50
        - Idle: -0.1
        """
        num_agents = len(state.positions)
        rewards = np.zeros(num_agents)
        
        for i in range(num_agents):
            # Movement reward
            movement = np.linalg.norm(
                next_state.positions[i] - state.positions[i]
            )
            rewards[i] += movement * 5.0
            
            # Collision penalty (check against constraints)
            if self._check_collision(next_state.positions[i], next_state.constraints):
                rewards[i] -= 10.0
            
            # Idle penalty
            if movement < 0.001:
                rewards[i] -= 0.1
        
        return rewards
    
    def extract_constraints(self, state: Optional[FlowrraState]) -> Dict[str, Any]:
        """Generate shelf obstacles."""
        # Generate warehouse shelf layout
        shelves = []
        
        # Create grid of shelves (leave aisles)
        for row in range(5):
            for col in range(8):
                if col % 2 == 1:  # Aisles every other column
                    continue
                x = 0.2 + col * 0.1
                y = 0.2 + row * 0.15
                shelves.append((np.array([x, y]), 0.04))  # 4% of world size
        
        return {
            'static_obstacles': shelves,
            'dynamic_obstacles': [],
            'boundaries': self.config.world_bounds,
            'loading_zones': [np.array([0.05, 0.05])],
        }
    
    def is_terminal(self, state: FlowrraState) -> bool:
        """Episode ends after fixed time or all tasks done."""
        return state.timestep >= 500
    
    def _check_collision(self, pos: np.ndarray, constraints: Dict) -> bool:
        """Check if position collides with any obstacle."""
        for obs_pos, radius in constraints['static_obstacles']:
            if np.linalg.norm(pos - obs_pos) < radius:
                return True
        return False


class SmartCityTrafficAdapter(DatasetAdapter):
    """
    Adapter for smart city traffic management datasets.
    
    Expected format:
    - Positions: (lat, lon) or (x, y) in city grid
    - Obstacles: Buildings, road boundaries
    - Actions: Signal timing, lane assignments
    - Goals: Minimize congestion, travel time
    """
    
    def load_dataset(self) -> Dict[str, Any]:
        """Load traffic simulation data (SUMO, real-world logs)."""
        # TODO: Load real traffic data
        return {
            'episodes': [],
            'metadata': {'domain': 'smart_city_traffic'},
            'statistics': {}
        }
    
    def get_initial_state(self, episode_idx: int = 0) -> FlowrraState:
        """Initialize vehicles at intersections."""
        num_vehicles = self.config.num_agents
        
        # Spawn at major intersections
        positions = np.random.uniform(0.1, 0.9, (num_vehicles, 2))
        velocities = np.random.uniform(-0.01, 0.01, (num_vehicles, 2))
        
        features = {
            'vehicle_type': np.random.randint(0, 3, num_vehicles),  # car, bus, truck
            'destination': np.random.uniform(0, 1, (num_vehicles, 2)),
            'urgency': np.random.uniform(0, 1, num_vehicles),
        }
        
        constraints = self.extract_constraints(None)
        
        return FlowrraState(
            positions=positions,
            velocities=velocities,
            features=features,
            constraints=constraints,
            timestep=0,
            domain='smart_city_traffic'
        )
    
    def convert_action(self, flowrra_action: np.ndarray) -> Any:
        """Convert to lane/speed adjustments."""
        # Actions map to acceleration/deceleration
        return flowrra_action  # Placeholder
    
    def compute_reward(
        self, 
        state: FlowrraState, 
        action: np.ndarray, 
        next_state: FlowrraState
    ) -> np.ndarray:
        """
        Traffic reward:
        - Progress toward destination: +1
        - Collision avoidance: +5
        - Congestion penalty: -2
        """
        num_agents = len(state.positions)
        rewards = np.zeros(num_agents)
        
        for i in range(num_agents):
            # Progress reward
            dest = state.features['destination'][i]
            old_dist = np.linalg.norm(state.positions[i] - dest)
            new_dist = np.linalg.norm(next_state.positions[i] - dest)
            
            rewards[i] += (old_dist - new_dist) * 10.0
            
            # Congestion penalty (if too close to others)
            for j in range(num_agents):
                if i != j:
                    dist = np.linalg.norm(
                        next_state.positions[i] - next_state.positions[j]
                    )
                    if dist < 0.05:
                        rewards[i] -= 2.0
        
        return rewards
    
    def extract_constraints(self, state: Optional[FlowrraState]) -> Dict[str, Any]:
        """Extract road network and buildings."""
        # Simplified: buildings as rectangular obstacles
        buildings = []
        for i in range(10):
            x = np.random.uniform(0.1, 0.9)
            y = np.random.uniform(0.1, 0.9)
            buildings.append((np.array([x, y]), 0.08))
        
        return {
            'static_obstacles': buildings,
            'dynamic_obstacles': [],
            'boundaries': self.config.world_bounds,
            'road_network': None,  # TODO: Add road graph
        }
    
    def is_terminal(self, state: FlowrraState) -> bool:
        """Episode ends when all vehicles reach destinations."""
        dests = state.features['destination']
        dists = np.linalg.norm(state.positions - dests, axis=1)
        return np.all(dists < 0.05) or state.timestep >= 1000


class SatelliteConstellationAdapter(DatasetAdapter):
    """
    Adapter for satellite constellation management.
    
    Expected format:
    - Positions: Orbital elements (or 3D Cartesian)
    - Obstacles: Debris, no-fly zones
    - Actions: Thrust vectors, antenna pointing
    - Goals: Coverage, collision avoidance
    """
    
    def load_dataset(self) -> Dict[str, Any]:
        """Load TLE data or constellation simulation."""
        # TODO: Load real satellite data (CelesTrak)
        return {
            'episodes': [],
            'metadata': {'domain': 'satellite_constellation'},
            'statistics': {}
        }
    
    def get_initial_state(self, episode_idx: int = 0) -> FlowrraState:
        """Initialize satellites in orbit."""
        num_sats = self.config.num_agents
        
        # Simplified: uniform distribution in orbital shell
        positions = np.random.uniform(0.3, 0.7, (num_sats, 3))
        
        # Orbital velocities (perpendicular to radius)
        velocities = np.zeros((num_sats, 3))
        for i in range(num_sats):
            r = positions[i]
            r_norm = r / np.linalg.norm(r)
            # Circular orbit velocity
            v_mag = 0.01  # Simplified
            velocities[i] = np.cross(r_norm, [0, 0, 1]) * v_mag
        
        features = {
            'power_level': np.ones(num_sats) * 80.0,
            'link_quality': np.random.uniform(0.5, 1.0, num_sats),
            'coverage_score': np.zeros(num_sats),
        }
        
        constraints = self.extract_constraints(None)
        
        return FlowrraState(
            positions=positions,
            velocities=velocities,
            features=features,
            constraints=constraints,
            timestep=0,
            domain='satellite_constellation'
        )
    
    def convert_action(self, flowrra_action: np.ndarray) -> Any:
        """Convert to thrust commands."""
        # Actions: 0-5 for +/-x, +/-y, +/-z thrust
        num_sats = len(flowrra_action)
        thrusts = np.zeros((num_sats, 3))
        
        thrust_mag = 0.001
        
        for i, action in enumerate(flowrra_action):
            if action == 0:
                thrusts[i] = [thrust_mag, 0, 0]
            elif action == 1:
                thrusts[i] = [-thrust_mag, 0, 0]
            elif action == 2:
                thrusts[i] = [0, thrust_mag, 0]
            elif action == 3:
                thrusts[i] = [0, -thrust_mag, 0]
            elif action == 4:
                thrusts[i] = [0, 0, thrust_mag]
            elif action == 5:
                thrusts[i] = [0, 0, -thrust_mag]
        
        return thrusts
    
    def compute_reward(
        self, 
        state: FlowrraState, 
        action: np.ndarray, 
        next_state: FlowrraState
    ) -> np.ndarray:
        """
        Satellite reward:
        - Coverage maintenance: +10
        - Collision avoidance: +20
        - Power efficiency: +1
        """
        num_sats = len(state.positions)
        rewards = np.zeros(num_sats)
        
        for i in range(num_sats):
            # Coverage reward (simplified)
            coverage_delta = (
                next_state.features['coverage_score'][i] - 
                state.features['coverage_score'][i]
            )
            rewards[i] += coverage_delta * 10.0
            
            # Collision avoidance
            min_dist = float('inf')
            for j in range(num_sats):
                if i != j:
                    dist = np.linalg.norm(
                        next_state.positions[i] - next_state.positions[j]
                    )
                    min_dist = min(min_dist, dist)
            
            if min_dist > 0.1:
                rewards[i] += 20.0
            elif min_dist < 0.05:
                rewards[i] -= 50.0  # Near collision!
            
            # Power efficiency
            power_used = (
                state.features['power_level'][i] - 
                next_state.features['power_level'][i]
            )
            rewards[i] += (1.0 - power_used / 100.0)
        
        return rewards
    
    def extract_constraints(self, state: Optional[FlowrraState]) -> Dict[str, Any]:
        """Extract debris and no-fly zones."""
        # Simplified: random debris objects
        debris = []
        for i in range(20):
            pos = np.random.uniform(0.2, 0.8, 3)
            debris.append((pos, 0.02))
        
        return {
            'static_obstacles': debris,
            'dynamic_obstacles': [],
            'boundaries': self.config.world_bounds,
            'earth_radius': 0.1,  # Avoid crashing into Earth!
        }
    
    def is_terminal(self, state: FlowrraState) -> bool:
        """Episode ends after orbital period."""
        return state.timestep >= 800


# =============================================================================
# ADAPTER FACTORY
# =============================================================================

def create_adapter(
    domain: str, 
    dataset_path: str, 
    config: FlowrraConfig
) -> DatasetAdapter:
    """
    Factory function to create the appropriate adapter.
    
    Args:
        domain: 'warehouse', 'traffic', or 'satellite'
        dataset_path: Path to dataset
        config: FLOWRRA configuration
    
    Returns:
        DatasetAdapter instance
    """
    adapters = {
        'warehouse': WarehouseRoboticsAdapter,
        'traffic': SmartCityTrafficAdapter,
        'satellite': SatelliteConstellationAdapter,
    }
    
    if domain not in adapters:
        raise ValueError(
            f"Unknown domain '{domain}'. "
            f"Supported: {list(adapters.keys())}"
        )
    
    return adapters[domain](dataset_path, config)
