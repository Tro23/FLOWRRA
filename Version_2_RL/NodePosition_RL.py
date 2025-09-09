"""
NodePosition_RL.py

Represents the physical state of a single node in the FLOWRRA environment.
This class holds all kinematic properties, such as position, velocity, and orientation,
and provides methods to update them based on actions. It now also handles its
own sensing capabilities and a local repulsion grid.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Dict, Any

# Prevents circular import issues while allowing type hinting
if TYPE_CHECKING:
    from EnvironmentA_RL import Environment_A

@dataclass
class Node_Position:
    """
    Data class for a node's physical state and sensing attributes.

    Attributes:
        id (int): A unique identifier for the node.
        pos (np.ndarray): The (x, y) coordinates of the node in a normalized [0,1) space.
        angle_idx (int): The discrete orientation of the node, from 0 to angle_steps-1.
        rotation_speed (float): Maximum number of angle indices the node can turn per step.
        move_speed (float): Maximum per-step displacement.
        angle_steps (int): The number of discrete steps in a full 360-degree rotation.
        target_angle_idx (int): The target orientation the node is currently rotating towards.
        last_pos (np.ndarray): The position of the node in the previous timestep, used for velocity calculation.
        
        # Sensor Attributes
        sensor_range (float): The range in which the node can detect other nodes and obstacles.
        
        # New: Per-node repulsion grid
        repulsion_grid: np.ndarray = field(default_factory=lambda: np.zeros((4, 4), dtype=np.float32))

    """
    id: int
    pos: np.ndarray
    angle_idx: int
    rotation_speed: float = 2.0
    move_speed: float = 0.01
    angle_steps: int = 360
    target_angle_idx: int = 0
    last_pos: np.ndarray = field(default_factory=lambda: np.zeros(2))
    sensor_range: float = 0.1

    def __post_init__(self):
        self.last_pos = self.pos.copy()
        
    def velocity(self) -> np.ndarray:
        """Calculates the current velocity based on last and current positions."""
        delta = self.pos - self.last_pos
        # Toroidal wrapping for velocity calculation
        toroidal_delta = np.mod(delta + 0.5, 1.0) - 0.5
        return toroidal_delta

    def get_angle_rad(self) -> float:
        """Returns the current angle in radians."""
        return (self.angle_idx / self.angle_steps) * 2 * np.pi

    def rotate_towards_target(self):
        """
        Rotates the node's angle_idx towards its target_angle_idx.
        """
        if self.angle_idx == self.target_angle_idx:
            return

        # Calculate clockwise and counter-clockwise distances
        # Angle difference is calculated in [0, 1] range of angle_steps
        diff = (self.target_angle_idx - self.angle_idx + self.angle_steps) % self.angle_steps
        
        if diff <= self.angle_steps / 2:
            # Rotate clockwise
            step = min(diff, self.rotation_speed)
            self.angle_idx = (self.angle_idx + int(step)) % self.angle_steps
        else:
            # Rotate counter-clockwise
            step = min(self.angle_steps - diff, self.rotation_speed)
            self.angle_idx = (self.angle_idx - int(step) + self.angle_steps) % self.angle_steps
    
    def move(self, dt: float = 1.0):
        """
        Moves the node forward in its current direction.
        """
        self.last_pos = self.pos.copy()
        angle_rad = self.get_angle_rad()
        direction_vector = np.array([np.cos(angle_rad), np.sin(angle_rad)])
        # New position is the current position plus the scaled direction vector
        self.pos = np.mod(self.pos + direction_vector * self.move_speed * dt, 1.0)
    
    def apply_position_action(self, pos_action: np.ndarray, dt: float = 1.0):
        """
        Applies a raw position change from the RL agent.
        """
        self.last_pos = self.pos.copy()
        self.pos = np.mod(self.pos + pos_action * dt, 1.0)

    def apply_angle_action(self, angle_action: int):
        """
        Applies a discrete angular action to the node.
        
        Args:
            angle_action (int): An index representing the desired rotation.
        """
        if angle_action == 0:  # Rotate left
            self.angle_idx = (self.angle_idx - int(self.rotation_speed) + self.angle_steps) % self.angle_steps
        elif angle_action == 1:  # Rotate right
            self.angle_idx = (self.angle_idx + int(self.rotation_speed)) % self.angle_steps
        # Other angle_action values (e.g., 2, 3) can be no-ops or other movements
        
    def sense_nodes(self, all_nodes: List[Node_Position]) -> List[Dict[str, Any]]:
        """
        Senses other nodes within the sensor range, excluding self.
        """
        detections = []
        for other_node in all_nodes:
            if other_node.id == self.id:
                continue

            delta = other_node.pos - self.pos
            # Use toroidal distance for sensing
            toroidal_delta = np.mod(delta + 0.5, 1.0) - 0.5
            distance = np.linalg.norm(toroidal_delta)

            if distance < self.sensor_range:
                relative_velocity = other_node.velocity() - self.velocity()
                bearing_rad = np.arctan2(toroidal_delta[1], toroidal_delta[0])
                
                detections.append({
                    'id': other_node.id,
                    'pos': other_node.pos.copy(),
                    'distance': float(distance),
                    'bearing_rad': float(bearing_rad),
                    'velocity': relative_velocity,
                    'type': 'node'
                })
        return detections
    
    def sense_obstacles(self, obstacle_states: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Senses obstacles within the sensor range.
        """
        detections = []
        for obstacle in obstacle_states:
            # Calculate toroidal distance to the obstacle
            delta = obstacle['pos'] - self.pos
            toroidal_delta = np.mod(delta + 0.5, 1.0) - 0.5
            distance = np.linalg.norm(toroidal_delta)

            if distance < self.sensor_range:
                relative_velocity = obstacle['velocity'] - self.velocity()
                bearing_rad = np.arctan2(toroidal_delta[1], toroidal_delta[0])

                detections.append({
                    'id': -1,  # A generic ID for obstacles
                    'pos': obstacle['pos'].copy(),
                    'distance': float(distance),
                    'bearing_rad': float(bearing_rad),
                    'velocity': relative_velocity,
                    'type': obstacle['type']
                })
        return detections
