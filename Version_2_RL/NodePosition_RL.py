"""
NodePosition_RL.py - FIXED VERSION

Fixed the action implementation to properly handle discrete actions
and added movement incentives.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Dict, Any

if TYPE_CHECKING:
    from EnvironmentA_RL import Environment_A

@dataclass
class Node_Position:
    """
    Data class for a node's physical state and sensing attributes.
    """
    id: int
    pos: np.ndarray
    angle_idx: int
    rotation_speed: float = 2.0
    move_speed: float = 0.015  # Increased from 0.01 for more movement
    angle_steps: int = 360
    target_angle_idx: int = 0
    last_pos: np.ndarray = field(default_factory=lambda: np.zeros(2))
    sensor_range: float = 0.15  # Increased sensor range

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
        """Rotates the node's angle_idx towards its target_angle_idx."""
        if self.angle_idx == self.target_angle_idx:
            return

        diff = (self.target_angle_idx - self.angle_idx + self.angle_steps) % self.angle_steps
        
        if diff <= self.angle_steps / 2:
            step = min(diff, self.rotation_speed)
            self.angle_idx = (self.angle_idx + int(step)) % self.angle_steps
        else:
            step = min(self.angle_steps - diff, self.rotation_speed)
            self.angle_idx = (self.angle_idx - int(step) + self.angle_steps) % self.angle_steps
    
    def move(self, dt: float = 1.0):
        """Moves the node forward in its current direction."""
        self.last_pos = self.pos.copy()
        angle_rad = self.get_angle_rad()
        direction_vector = np.array([np.cos(angle_rad), np.sin(angle_rad)])
        self.pos = np.mod(self.pos + direction_vector * self.move_speed * dt, 1.0)
    
    def apply_position_action(self, pos_action: int, dt: float = 1.0):
        """
        FIXED: Apply discrete position action properly.
        
        Args:
            pos_action (int): Discrete action index (0=left, 1=right, 2=up, 3=down)
            dt (float): Time step
        """
        self.last_pos = self.pos.copy()
        
        # Convert discrete action to movement vector
        if pos_action == 0:  # Move left
            delta = np.array([-self.move_speed, 0.0])
        elif pos_action == 1:  # Move right  
            delta = np.array([self.move_speed, 0.0])
        elif pos_action == 2:  # Move up
            delta = np.array([0.0, self.move_speed])
        elif pos_action == 3:  # Move down
            delta = np.array([0.0, -self.move_speed])
        else:
            delta = np.array([0.0, 0.0])  # No movement
            
        # Apply movement with toroidal wrapping
        self.pos = np.mod(self.pos + delta * dt, 1.0)

    def apply_angle_action(self, angle_action: int):
        """
        Apply discrete angular action to the node.
        
        Args:
            angle_action (int): Discrete angle action (0=left, 1=right, 2=forward, 3=noop)
        """
        if angle_action == 0:  # Rotate left
            self.angle_idx = (self.angle_idx - int(self.rotation_speed) + self.angle_steps) % self.angle_steps
        elif angle_action == 1:  # Rotate right
            self.angle_idx = (self.angle_idx + int(self.rotation_speed)) % self.angle_steps
        elif angle_action == 2:  # Move forward in current direction
            self.move()
        # angle_action == 3 is no-op
        
    def sense_nodes(self, all_nodes: List[Node_Position]) -> List[Dict[str, Any]]:
        """Sense other nodes within sensor range, excluding self."""
        detections = []
        for other_node in all_nodes:
            if other_node.id == self.id:
                continue

            delta = other_node.pos - self.pos
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
        """Sense obstacles within sensor range."""
        detections = []
        for obstacle in obstacle_states:
            delta = obstacle['pos'] - self.pos
            toroidal_delta = np.mod(delta + 0.5, 1.0) - 0.5
            distance = np.linalg.norm(toroidal_delta)

            if distance < self.sensor_range:
                relative_velocity = obstacle['velocity'] - self.velocity()
                bearing_rad = np.arctan2(toroidal_delta[1], toroidal_delta[0])

                detections.append({
                    'id': -1,
                    'pos': obstacle['pos'].copy(),
                    'distance': float(distance),
                    'bearing_rad': float(bearing_rad),
                    'velocity': relative_velocity,
                    'type': obstacle['type']
                })
        return detections