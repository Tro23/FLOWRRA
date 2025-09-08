"""
NodePosition_RL.py

Represents the physical state of a single node in the FLOWRRA environment.
This class holds all kinematic properties, such as position, velocity, and orientation,
and provides methods to update them based on actions. It now also handles its
own sensing capabilities.
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
        sensor_range (float): The maximum distance the sensor can detect.
    """
    id: int
    pos: np.ndarray = field(default_factory=lambda: np.random.rand(2))
    angle_idx: int = 0
    rotation_speed: float = 3.0
    move_speed: float = 0.05
    angle_steps: int = 360
    target_angle_idx: int = 0
    last_pos: np.ndarray = field(default_factory=lambda: np.random.rand(2))
    sensor_range: float = 0.2
    
    def step_rotate_and_move(self, dt: float = 1.0):
        """
        Rotates towards the target angle and moves forward.
        """
        # Calculate the shortest angle difference
        diff = (self.target_angle_idx - self.angle_idx + self.angle_steps / 2) % self.angle_steps - self.angle_steps / 2
        
        # Apply rotation speed limit
        turn = np.clip(diff, -self.rotation_speed, self.rotation_speed)
        
        # Update angle
        self.angle_idx = int((self.angle_idx + turn + self.angle_steps) % self.angle_steps)
        
        # Calculate velocity vector from new angle
        angle_rad = (self.angle_idx / self.angle_steps) * 2 * np.pi
        velocity_vector = np.array([np.cos(angle_rad), np.sin(angle_rad)]) * self.move_speed
        
        # Store last position for velocity calculation
        self.last_pos = self.pos.copy()
        
        # Update position
        self.pos = self.pos + velocity_vector * dt
        
        # Clamp position to a normalized [0,1) space
        self.pos = np.mod(self.pos, 1.0)
        
    def velocity(self, dt: float = 1.0) -> np.ndarray:
        """Calculates the node's current velocity based on its position change."""
        # Handle toroidal space for accurate velocity
        vel = self.pos - self.last_pos
        # Check for wrap-around movement
        if np.abs(vel[0]) > 0.5:
            vel[0] -= np.sign(vel[0]) * 1.0
        if np.abs(vel[1]) > 0.5:
            vel[1] -= np.sign(vel[1]) * 1.0
        return vel / dt

    def sense_other_nodes(self, all_nodes: List[Node_Position]) -> List[Dict[str, Any]]:
        """
        Senses other nodes within the sensor range.
        """
        detections = []
        for other_node in all_nodes:
            if other_node.id == self.id:
                continue
            
            # Calculate toroidal distance
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
