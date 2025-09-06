"""
NodePosition.py

Represents the physical state of a single node in the FLOWRRA environment.
This class holds all kinematic properties, such as position, velocity, and orientation,
and provides methods to update them based on actions.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field

@dataclass
class Node_Position:
    """
    Data class for a node's physical state.

    Attributes:
        id (int): A unique identifier for the node.
        pos (np.ndarray): The (x, y) coordinates of the node in a normalized [0,1) space.
        angle_idx (int): The discrete orientation of the node, from 0 to angle_steps-1.
        rotation_speed (float): Maximum number of angle indices the node can turn per step.
        move_speed (np.ndarray): Maximum per-step displacement vector.
        angle_steps (int): The number of discrete steps in a full 360-degree rotation.
        target_angle_idx (int): The target orientation the node is currently rotating towards.
        last_pos (np.ndarray): The position of the node in the previous timestep, used for velocity calculation.
    """
    id: int
    pos: np.ndarray = field(default_factory=lambda: np.random.rand(2))
    angle_idx: int = 0
    rotation_speed: float = 2.0  # Default rotation speed in angle indices per step
    move_speed: np.ndarray = field(default_factory=lambda: np.array([0.01, 0.01]))
    angle_steps: int = 360
    target_angle_idx: int | None = None
    last_pos: np.ndarray | None = None

    def __post_init__(self):
        """Initializes last_pos to the current position if not provided."""
        if self.last_pos is None:
            self.last_pos = self.pos.copy()

    def step_rotate_towards_target(self, dt: float = 1.0):
        """
        Rotates the node towards its target angle, respecting rotation speed limits.
        Calculates the shortest circular path to the target.
        """
        if self.target_angle_idx is None:
            return

        # Calculate the shortest difference on a circular domain (e.g., a clock face)
        a = self.angle_idx
        b = self.target_angle_idx
        diff = (b - a + self.angle_steps // 2) % self.angle_steps - self.angle_steps // 2

        # Clamp the rotation to the maximum speed
        move = np.clip(diff, -self.rotation_speed * dt, self.rotation_speed * dt)
        self.angle_idx = int((self.angle_idx + move) % self.angle_steps)

    def move(self, action_vector: np.ndarray, dt: float = 1.0, bounds: str = 'toroidal'):
        """
        Moves the node based on an action vector, respecting speed limits and world boundaries.

        Args:
            action_vector (np.ndarray): The desired movement vector.
            dt (float): Timestep delta.
            bounds (str): The boundary condition ('toroidal' for wrap-around, or None).
        """
        # Clamp the movement to the maximum speed
        displacement = np.clip(action_vector, -self.move_speed * dt, self.move_speed * dt)

        # Update position history
        self.last_pos = self.pos.copy()
        self.pos = self.pos + displacement

        # Handle world boundaries
        if bounds == 'toroidal':
            self.pos = np.mod(self.pos, 1.0)
        elif bounds == 'clamp':
            self.pos = np.clip(self.pos, 0.0, 1.0)

    def velocity(self, dt: float = 1.0) -> np.ndarray:
        """Calculates the node's current velocity based on its position change."""
        return (self.pos - self.last_pos) / dt
