"""
node.py

N-dimensional node representation supporting both 2D and 3D modes.
Handles position, orientation, movement, and sensing in arbitrary dimensions.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class NodePositionND:
    """
    N-dimensional node with position, orientation, and sensing capabilities.

    For 2D: Uses single angle (azimuth)
    For 3D: Uses spherical coordinates (azimuth, elevation)
    """

    id: int
    pos: np.ndarray  # N-dimensional position [0,1]^N
    dimensions: int  # 2 or 3

    last_pos: Optional[np.ndarray] = field(default=None)

    # Orientation (discretized angles)
    azimuth_idx: int = 0
    elevation_idx: int = 0
    azimuth_steps: int = 24
    elevation_steps: int = 12

    # Movement parameters
    rotation_speed: float = 2.0
    move_speed: float = 0.015
    sensor_range: float = 0.15

    # State tracking
    target_azimuth_idx: int = 0
    target_elevation_idx: int = 0

    # World bounds for toroidal wrapping
    world_bounds: Tuple[float, ...] = field(default_factory=lambda: (1.0, 1.0, 1.0))

    def __post_init__(self):
        """Initialize derived properties."""
        # CHANGE 2: Initialize last_pos based on self.dimensions
        if self.last_pos is None:
            self.last_pos = self.pos.copy()  # Use current position as initial last_pos

        # Ensure pos and last_pos have the correct shape
        if self.pos.shape != (self.dimensions,):
            raise ValueError(
                f"Initial position shape {self.pos.shape} does not match dimensions ({self.dimensions},)"
            )

        if self.last_pos.shape != (self.dimensions,):
            raise ValueError(
                f"last_pos shape {self.last_pos.shape} does not match dimensions ({self.dimensions},)"
            )

        # Validate dimensions
        assert self.dimensions in [2, 3], "Only 2D and 3D supported"
        assert len(self.pos) == self.dimensions, (
            f"Position must have {self.dimensions} elements"
        )

        # For 2D, elevation is fixed
        if self.dimensions == 2:
            self.elevation_idx = 0
            self.elevation_steps = 1
            self.world_bounds = self.world_bounds[:2]

        # NEW: Add for BenchMARL integration
        self.last_action = 0  # Stores most recent action index

    def velocity(self) -> np.ndarray:
        """Calculates velocity with toroidal wrapping."""
        delta = self.pos - self.last_pos
        # Toroidal distance (shortest path through wrap-around)
        toroidal_delta = np.mod(delta + 0.5, 1.0) - 0.5
        return toroidal_delta

    def get_direction_vector(self) -> np.ndarray:
        """
        Returns the unit direction vector based on current orientation.

        2D: [cos(θ), sin(θ)]
        3D: [cos(az)cos(el), sin(az)cos(el), sin(el)]
        """
        azimuth_rad = (self.azimuth_idx / self.azimuth_steps) * 2 * np.pi

        if self.dimensions == 2:
            return np.array([np.cos(azimuth_rad), np.sin(azimuth_rad)])

        else:  # 3D
            # Elevation: 0 = horizontal, ±π/2 = up/down
            elevation_rad = ((self.elevation_idx / self.elevation_steps) - 0.5) * np.pi

            return np.array(
                [
                    np.cos(azimuth_rad) * np.cos(elevation_rad),
                    np.sin(azimuth_rad) * np.cos(elevation_rad),
                    np.sin(elevation_rad),
                ]
            )

    def get_spherical_coords(self) -> Tuple[float, float]:
        """Returns (azimuth, elevation) in radians."""
        azimuth = (self.azimuth_idx / self.azimuth_steps) * 2 * np.pi
        elevation = ((self.elevation_idx / self.elevation_steps) - 0.5) * np.pi
        return azimuth, elevation

    def move(self, dt: float = 1.0):
        """Moves forward in current direction."""
        self.last_pos = self.pos.copy()
        direction = self.get_direction_vector()
        self.pos = np.mod(self.pos + direction * self.move_speed * dt, 1.0)

    def apply_directional_action(self, action: int, dt: float = 1.0):
        """
        Applies discrete directional movement.

        2D actions: 0=left(-x), 1=right(+x), 2=up(+y), 3=down(-y)
        3D actions: 0-3 same as 2D, 4=up(+z), 5=down(-z)
        """
        self.last_pos = self.pos.copy()

        deltas_2d = [
            np.array([-1, 0]),  # Left
            np.array([1, 0]),  # Right
            np.array([0, 1]),  # Up
            np.array([0, -1]),  # Down
        ]

        deltas_3d = deltas_2d + [
            np.array([0, 0, 1]),  # Up (Z)
            np.array([0, 0, -1]),  # Down (Z)
        ]

        deltas = deltas_2d if self.dimensions == 2 else deltas_3d

        if 0 <= action < len(deltas):
            # Pad 2D deltas with 0 for 3D compatibility
            if self.dimensions == 3 and action < 4:
                delta = np.append(deltas[action], 0)
            else:
                delta = deltas[action]

            self.pos = np.mod(self.pos + delta * self.move_speed * dt, 1.0)

    def apply_rotation_action(self, action: int):
        """
        Applies discrete rotation.

        Actions:
        0 = rotate azimuth left
        1 = rotate azimuth right
        2 = rotate elevation up (3D only)
        3 = rotate elevation down (3D only)
        4+ = no-op
        """
        if action == 0:  # Azimuth left
            self.azimuth_idx = (
                self.azimuth_idx - int(self.rotation_speed)
            ) % self.azimuth_steps
        elif action == 1:  # Azimuth right
            self.azimuth_idx = (
                self.azimuth_idx + int(self.rotation_speed)
            ) % self.azimuth_steps
        elif action == 2 and self.dimensions == 3:  # Elevation up
            self.elevation_idx = min(self.elevation_idx + 1, self.elevation_steps - 1)
        elif action == 3 and self.dimensions == 3:  # Elevation down
            self.elevation_idx = max(self.elevation_idx - 1, 0)

    def rotate_towards_target(self):
        """Smoothly rotates towards target orientation."""
        # Azimuth rotation
        if self.azimuth_idx != self.target_azimuth_idx:
            diff = (
                self.target_azimuth_idx - self.azimuth_idx + self.azimuth_steps
            ) % self.azimuth_steps

            if diff <= self.azimuth_steps / 2:
                step = min(diff, self.rotation_speed)
                self.azimuth_idx = (self.azimuth_idx + int(step)) % self.azimuth_steps
            else:
                step = min(self.azimuth_steps - diff, self.rotation_speed)
                self.azimuth_idx = (self.azimuth_idx - int(step)) % self.azimuth_steps

        # Elevation rotation (3D only)
        if self.dimensions == 3 and self.elevation_idx != self.target_elevation_idx:
            if self.elevation_idx < self.target_elevation_idx:
                self.elevation_idx = min(
                    self.elevation_idx + 1, self.target_elevation_idx
                )
            else:
                self.elevation_idx = max(
                    self.elevation_idx - 1, self.target_elevation_idx
                )

    def sense_nodes(self, all_nodes: List["NodePositionND"]) -> List[Dict[str, Any]]:
        """
        Detects other nodes within sensor range.

        Returns list of detections with:
        - id, pos, distance, bearing (angles), velocity, type
        """
        detections = []

        for other in all_nodes:
            if other.id == self.id:
                continue

            # Toroidal distance
            delta = other.pos - self.pos
            toroidal_delta = np.mod(delta + 0.5, 1.0) - 0.5
            distance = np.linalg.norm(toroidal_delta)

            if distance < self.sensor_range:
                relative_velocity = other.velocity() - self.velocity()

                # Calculate bearing angles
                if self.dimensions == 2:
                    bearing_azimuth = np.arctan2(toroidal_delta[1], toroidal_delta[0])
                    bearing_elevation = 0.0
                else:  # 3D
                    bearing_azimuth = np.arctan2(toroidal_delta[1], toroidal_delta[0])
                    bearing_elevation = np.arctan2(
                        toroidal_delta[2],
                        np.sqrt(toroidal_delta[0] ** 2 + toroidal_delta[1] ** 2),
                    )

                detections.append(
                    {
                        "id": other.id,
                        "pos": other.pos.copy(),
                        "distance": float(distance),
                        "bearing_azimuth": float(bearing_azimuth),
                        "bearing_elevation": float(bearing_elevation),
                        "velocity": relative_velocity,
                        "type": "node",
                    }
                )

        return detections

    def sense_obstacles(
        self, obstacle_states: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Detects obstacles within sensor range.

        obstacle_states format:
        [{'pos': np.array, 'velocity': np.array, 'type': 'fixed'/'moving'}, ...]
        """
        detections = []

        for obs in obstacle_states:
            delta = obs["pos"] - self.pos
            toroidal_delta = np.mod(delta + 0.5, 1.0) - 0.5
            distance = np.linalg.norm(toroidal_delta)

            if distance < self.sensor_range:
                relative_velocity = obs["velocity"] - self.velocity()

                # Calculate bearing
                if self.dimensions == 2:
                    bearing_azimuth = np.arctan2(toroidal_delta[1], toroidal_delta[0])
                    bearing_elevation = 0.0
                else:
                    bearing_azimuth = np.arctan2(toroidal_delta[1], toroidal_delta[0])
                    bearing_elevation = np.arctan2(
                        toroidal_delta[2],
                        np.sqrt(toroidal_delta[0] ** 2 + toroidal_delta[1] ** 2),
                    )

                detections.append(
                    {
                        "id": -1,
                        "pos": obs["pos"].copy(),
                        "distance": float(distance),
                        "bearing_azimuth": float(bearing_azimuth),
                        "bearing_elevation": float(bearing_elevation),
                        "velocity": relative_velocity,
                        "type": obs["type"],
                    }
                )

        return detections

    def get_state_vector(
        self,
        local_repulsion_grid: np.ndarray,
        node_detections: List[Dict],
        obstacle_detections: List[Dict],
        max_neighbors: int = 5,
        max_obstacles: int = 5,
    ) -> np.ndarray:
        """
        Constructs the state vector for this node for GNN input.

        Returns:
        - pos (N)
        - velocity (N)
        - orientation (2 for 3D: azimuth, elevation; 1 for 2D)
        - node detections (max_neighbors * 5): [distance, bearing_az, bearing_el, vel_x, vel_y, (vel_z)]
        - obstacle detections (max_obstacles * 5)
        - local repulsion grid (flattened)
        """
        # Sort and pad detections
        node_det_sorted = sorted(node_detections, key=lambda x: x["distance"])[
            :max_neighbors
        ]
        obs_det_sorted = sorted(obstacle_detections, key=lambda x: x["distance"])[
            :max_obstacles
        ]

        # Node detection features
        node_features = []
        for i in range(max_neighbors):
            if i < len(node_det_sorted):
                d = node_det_sorted[i]
                vel = list(d["velocity"])
                node_features.extend(
                    [d["distance"], d["bearing_azimuth"], d["bearing_elevation"]] + vel
                )
            else:
                # Padding with sentinel values
                node_features.extend([1.0, 0.0, 0.0] + [0.0] * self.dimensions)

        # Obstacle detection features
        obs_features = []
        for i in range(max_obstacles):
            if i < len(obs_det_sorted):
                d = obs_det_sorted[i]
                vel = list(d["velocity"])
                obs_features.extend(
                    [d["distance"], d["bearing_azimuth"], d["bearing_elevation"]] + vel
                )
            else:
                obs_features.extend([1.0, 0.0, 0.0] + [0.0] * self.dimensions)

        # Orientation
        if self.dimensions == 2:
            orientation = [self.get_spherical_coords()[0]]  # Just azimuth
        else:
            orientation = list(self.get_spherical_coords())  # Both angles

        # Combine all features
        state = np.concatenate(
            [
                self.pos,
                self.velocity(),
                orientation,
                node_features,
                obs_features,
                local_repulsion_grid.flatten(),
            ]
        )

        return state.astype(np.float32)
