"""
obstacles.py

Manages static and dynamic obstacles in the FLOWRRA environment.
Optimized for high-speed parallel execution on AWS.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
from collections import defaultdict
import numpy as np


@dataclass
class Obstacle:
    """Represents an obstacle in the environment."""

    id: int
    pos: np.ndarray  # N-dimensional position
    radius: float
    velocity: np.ndarray  # For moving obstacles
    is_static: bool = True
    dimensions: int = 2

    def update(self, dt: float = 1.0):
        """Update obstacle position if moving."""
        if not self.is_static:
            self.pos = np.mod(self.pos + self.velocity * dt, 1.0)

    def get_state_dict(self) -> Dict[str, Any]:
        """Returns state dict for sensing."""
        return {
            "id": self.id,
            "pos": self.pos.copy(),
            "velocity": self.velocity.copy()
            if not self.is_static
            else np.zeros(self.dimensions),
            "type": "static" if self.is_static else "moving",
            "radius": self.radius,
        }

    def check_collision(self, point: np.ndarray, safety_margin: float = 0.0) -> bool:
        """Check if a point collides with this obstacle."""
        # Toroidal distance
        delta = point - self.pos
        toroidal_delta = np.mod(delta + 0.5, 1.0) - 0.5
        distance = np.linalg.norm(toroidal_delta)
        return distance < (self.radius + safety_margin)

    def check_line_intersection(self, p1: np.ndarray, p2: np.ndarray) -> bool:
        """
        Check if a line segment between two points intersects this obstacle.
        Used for detecting loop breaks.
        """
        # Toroidal distance to both points
        delta1 = p1 - self.pos
        toroidal_delta1 = np.mod(delta1 + 0.5, 1.0) - 0.5
        dist1 = np.linalg.norm(toroidal_delta1)

        delta2 = p2 - self.pos
        toroidal_delta2 = np.mod(delta2 + 0.5, 1.0) - 0.5
        dist2 = np.linalg.norm(toroidal_delta2)

        if dist1 > self.radius and dist2 > self.radius:
            segment = p2 - p1
            segment_toroidal = np.mod(segment + 0.5, 1.0) - 0.5
            to_obstacle = self.pos - p1
            to_obstacle_toroidal = np.mod(to_obstacle + 0.5, 1.0) - 0.5

            segment_length_sq = np.dot(segment_toroidal, segment_toroidal)
            if segment_length_sq < 1e-8:
                return dist1 < self.radius

            t = np.clip(
                np.dot(to_obstacle_toroidal, segment_toroidal) / segment_length_sq, 0, 1
            )
            closest_point = p1 + t * segment_toroidal
            delta_closest = closest_point - self.pos
            toroidal_delta_closest = np.mod(delta_closest + 0.5, 1.0) - 0.5
            dist_closest = np.linalg.norm(toroidal_delta_closest)

            return dist_closest < self.radius

        return True


class ObstacleManager:
    """Manages all obstacles in the environment."""

    def __init__(self, dimensions: int = 2):
        self.dimensions = dimensions
        self.obstacles: List[Obstacle] = []
        self.next_id = 0
        
        # Internal Add-on: High-speed grid lookup
        self.grid_hash = defaultdict(list)
        self.grid_res = 10 

    def _sync_grid_hash(self):
        """Internal helper to keep the hash map in sync with self.obstacles."""
        self.grid_hash.clear()
        for obs in self.obstacles:
            gx = int(np.clip(obs.pos[0] * self.grid_res, 0, self.grid_res - 1))
            gy = int(np.clip(obs.pos[1] * self.grid_res, 0, self.grid_res - 1))
            self.grid_hash[(gx, gy)].append(obs)

    def add_static_obstacle(self, pos: np.ndarray, radius: float) -> Obstacle:
        """Add a static obstacle."""
        obs = Obstacle(
            id=self.next_id,
            pos=pos.copy(),
            radius=radius,
            velocity=np.zeros(self.dimensions),
            is_static=True,
            dimensions=self.dimensions,
        )
        self.obstacles.append(obs)
        self.next_id += 1
        self._sync_grid_hash()
        return obs

    def add_moving_obstacle(
        self, pos: np.ndarray, radius: float, velocity: np.ndarray
    ) -> Obstacle:
        """Add a moving obstacle."""
        obs = Obstacle(
            id=self.next_id,
            pos=pos.copy(),
            radius=radius,
            velocity=velocity.copy(),
            is_static=False,
            dimensions=self.dimensions,
        )
        self.obstacles.append(obs)
        self.next_id += 1
        self._sync_grid_hash()
        return obs

    def update_all(self, dt: float = 1.0):
        """Update all moving obstacles."""
        for obs in self.obstacles:
            obs.update(dt)
        self._sync_grid_hash()

    def get_all_states(self) -> List[Dict[str, Any]]:
        """Get state dicts for all obstacles."""
        return [obs.get_state_dict() for obs in self.obstacles]

    def check_collision(
        self, point: np.ndarray, safety_margin: float = 0.0
    ) -> Tuple[bool, List[int]]:
        """
        Check if a point collides with any obstacle.
        Returns (collision_detected, list_of_obstacle_ids)
        """
        # Speed boost: Use grid hashing to narrow the search
        gx = int(np.clip(point[0] * self.grid_res, 0, self.grid_res - 1))
        gy = int(np.clip(point[1] * self.grid_res, 0, self.grid_res - 1))
        
        colliding_ids = []
        # We only check the subset of obstacles in the local grid cell
        for obs in self.grid_hash.get((gx, gy), []):
            if obs.check_collision(point, safety_margin):
                colliding_ids.append(obs.id)
        return len(colliding_ids) > 0, colliding_ids

    def check_line_intersection(
        self, p1: np.ndarray, p2: np.ndarray
    ) -> Tuple[bool, List[int]]:
        """
        Check if a line segment intersects any obstacle.
        Returns (intersection_detected, list_of_obstacle_ids)
        """
        intersecting_ids = []
        for obs in self.obstacles:
            if obs.check_line_intersection(p1, p2):
                intersecting_ids.append(obs.id)
        return len(intersecting_ids) > 0, intersecting_ids

    def get_obstacle_by_id(self, obs_id: int) -> Obstacle:
        """Get obstacle by ID."""
        for obs in self.obstacles:
            if obs.id == obs_id:
                return obs
        return None

    def remove_obstacle(self, obs_id: int):
        """Remove an obstacle by ID."""
        self.obstacles = [obs for obs in self.obstacles if obs.id != obs_id]
        self._sync_grid_hash()

    def clear_all(self):
        """Remove all obstacles."""
        self.obstacles.clear()
        self.next_id = 0
        self.grid_hash.clear()