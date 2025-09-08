"""
EnvironmentB_RL.py

Defines the external world in which the FLOWRRA agent operates.
This environment contains both static (fixed) and dynamic (moving) obstacles.
It is responsible for updating the state of these obstacles each timestep.
"""
import numpy as np
import random
from typing import List, Dict, Any, Tuple

class EnvironmentB:
    """
    The external environment with fixed and moving obstacles.

    Attributes:
        grid_size (int): The resolution of the grid on which obstacles exist.
        fixed_blocks (set): A set of (x,y) tuples for static obstacles.
        moving_blocks (list): A list of (x,y) tuples for dynamic obstacles.
        last_moving_blocks (list): The previous positions of moving obstacles, for velocity calculation.
    """
    def __init__(self, grid_size: int = 60, num_fixed: int = 10, num_moving: int = 4, seed: int | None = None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            
        self.grid_size = grid_size
        self.num_fixed = num_fixed
        self.num_moving = num_moving
        self.seed = seed
        self.reset()
        
    def reset(self):
        """
        Resets the environment, generating new positions for fixed and moving
        obstacles.
        """
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)
            
        self.fixed_blocks = self._generate_fixed_obstacles()
        self.moving_blocks = self._initialize_moving_blocks()
        self.last_moving_blocks = list(self.moving_blocks)
        self.all_blocks = self.fixed_blocks.union(set(self.moving_blocks))
        
    def _generate_fixed_obstacles(self) -> set:
        """Generates a random set of fixed obstacle positions on the grid."""
        fixed = set()
        while len(fixed) < self.num_fixed:
            x = random.randrange(0, self.grid_size)
            y = random.randrange(0, self.grid_size)
            fixed.add((x, y))
        return fixed
    
    def _initialize_moving_blocks(self) -> list:
        """Generates a random list of moving obstacle positions."""
        moving = []
        while len(moving) < self.num_moving:
            x = random.randrange(0, self.grid_size)
            y = random.randrange(0, self.grid_size)
            if (x, y) not in self.fixed_blocks:
                moving.append((x, y))
        return moving

    def step(self):
        """
        Advances the state of the moving obstacles by one step.
        Each moving obstacle attempts to move one step in a random direction.
        """
        self.last_moving_blocks = list(self.moving_blocks)
        updated_positions = []
        occupied_for_step = set(self.fixed_blocks)

        for x, y in self.moving_blocks:
            moved = False
            
            # Prioritize moving away from other blocks
            possible_moves = [(x+dx, y+dy) for dx in [-1, 0, 1] for dy in [-1, 0, 1] if dx != 0 or dy != 0]
            random.shuffle(possible_moves)

            for next_x, next_y in possible_moves:
                if 0 <= next_x < self.grid_size and 0 <= next_y < self.grid_size and (next_x, next_y) not in occupied_for_step:
                    updated_positions.append((next_x, next_y))
                    occupied_for_step.add((next_x, next_y))
                    moved = True
                    break
                
                # If move was not valid, add its old position back
                occupied_for_step.add((x,y))

            if not moved:
                updated_positions.append((x, y)) # If no valid move, stay put

        self.moving_blocks = updated_positions
        self.all_blocks = self.fixed_blocks.union(set(self.moving_blocks))

    def get_obstacle_states(self, dt: float = 1.0) -> list[dict[str, Any]]:
        """
        Provides the state of all obstacles in continuous [0,1) coordinates.

        Returns:
            A list of dictionaries, each containing an obstacle's continuous
            position, velocity, and type.
        """
        states = []
        # Fixed obstacles (zero velocity)
        for fx, fy in self.fixed_blocks:
            pos = np.array([(fx + 0.5) / self.grid_size, (fy + 0.5) / self.grid_size])
            states.append({'pos': pos, 'velocity': np.zeros(2), 'type': 'fixed'})

        # Moving obstacles
        for i, (mx, my) in enumerate(self.moving_blocks):
            pos = np.array([(mx + 0.5) / self.grid_size, (my + 0.5) / self.grid_size])
            last_mx, last_my = self.last_moving_blocks[i]
            last_pos = np.array([(last_mx + 0.5) / self.grid_size, (last_my + 0.5) / self.grid_size])
            velocity = (pos - last_pos) / dt
            states.append({'pos': pos, 'velocity': velocity, 'type': 'moving'})
        return states
