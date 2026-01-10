"""
Environment_B.py

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
        self.fixed_blocks = self._generate_fixed_obstacles()
        self.moving_blocks = self._initialize_moving_blocks()
        self.last_moving_blocks = list(self.moving_blocks) # Store initial state for velocity calc
        self.all_blocks = self.fixed_blocks.union(set(self.moving_blocks))

    def _generate_fixed_obstacles(self) -> set[tuple[int, int]]:
        """Randomly places fixed obstacles on the grid."""
        blocks = set()
        while len(blocks) < self.num_fixed:
            x, y = random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1)
            blocks.add((x, y))
        return blocks

    def _initialize_moving_blocks(self) -> list[tuple[int, int]]:
        """Randomly places moving obstacles, avoiding fixed obstacles."""
        positions = []
        occupied = set(self.fixed_blocks)
        while len(positions) < self.num_moving:
            x, y = random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1)
            if (x, y) not in occupied:
                positions.append((x, y))
                occupied.add((x, y))
        return positions

    def step(self):
        """Moves the moving blocks randomly by one grid cell."""
        self.last_moving_blocks = list(self.moving_blocks) # Store current pos before moving

        updated_positions = []
        # Create a temporary set of occupied positions for this step's collision checks
        occupied_for_step = self.fixed_blocks.union(set(self.moving_blocks))

        for i, (x, y) in enumerate(self.moving_blocks):
            possible_moves = [(0, 1), (0, -1), (1, 0), (-1, 0), (0, 0)] # Allow staying still
            random.shuffle(possible_moves)
            moved = False
            for dx, dy in possible_moves:
                nx, ny = x + dx, y + dy
                new_pos = (nx, ny)

                # Temporarily remove the current block's old position to allow other blocks to move into it
                occupied_for_step.remove((x,y))
                
                # Check bounds and if the new position is not occupied by another block
                if (0 <= nx < self.grid_size and 0 <= ny < self.grid_size and
                        new_pos not in occupied_for_step):
                    updated_positions.append(new_pos)
                    occupied_for_step.add(new_pos) # Add its new position
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
