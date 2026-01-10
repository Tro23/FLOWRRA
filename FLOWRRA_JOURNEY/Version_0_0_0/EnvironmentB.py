import numpy as np
import random

class EnvironmentB:
    def __init__(self, grid_size=30, num_fixed=15, num_moving=5):
        self.grid_size = grid_size
        self.num_fixed = num_fixed
        self.num_moving = num_moving
        self.grid = np.zeros((grid_size, grid_size))
        self.fixed_blocks = self._generate_fixed_obstacles()
        self.moving_blocks = self._initialize_moving_blocks()

    def _generate_fixed_obstacles(self):
        blocks = set()
        while len(blocks) < self.num_fixed:
            x, y = random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1)
            if self.grid[x][y] == 0:
                blocks.add((x, y))
                self.grid[x][y] = -1
        return blocks

    def _initialize_moving_blocks(self):
        positions = []
        occupied = set(self.fixed_blocks)
        while len(positions) < self.num_moving:
            x, y = random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1)
            if (x, y) not in occupied:
                positions.append((x, y))
                occupied.add((x, y))
        return positions

    def step(self):
        updated_positions = []
        for i, (x, y) in enumerate(self.moving_blocks):
            possible_moves = [(0, 1), (0, -1), (1, 0), (-1, 0)]
            random.shuffle(possible_moves)
            moved = False
            for dx, dy in possible_moves:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                    if (nx, ny) not in self.fixed_blocks and (nx, ny) not in self.moving_blocks:
                        self.moving_blocks[i] = (nx, ny)
                        moved = True
                        break
            updated_positions.append(self.moving_blocks[i])
        return updated_positions
