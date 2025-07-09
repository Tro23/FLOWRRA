import math
import numpy as np
import random

class EnvironmentA:
    def __init__(self, grid_size=30, num_nodes=12, angle_steps=36, rotation_speed=2, move_speed=1):
        self.grid_size = grid_size
        self.num_nodes = num_nodes
        self.angle_steps = angle_steps
        self.rotation_speed = rotation_speed
        self.move_speed = move_speed
        self.nodes = self._initialize_nodes()

    def _get_valid_positions(self):
        positions = set()
        while len(positions) < self.num_nodes:
            x, y = random.randint(1, self.grid_size - 2), random.randint(1, self.grid_size - 2)
            positions.add((x, y))
        return list(positions)

    def _calculate_angle_idx(self, from_pos, to_pos):
        dx, dy = to_pos[0] - from_pos[0], to_pos[1] - from_pos[1]
        angle_rad = math.atan2(dy, dx)
        angle_deg = math.degrees(angle_rad) % 360
        return int(round(angle_deg / (360 / self.angle_steps))) % self.angle_steps

    def _eye_direction_vector(self, angle_idx):
        rad = np.deg2rad(angle_idx * (360 / self.angle_steps))
        return np.cos(rad), np.sin(rad)

    def _initialize_nodes(self):
        positions = self._get_valid_positions()
        loop = self._generate_loop(positions)
        nodes = []
        for i, pos in enumerate(loop):
            nodes.append({
                'id': i,
                'pos': pos,
                'eye_angle_idx': random.randint(0, self.angle_steps - 1),
                'target_angle_idx': 0
            })
        return nodes

    def _generate_loop(self, positions):
        if not positions:
            return []
        final_list = [positions[0]]
        current = positions[0]
        remaining = positions[1:]
        while remaining:
            distances = [(math.hypot(x - current[0], y - current[1]), (x, y)) for x, y in remaining]
            _, next_pos = min(distances)
            final_list.append(next_pos)
            remaining.remove(next_pos)
            current = next_pos
        final_list.append(final_list[0])  # close the loop
        return final_list

    def step(self):
        for i in range(self.num_nodes):
            node = self.nodes[i]
            next_node = self.nodes[(i + 1) % self.num_nodes]

            # Move node randomly if possible
            moved = False
            for dx, dy in random.sample([(0, 1), (0, -1), (1, 0), (-1, 0)], 4):
                new_x, new_y = node['pos'][0] + dx, node['pos'][1] + dy
                if 0 <= new_x < self.grid_size and 0 <= new_y < self.grid_size:
                    if all((new_x, new_y) != n['pos'] for j, n in enumerate(self.nodes) if j != i):
                        node['pos'] = (new_x, new_y)
                        moved = True
                        break

            # Update target and rotate eye
            node['target_angle_idx'] = self._calculate_angle_idx(node['pos'], next_node['pos'])
            diff = (node['target_angle_idx'] - node['eye_angle_idx']) % self.angle_steps
            if diff > self.angle_steps // 2:
                diff -= self.angle_steps
            if diff > 0:
                node['eye_angle_idx'] = (node['eye_angle_idx'] + self.rotation_speed) % self.angle_steps
            elif diff < 0:
                node['eye_angle_idx'] = (node['eye_angle_idx'] - self.rotation_speed) % self.angle_steps

        return [(node['pos'], node['eye_angle_idx']) for node in self.nodes]
