import math
import numpy as np
import random

class EnvironmentA:
    def __init__(self, grid_size=30, num_nodes=12, angle_steps=36, rotation_speed=2, move_speed=1, initial_loop_data=None):
        self.grid_size = grid_size
        self.num_nodes = num_nodes # This will be updated if initial_loop_data is provided
        self.angle_steps = angle_steps
        self.rotation_speed = rotation_speed
        self.move_speed = move_speed

        if initial_loop_data is not None:
            # If initial_loop_data is provided, use it to set up nodes
            # initial_loop_data is expected to be a list of [x, y, angle] points for each node
            # The last point in initial_loop_data from WFC is the closing point for plotting, so exclude it for node count
            self.num_nodes = len(initial_loop_data) - 1
            
            # Extract just positions (x,y) for _generate_loop to re-order them
            # _generate_loop sorts points by proximity, discarding the original order/angles
            positions_from_wfc = [tuple(p[:2]) for p in initial_loop_data[:-1]] # Exclude the closing point
            # _generate_loop re-orders these positions based on nearest neighbor to form a new loop sequence
            # The output 'loop_positions' will be list of (x,y) tuples, with the first point repeated at the end.
            loop_positions = self._generate_loop(positions_from_wfc)

            self.nodes = []
            for i in range(self.num_nodes): # Iterate through actual nodes
                current_pos = loop_positions[i]
                next_pos_in_loop = loop_positions[i + 1] # Next point in the re-ordered loop (handles wrapping for last node via _generate_loop)

                # Find the original angle for 'current_pos' from the WFC output (initial_loop_data)
                # This requires finding which original WFC point corresponds to current_pos
                original_angle = None
                for original_wfc_point in initial_loop_data[:-1]: # Search in original WFC points
                    if int(original_wfc_point[0]) == current_pos[0] and int(original_wfc_point[1]) == current_pos[1]:
                        original_angle = int(original_wfc_point[2])
                        break
                if original_angle is None: # Fallback if for some reason exact match isn't found
                    original_angle = random.randint(0, self.angle_steps - 1)

                target_angle = self._calculate_angle_idx(current_pos, next_pos_in_loop)

                self.nodes.append({
                    'id': i,
                    'pos': current_pos,
                    'eye_angle_idx': original_angle, # Use the sampled angle from WFC output
                    'target_angle_idx': target_angle # Set initial target correctly
                })
        else:
            # Fallback to original random initialization if no initial_loop_data
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

    def _initialize_nodes(self): # Original method for random initialization
        positions = self._get_valid_positions()
        loop_positions = self._generate_loop(positions) # This returns just (x,y) positions, closing point included
        nodes = []
        for i, pos in enumerate(loop_positions[:-1]): # Exclude the closing point as it's a duplicate of the first
            nodes.append({
                'id': i,
                'pos': pos,
                'eye_angle_idx': random.randint(0, self.angle_steps - 1),
                'target_angle_idx': self._calculate_angle_idx(pos, loop_positions[(i + 1)]) # Next node in this list
            })
        return nodes

    def _generate_loop(self, positions): # This method re-orders given positions
        if not positions:
            return []
        final_list = [positions[0]]
        current = positions[0]
        remaining = list(positions[1:]) # Make a mutable copy
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
            # Ensure next_node is correctly determined from the current sequence of nodes, which is self.nodes
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