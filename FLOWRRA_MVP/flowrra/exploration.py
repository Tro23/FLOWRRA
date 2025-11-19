import numpy as np

class ExplorationMap:
    def __init__(self, bounds, resolution):
        self.bounds = bounds
        self.res = resolution
        self.shape = tuple(int(b / resolution) for b in bounds)
        
        # 0 = Unexplored, 1 = Explored
        self.grid = np.zeros(self.shape, dtype=np.float32)
        self.total_cells = np.prod(self.shape)
        
        # Pre-compute indices for speed
        x = np.linspace(0, bounds[0], self.shape[0])
        y = np.linspace(0, bounds[1], self.shape[1])
        self.xv, self.yv = np.meshgrid(x, y, indexing='ij')

    def update(self, nodes):
        """
        Updates the map based on node positions.
        Returns: Total NEW cells discovered in this step (for global reward).
        """
        initial_coverage = np.sum(self.grid)
        
        for node in nodes:
            # Vectorized distance check (simple circle stamp)
            # In production, use a KDTree or Bounding Box for speed
            dist_sq = (self.xv - node.pos[0])**2 + (self.yv - node.pos[1])**2
            mask = dist_sq <= (node.sensor_range**2)
            self.grid[mask] = 1.0
            
        final_coverage = np.sum(self.grid)
        return final_coverage - initial_coverage

    def get_frontier_vector(self, node_pos):
        """
        Returns a normalized vector pointing to the nearest Unexplored area.
        Used as input for the GNN.
        """
        # Find indices of all 0s (Unexplored)
        unexplored_indices = np.argwhere(self.grid == 0)
        
        if len(unexplored_indices) == 0:
            return np.zeros(len(self.bounds)) # Map done
            
        # Convert grid indices back to world coordinates
        unexplored_world = unexplored_indices * self.res
        
        # Find nearest (Simple L2 norm) - heavy compute, optimize for scale
        # Heuristic: Just sample 100 random unexplored points for speed
        if len(unexplored_world) > 100:
            indices = np.random.choice(len(unexplored_world), 100, replace=False)
            samples = unexplored_world[indices]
        else:
            samples = unexplored_world

        dists = np.linalg.norm(samples - node_pos, axis=1)
        nearest_idx = np.argmin(dists)
        target = samples[nearest_idx]
        
        # Create vector
        vec = target - node_pos
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else np.zeros_like(vec)

    def get_coverage_percentage(self):
        return (np.sum(self.grid) / self.total_cells) * 100