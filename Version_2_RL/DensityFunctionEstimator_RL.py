"""
DensityFunctionEstimator_RL.py

This is the core of the v2 system, implementing the speed-aware repulsive density field.
It translates the mathematical design for "comet-tail" repulsion into a practical,
grid-based implementation. It handles splatting, decay, and diffusion of the repulsion
field, which guides node movement.

This version is modified to generate a LOCAL 4x4 repulsion grid for each node,
instead of a single global grid.
"""
import numpy as np
from typing import List, Dict, Any

class Density_Function_Estimator:
    """
    Manages the repulsive density field based on sensor detections.

    This class implements the "comet-tail" repulsion by projecting repulsion kernels
    forward in time based on detected object velocities.

    Attributes:
        eta (float): Learning rate for splatting repulsion.
        gamma_f (float): Decay factor for forward-projected "comet-tail" kernels.
        k_f (int): Number of future steps to project repulsion for.
        sigma_f (float): Width (standard deviation) of the Gaussian kernel for splats.
        decay_lambda (float): Per-step decay rate for the entire repulsion field.
        blur_delta (float): Mix factor for diffusion (blurring) to maintain smoothness.
        blur_kernel (np.ndarray): The convolution kernel used for blurring.
    """
    def __init__(self,
                 grid_shape: tuple[int, int] = (60, 60),
                 eta: float = 0.5,
                 gamma_f: float = 0.9,
                 k_f: int = 5,
                 sigma_f: float = 0.05,
                 decay_lambda: float = 0.01,
                 blur_delta: float = 0.0):
        # We still need a global grid for visualization, but calculations are local now.
        self.grid_shape = grid_shape
        self.repulsion_field = np.zeros(grid_shape)
        self.eta = eta
        self.gamma_f = gamma_f
        self.k_f = k_f
        self.sigma_f = sigma_f
        self.decay_lambda = decay_lambda
        self.blur_delta = blur_delta
        
    def reset(self):
        self.repulsion_field = np.zeros(self.grid_shape)

    def get_repulsion_potential_for_node(self,
                                         node_pos: np.ndarray,
                                         all_node_positions: np.ndarray,
                                         all_obstacle_states: List[Dict[str, Any]]) -> np.ndarray:
        """
        Calculates and returns a LOCAL 4x4 repulsion potential grid for a single node.
        
        Args:
            node_pos (np.ndarray): The (x,y) position of the node.
            all_node_positions (np.ndarray): All node positions for neighbor repulsion.
            all_obstacle_states (List[Dict]): All obstacle states for obstacle repulsion.
            
        Returns:
            np.ndarray: A 4x4 grid of repulsion values.
        """
        local_grid = np.zeros((4, 4))
        
        # Combine all sensor-detectable sources of repulsion
        repulsion_sources = []
        
        # Add other nodes as repulsion sources
        for other_pos in all_node_positions:
            if np.linalg.norm(node_pos - other_pos) > 1e-6: # Avoid self
                repulsion_sources.append({'pos': other_pos, 'velocity': np.zeros(2), 'type': 'node'})
                
        # Add obstacles as repulsion sources
        for obstacle in all_obstacle_states:
            repulsion_sources.append({'pos': obstacle['pos'], 'velocity': obstacle['velocity'], 'type': obstacle['type']})
            
        # Iterate over each source and splat onto the local grid
        for source in repulsion_sources:
            for k in range(self.k_f):
                # Calculate future position
                future_pos = source['pos'] + source['velocity'] * k
                
                # Check if future position is within the local grid bounds
                # We assume a fixed grid size relative to the node
                local_grid_x = int(np.clip((future_pos[0] - node_pos[0] + 0.5) * 4, 0, 3))
                local_grid_y = int(np.clip((future_pos[1] - node_pos[1] + 0.5) * 4, 0, 3))
                
                # Apply Gaussian kernel to the local grid
                kernel_value = np.exp(-0.5 * (k / self.sigma_f)**2)
                
                local_grid[local_grid_y, local_grid_x] += self.eta * (self.gamma_f ** k) * kernel_value

        return local_grid

    def update_from_sensor_data(self,
                               all_nodes: List[Any],
                               all_obstacle_states: List[Dict[str, Any]]):
        """
        Maintains a single global grid for visualization purposes only.
        This function now recalculates the global field based on all local grids.
        """
        self.repulsion_field *= (1 - self.decay_lambda)
        
        for node in all_nodes:
            # Get the local grid for this node
            local_grid = self.get_repulsion_potential_for_node(
                node_pos=node.pos,
                all_node_positions=np.array([n.pos for n in all_nodes]),
                all_obstacle_states=all_obstacle_states
            )
            
            # Splat the local grid back onto the global grid for visualization
            local_grid_x = np.arange(4) - 2
            local_grid_y = np.arange(4) - 2
            
            for i in range(4):
                for j in range(4):
                    grid_x = int(np.clip((node.pos[0] + local_grid_x[i]) * self.grid_shape[0], 0, self.grid_shape[0] - 1))
                    grid_y = int(np.clip((node.pos[1] + local_grid_y[j]) * self.grid_shape[1], 0, self.grid_shape[1] - 1))
                    self.repulsion_field[grid_y, grid_x] += local_grid[j, i]
                    
    def get_full_repulsion_grid(self) -> np.ndarray:
        """
        Returns the full global repulsion grid for visualization.
        """
        return self.repulsion_field
