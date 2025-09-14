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
                 blur_delta: float = 0.2,
                 beta: float = 0.7):
        # We still need a global grid for visualization, but calculations are local now.
        self.grid_shape = grid_shape
        self.repulsion_field = np.zeros(grid_shape)
        self.eta = eta
        self.gamma_f = gamma_f
        self.k_f = k_f
        self.sigma_f = sigma_f
        self.decay_lambda = decay_lambda
        self.blur_delta = blur_delta
        self.beta = beta
        
    def reset(self):
        self.repulsion_field = np.zeros(self.grid_shape)

    def get_repulsion_potential_for_node(self,
                                         node_pos: np.ndarray,
                                         repulsion_sources: List[Dict[str, Any]]) -> np.ndarray:
        """
        Calculates and returns a LOCAL 4x4 repulsion potential grid for a single node
        based ONLY on its detected sources.
        
        Args:
            node_pos (np.ndarray): The (x,y) position of the node.
            repulsion_sources (List[Dict]): The list of sensor detections.
            
        Returns:
            np.ndarray: A 4x4 grid of repulsion values.
        """
        local_grid = np.zeros((5, 5))
        
        # Iterate over each source detected by the sensor and splat onto the local grid
        for source in repulsion_sources:
            for k in range(self.k_f):
                # Calculate future position
                future_pos = source['pos'] + source['velocity'] * k
                
                # Check if future position is within the local grid bounds
                local_grid_x = int(np.clip((future_pos[0] - node_pos[0] + 0.5) * 4, 0, 4))
                local_grid_y = int(np.clip((future_pos[1] - node_pos[1] + 0.5) * 4, 0, 4))
                
                # Apply Gaussian kernel to the local grid
                kernel_value = np.exp(-0.5 * (k / self.sigma_f)**2)
                
                local_grid[local_grid_y, local_grid_x] += self.eta * (self.gamma_f ** k) * kernel_value

        return self.beta * local_grid

    def update_from_sensor_data(self,
                               all_nodes: List[Any],
                               all_obstacle_states: List[Dict[str, Any]]):
        """
        Maintains a single global grid for visualization purposes only.
        This function now recalculates the global field based on all local grids.
        """
        self.repulsion_field *= (1 - self.decay_lambda)
        
        for node in all_nodes:
            # Get sensor detections for this node
            all_detections = node.sense_nodes(all_nodes) + node.sense_obstacles(all_obstacle_states)
            
            # Get the local grid based on detections
            local_grid = self.get_repulsion_potential_for_node(
                node_pos=node.pos,
                repulsion_sources=all_detections
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