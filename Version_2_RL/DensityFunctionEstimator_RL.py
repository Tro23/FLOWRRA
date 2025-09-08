"""
DensityFunctionEstimator_RL.py

This is the core of the v2 system, implementing the speed-aware repulsive density field.
It translates the mathematical design for "comet-tail" repulsion into a practical,
grid-based implementation. It handles splatting, decay, and diffusion of the repulsion
field, which guides node movement.
"""
import numpy as np
from typing import List, Dict, Any

class Density_Function_Estimator:
    """
    Manages the repulsive density field based on sensor detections.

    This class implements the "comet-tail" repulsion by projecting repulsion kernels
    forward in time based on detected object velocities.

    Attributes:
        grid_shape (tuple): The (width, height) resolution of the density grid.
        repulsion_field (np.ndarray): The grid storing repulsive potential values.
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
                 eta: float = 0.09,
                 gamma_f: float = 0.4,
                 k_f: int = 4,
                 sigma_f: float = 4.0,
                 decay_lambda: float = 0.003,
                 blur_delta: float = 0.1,
                 beta: float = 0.8):
        self.grid_shape = grid_shape
        self.width, self.height = grid_shape
        self.eta = eta
        self.gamma_f = gamma_f
        self.k_f = k_f
        self.sigma_f = sigma_f
        self.decay_lambda = decay_lambda
        self.blur_delta = blur_delta
        self.beta = beta,
        self.repulsion_field: np.ndarray = np.zeros(grid_shape)
        self.blur_kernel = self._create_blur_kernel()

    def reset(self):
        """
        Resets the repulsion field to a clean state for a new episode.
        """
        self.repulsion_field = np.zeros(self.grid_shape)

    def _create_blur_kernel(self):
        """Creates a simple 3x3 blurring kernel for diffusion."""
        kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float32)
        return kernel / kernel.sum()

    def _apply_gaussian_kernel(self, pos: np.ndarray, field: np.ndarray, std_dev: float, value: float):
        """
        Applies a 2D Gaussian splat to the given field at a continuous position.
        """
        grid_y, grid_x = np.ogrid[:self.height, :self.width]
        dist_sq = ((grid_x - pos[0] * self.width)**2 + (grid_y - pos[1] * self.height)**2)
        kernel = np.exp(-dist_sq / (2 * std_dev**2))
        field += value * kernel

    def step_dynamics(self):
        """
        Applies decay and diffusion to the repulsion field.
        """
        # Decay the field
        self.repulsion_field *= (1 - self.decay_lambda)
        
        # Diffusion (blurring)
        blurred_field = np.zeros_like(self.repulsion_field)
        for i in range(1, self.height - 1):
            for j in range(1, self.width - 1):
                blurred_field[i, j] = np.sum(self.repulsion_field[i-1:i+2, j-1:j+2] * self.blur_kernel)
        self.repulsion_field = (1 - self.blur_delta) * self.repulsion_field + self.blur_delta * blurred_field
        
    def splat_repulsion(self, detections: List[Dict[str, Any]]):
        """
        Projects repulsion from detected obstacles onto the field.
        """
        for det in detections:
            # Main repulsion splat
            self._apply_gaussian_kernel(det['pos'], self.repulsion_field, self.sigma_f, self.eta)
            
            # Comet-tail projection for moving objects
            if np.linalg.norm(det['velocity']) > 0:
                for k in range(1, self.k_f + 1):
                    projected_pos = det['pos'] + k * det['velocity']
                    projected_value = self.eta * (self.gamma_f ** k)
                    self._apply_gaussian_kernel(projected_pos, self.repulsion_field, self.sigma_f, projected_value)

    def get_potential_at_positions(self, positions: np.ndarray) -> np.ndarray:
        """
        Samples the raw repulsion potential at a set of continuous positions.

        Args:
            positions (np.ndarray): An (N, 2) array of positions in [0,1) space.

        Returns:
            An (N,) array of potential values.
        """
        if positions.shape[0] == 0:
            return np.array([])
        # positions assumed normalized [0,1)
        grid_x = np.clip((positions[:,0] * self.width).astype(int), 0, self.width - 1)
        grid_y = np.clip((positions[:,1] * self.height).astype(int), 0, self.height - 1)
        return self.beta * self.repulsion_field[grid_y, grid_x]
    
    def get_density_at_positions(self, positions: np.ndarray, loop_center: np.ndarray = None) -> np.ndarray:
        """
        Samples the normalized probability density at a set of continuous positions.
        High density = good areas (low repulsion)
        Low density = bad areas (high repulsion)
        """
        if positions.shape[0] == 0:
            return np.array([])
        
        # 1. Get raw repulsion potential (high values = bad areas)
        repulsion_potentials = self.get_potential_at_positions(positions)
        
        # 2. Total potential (high = bad areas to avoid)
        total_repulsion_potentials = 1 - np.exp(-(repulsion_potentials))
        
        # 3. Apply additional potentials (e.g., global coherence)
        # This part of the design can be adjusted to include other factors if needed.
        
        # 4. Convert potential to density
        density = 1 - np.clip(total_repulsion_potentials,0,1)

        return density
        
