"""
DensityFunctionEstimator.py

This is the core of the v2 system, implementing the speed-aware repulsive density field.
It translates the mathematical design for "comet-tail" repulsion into a practical,
grid-based implementation. It handles splatting, decay, and diffusion of the repulsion
field, which guides node movement.
"""
import numpy as np

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
                 eta: float = 0.08,         # Repulsion learning rate
                 gamma_f: float = 0.8,       # Forward decay for comet-tail
                 k_f: int = 5,             # Forward projection steps
                 sigma_f: float = 4,       # Kernel width in grid cells
                 decay_lambda: float = 0.005, # Field decay rate
                 blur_delta: float = 0.1, # Diffusion/blur mix factor
                 angle_steps: int = 360,   # Discrete angle resolution
                 beta: float = 0.4):       # Repulsion Weight

        self.grid_shape = grid_shape
        self.repulsion_field = np.zeros(grid_shape)

        # Parameters from the design document
        self.eta = eta
        self.gamma_f = gamma_f
        self.k_f = k_f
        self.sigma_f = sigma_f
        self.decay_lambda = decay_lambda
        self.blur_delta = blur_delta
        self.angle_steps = angle_steps
        self.beta = beta

        # Pre-compute grid coordinates and a blur kernel for efficiency
        self._grid_y, self._grid_x = np.mgrid[0:grid_shape[0], 0:grid_shape[1]]
        self.width = grid_shape[1]
        self.height = grid_shape[0]
        self.blur_kernel = self._create_blur_kernel()

    def _create_blur_kernel(self):
        """Creates a simple 3x3 averaging kernel for diffusion."""
        k = np.array([[0.5, 1.0, 0.5], [1.0, 2.0, 1.0], [0.5, 1.0, 0.5]])
        return k / k.sum()

    def _splat_gaussian(self, center_x_grid: float, center_y_grid: float, strength: float):
        """
        Adds a single Gaussian kernel to the repulsion field.
        This is the fundamental operation for adding repulsion.
        """
        """center_x_grid, center_y_grid are in grid coordinates (0..width-1), (0..height-1)."""
        g = strength * np.exp(-((self._grid_x - center_x_grid)**2 + (self._grid_y - center_y_grid)**2) / (2 * (self.sigma_f**2)))
        self.repulsion_field += g

    def update_from_detections(self, all_detections: list[dict[str, any]]):
        """
        Updates the repulsion field based on all sensor detections.
        Handles both nodes (with angle data) and obstacles (without).
        """
        if not all_detections:
            return

        for det in all_detections:
            # 1. Get core detection data
            pos = np.array(det['pos'])        # normalized [0,1)
            velocity = np.array(det.get('velocity', np.zeros(2)))
        
            # 2. Create the repulsion splat from position & velocity
            # This is the "comet-tail" logic, which applies to both nodes and obstacles.

            # Convert normalized pos -> grid coords BEFORE splatting
            center_norm = pos + velocity * self.gamma_f   # still normalized
            # clamp to [0,1)
            center_norm = np.clip(center_norm, 0.0, 0.999999)
            # convert to grid coordinates (x,y)
            center_x_grid = center_norm[0] * (self.width - 1)
            center_y_grid = center_norm[1] * (self.height - 1)

            # Strength uses eta and optionally signal if available
            signal = float(det.get('signal', 1.0))
            strength = self.eta * signal

            self._splat_gaussian(center_x_grid, center_y_grid, strength)

            # 3. Create a **directional** repulsion based on eye angle.
            # This only applies to the nodes that have an angle_idx.
            if 'angle_idx' in det:
                angle_idx = det['angle_idx']
                # convert discrete index -> angle in radians using estimator's angle_steps
                eye_angle_rad = (angle_idx / float(self.angle_steps)) * 2.0 * np.pi
                eye_vector = np.array([np.cos(eye_angle_rad), np.sin(eye_angle_rad)])
                cone_center_norm = pos + eye_vector * 0.03  # small offset in normalized space
                cone_center_norm = np.clip(cone_center_norm, 0.0, 0.999999)
                cx = cone_center_norm[0] * (self.width - 1)
                cy = cone_center_norm[1] * (self.height - 1)
                self._splat_gaussian(cx, cy, strength * 0.6)

    def step_dynamics(self):
        """
        Applies decay and diffusion to the repulsion field.
        This should be called once per simulation step.
        """
        # 1. Decay (forgetting old scars)
        self.repulsion_field *= (1.0 - self.decay_lambda)

        # 2. Smooth (diffusion) using convolution
        # We use a simplified approach here for performance. A proper convolution would be better.
        blurred = np.copy(self.repulsion_field)
        for _ in range(2): # Apply a simple blur multiple times
            blurred = (blurred + np.roll(blurred,1,axis=0) + np.roll(blurred,-1,axis=0) +
                       np.roll(blurred,1,axis=1) + np.roll(blurred,-1,axis=1)) / 5
        self.repulsion_field = (1.0 - self.blur_delta) * self.repulsion_field + self.blur_delta * blurred

        # 3. Clip to prevent unbounded growth
        np.clip(self.repulsion_field, 0, 10.0, out=self.repulsion_field)

    def get_potential_at_positions(self, positions: np.ndarray) -> np.ndarray:
        """
        Samples the repulsion potential at a set of continuous positions.

        Args:
            positions (np.ndarray): An (N, 2) array of positions in [0,1) space.

        Returns:
            An (N,) array of potential values.
        """
        if positions.shape[0] == 0:
            return np.array([])
        # positions assumed normalized [0,1)
        #simplest nearest neighbor
        grid_x = np.clip((positions[:,0] * (self.width - 1)).astype(int), 0, self.width - 1)
        grid_y = np.clip((positions[:,1] * (self.height - 1)).astype(int), 0, self.height - 1)
        return self.repulsion_field[grid_y, grid_x]

    
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
        total_potentials = self.beta * repulsion_potentials
        
        # 3. Convert to density: exp(-U) so high repulsive potential -> low repulsive density
        densities = 1.0 - np.clip(1.0 - np.exp(-total_potentials), 0, 1)
        
        '''# 4. Clip to reasonable range to avoid numerical issues
        densities = np.clip(densities, 0, 1)'''
        
        return densities
