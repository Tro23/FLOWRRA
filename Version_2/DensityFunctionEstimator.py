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
                 grid_shape: tuple[int, int] = (64, 64),
                 eta: float = 0.1,         # Repulsion learning rate
                 gamma_f: float = 0.8,       # Forward decay for comet-tail
                 k_f: int = 5,             # Forward projection steps
                 sigma_f: float = 2.5,       # Kernel width in grid cells
                 decay_lambda: float = 0.01, # Field decay rate
                 blur_delta: float = 0.1):   # Diffusion/blur mix factor

        self.grid_shape = grid_shape
        self.repulsion_field = np.zeros(grid_shape)

        # Parameters from the design document
        self.eta = eta
        self.gamma_f = gamma_f
        self.k_f = k_f
        self.sigma_f = sigma_f
        self.decay_lambda = decay_lambda
        self.blur_delta = blur_delta

        # Pre-compute grid coordinates and a blur kernel for efficiency
        self._grid_y, self._grid_x = np.mgrid[0:grid_shape[0], 0:grid_shape[1]]
        self.blur_kernel = self._create_blur_kernel()

    def _create_blur_kernel(self):
        """Creates a simple 3x3 averaging kernel for diffusion."""
        k = np.array([[0.5, 1.0, 0.5], [1.0, 2.0, 1.0], [0.5, 1.0, 0.5]])
        return k / k.sum()

    def _splat_gaussian(self, center_x: float, center_y: float, strength: float):
        """
        Adds a single Gaussian kernel to the repulsion field.
        This is the fundamental operation for adding repulsion.
        """
        g = strength * np.exp(-((self._grid_x - center_x)**2 + (self._grid_y - center_y)**2) / (2 * self.sigma_f**2))
        self.repulsion_field += g

    def update_from_detections(self, all_detections: list[dict[str, any]]):
        """
        Updates the repulsion field based on all sensor detections.
        We now use the eye angle to create a directional repulsion component.
        """
        if not all_detections:
            return

        for det in all_detections:
            # 1. Get core detection data
            pos = det['pos']
            velocity = det['velocity']
            angle_idx = det['angle_idx']  # <-- GET THE NEW DATA

            # 2. Create the repulsion splat from position & velocity
            # This is your existing "comet-tail" logic, which is still important.
            kernel_center = pos + velocity * self.gamma_f

            # 3. Create a **directional** repulsion based on eye angle
            # This is the new logic to connect eyes to the density field.
            eye_angle_rad = (angle_idx / 360) * 2 * np.pi
            eye_vector = np.array([np.cos(eye_angle_rad), np.sin(eye_angle_rad)])

            # We can create a small "cone" of repulsion in the direction of the eye
            # This tells the system that looking in a certain direction creates a "bad" field
            cone_center = pos + eye_vector * 0.05  # A small distance in front of the node

            # Combine existing repulsion with directional repulsion
            # We'll splat a kernel at both the original position and a new 'cone' position.
            self._splat_gaussian(kernel_center, strength =self.sigma_f)
            self._splat_gaussian(cone_center, strength = self.sigma_f * 0.5) # Smaller, more localized splat

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
        # Convert to grid coordinates
        grid_coords = positions * np.array([self.grid_shape[1], self.grid_shape[0]])
        # Simple nearest-neighbor sampling. Bilinear interpolation would be more accurate.
        grid_x = np.clip(grid_coords[:, 0].astype(int), 0, self.grid_shape[1] - 1)
        grid_y = np.clip(grid_coords[:, 1].astype(int), 0, self.grid_shape[0] - 1)
        return self.repulsion_field[grid_y, grid_x]
    
    def get_density_at_positions(self, positions: np.ndarray) -> np.ndarray:
        """
        Samples the normalized probability density at a set of continuous positions,
        as defined by the design document.
        """
        if positions.shape[0] == 0:
            return np.array([])
        
        # 1. Get raw repulsion potential
        repulsion_potentials = self.get_potential_at_positions(positions)
        
        # 2. Add the base potential U_pos. (Assuming U_pos is a constant for simplicity, e.g., 0.5)
        # The design document states U_pos is a "smooth, fixed prior".
        U_pos = 0.5 
        
        # 3. Calculate total potential U(x,t) = U_pos + beta * r(x,t)
        # Note: 'beta' (repulsion weight) is currently missing from your config, 
        # so let's add it to the main_runner config and pass it here. 
        beta = 1.0 # This needs to be a parameter, as suggested in the design doc.
        total_potentials = U_pos + beta * repulsion_potentials
        
        # 4. Convert potential to unnormalized density: exp(-U(x,t))
        unnormalized_density = np.exp(-total_potentials)
        
        # 5. Normalize density by a simple max-value for the coherence calculation
        # The design document's full normalization requires an integral,
        # but for the coherence metric, normalizing by rho_max is simpler and sufficient.
        rho_max = np.exp(-U_pos) # max density occurs where repulsion is zero
        normalized_density = unnormalized_density / rho_max

        return normalized_density
