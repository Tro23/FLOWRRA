from sklearn.neighbors import KernelDensity
import numpy as np

class DensityFunctionEstimator:
    def __init__(self, bandwidth=1.0):
        self.kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth)
        self.trained = False

    def fit(self, data):
        """
        Data format: List of [x, y, angle_idx] vectors.
        """
        self.kde.fit(np.array(data))
        self.trained = True

    def sample(self, num_samples=10):
        if not self.trained:
            raise ValueError("Density estimator not fitted yet.")
        return self.kde.sample(num_samples)

    def score_samples(self, data_points):
        """
        Returns log-density estimates for each point.
        """
        return self.kde.score_samples(np.array(data_points))

from EnvironmentA import EnvironmentA
from EnvironmentB import EnvironmentB

#initialize both environments 
env_a = EnvironmentA()
env_b = EnvironmentB()

#Collecting data from Environment A

data = []
for _ in range(500):
    state = env_a.step()
    data.extend([[x, y, angle] for (x, y), angle in state])

# Fit the density estimator

estimator = DensityFunctionEstimator(bandwidth=1.5)
estimator.fit(data)


#Visualizing the Density Estimation.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity

# Simulate EnvironmentA output data
def simulate_env_a_data(grid_size=30, num_nodes=12, angle_steps=36, steps=500):
    data = []
    for _ in range(steps):
        for _ in range(num_nodes):
            x = np.random.randint(0, grid_size)
            y = np.random.randint(0, grid_size)
            angle_idx = np.random.randint(0, angle_steps)
            data.append([x, y, angle_idx])
    return np.array(data)

# Generate synthetic data
env_a_data = simulate_env_a_data()

# Fit Kernel Density Estimator
kde = KernelDensity(kernel='gaussian', bandwidth=1.5)
kde.fit(env_a_data)

# Create a grid for evaluation
x_vals = np.linspace(0, 30, 60)
y_vals = np.linspace(0, 30, 60)
angle_vals = np.linspace(0, 36, 36)
xx, yy, aa = np.meshgrid(x_vals, y_vals, angle_vals)
grid_points = np.vstack([xx.ravel(), yy.ravel(), aa.ravel()]).T

# Evaluate log density
log_density = kde.score_samples(grid_points)
density = np.exp(log_density)
density_grid = density.reshape(xx.shape)

# Visualize density for a fixed angle slice (e.g., angle_idx = 36)
fixed_angle_idx = 36
slice_index = np.argmin(np.abs(angle_vals - fixed_angle_idx))
density_slice = density_grid[:, :, slice_index]

# Plot the density slice
plt.figure(figsize=(8, 6))
plt.contourf(x_vals, y_vals, density_slice, levels=100, cmap='viridis')
plt.colorbar(label='Density')
plt.title(f'Kernel Density Estimation (Angle Index = {fixed_angle_idx})')
plt.xlabel('X')
plt.ylabel('Y')
plt.tight_layout()
plt.show()
