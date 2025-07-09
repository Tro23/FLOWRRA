import numpy as np
import math
import matplotlib.pyplot as plt
from EnvironmentA import EnvironmentA
from density_function_estimator import DensityFunctionEstimator



class WaveFunctionCollapse:
    def __init__(self, estimator, grid_size=30, angle_steps=36):
        self.estimator = estimator
        self.grid_size = grid_size
        self.angle_steps = angle_steps

    def generate_loop(self, positions):
        """Generates a loop by connecting nearest nodes iteratively."""
        if len(positions) == 0:
            return []

        loop = [positions[0]]
        current = positions[0]
        remaining = positions[1:].tolist()

        while remaining:
            distances = [(math.hypot(x - current[0], y - current[1]), [x, y, a]) for x, y, a in remaining]
            _, next_point = min(distances)
            loop.append(next_point)
            remaining.remove(next_point)
            current = next_point

        loop.append(loop[0])  # Close the loop
        return np.array(loop)

    def collapse(self, num_nodes=12):
        """Performs wave function collapse by sampling and looping."""
        samples = self.estimator.sample(num_nodes)

        # Clip within grid and angle bounds
        samples[:, 0] = np.clip(samples[:, 0], 0, self.grid_size - 1)
        samples[:, 1] = np.clip(samples[:, 1], 0, self.grid_size - 1)
        samples[:, 2] = np.clip(samples[:, 2], 0, self.angle_steps - 1)

        samples_int = np.round(samples).astype(int)
        return self.generate_loop(samples_int)


# 1. Simulate EnvironmentA data
env = EnvironmentA()
env_data = []
for _ in range(500):
    state = env.step()
    env_data.extend([[x, y, angle] for (x, y), angle in state])
env_data = np.array(env_data)

# 2. Fit estimator
estimator = DensityFunctionEstimator(bandwidth=1.5)
estimator.fit(env_data)

# 3. Instantiate WFC
wfc = WaveFunctionCollapse(estimator)

# 4. Run WFC
collapsed_loop = wfc.collapse(num_nodes=12)

# 5. Visualize
x, y = collapsed_loop[:, 0] + 0.5, collapsed_loop[:, 1] + 0.5
plt.figure(figsize=(6, 6))
plt.plot(x, y, 'o-', color='orange', linewidth=2, markersize=6)
plt.xlim(0, 30)
plt.ylim(0, 30)
plt.gca().set_aspect('equal')
plt.title("Wave Function Collapse: Loop")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()