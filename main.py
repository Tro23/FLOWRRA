# main.py

from EnvironmentA import EnvironmentA
from EnvironmentB import EnvironmentB
from density_function_estimator import DensityFunctionEstimator
from wave_function_collapse import WaveFunctionCollapse
from environmentAVisualizer import EnvironmentAVisualizer
import numpy as np
import matplotlib.pyplot as plt

class StageMinusOneController:
    def __init__(self, grid_size=30, num_nodes=12, bandwidth=1.5):
        self.grid_size = grid_size
        self.num_nodes = num_nodes
        self.bandwidth = bandwidth
        self.estimator = DensityFunctionEstimator(bandwidth=bandwidth)
        self.env_b = EnvironmentB(grid_size=grid_size)
        self.env_a = None
        self.wfc = WaveFunctionCollapse(self.estimator, grid_size=grid_size)

    def collect_data_from_environment_a(self, steps=500):
        temp_env_a = EnvironmentA(grid_size=self.grid_size, num_nodes=self.num_nodes)
        data = []
        for _ in range(steps):
            state = temp_env_a.step()
            data.extend([[x, y, angle] for (x, y), angle in state])
        self.estimator.fit(data)
        return data

    def collapse_and_initialize_env_a(self):
        loop = self.wfc.collapse(num_nodes=self.num_nodes)
        self.env_a = EnvironmentA(initial_loop_data=loop.tolist())
        return loop

    def run_with_obstacles_and_update(self, steps=100):
        log_scores = []
        all_new_data = []
        for _ in range(steps):
            obstacles = self.env_b.step()
            state = self.env_a.step(external_obstacles=obstacles)
            state_arr = np.array([[x, y, angle] for (x, y), angle in state])
            log_score = self.estimator.kde.score(state_arr)
            log_scores.append(log_score)
            all_new_data.extend(state_arr)

        self.estimator.fit(all_new_data)

        plt.figure(figsize=(8, 4))
        plt.plot(log_scores, color='green', linewidth=2)
        plt.title("Feedback Log-Likelihood over Time with Obstacles")
        plt.xlabel("Step")
        plt.ylabel("Log-Likelihood")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()
        return log_scores, all_new_data

    def visualize_coherence(self, loop):
        angles = loop[:, 2]
        counts, _ = np.histogram(angles, bins=range(37), density=False)
        probabilities = counts / np.sum(counts)
        probabilities = probabilities[probabilities > 0]

        from scipy.stats import entropy
        h_value = entropy(probabilities, base=2)

        x, y = loop[:, 0] + 0.5, loop[:, 1] + 0.5
        plt.figure(figsize=(6, 6))
        plt.plot(x, y, 'o-', color='blue', linewidth=2, markersize=6)
        plt.xlim(0, self.grid_size)
        plt.ylim(0, self.grid_size)
        plt.gca().set_aspect('equal')
        plt.title(f"Wave Function Collapse: Loop with Entropy Value of {h_value:.4f}")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()
        return h_value


if __name__ == "__main__":
    controller = StageMinusOneController()
    controller.collect_data_from_environment_a()
    loop = controller.collapse_and_initialize_env_a()
    entropy_value = controller.visualize_coherence(loop)
    log_scores = controller.run_with_obstacles_and_update()

    visualizer = EnvironmentAVisualizer(controller.env_a)
    visualizer.render()
