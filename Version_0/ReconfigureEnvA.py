import numpy as np
import matplotlib.pyplot as plt
from EnvironmentA import EnvironmentA
from density_function_estimator import DensityFunctionEstimator
from wave_function_collapse import WaveFunctionCollapse
import matplotlib.patches as patches

# Step 1: Simulate data using EnvironmentA
env_a = EnvironmentA()
simulated_data = []
for _ in range(100):
    state = env_a.step()
    simulated_data.extend([[x, y, angle] for (x, y), angle in state])
simulated_data = np.array(simulated_data)

# Step 2: Fit KDE
estimator = DensityFunctionEstimator(bandwidth=1.5)
estimator.fit(simulated_data)

# Step 3: Instantiate Wave Function Collapse
wfc = WaveFunctionCollapse(estimator)

#Step 4: Run WFC
collapsed_loop = wfc.collapse(num_nodes=12)

# Step 5: Initialize new EnvironmentA from collapsed loop
env_a_reinit = EnvironmentA(initial_loop_data=collapsed_loop.tolist())

# Step 5: Run for N steps and compute feedback scores
log_scores = []
steps = 100
for _ in range(steps):
    state = env_a_reinit.step()
    state_arr = np.array([[x, y, angle] for (x, y), angle in state])
    log_score = estimator.kde.score(state_arr)
    log_scores.append(log_score)

# Step 6: Plot feedback score over time
plt.figure(figsize=(8, 4))
plt.plot(log_scores, color='purple', linewidth=2)
plt.title("FLOWRRA Feedback Score Over Time (Log-Likelihood)")
plt.xlabel("Step")
plt.ylabel("Log-Likelihood")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# Step 7: Visualizing the Reconfigured Environment A

from environmentAVisualizer import EnvironmentAVisualizer

visualizer = EnvironmentAVisualizer(env_a_reinit)
visualizer.render()