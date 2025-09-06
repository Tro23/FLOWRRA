"""
Main runner script for the FLOWRRA v2 simulation.

This script sets up the configuration, initializes the Flowrra instance,
and runs the main simulation loop for a specified number of steps.
"""
import time
from FLOWRRA import Flowrra
import logging

# --- Main Execution ---
if __name__ == '__main__':
    # Configuration dictionary for the simulation
    config = {
        'num_nodes': 12,
        'seed': 42,
        'grid_size': (80, 80),
        'visual_dir': 'flowrra_visuals_v2',
        'logfile': 'flowrra_log_v2.csv',

        # EnvironmentB Params
        'env_b_grid_size': 40,
        'env_b_num_fixed': 20,
        'env_b_num_moving': 8,

        # Density Estimator Params
        'eta': 0.02,          # Repulsion learning rate
        'gamma_f': 0.8,      # Comet-tail decay
        'k_f': 5,             # Comet-tail length
        'sigma_f': 2.0,       # Repulsion kernel width
        'decay_lambda': 0.02, # Field decay rate

        # WFC Params
        'history_length': 300,
        'tail_length': 20,
        'collapse_threshold': 0.35,
        'tau': 8,             # Steps below threshold to trigger collapse
    }

    # Simulation parameters
    total_steps = 2000
    visualize_every_n_steps = 200

    # --- Initialize and Run ---
    model = Flowrra(config)
    start_time = time.time()
    logging.info("--- Starting FLOWRRA v2 Simulation ---")
    logging.info(f"Config: {config}")

    for i in range(total_steps):
        should_visualize = (i % visualize_every_n_steps == 0)
        output = model.STEP(visualize=should_visualize)

        if i % 50 == 0:
            logging.info(f"Step {i}/{total_steps} | "
                         f"Coherence: {output['coherence']:.4f} | "
                         f"Reward: {output['reward']:.4f} | "
                         f"Collapsed: {output['collapse_event']}")

    end_time = time.time()
    logging.info("--- Simulation Complete ---")
    logging.info(f"Total runtime: {end_time - start_time:.2f} seconds.")
    logging.info(f"Log data saved to: {config['logfile']}")
    logging.info(f"Visualizations saved in: {config['visual_dir']}/")

    import pandas as pd
    import matplotlib.pyplot as plt

    # One-liner to read and plot the coherence
    df = pd.read_csv('flowrra_log_v2.csv')
    df.plot(x='t', y='coherence', title='Coherence Over Time')
    plt.savefig("Coherence_Over_Time.png")

