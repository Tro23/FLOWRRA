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
        'grid_size': (60, 60),
        'visual_dir': 'flowrra_visuals_v2',
        'logfile': 'flowrra_log_v2.csv',

        # EnvironmentB Params
        'env_b_grid_size': 60,
        'env_b_num_fixed': 10,
        'env_b_num_moving': 4,

        # Density Estimator Params - ADJUSTED
        'eta': 0.01,              # Decreased repulsion learning rate (was 0.08)
        'gamma_f': 0.4,           # Reduced comet-tail decay for tighter response (was 0.8)
        'k_f': 4,                 # Shorter comet-tail for more immediate response (was 5)
        'sigma_f': 2.0,           # kernel for smoother field (was 4.0)
        'decay_lambda': 0.003,    # Slower decay to maintain memory (was 0.005)
        'beta': 0.2,              # Decreased repulsion weight (was 0.8)

        # WFC Params - ADJUSTED
        'history_length': 200,
        'tail_length': 15,
        'collapse_threshold': 0.25,  # Lower threshold for earlier intervention (was 0.35)
        'tau': 5,                    # Less patience before collapse (was 8)
    }

    # Simulation parameters
    total_steps = 2000
    visualize_every_n_steps = 100 # Reduced for more frequent visualization

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
    try:
        df = pd.read_csv('flowrra_log_v2.csv')
        df.plot(x='t', y='coherence', title='Coherence Over Time')
        plt.savefig("Coherence_Over_Time.png")
        plt.show()
    except FileNotFoundError:
        logging.error("Log file not found. Could not generate coherence plot.")
