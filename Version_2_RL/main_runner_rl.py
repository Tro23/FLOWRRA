"""
Main runner script for the FLOWRRA RL simulation.

This script sets up the configuration, initializes the Flowrra_RL instance,
and runs the main training loop, then the deployment phase.
"""
import time
from FLOWRRA_RL import Flowrra_RL
from utils_rl import create_gif
import logging
import os
import shutil

# --- Main Execution ---
if __name__ == '__main__':
    # Configuration dictionary for the simulation
    config = {
        'num_nodes': 12,
        'seed': 42,
        'grid_size': (60, 60),
        'visual_dir': 'flowrra_rl_visuals',
        'logfile': 'flowrra_rl_log.csv',

        # EnvironmentB Params
        'env_b_grid_size': 60,
        'env_b_num_fixed': 20,
        'env_b_num_moving': 6,

        # RL Params
        'state_size': 122, # Hardcoded based on current state vector size
        'action_size': 4,  # Turn Left, Turn Right, Move Up, Move Down
        'batch_size': 64,
        'gamma': 0.99,
        'lr': 0.0005,
        'target_update_freq': 500,
        
        # NEW: Repulsion-based collapse threshold
        'repulsion_collapse_threshold': 0.4,

        # Training Parameters
        'total_training_steps': 20000,
        'episode_steps': 200,
        'visualize_every_n_steps': 1000
    }

    # Ensure the visuals directory is clean
    if os.path.exists(config['visual_dir']):
        shutil.rmtree(config['visual_dir'])
    os.makedirs(config['visual_dir'])

    # --- Initialize and Run Training ---
    model = Flowrra_RL(config)
    
    start_time = time.time()
    logging.info("--- Starting FLOWRRA RL Training ---")
    model.train(
        total_steps=config['total_training_steps'],
        episode_steps=config['episode_steps'],
        visualize_every_n_steps=config['visualize_every_n_steps']
    )
    end_time = time.time()
    logging.info("--- Training Complete ---")
    logging.info(f"Total training runtime: {end_time - start_time:.2f} seconds.")
    
    # --- Run Deployment ---
    logging.info("--- Starting FLOWRRA RL Deployment ---")
    model.deploy(
        total_steps=150,
        visualize_every_n_steps=1
    )
    logging.info("--- Deployment Complete ---")

    # --- Create GIF Visualization ---
    logging.info("--- Creating GIF ---")
    create_gif(
        image_folder=config['visual_dir'],
        output_gif='flowrra_rl_deployment.gif',
        pattern="deploy_t_*.png", # Use images from the deployment phase
        duration=1000 # milliseconds per frame
    )
    logging.info(f"GIF saved to flowrra_rl_deployment.gif")