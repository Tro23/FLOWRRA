"""
Main runner script for the FLOWRRA RL simulation.
Initializes the Flowrra_RL instance and the shared RL agent, runs training,
then runs deployment and creates the GIF.
"""
import time
import logging
import os
import shutil
import numpy as np

from FLOWRRA_RL import Flowrra_RL
from RLAgent import SharedRLAgent
from utils_rl import create_gif

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("main_runner")

if __name__ == '__main__':
    # Configuration
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

        # RL Params (some will be computed)
        'action_size': 4,  # 0: left, 1: right, 2: noop, 3: noop (expand if desired)
        'batch_size': 64,
        'gamma': 0.99,
        'lr': 0.0005,
        'target_update_freq': 10,

        # Repulsion collapse threshold
        'repulsion_collapse_threshold': 0.4,

        # Training Parameters
        'total_training_steps': 20000,
        'episode_steps': 200,
        'visualize_every_n_steps': 1000,

        # Model save path
        'model_save_path': 'flowrra_shared_agent.pth'
    }

    # Clean visuals
    if os.path.exists(config['visual_dir']):
        shutil.rmtree(config['visual_dir'])
    os.makedirs(config['visual_dir'], exist_ok=True)

    # Initialize environment orchestrator
    model = Flowrra_RL(config)

    # Derive dynamic state size from the environment
    model.env.reset()
    model.env_b.reset()
    model.density_estimator.reset()
    model.wfc.reset()
    example_state = model.get_state()
    state_size = int(len(example_state))
    num_nodes = config['num_nodes']
    action_size = config['action_size']

    logger.info(f"Derived state_size={state_size} num_nodes={num_nodes} action_size={action_size}")

    # Initialize shared agent
    agent = SharedRLAgent(state_size=state_size, num_nodes=num_nodes, action_size=action_size,
                          lr=config['lr'], gamma=config['gamma'], buffer_capacity=50000, seed=config['seed'])

    # Attach agent to model and run training
    start_time = time.time()
    logger.info("--- Starting FLOWRRA RL Training ---")
    model.train(
        total_steps=config['total_training_steps'],
        episode_steps=config['episode_steps'],
        visualize_every_n_steps=config['visualize_every_n_steps'],
        agent=agent
    )
    end_time = time.time()
    logger.info(f"--- Training Complete (took {end_time - start_time:.1f}s) ---")

    # Save agent
    agent.save(config['model_save_path'])

    # Deployment (greedy)
    logger.info("--- Starting FLOWRRA RL Deployment ---")
    model.attach_agent(agent)
    model.deploy(total_steps=150, visualize_every_n_steps=1)
    logger.info("--- Deployment Complete ---")

    # Create GIF from deployment images
    logger.info("--- Creating GIF ---")
    create_gif(
        image_folder=config['visual_dir'],
        output_gif='flowrra_rl_deployment.gif',
        pattern="deploy_t_*.png",
        duration=1000
    )
    logger.info("GIF saved to flowrra_rl_deployment.gif")
