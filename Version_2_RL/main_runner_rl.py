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
        'action_size': 16,  # 4 position actions * 4 angle actions = 16
        'gamma': 0.99,
        'lr': 0.001,
        'buffer_capacity': 50000,
        'batch_size': 64,
        'epsilon_decay': 0.9999,
        'epsilon_min': 0.01,

        # Training & Visualization Params
        'total_training_steps': 10000,
        'episode_steps': 200,
        'visualize_every_n_steps': 1000,
        'model_save_path': 'flowrra_rl_agent.pth',
    }

    # Clean up previous runs
    if os.path.exists(config['visual_dir']):
        shutil.rmtree(config['visual_dir'])
    
    # Initialize FLOWRRA model to get state dimensions
    model = Flowrra_RL(config=config)
    
    # Derive state and action sizes from the model
    initial_state = model.get_state()
    state_size = len(initial_state)
    num_nodes = model.num_nodes
    action_size = model.combined_action_size

    logger.info(f"Derived state_size={state_size} num_nodes={num_nodes} action_size={action_size}")

    # Initialize shared agent
    agent = SharedRLAgent(
        state_size=state_size, 
        num_nodes=num_nodes, 
        action_size=action_size,
        lr=config['lr'], 
        gamma=config['gamma'], 
        buffer_capacity=config['buffer_capacity'], 
        seed=config['seed']
    )

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
    model.deploy(total_steps=config['episode_steps'], visualize_every_n_steps=1)
    logger.info("--- Deployment Complete ---")

    # Create GIF of the deployment
    logger.info("--- Creating GIF from deployment visuals ---")
    gif_path = os.path.join(config['visual_dir'], 'deployment.gif')
    create_gif(config['visual_dir'], gif_path, pattern="deploy_*.png")
    logger.info(f"--- GIF created at {gif_path} ---")
