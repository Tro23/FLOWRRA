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
    # IMPROVED Configuration
    config = {
        'num_nodes': 8,  # Reduced for easier learning
        'seed': 42,
        'grid_size': (60, 60),
        'visual_dir': 'flowrra_rl_visuals',
        'logfile': 'flowrra_rl_log.csv',

        # EnvironmentB Params
        'env_b_grid_size': 60,
        'env_b_num_fixed': 8,   # Reduced obstacles
        'env_b_num_moving': 4,  # Reduced moving obstacles

        # RL Params
        'action_size': 16,
        'gamma': 0.95,          # Slightly reduced for faster learning
        'lr': 0.0005,           # Reduced learning rate for stability
        'buffer_capacity': 10000,
        'batch_size': 32,       # Smaller batch size

        # Training & Visualization Params
        'total_training_steps': 4000,   # More training steps
        'episode_steps': 200,           # Shorter episodes
        'visualize_every_n_steps': 100,
        'model_save_path': 'flowrra_rl_agent.pth',
    }

    # Clean up previous runs
    if os.path.exists(config['visual_dir']):
        shutil.rmtree(config['visual_dir'])
    
    # Initialize FLOWRRA model
    model = Flowrra_RL(config=config)
    
    # Get dimensions
    initial_state = model.get_state()
    state_size = len(initial_state)
    num_nodes = model.num_nodes
    action_size = model.combined_action_size

    logger.info(f"State_size={state_size}, num_nodes={num_nodes}, action_size={action_size}")

    # Initialize agent
    agent = SharedRLAgent(
        state_size=state_size, 
        num_nodes=num_nodes, 
        action_size=action_size,
        lr=config['lr'], 
        gamma=config['gamma'], 
        buffer_capacity=config['buffer_capacity'], 
        seed=config['seed']
    )

    # Warm-up phase
    logger.info("--- Starting Warm-up Phase ---")
    warm_up_steps = 50
    model.reset()
    
    for step in range(warm_up_steps):
        state = model.get_state()
        actions = np.random.randint(0, action_size, size=num_nodes)
        rewards, done, info = model.step(list(actions))
        next_state = model.get_state()
        agent.memory.push(state, actions, rewards, next_state, done)
        
        if step % 100 == 0:
            logger.info(f"Warm-up step {step}/{warm_up_steps}")
            
    logger.info("--- Warm-up Complete ---")

    # Training
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

    # Deployment with moderate exploration to see swarm behavior
    logger.info("--- Starting FLOWRRA RL Deployment ---")
    model.attach_agent(agent)
    model.deploy(total_steps=config['episode_steps'], visualize_every_n_steps=50)
    logger.info("--- Deployment Complete ---")

    # Create GIF
    logger.info("--- Creating GIF from deployment visuals ---")
    gif_path = os.path.join(config['visual_dir'], 'flowrra_rl_deployment.gif')
    create_gif(config['visual_dir'], gif_path, pattern="deploy_*.png")
    logger.info(f"--- GIF created at {gif_path} ---")