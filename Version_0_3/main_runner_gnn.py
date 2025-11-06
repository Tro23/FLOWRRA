"""
main_runner_gnn.py

Complete training pipeline for FLOWRRA-GNN.
Demonstrates 2D and 3D modes with easy configuration switching.
"""
import time
import logging
import os
import shutil
import numpy as np

# Import all components
from config import (
    get_2d_config, get_3d_config, get_fast_prototype_config,
    validate_config, print_config_summary
)
from FLOWRRA_GNN import FLOWRRA_GNN
from GNNAgent import GNNAgent

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("main_runner_gnn")

def main():
    """Main training and deployment pipeline."""
    
    # ==========================================================================
    # CONFIGURATION SELECTION
    # ==========================================================================
    
    # Choose your configuration:
    # - get_2d_config(): Full 2D experiment
    # - get_3d_config(): Full 3D experiment
    # - get_fast_prototype_config(): Quick 2D test
    
    config = get_3d_config()  # <--- CHANGE THIS TO SWITCH MODES
    
    # You can also customize config after loading:
    config['node']['num_nodes'] = 20  # Example: change number of nodes
    # config['repulsion']['local_grid_size'] = (7, 7, 7)  # Larger local grids
    # config['gnn']['hidden_dim'] = 256  # Larger network
    
    # Validate and display
    validate_config(config)
    print_config_summary(config)
    
    # ==========================================================================
    # INITIALIZE FLOWRRA
    # ==========================================================================
    
    logger.info("Initializing FLOWRRA-GNN system...")
    
    # Clean up previous runs
    if os.path.exists(config['viz']['visual_dir']):
        shutil.rmtree(config['viz']['visual_dir'])
    
    model = FLOWRRA_GNN(config=config)
    
    # ==========================================================================
    # INITIALIZE GNN AGENT
    # ==========================================================================
    
    logger.info("Initializing GNN agent...")
    
    agent = GNNAgent(
        node_feature_dim=model.node_feature_dim,
        edge_feature_dim=config['gnn']['edge_feature_dim'],
        action_size=model.action_size,
        hidden_dim=config['gnn']['hidden_dim'],
        num_layers=config['gnn']['num_layers'],
        n_heads=config['gnn']['num_heads'],
        lr=config['training']['learning_rate'],
        gamma=config['training']['gamma'],
        buffer_capacity=config['training']['buffer_capacity'],
        dropout=config['gnn']['dropout'],
        seed=config.get('seed', 42)
    )
    
    logger.info(f"Agent initialized with {sum(p.numel() for p in agent.policy_net.parameters())} parameters")
    
    # ==========================================================================
    # WARM-UP PHASE
    # ==========================================================================
    
    logger.info("Starting warm-up phase...")
    warm_up_steps = config['training'].get('warm_up_steps', 200)
    model.reset()
    
    for step in range(warm_up_steps):
        node_features, adj_matrix = model.get_state()
        
        # Random actions
        actions = np.random.randint(0, model.action_size, size=model.num_nodes)
        
        rewards, done, info = model.step(actions)
        next_node_features, next_adj_matrix = model.get_state()
        
        # Store in replay buffer
        agent.memory.push(
            node_features=node_features,
            adj_matrix=adj_matrix,
            actions=actions,
            rewards=rewards,
            next_node_features=next_node_features,
            next_adj_matrix=next_adj_matrix,
            done=done
        )
        
        if step % 50 == 0:
            logger.info(f"Warm-up: {step}/{warm_up_steps}")
    
    logger.info(f"Warm-up complete. Buffer size: {len(agent.memory)}")
    
    # ==========================================================================
    # TRAINING
    # ==========================================================================
    
    logger.info("=" * 70)
    logger.info("STARTING TRAINING")
    logger.info("=" * 70)
    
    start_time = time.time()
    
    model.train(
        total_steps=config['training']['total_training_steps'],
        episode_steps=config['training']['episode_steps'],
        visualize_every_n_steps=config['viz']['visualize_every_n_steps'],
        agent=agent
    )
    
    training_time = time.time() - start_time
    logger.info(f"Training completed in {training_time:.1f}s ({training_time/60:.1f}m)")
    
    # ==========================================================================
    # SAVE MODEL
    # ==========================================================================
    
    model_path = config['viz']['model_save_path']
    agent.save(model_path)
    logger.info(f"Model saved to {model_path}")
    
    # ==========================================================================
    # DEPLOYMENT
    # ==========================================================================
    
    logger.info("=" * 70)
    logger.info("STARTING DEPLOYMENT")
    logger.info("=" * 70)
    
    model.attach_agent(agent)
    model.deploy(
        total_steps=config['viz']['deployment_steps'],
        visualize_every_n_steps=1,
        num_episodes=config['viz']['deployment_episodes']
    )
    
    logger.info("Deployment complete!")
    
    # ==========================================================================
    # ANALYSIS & VISUALIZATION
    # ==========================================================================
    
    logger.info("Generating analysis plots...")
    
    try:
        import pandas as pd
        import matplotlib.pyplot as plt
        
        # Load training log
        df = pd.read_csv(config['viz']['log_file'])
        
        # Plot coherence over time
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Coherence
        axes[0, 0].plot(df['step'], df['avg_coherence'])
        axes[0, 0].set_title('Coherence Over Time')
        axes[0, 0].set_xlabel('Step')
        axes[0, 0].set_ylabel('Coherence')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Rewards
        axes[0, 1].plot(df['step'], df['total_reward'])
        axes[0, 1].set_title('Total Reward Over Time')
        axes[0, 1].set_xlabel('Step')
        axes[0, 1].set_ylabel('Reward')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Exploration reward
        axes[1, 0].plot(df['step'], df['exploration_reward'])
        axes[1, 0].set_title('Exploration Reward Over Time')
        axes[1, 0].set_xlabel('Step')
        axes[1, 0].set_ylabel('Exploration Reward')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Loss
        axes[1, 1].plot(df['step'], df['loss'])
        axes[1, 1].set_title('Training Loss Over Time')
        axes[1, 1].set_xlabel('Step')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('training_analysis.png', dpi=150)
        logger.info("Training analysis saved to training_analysis.png")
        
        # Print summary statistics
        logger.info("\n" + "=" * 70)
        logger.info("TRAINING SUMMARY")
        logger.info("=" * 70)
        logger.info(f"Final coherence: {df['avg_coherence'].iloc[-100:].mean():.4f}")
        logger.info(f"Final exploration reward: {df['exploration_reward'].iloc[-100:].mean():.4f}")
        logger.info(f"Total collapses: {(df['wfc_reinit'] != 'none').sum()}")
        logger.info(f"Average reward (last 1000 steps): {df['total_reward'].iloc[-1000:].mean():.4f}")
        logger.info("=" * 70)
        
    except Exception as e:
        logger.warning(f"Could not generate analysis plots: {e}")
    
    # ==========================================================================
    # CREATE GIF (if in 2D mode)
    # ==========================================================================
    
    if config['spatial']['dimensions'] == 2:
        try:
            from utils_rl import create_gif
            gif_path = os.path.join(config['viz']['visual_dir'], 'deployment.gif')
            create_gif(
                config['viz']['visual_dir'],
                gif_path,
                pattern='deploy_*.png',
                duration=100
            )
            logger.info(f"GIF created at {gif_path}")
        except Exception as e:
            logger.warning(f"Could not create GIF: {e}")
    
    logger.info("\n🎉 FLOWRRA-GNN pipeline complete!")


def compare_configs():
    """
    Utility function to compare different configurations.
    Useful for hyperparameter tuning experiments.
    """
    configs = {
        '2D-Fast': get_fast_prototype_config(),
        '2D-Full': get_2d_config(),
        '3D-Full': get_3d_config(),
    }
    
    print("\n" + "=" * 70)
    print("CONFIGURATION COMPARISON")
    print("=" * 70)
    
    for name, cfg in configs.items():
        print(f"\n{name}:")
        print(f"  Dimensions: {cfg['spatial']['dimensions']}D")
        print(f"  Nodes: {cfg['node']['num_nodes']}")
        print(f"  Local grid: {cfg['repulsion']['local_grid_size']}")
        print(f"  Hidden dim: {cfg['gnn']['hidden_dim']}")
        print(f"  Training steps: {cfg['training']['total_training_steps']}")
        
        # Estimate node feature dimension
        dims = cfg['spatial']['dimensions']
        grid_size = cfg['repulsion']['local_grid_size']
        grid_elements = np.prod(grid_size)
        orientation_dim = 1 if dims == 2 else 2
        feature_dim = dims + dims + orientation_dim + 5*(3+dims) + 5*(3+dims) + grid_elements
        print(f"  Node feature dim: {feature_dim}")


if __name__ == '__main__':
    # Uncomment to compare configurations:
    # compare_configs()
    
    # Run main pipeline
    main()