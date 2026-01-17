"""
flowrra_backend.py

Domain-Agnostic FLOWRRA Backend

This is the main entry point that:
1. Loads dataset via adapter
2. Initializes federated holonic system
3. Runs training/inference
4. Exports results

Usage:
    python flowrra_backend.py --domain warehouse --dataset ./data/warehouse.json --episodes 1000
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from dataset_adapter import (
    FlowrraConfig,
    FlowrraState,
    create_adapter,
)


class FlowrraBackend:
    """
    Main backend orchestrator.
    
    Connects dataset adapter → federated system → training loop.
    """
    
    def __init__(
        self,
        domain: str,
        dataset_path: str,
        config: FlowrraConfig,
        mode: str = 'training'
    ):
        self.domain = domain
        self.mode = mode
        self.config = config
        
        # Create adapter
        self.adapter = create_adapter(domain, dataset_path, config)
        
        # Load dataset
        print(f"\n[Backend] Loading {domain} dataset from {dataset_path}")
        self.dataset = self.adapter.load_dataset()
        print(f"[Backend] ✅ Dataset loaded: {len(self.dataset.get('episodes', []))} episodes")
        
        # Initialize federated system (YOUR existing code!)
        self.federation = None
        self.holons: Dict[int, Any] = {}
        
        # Metrics
        self.episode_rewards: List[float] = []
        self.training_metrics: List[Dict] = []
        
    def initialize_federation(self):
        """
        Initialize federated holonic system using YOUR existing code.
        
        This is where we bridge to federation/manager.py and holon/holon_core.py
        """
        from federation.manager import FederationManager
        from holon.holon_core import Holon
        
        print(f"\n[Backend] Initializing federated system with {self.config.num_holons} holons")
        
        # Create federation manager
        self.federation = FederationManager(
            num_holons=self.config.num_holons,
            world_bounds=self.config.world_bounds,
            breach_threshold=0.1,
            coordination_mode='positional'
        )
        
        # Create holons
        partition_assignments = self.federation.get_partition_assignments()
        
        for partition_id, partition in partition_assignments.items():
            holon = Holon(
                holon_id=partition_id,
                partition_id=partition_id,
                spatial_bounds={
                    'x': partition.bounds_x,
                    'y': partition.bounds_y
                },
                config=self._convert_config_to_dict(),
                mode=self.mode
            )
            self.holons[partition_id] = holon
        
        print(f"[Backend] ✅ Created {len(self.holons)} holons")
        
        # Distribute agents to holons
        self._distribute_agents_to_holons()
        
    def _convert_config_to_dict(self) -> Dict:
        """Convert FlowrraConfig to the dictionary format YOUR code expects."""
        # This bridges the adapter's config to your existing CONFIG structure
        
        # Map action space
        if self.config.action_space_type == 'discrete':
            if self.config.dimensions == 2:
                action_size = 4  # left, right, up, down
            else:
                action_size = 6  # +/-x, +/-y, +/-z
        else:
            action_size = self.config.dimensions  # Continuous
        
        return {
            'spatial': {
                'dimensions': self.config.dimensions,
                'world_bounds': self.config.world_bounds,
            },
            'node': {
                'total_nodes': self.config.num_agents,
                'num_nodes_per_holon': self.config.num_agents // self.config.num_holons,
                'move_speed': self.config.agent_speed,
                'sensor_range': self.config.sensor_range,
            },
            'federation': {
                'num_holons': self.config.num_holons,
                'world_bounds': self.config.world_bounds,
                'breach_threshold': 0.1,
                'coordination_mode': 'positional',
                'enable_dynamic_splitting': False,
            },
            'holon': {
                'mode': self.mode,
                'independent_training': True,
                'share_experience': False,
                'use_r_gnn': False,
                'enable_strategic_freezing': self.config.enable_frozen_nodes,
            },
            'gnn': {
                'hidden_dim': 128,
                'num_layers': 3,
                'n_heads': 4,
                'lr': 0.0001,
                'gamma': 0.98,
                'dropout': 0.1,
                'buffer_capacity': 15000,
                'stability_coef': 0.55,
            },
            'loop': {
                'ideal_distance': 0.12,
                'stiffness': 0.45,
                'break_threshold': 0.32,
            },
            'patrol': {
                'enabled': True,
                'waypoint_threshold': 0.15,
                'stick_prob': 0.05,
                'bias_force': 0.002,
            },
            'rewards': {
                'r_flow': 5.0,
                'r_collision': 30.0,
                'r_idle': 2.0,
                'r_loop_integrity': 10.0,
                'r_collapse_penalty': 25.0,
                'r_explore': 12.0,
                'r_reconnection_spatial': 40.0,
                'r_reconnection_temporal': 10.0,
                'r_boundary_breach': 10.0,
                'r_frozen_node_bonus': 50.0,
                'r_frozen_utility': 1.0,
                'r_reconnection': 5.0,
            },
            'repulsion': {
                'local_grid_size': (5, 5) if self.config.dimensions == 2 else (5, 5, 5),
                'global_grid_shape': (60, 60) if self.config.dimensions == 2 else (60, 60, 60),
                'eta': 0.5,
                'gamma_f': 0.9,
                'k_f': 5,
                'sigma_f': 0.05,
                'decay_lambda': 0.9,
                'blur_delta': 0.1,
                'beta': 0.3,
            },
            'exploration': {
                'map_resolution': 0.01,
                'sensor_range': 0.20,
            },
            'wfc': {
                'history_length': 150,
                'tail_length': 8,
                'collapse_threshold': 0.55,
                'tau': 2,
                'spatial_search_radius_mult': 1.2,
                'spatial_samples': 32,
                'spatial_accept_threshold': 0.60,
                'spatial_improvement_min': 0.50,
            },
            'obstacles': [],  # Will be populated from adapter
            'moving_obstacles': [],
            'training': {
                'num_episodes': 50,
                'steps_per_episode': 1200,
                'target_update_frequency': 100,
                'save_frequency': 100,
                'metrics_save_frequency': 20,
            },
            'visualization': {
                'show_partitions': True,
                'show_breach_alerts': True,
                'show_frozen_nodes': True,
                'frozen_node_color': 'gold',
                'partition_color_scheme': 'rainbow',
                'render_frequency': 50,
                'save_history': True,
            },
        }
    
    def _distribute_agents_to_holons(self):
        """
        Distribute agents (nodes) across holons based on initial state.
        """
        from holon.node import NodePositionND
        
        # Get initial state from adapter
        initial_state = self.adapter.get_initial_state()
        
        # Create NodePositionND objects
        all_nodes = []
        for i, pos in enumerate(initial_state.positions):
            node = NodePositionND(
                id=i,
                pos=pos.copy(),
                dimensions=self.config.dimensions
            )
            node.sensor_range = self.config.sensor_range
            node.move_speed = self.config.agent_speed
            all_nodes.append(node)
        
        # Distribute to holons based on spatial partitioning
        nodes_per_holon = len(all_nodes) // self.config.num_holons
        
        for holon_id, holon in self.holons.items():
            start_idx = holon_id * nodes_per_holon
            end_idx = start_idx + nodes_per_holon
            holon_nodes = all_nodes[start_idx:end_idx]
            
            # Initialize orchestrator with nodes
            holon.initialize_orchestrator_with_nodes(holon_nodes)
            
            # Load obstacles from adapter
            constraints = initial_state.constraints
            self._load_constraints_to_holon(holon, constraints)
        
        print(f"[Backend] ✅ Distributed {len(all_nodes)} agents across {len(self.holons)} holons")
    
    def _load_constraints_to_holon(self, holon, constraints: Dict):
        """Load obstacles from adapter into holon's orchestrator."""
        if holon.orchestrator is None:
            return
        
        # Clear existing obstacles
        holon.orchestrator.obstacle_manager.obstacles.clear()
        
        # Add static obstacles
        for obs_pos, obs_radius in constraints.get('static_obstacles', []):
            # Convert to holon's local coordinates
            local_pos = holon._to_local(obs_pos)
            h_width = holon.x_max - holon.x_min
            local_radius = obs_radius / h_width
            
            holon.orchestrator.obstacle_manager.add_static_obstacle(
                local_pos, local_radius
            )
        
        # Add dynamic obstacles
        for obs_pos, obs_radius, obs_vel in constraints.get('dynamic_obstacles', []):
            local_pos = holon._to_local(obs_pos)
            h_width = holon.x_max - holon.x_min
            local_radius = obs_radius / h_width
            
            holon.orchestrator.obstacle_manager.add_moving_obstacle(
                local_pos, local_radius, obs_vel
            )
        
        print(f"[Holon {holon.holon_id}] Loaded {len(constraints.get('static_obstacles', []))} obstacles")
    
    def train(self, num_episodes: int):
        """
        Main training loop.
        
        This uses YOUR existing training logic from main.py
        """
        print(f"\n[Backend] Starting training for {num_episodes} episodes")
        
        steps_per_episode = 1200  # TODO: Make configurable
        
        for episode in range(1, num_episodes + 1):
            episode_reward = self._train_episode(episode, num_episodes, steps_per_episode)
            self.episode_rewards.append(episode_reward)
            
            if episode % 10 == 0:
                self._print_progress(episode, num_episodes)
            
            if episode % 100 == 0:
                self._save_checkpoint(episode)
        
        print(f"\n[Backend] ✅ Training complete!")
        self._save_final_results()
    
    def _train_episode(
        self, 
        episode_num: int, 
        total_episodes: int,
        steps_per_episode: int
    ) -> float:
        """Execute one training episode."""
        total_reward = 0.0
        
        for step in range(steps_per_episode):
            # === Phase A: Holons execute step ===
            holon_rewards = []
            
            for holon in self.holons.values():
                try:
                    reward = holon.step(step, total_episodes)
                    holon_rewards.append(reward)
                    total_reward += reward
                except Exception as e:
                    print(f"[Backend] ERROR in Holon {holon.holon_id}: {e}")
                    holon_rewards.append(0.0)
            
            # === Phase B: Federation cycle ===
            holon_states = {
                holon_id: holon.get_state_summary()
                for holon_id, holon in self.holons.items()
            }
            
            breach_alerts = self.federation.step(holon_states)
            
            # === Phase C: Send breach alerts ===
            for holon_id, alerts in breach_alerts.items():
                if alerts:
                    self.holons[holon_id].receive_breach_alerts(alerts)
        
        avg_reward = total_reward / (len(self.holons) * steps_per_episode)
        return avg_reward
    
    def _print_progress(self, episode: int, total_episodes: int):
        """Print training progress."""
        fed_stats = self.federation.get_statistics()
        avg_reward = self.episode_rewards[-1] if self.episode_rewards else 0.0
        
        print(f"\n{'=' * 70}")
        print(f"Episode {episode}/{total_episodes}")
        print(f"Avg Reward: {avg_reward:.3f} | Total Breaches: {fed_stats['total_breaches']}")
        
        for holon in self.holons.values():
            stats = holon.get_statistics()
            print(f"  Holon {stats['holon_id']}: "
                  f"reward={stats['avg_reward']:.3f}, "
                  f"active={stats['num_active_nodes']}, "
                  f"frozen={stats['num_frozen_nodes']}")
        print(f"{'=' * 70}\n")
    
    def _save_checkpoint(self, episode: int):
        """Save training checkpoint."""
        checkpoint_dir = Path("checkpoints")
        checkpoint_dir.mkdir(exist_ok=True)
        
        for holon_id, holon in self.holons.items():
            filepath = checkpoint_dir / f"{self.domain}_holon_{holon_id}_ep{episode}.pt"
            try:
                holon.save(str(filepath))
            except Exception as e:
                print(f"[Backend] Warning: Could not save Holon {holon_id}: {e}")
        
        print(f"[Backend] ✅ Checkpoint saved at episode {episode}")
    
    def _save_final_results(self):
        """Save final training results."""
        results_dir = Path("results") / self.domain
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Final statistics
        final_stats = {
            'domain': self.domain,
            'total_episodes': len(self.episode_rewards),
            'final_avg_reward': self.episode_rewards[-1] if self.episode_rewards else 0.0,
            'best_reward': max(self.episode_rewards) if self.episode_rewards else 0.0,
            'federation': self.federation.get_statistics(),
            'holons': {
                holon_id: holon.get_statistics()
                for holon_id, holon in self.holons.items()
            },
            'config': self._convert_config_to_dict(),
        }
        
        filepath = results_dir / f"{self.domain}_final_results.json"
        with open(filepath, 'w') as f:
            json.dump(final_stats, f, indent=2, default=str)
        
        # Episode rewards
        rewards_filepath = results_dir / f"{self.domain}_rewards.npy"
        np.save(rewards_filepath, np.array(self.episode_rewards))
        
        print(f"\n[Backend] ✅ Results saved to {results_dir}")
        print(f"  - Final results: {filepath}")
        print(f"  - Rewards: {rewards_filepath}")
    
    def deploy(self, checkpoint_path: str):
        """
        Load trained model and run in deployment mode.
        
        Args:
            checkpoint_path: Path to checkpoint directory
        """
        print(f"\n[Backend] Loading checkpoint from {checkpoint_path}")
        
        checkpoint_dir = Path(checkpoint_path)
        
        # Load holon models
        for holon_id, holon in self.holons.items():
            filepath = checkpoint_dir / f"{self.domain}_holon_{holon_id}_ep*.pt"
            
            # Find latest checkpoint
            checkpoints = list(checkpoint_dir.glob(f"{self.domain}_holon_{holon_id}_ep*.pt"))
            if checkpoints:
                latest = max(checkpoints, key=lambda p: int(p.stem.split('ep')[1]))
                holon.load(str(latest))
                print(f"[Backend] ✅ Loaded Holon {holon_id} from {latest}")
        
        print(f"[Backend] Deployment mode ready")
        
        # Run deployment episode
        self._deploy_episode()
    
    def _deploy_episode(self):
        """Run one deployment episode (no training)."""
        steps = 1000
        
        for step in range(steps):
            # Execute holon steps (no learning)
            for holon in self.holons.values():
                holon.step(step, 1)
            
            # Federation coordination
            holon_states = {
                holon_id: holon.get_state_summary()
                for holon_id, holon in self.holons.items()
            }
            
            self.federation.step(holon_states)
            
            if step % 100 == 0:
                print(f"[Deployment] Step {step}/{steps}")
        
        print(f"[Backend] ✅ Deployment episode complete")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="FLOWRRA Modular Backend")
    
    parser.add_argument(
        '--domain',
        type=str,
        required=True,
        choices=['warehouse', 'traffic', 'satellite'],
        help='Domain to train on'
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        default='./data',
        help='Path to dataset'
    )
    
    parser.add_argument(
        '--episodes',
        type=int,
        default=100,
        help='Number of training episodes'
    )
    
    parser.add_argument(
        '--holons',
        type=int,
        default=4,
        help='Number of holons (must be perfect square)'
    )
    
    parser.add_argument(
        '--agents',
        type=int,
        default=16,
        help='Total number of agents'
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        default='training',
        choices=['training', 'deployment'],
        help='Operation mode'
    )
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        help='Checkpoint path for deployment mode'
    )
    
    args = parser.parse_args()
    
    # Create configuration
    if args.domain == 'warehouse':
        dimensions = 2
        world_bounds = (1.0, 1.0)
        agent_speed = 0.01
    elif args.domain == 'traffic':
        dimensions = 2
        world_bounds = (1.0, 1.0)
        agent_speed = 0.015
    elif args.domain == 'satellite':
        dimensions = 3
        world_bounds = (1.0, 1.0, 1.0)
        agent_speed = 0.005
    else:
        raise ValueError(f"Unknown domain: {args.domain}")
    
    config = FlowrraConfig(
        dimensions=dimensions,
        world_bounds=world_bounds,
        num_agents=args.agents,
        agent_speed=agent_speed,
        sensor_range=0.2,
        action_space_type='discrete',
        action_space_size=4 if dimensions == 2 else 6,
        num_holons=args.holons,
        enable_wfc=True,
        enable_frozen_nodes=True,
    )
    
    # Create backend
    backend = FlowrraBackend(
        domain=args.domain,
        dataset_path=args.dataset,
        config=config,
        mode=args.mode
    )
    
    # Initialize federation
    backend.initialize_federation()
    
    # Run
    if args.mode == 'training':
        backend.train(args.episodes)
    else:
        if not args.checkpoint:
            print("ERROR: --checkpoint required for deployment mode")
            return 1
        backend.deploy(args.checkpoint)
    
    return 0


if __name__ == '__main__':
    exit(main())
