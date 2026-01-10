"""
FLOWRRA_GNN.py

Main orchestrator for GNN-based FLOWRRA system.
Supports both 2D and 3D with configurable hyperparameters.
"""
import logging
import csv
import os
import numpy as np
import time
from typing import List, Dict, Any, Tuple, Optional

from NodePositionND import NodePositionND
from DensityFunctionEstimatorND import DensityFunctionEstimatorND
from GNNAgent import GNNAgent, build_adjacency_matrix
from WaveFunctionCollapse_RL import Wave_Function_Collapse  # Reuse from original

logger = logging.getLogger("FLOWRRA_GNN")
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(h)
logger.setLevel(logging.INFO)

class FLOWRRA_GNN:
    """
    GNN-based FLOWRRA orchestrator with N-dimensional support.
    
    Key improvements over Q-learning version:
    - Scales to 100+ nodes
    - Permutation invariant
    - Graph-structured learning
    - Configurable 2D/3D
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize FLOWRRA with configuration dictionary.
        
        config structure:
        {
            'spatial': {...},
            'repulsion': {...},
            'node': {...},
            'gnn': {...},
            'training': {...},
            'viz': {...}
        }
        """
        self.config = config
        
        # Extract config sections
        spatial_cfg = config['spatial']
        repulsion_cfg = config['repulsion']
        node_cfg = config['node']
        gnn_cfg = config['gnn']
        train_cfg = config['training']
        viz_cfg = config['viz']
        
        self.dimensions = spatial_cfg['dimensions']
        self.num_nodes = node_cfg['num_nodes']
        self.sensor_range = node_cfg['sensor_range']
        
        # Initialize nodes
        self.nodes: List[NodePositionND] = []
        self._initialize_nodes(node_cfg, spatial_cfg)
        
        # Density estimator
        self.density_estimator = DensityFunctionEstimatorND(
            dimensions=self.dimensions,
            local_grid_size=repulsion_cfg['local_grid_size'],
            global_grid_shape=repulsion_cfg['global_grid_shape'],
            eta=repulsion_cfg['eta'],
            gamma_f=repulsion_cfg['gamma_f'],
            k_f=repulsion_cfg['k_f'],
            sigma_f=repulsion_cfg['sigma_f'],
            decay_lambda=repulsion_cfg['decay_lambda'],
            blur_delta=repulsion_cfg['blur_delta'],
            beta=repulsion_cfg['beta']
        )
        
        # Wave function collapse
        wfc_cfg = config.get('wfc', {})
        self.wfc = Wave_Function_Collapse(
            history_length=wfc_cfg.get('history_length', 200),
            tail_length=wfc_cfg.get('tail_length', 15),
            collapse_threshold=wfc_cfg.get('collapse_threshold', 0.88),
            tau=wfc_cfg.get('tau', 2)
        )
        
        # Compute dimensions for GNN
        self.node_feature_dim = self._compute_node_feature_dim(repulsion_cfg)
        self.edge_feature_dim = gnn_cfg['edge_feature_dim']
        
        # Action space size
        if self.dimensions == 2:
            self.action_size = 4 * 4  # 4 directions × 4 rotation actions
        else:
            self.action_size = 6 * 6  # 6 directions × 6 rotation actions (includes elevation)
        
        # GNN agent (initialized later via attach_agent)
        self.agent: Optional[GNNAgent] = None
        
        # Logging
        self.log_file = viz_cfg['log_file']
        self.visual_dir = viz_cfg['visual_dir']
        self.t = 0
        
        # Track previous positions for movement rewards
        self.prev_positions = None
        
        logger.info(f"FLOWRRA-GNN initialized: {self.dimensions}D, {self.num_nodes} nodes")
    
    def _initialize_nodes(self, node_cfg: Dict, spatial_cfg: Dict):
        """Initialize nodes with random positions."""
        self.nodes = []
        dims = spatial_cfg['dimensions']
        
        for i in range(node_cfg['num_nodes']):
            pos = np.random.rand(dims)
            
            node = NodePositionND(
                id=i,
                pos=pos,
                dimensions=dims,
                azimuth_idx=np.random.randint(0, node_cfg['azimuth_steps']),
                elevation_idx=np.random.randint(0, node_cfg.get('elevation_steps', 1)),
                azimuth_steps=node_cfg['azimuth_steps'],
                elevation_steps=node_cfg.get('elevation_steps', 1),
                rotation_speed=node_cfg['rotation_speed'],
                move_speed=node_cfg['move_speed'],
                sensor_range=node_cfg['sensor_range'],
                world_bounds=spatial_cfg['world_bounds']
            )
            self.nodes.append(node)
        
        self.prev_positions = np.array([n.pos.copy() for n in self.nodes])
    
    def _compute_node_feature_dim(self, repulsion_cfg: Dict) -> int:
        """
        Compute the dimensionality of node feature vectors.
        
        Features:
        - position (N)
        - velocity (N)
        - orientation (1 for 2D, 2 for 3D)
        - neighbor detections (max_neighbors × features_per_detection)
        - obstacle detections (max_obstacles × features_per_detection)
        - local repulsion grid (flattened)
        """
        dims = self.dimensions
        max_neighbors = 5
        max_obstacles = 5
        
        # Detection features: distance(1) + bearing_az(1) + bearing_el(1) + velocity(N)
        features_per_detection = 3 + dims
        
        local_grid_size = repulsion_cfg['local_grid_size']
        grid_elements = np.prod(local_grid_size)
        
        orientation_dim = 1 if dims == 2 else 2
        
        total_dim = (
            dims +  # position
            dims +  # velocity
            orientation_dim +  # orientation
            max_neighbors * features_per_detection +  # neighbor detections
            max_obstacles * features_per_detection +  # obstacle detections
            grid_elements  # local repulsion grid
        )
        
        logger.info(f"Node feature dimension: {total_dim}")
        return total_dim
    
    def reset(self):
        """Reset environment for new episode."""
        spatial_cfg = self.config['spatial']
        node_cfg = self.config['node']
        
        self._initialize_nodes(node_cfg, spatial_cfg)
        self.density_estimator.reset()
        self.wfc.reset()
        self.t = 0
        
        self.prev_positions = np.array([n.pos.copy() for n in self.nodes])
    
    def get_state(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get current state as graph structure.
        
        Returns:
            node_features: [num_nodes, node_feature_dim]
            adj_matrix: [num_nodes, num_nodes]
        """
        # Placeholder obstacles (extend this with EnvironmentB later)
        obstacle_states = []
        
        node_features_list = []
        
        for node in self.nodes:
            # Get detections
            node_detections = node.sense_nodes(self.nodes)
            obstacle_detections = node.sense_obstacles(obstacle_states)
            
            # Compute local repulsion
            all_detections = node_detections + obstacle_detections
            local_repulsion = self.density_estimator.get_repulsion_potential_for_node(
                node_pos=node.pos,
                repulsion_sources=all_detections
            )
            
            # Build node feature vector
            node_state = node.get_state_vector(
                local_repulsion_grid=local_repulsion,
                node_detections=node_detections,
                obstacle_detections=obstacle_detections,
                max_neighbors=5,
                max_obstacles=5
            )
            
            node_features_list.append(node_state)
        
        node_features = np.array(node_features_list, dtype=np.float32)
        
        # Build adjacency matrix
        adj_matrix = build_adjacency_matrix(self.nodes, self.sensor_range)
        
        return node_features, adj_matrix
    
    def calculate_coherence(self) -> float:
        """Calculate system coherence based on repulsion."""
        obstacle_states = []
        coherences = []
        
        for node in self.nodes:
            all_detections = node.sense_nodes(self.nodes) + \
                           node.sense_obstacles(obstacle_states)
            
            local_repulsion = self.density_estimator.get_repulsion_potential_for_node(
                node_pos=node.pos,
                repulsion_sources=all_detections
            )
            
            # Sample repulsion at node's position
            center_idx = tuple(np.array(local_repulsion.shape) // 2)
            repulsion_at_node = local_repulsion[center_idx]
            
            coherence = 1.0 / (1.0 + repulsion_at_node)
            coherences.append(coherence)
        
        return np.mean(coherences)
    
    def calculate_exploration_reward(self) -> float:
        """Reward movement and spatial diversity."""
        if self.prev_positions is None:
            return 0.0
        
        current_positions = np.array([n.pos.copy() for n in self.nodes])
        
        # Movement reward
        movements = []
        for i in range(self.num_nodes):
            delta = current_positions[i] - self.prev_positions[i]
            toroidal_delta = np.mod(delta + 0.5, 1.0) - 0.5
            movement = np.linalg.norm(toroidal_delta)
            movements.append(movement)
        
        movement_reward = np.mean(movements) * self.config['training']['movement_weight']
        
        # Spread reward (encourage spatial diversity)
        pairwise_distances = []
        for i in range(self.num_nodes):
            for j in range(i+1, self.num_nodes):
                delta = current_positions[i] - current_positions[j]
                toroidal_delta = np.mod(delta + 0.5, 1.0) - 0.5
                dist = np.linalg.norm(toroidal_delta)
                pairwise_distances.append(dist)
        
        avg_dist = np.mean(pairwise_distances)
        optimal_dist = 0.3
        spread_reward = max(0.0, 1.0 - abs(avg_dist - optimal_dist))
        
        self.prev_positions = current_positions.copy()
        
        return movement_reward + spread_reward * 1.5
    
    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, bool, Dict[str, Any]]:
        """
        Execute one simulation step.
        
        Args:
            actions: [num_nodes] combined actions (direction + rotation)
        
        Returns:
            rewards: [num_nodes] rewards
            done: episode termination flag
            info: diagnostic information
        """
        info = {'wfc_reinit': 'none'}
        
        # Decode combined actions
        if self.dimensions == 2:
            direction_actions = actions // 4
            rotation_actions = actions % 4
        else:
            direction_actions = actions // 6
            rotation_actions = actions % 6
        
        # Apply direction actions
        for node, dir_act in zip(self.nodes, direction_actions):
            node.apply_directional_action(int(dir_act))
        
        # Update density field
        obstacle_states = []
        self.density_estimator.update_from_sensor_data(
            all_nodes=self.nodes,
            all_obstacle_states=obstacle_states
        )
        
        coherence_after_move = self.calculate_coherence()
        exploration_reward = self.calculate_exploration_reward()
        
        # Check for collapse
        if coherence_after_move < self.wfc.collapse_threshold:
            rewards = np.full(self.num_nodes, self.config['training']['collapse_penalty'])
            info['wfc_reinit'] = 'collapse_after_move'
            info['exploration_reward'] = 0.0
            return rewards, False, info
        
        # Apply rotation actions
        for node, rot_act in zip(self.nodes, rotation_actions):
            node.apply_rotation_action(int(rot_act))
        
        # Final coherence
        self.density_estimator.update_from_sensor_data(
            all_nodes=self.nodes,
            all_obstacle_states=obstacle_states
        )
        
        final_coherence = self.calculate_coherence()
        
        # Compute rewards
        if final_coherence < self.wfc.collapse_threshold:
            rewards = np.full(self.num_nodes, self.config['training']['collapse_penalty'])
            info['wfc_reinit'] = 'collapse_after_rotation'
            info['exploration_reward'] = 0.0
        else:
            coherence_reward = (coherence_after_move + final_coherence) * \
                             self.config['training']['coherence_weight']
            total_reward = coherence_reward + exploration_reward + 0.1
            rewards = np.full(self.num_nodes, total_reward)
            info['exploration_reward'] = exploration_reward
        
        self.t += 1
        done = False
        
        return rewards, done, info
    
    def attach_agent(self, agent: GNNAgent):
        """Attach GNN agent."""
        self.agent = agent
    
    def train(self, total_steps: int, episode_steps: int, 
             visualize_every_n_steps: int, agent: GNNAgent):
        """Train the GNN agent."""
        self.attach_agent(agent)
        
        os.makedirs(self.visual_dir, exist_ok=True)
        
        # Initialize log file
        log_header = ['step', 'episode', 'total_reward', 'avg_coherence', 
                     'exploration_reward', 'loss', 'wfc_reinit']
        
        with open(self.log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(log_header)
        
        logger.info("=== Starting FLOWRRA-GNN Training ===")
        self.reset()
        
        for step in range(total_steps):
            episode_num = step // episode_steps
            
            if step % episode_steps == 0:
                self.reset()
            
            # Get state
            node_features, adj_matrix = self.get_state()
            
            # Choose actions
            actions = self.agent.choose_actions(
                node_features=node_features,
                adj_matrix=adj_matrix,
                episode_number=episode_num,
                total_episodes=total_steps // episode_steps
            )
            
            # Execute step
            rewards, done, info = self.step(actions)
            
            # Get next state
            next_node_features, next_adj_matrix = self.get_state()
            
            # Store transition
            self.agent.memory.push(
                node_features=node_features,
                adj_matrix=adj_matrix,
                actions=actions,
                rewards=rewards,
                next_node_features=next_node_features,
                next_adj_matrix=next_adj_matrix,
                done=done
            )
            
            # Learn
            loss = 0.0
            if len(self.agent.memory) >= self.agent.batch_size:
                loss = self.agent.learn()
                
                if step % self.config['training'].get('target_update_freq', 100) == 0:
                    self.agent.update_target_network()
            
            # Logging
            total_reward = np.sum(rewards)
            avg_coherence = self.calculate_coherence()
            exploration_reward = info.get('exploration_reward', 0.0)
            
            with open(self.log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([step, episode_num, total_reward, avg_coherence,
                               exploration_reward, loss, info['wfc_reinit']])
            
            if step % 50 == 0:
                logger.info(f"Step {step} | Ep {episode_num} | "
                          f"Coherence: {avg_coherence:.4f} | "
                          f"Exploration: {exploration_reward:.4f} | "
                          f"Loss: {loss:.4f}")
            
            # Visualization (placeholder - implement based on dimension)
            if step % visualize_every_n_steps == 0:
                self._visualize(step)
        
        logger.info("=== Training Complete ===")
    
    def _visualize(self, step: int):
        """Visualization placeholder."""
        # Implement 2D/3D visualization based on self.dimensions
        pass
    
    def deploy(self, total_steps: int = 50, visualize_every_n_steps: int = 1,
              num_episodes: int = 3):
        """Deploy trained agent."""
        if self.agent is None:
            raise ValueError("Agent not attached!")
        
        logger.info("=== Starting Deployment ===")
        
        for ep in range(num_episodes):
            self.reset()
            
            for step in range(total_steps):
                node_features, adj_matrix = self.get_state()
                actions = self.agent.choose_actions(
                    node_features, adj_matrix,
                    episode_number=ep, total_episodes=num_episodes,
                    eps_peak=0.3  # Lower exploration
                )
                
                rewards, done, info = self.step(actions)
                
                if step % visualize_every_n_steps == 0:
                    self._visualize(step)
                
                if done:
                    break
        
        logger.info("=== Deployment Complete ===")