"""
core.py

FLOWRRA Orchestrator - Main simulation loop with GNN agents.

FIXES:
- Added simple collision detection (distance-based)
- Fixed WFC method calls with correct parameters
- Improved reward calculation
- Added replay buffer population
- Added target network updates
"""
import numpy as np
from typing import List

from .config import CONFIG
from .node import NodePositionND
from .density import DensityFunctionEstimatorND
from .exploration import ExplorationMap
from .agent import GNNAgent, build_adjacency_matrix
from .recovery import Wave_Function_Collapse


class FLOWRRA_Orchestrator:
    """
    Main orchestrator for FLOWRRA exploration swarm.
    
    Manages:
    - Node initialization and physics
    - Density field computation
    - GNN policy execution
    - Exploration map updates
    - Wave function collapse recovery
    """
    
    def __init__(self):
        self.cfg = CONFIG
        self.dims = self.cfg['spatial']['dimensions']
        
        # Components
        self.map = ExplorationMap(
            self.cfg['spatial']['world_bounds'], 
            self.cfg['exploration']['map_resolution']
        )
        
        self.density = DensityFunctionEstimatorND(
            dimensions=self.dims,
            local_grid_size=self.cfg['repulsion']['local_grid_size'],
            global_grid_shape=self.cfg['repulsion']['global_grid_shape']
        )
        
        self.wfc = Wave_Function_Collapse()
        
        # Init Nodes
        self.nodes = [
            NodePositionND(i, np.random.rand(self.dims), self.dims) 
            for i in range(self.cfg['node']['num_nodes'])
        ]
        
        # Set sensor range for all nodes
        for node in self.nodes:
            node.sensor_range = self.cfg['exploration']['sensor_range']
            node.move_speed = self.cfg['node']['move_speed']
        
        # Calculate GNN input dimension
        input_dim = self._calculate_input_dim()
        
        # GNN Setup
        action_size = 4 if self.dims == 2 else 6  # Directional actions
        
        self.gnn = GNNAgent(
            node_feature_dim=input_dim,
            edge_feature_dim=0,
            action_size=action_size,
            hidden_dim=self.cfg['gnn']['hidden_dim'],
            lr=self.cfg['gnn'].get('lr', 0.0003),
            gamma=self.cfg['gnn'].get('gamma', 0.95),
        )
        
        # State tracking
        self.history = []
        self.current_episode = 0
        self.step_count = 0
        self.last_state = None  # For replay buffer
        
        # Performance tracking
        self.total_reward = 0.0
        self.episode_rewards = []
    
    def _calculate_input_dim(self) -> int:
        """Calculate the dimension of the GNN input vector."""
        dummy_rep = np.zeros(np.prod(self.cfg['repulsion']['local_grid_size']))
        dummy_detections = []
        return len(self.nodes[0].get_state_vector(dummy_rep, dummy_detections, dummy_detections))
    
    def calculate_coherence(self, rewards: np.ndarray) -> float:
        """
        Estimates flow coherence based on recent rewards.
        
        Positive rewards (exploration, movement) = High Coherence
        Negative rewards (collision, idle) = Low Coherence
        
        Args:
            rewards: Array of rewards for all nodes
            
        Returns:
            Coherence score between 0 and 1
        """
        mean_reward = np.mean(rewards)
        # Sigmoid squashing to [0, 1]
        coherence = 1.0 / (1.0 + np.exp(-mean_reward))
        return coherence
    
    def _check_collisions(self, node: NodePositionND) -> bool:
        """
        Simple collision detection based on distance to nearest neighbor.
        
        Args:
            node: Node to check for collisions
            
        Returns:
            True if collision detected, False otherwise
        """
        detections = node.sense_nodes(self.nodes)
        if not detections:
            return False
        
        # Find closest neighbor
        closest_dist = min([d['distance'] for d in detections])
        
        # Collision threshold (small distance)
        collision_threshold = 0.03
        return closest_dist < collision_threshold
    
    def step(self, episode_step: int, total_episodes: int = 5000) -> float:
        """
        Execute one simulation step.
        
        Args:
            episode_step: Current step number
            total_episodes: Total episodes for exploration schedule
            
        Returns:
            Average reward for this step
        """
        # --- 1. Density Update and Map Coverage ---
        self.density.update_from_sensor_data(all_nodes=self.nodes, all_obstacle_states=[])
        
        # --- 2. Build State Representations ---
        node_features = []
        adj_mat = build_adjacency_matrix(self.nodes, self.cfg['exploration']['sensor_range'])
        
        for node in self.nodes:
            # 2a. SENSE: Get raw detections
            node_detections = node.sense_nodes(self.nodes)
            obstacle_detections = node.sense_obstacles([])
            
            # 2b. LOCAL REPULSION: Get density field
            repulsion_sources = node_detections + obstacle_detections
            local_grid = self.density.get_repulsion_potential_for_node(
                node_pos=node.pos,
                repulsion_sources=repulsion_sources
            )
            
            # 2c. GNN STATE: Construct feature vector
            feats = node.get_state_vector(
                local_repulsion_grid=local_grid,
                node_detections=node_detections,
                obstacle_detections=obstacle_detections
            )
            node_features.append(feats)
        
        node_features_array = np.array(node_features, dtype=np.float32)
        
        # --- 3. GNN ACTION SELECTION ---
        actions = self.gnn.choose_actions(
            node_features=node_features_array,
            adj_matrix=adj_mat,
            episode_number=self.current_episode,
            total_episodes=total_episodes
        )
        
        # --- 4. Physics & Reward Calculation ---
        step_rewards = []
        
        for i, node in enumerate(self.nodes):
            action_id = actions[i]
            
            # Store old position for coverage calculation
            old_pos = node.pos.copy()
            
            # Apply action
            node.apply_directional_action(action_id, dt=1.0)
            
            # Calculate movement magnitude
            move_mag = np.linalg.norm(node.velocity())
            
            # Collision check
            has_collision = self._check_collisions(node)
            r_coll = -self.cfg['rewards']['r_collision'] if has_collision else 0.0
            
            # Movement reward (encouraging exploration)
            r_flow = self.cfg['rewards']['r_flow'] * move_mag
            
            # Idle penalty (if not moving)
            r_idle = -self.cfg['rewards']['r_idle'] if move_mag < 0.001 else 0.0
            
            # Total reward for this node
            reward = r_flow + r_coll + r_idle
            step_rewards.append(reward)
        
        step_rewards_array = np.array(step_rewards, dtype=np.float32)
        
        # Update map and calculate exploration reward
        new_coverage = self.map.update(self.nodes)
        r_explore = new_coverage * self.cfg['rewards']['r_explore']
        
        # Add exploration reward to all nodes (global reward)
        step_rewards_array += r_explore
        
        # --- 5. Store Experience in Replay Buffer ---
        if self.last_state is not None:
            last_features, last_adj, last_actions = self.last_state
            
            # Store transition
            self.gnn.memory.push(
                node_features=last_features,
                adj_matrix=last_adj,
                actions=last_actions,
                rewards=step_rewards_array,
                next_node_features=node_features_array,
                next_adj_matrix=adj_mat,
                done=False  # Exploration is continuous
            )
            
            # Learn from experience
            if len(self.gnn.memory) >= self.gnn.batch_size:
                loss = self.gnn.learn()
        
        # Store current state for next step
        self.last_state = (node_features_array, adj_mat, actions)
        
        # --- 6. Update Target Network Periodically ---
        if self.step_count % 100 == 0:
            self.gnn.update_target_network()
        
        # --- 7. WFC Safety Loop ---
        current_coherence = self.calculate_coherence(step_rewards_array)
        
        # Log state to WFC history
        self.wfc.assess_loop_coherence(current_coherence, self.nodes)
        
        # Trigger recovery if needed
        if self.wfc.needs_recovery():
            print(f"\n[!] Step {episode_step}: Coherence lost ({current_coherence:.2f}). WFC Triggered.")
            recovery_info = self.wfc.collapse_and_reinitialize(self.nodes)
            print(f"[WFC] Recovery: {recovery_info}")
            
            # Reset last_state after recovery
            self.last_state = None
        
        # --- 8. Record State ---
        self.record_state(episode_step)
        
        # Update tracking
        self.step_count += 1
        avg_reward = float(np.mean(step_rewards_array))
        self.total_reward += avg_reward
        
        return avg_reward
    
    def record_state(self, t: int):
        """Record current state for visualization."""
        snap = {
            "time": t,
            "nodes": [{"id": n.id, "pos": n.pos.tolist()} for n in self.nodes]
        }
        self.history.append(snap)
    
    def get_statistics(self) -> dict:
        """Get current simulation statistics."""
        coverage = self.map.get_coverage_percentage()
        
        return {
            'step': self.step_count,
            'coverage': coverage,
            'avg_reward': self.total_reward / max(1, self.step_count),
            'num_nodes': len(self.nodes),
            'buffer_size': len(self.gnn.memory)
        }