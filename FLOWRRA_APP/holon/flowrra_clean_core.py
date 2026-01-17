"""
core.py - DOMAIN-AGNOSTIC VERSION

Clean orchestrator that works with ANY domain via feature extractors.

Key Changes:
1. Accepts FeatureExtractor protocol for state representation
2. Accepts RewardCalculator protocol for domain rewards
3. Dynamic GNN input sizing
4. Modular constraint handling
"""

from typing import Any, Callable, Dict, List, Optional, Protocol, Set

import numpy as np

from .agent import GNNAgent, build_adjacency_matrix
from .density import DensityFunctionEstimatorND
from .exploration import ExplorationMap
from .loop import LoopStructure
from .node import NodePositionND
from .obstacles import ObstacleManager
from .recovery import Wave_Function_Collapse


# =============================================================================
# PROTOCOL DEFINITIONS (Domain Abstraction)
# =============================================================================

class FeatureExtractor(Protocol):
    """Protocol for extracting state features from nodes."""
    
    def extract_features(
        self,
        node: NodePositionND,
        local_grid: np.ndarray,
        node_detections: List[Dict],
        constraint_detections: List[Dict],
        **domain_context
    ) -> np.ndarray:
        """
        Extract feature vector for a single node.
        
        Args:
            node: The node to extract features for
            local_grid: Local affordance/density grid
            node_detections: Detected neighboring nodes
            constraint_detections: Detected obstacles/constraints
            **domain_context: Additional domain-specific info
            
        Returns:
            Feature vector (1D numpy array)
        """
        ...


class RewardCalculator(Protocol):
    """Protocol for computing domain-specific rewards."""
    
    def compute_reward(
        self,
        node: NodePositionND,
        action: int,
        old_pos: np.ndarray,
        new_pos: np.ndarray,
        collided: bool,
        loop_integrity: float,
        **domain_context
    ) -> float:
        """
        Compute reward for a single node's action.
        
        Args:
            node: The node that acted
            action: Action taken
            old_pos: Position before action
            new_pos: Position after action
            collided: Whether collision occurred
            loop_integrity: Current loop integrity [0, 1]
            **domain_context: Domain-specific data
            
        Returns:
            Scalar reward value
        """
        ...


class ConstraintChecker(Protocol):
    """Protocol for checking domain-specific constraints."""
    
    def extract_constraints(self, **domain_state) -> Dict[str, Any]:
        """
        Extract constraints from current domain state.
        
        Returns:
            Dictionary with:
            - 'static': List of (pos, radius) static obstacles
            - 'dynamic': List of (pos, radius, velocity) moving obstacles
            - 'boundaries': World boundaries
            - 'custom': Domain-specific constraints
        """
        ...
    
    def check_violation(
        self,
        node: NodePositionND,
        **domain_state
    ) -> Optional[Dict[str, Any]]:
        """
        Check if node violates any domain constraints.
        
        Returns:
            Violation info dict if violated, None otherwise
        """
        ...


# =============================================================================
# DEFAULT IMPLEMENTATIONS (For backward compatibility)
# =============================================================================

class DefaultFeatureExtractor:
    """Default feature extractor (warehouse-like behavior)."""
    
    def __init__(self, include_velocity: bool = True, include_orientation: bool = True):
        self.include_velocity = include_velocity
        self.include_orientation = include_orientation
    
    def extract_features(
        self,
        node: NodePositionND,
        local_grid: np.ndarray,
        node_detections: List[Dict],
        constraint_detections: List[Dict],
        **domain_context
    ) -> np.ndarray:
        """Extract standard features."""
        features = []
        
        # 1. Position (normalized)
        features.extend(node.pos.tolist())
        
        # 2. Velocity
        if self.include_velocity:
            features.extend(node.velocity().tolist())
        
        # 3. Orientation (if applicable)
        if self.include_orientation and hasattr(node, 'azimuth_idx'):
            features.append(node.azimuth_idx / 16.0)  # Normalize
        
        # 4. Local grid (flattened)
        features.extend(local_grid.flatten().tolist())
        
        # 5. Detection counts
        features.append(len(node_detections) / 10.0)  # Normalize
        features.append(len(constraint_detections) / 10.0)
        
        return np.array(features, dtype=np.float32)


class DefaultRewardCalculator:
    """Default reward calculator (exploration + loop integrity)."""
    
    def __init__(self, config: Dict[str, Any]):
        self.cfg = config['rewards']
    
    def compute_reward(
        self,
        node: NodePositionND,
        action: int,
        old_pos: np.ndarray,
        new_pos: np.ndarray,
        collided: bool,
        loop_integrity: float,
        **domain_context
    ) -> float:
        """Compute standard reward."""
        reward = 0.0
        
        # Movement reward
        move_mag = np.linalg.norm(new_pos - old_pos)
        if not collided:
            reward += self.cfg['r_flow'] * move_mag
        
        # Collision penalty
        if collided:
            reward -= self.cfg['r_collision']
        
        # Idle penalty
        if move_mag < 0.001:
            reward -= self.cfg['r_idle']
        
        # Loop integrity bonus
        reward += self.cfg['r_loop_integrity'] * loop_integrity
        
        # Collapse penalty
        if loop_integrity < 0.7:
            reward -= self.cfg['r_collapse_penalty']
        
        return reward


# =============================================================================
# CLEAN ORCHESTRATOR
# =============================================================================

class FLOWRRA_Orchestrator:
    """
    Domain-agnostic FLOWRRA orchestrator.
    
    Uses dependency injection for domain-specific logic.
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        feature_extractor: Optional[FeatureExtractor] = None,
        reward_calculator: Optional[RewardCalculator] = None,
        constraint_checker: Optional[ConstraintChecker] = None,
        mode: str = "training"
    ):
        self.cfg = config
        self.dims = config['spatial']['dimensions']
        self.mode = mode
        
        # Inject dependencies (or use defaults)
        self.feature_extractor = feature_extractor or DefaultFeatureExtractor()
        self.reward_calculator = reward_calculator or DefaultRewardCalculator(config)
        self.constraint_checker = constraint_checker
        
        # Core components
        self.total_steps = config['training']['steps_per_episode']
        
        self.map = ExplorationMap(
            config['spatial']['world_bounds'],
            config['exploration']['map_resolution']
        )
        
        self.density = DensityFunctionEstimatorND(
            dimensions=self.dims,
            local_grid_size=config['repulsion']['local_grid_size'],
            global_grid_shape=config['repulsion']['global_grid_shape']
        )
        
        self.wfc = Wave_Function_Collapse(
            history_length=config['wfc']['history_length'],
            tail_length=config['wfc']['tail_length'],
            collapse_threshold=config['wfc']['collapse_threshold'],
            tau=config['wfc']['tau'],
            global_grid_shape=config['repulsion']['global_grid_shape'],
            local_grid_size=config['repulsion']['local_grid_size']
        )
        
        self.obstacle_manager = ObstacleManager(dimensions=self.dims)
        
        self.loop = LoopStructure(
            ideal_distance=config['loop']['ideal_distance'],
            stiffness=config['loop']['stiffness'],
            break_threshold=config['loop']['break_threshold'],
            dimensions=self.dims
        )
        
        # Initialize nodes (will be set externally)
        self.nodes: List[NodePositionND] = []
        
        # Calculate GNN input dimension dynamically
        input_dim = self._calculate_input_dim()
        
        action_size = 4 if self.dims == 2 else 6
        
        self.gnn = GNNAgent(
            node_feature_dim=input_dim,
            edge_feature_dim=0,
            action_size=action_size,
            hidden_dim=config['gnn']['hidden_dim'],
            lr=config['gnn'].get('lr', 0.0003),
            gamma=config['gnn'].get('gamma', 0.95),
            stability_coef=config['gnn'].get('stability_coef', 0.5)
        )
        
        # Frozen node management
        self.frozen_nodes: Set[int] = set()
        self.frozen_events: List[Dict] = []
        
        # State tracking
        self.history = []
        self.current_episode = 0
        self.step_count = 0
        self.last_state = None
        
        # Metrics
        self.total_reward = 0.0
        self.episode_rewards = []
        self.metrics_history = []
        self.training_losses = []
        
        # Event tracking
        self.loop_break_events = []
        self.wfc_trigger_events = []
        self.collision_events = []
        self.total_reconnections = 0
        
        # Actions storage for BenchMARL
        self._last_actions = None
        self._action_history = []
    
    def _calculate_input_dim(self) -> int:
        """
        Dynamically calculate GNN input dimension.
        
        Creates a dummy node and extracts features to determine size.
        """
        # Create dummy node
        dummy_node = NodePositionND(0, np.zeros(self.dims), self.dims)
        dummy_node.sensor_range = self.cfg['exploration']['sensor_range']
        
        # Create dummy inputs
        dummy_grid = np.zeros(np.prod(self.cfg['repulsion']['local_grid_size']))
        dummy_detections = []
        
        # Extract features
        features = self.feature_extractor.extract_features(
            dummy_node,
            dummy_grid,
            dummy_detections,
            dummy_detections
        )
        
        return len(features)
    
    # =========================================================================
    # NODE FREEZING (Domain-agnostic)
    # =========================================================================
    
    def freeze_node(self, node_id: int, reason: str = "mission_complete"):
        """Freeze a node - it becomes a static landmark."""
        node = next((n for n in self.nodes if n.id == node_id), None)
        if node is None or node_id in self.frozen_nodes:
            return
        
        self.frozen_nodes.add(node_id)
        self.gnn.freeze_node(node_id, node.pos)
        self._repatch_loop_around_frozen_nodes()
        
        self.frozen_events.append({
            'timestep': self.step_count,
            'episode': self.current_episode,
            'node_id': node_id,
            'position': node.pos.copy(),
            'reason': reason
        })
        
        print(f"[Orchestrator] 🧊 NODE {node_id} FROZEN ({reason})")
    
    def unfreeze_node(self, node_id: int):
        """Unfreeze a node."""
        if node_id not in self.frozen_nodes:
            return
        
        self.frozen_nodes.remove(node_id)
        self.gnn.unfreeze_node(node_id)
        self._repatch_loop_around_frozen_nodes()
        
        print(f"[Orchestrator] 🔥 NODE {node_id} UNFROZEN")
    
    def get_active_nodes(self) -> List[NodePositionND]:
        """Get active (non-frozen) nodes."""
        return [n for n in self.nodes if n.id not in self.frozen_nodes]
    
    def get_frozen_nodes(self) -> List[NodePositionND]:
        """Get frozen nodes."""
        return [n for n in self.nodes if n.id in self.frozen_nodes]
    
    def _repatch_loop_around_frozen_nodes(self):
        """Rebuild loop connections, skipping frozen nodes."""
        active_nodes = self.get_active_nodes()
        
        if len(active_nodes) < 3:
            return
        
        self.loop.connections.clear()
        
        from .loop import Connection
        
        for i in range(len(active_nodes)):
            node_a = active_nodes[i]
            node_b = active_nodes[(i + 1) % len(active_nodes)]
            
            conn = Connection(
                node_a_id=node_a.id,
                node_b_id=node_b.id,
                ideal_distance=self.loop.ideal_distance
            )
            self.loop.connections.append(conn)
    
    # =========================================================================
    # MAIN STEP FUNCTION (Clean)
    # =========================================================================
    
    def step(self, episode_step: int, total_episodes: int = 1000) -> float:
        """
        Execute one simulation step (domain-agnostic).
        
        Returns:
            Average reward across all nodes
        """
        # 1. Update obstacles
        self.obstacle_manager.update_all()
        
        # 2. Check loop breaks (active nodes only)
        broken = self.loop.check_breaks(
            self.get_active_nodes(),
            self.obstacle_manager,
            self.step_count
        )
        
        # 3. Attempt reconnections
        reconnected = self.loop.attempt_reconnection(
            self.get_active_nodes(),
            self.obstacle_manager,
            self.step_count
        )
        
        reconnection_bonus = len(reconnected) * self.cfg['rewards'].get('r_reconnection', 5.0)
        
        # 4. Calculate spring forces (active nodes only)
        spring_forces = self.loop.calculate_spring_forces(self.get_active_nodes())
        
        # 5. Update density field (all nodes)
        self.density.update_from_sensor_data(
            all_nodes=self.nodes,
            all_obstacle_states=self.obstacle_manager.get_all_states()
        )
        
        # 6. Build state representations
        node_features, local_grids, node_ids = self._build_state_representations()
        
        adj_mat = build_adjacency_matrix(
            self.nodes,
            self.cfg['exploration']['sensor_range']
        )
        
        # 7. GNN action selection
        actions = self.gnn.choose_actions(
            node_features=node_features,
            adj_matrix=adj_mat,
            episode_number=self.current_episode,
            total_episodes=total_episodes,
            node_ids=node_ids
        )
        
        # Store actions for BenchMARL
        self._last_actions = actions.copy() if actions is not None else np.zeros(len(self.nodes))
        self._action_history.append({
            'timestep': self.step_count,
            'actions': self._last_actions.copy()
        })
        if len(self._action_history) > 100:
            self._action_history.pop(0)
        
        # 8. Execute actions and compute rewards
        step_rewards = self._execute_actions_and_get_rewards(
            actions,
            spring_forces,
            episode_step,
            total_episodes
        )
        
        # Add reconnection bonus
        if reconnection_bonus > 0:
            step_rewards += reconnection_bonus
        
        # 9. Update exploration map
        new_coverage_diff = self.map.update(self.nodes)
        step_rewards += new_coverage_diff * self.cfg['rewards']['r_explore']
        
        # 10. Store experience (if training)
        training_loss = self._store_experience_and_learn(
            node_features, adj_mat, actions, step_rewards, node_ids
        )
        
        # 11. WFC recovery if needed
        loop_integrity = self.loop.calculate_integrity()
        self._check_and_recover_if_needed(
            loop_integrity, step_rewards, local_grids
        )
        
        # 12. Record metrics
        self._record_metrics(step_rewards, loop_integrity, training_loss)
        
        # 13. Update tracking
        self.step_count += 1
        avg_reward = float(np.mean(step_rewards)) if len(step_rewards) > 0 else 0.0
        self.total_reward += avg_reward
        
        return avg_reward
    
    def _build_state_representations(self):
        """Build state representations using feature extractor."""
        node_features = []
        local_grids = []
        node_ids = [n.id for n in self.nodes]
        
        for node in self.nodes:
            # Get detections
            node_detections = node.sense_nodes(self.nodes)
            constraint_detections = node.sense_obstacles(
                self.obstacle_manager.get_all_states()
            )
            
            # Get affordance field
            repulsion_sources = node_detections + constraint_detections
            local_grid = self.density.get_affordance_potential_for_node(
                node_pos=node.pos,
                repulsion_sources=repulsion_sources
            )
            local_grids.append(local_grid)
            
            # Extract features using injected extractor
            features = self.feature_extractor.extract_features(
                node=node,
                local_grid=local_grid,
                node_detections=node_detections,
                constraint_detections=constraint_detections
            )
            
            node_features.append(features)
        
        return np.array(node_features, dtype=np.float32), local_grids, node_ids
    
    def _execute_actions_and_get_rewards(
        self,
        actions: np.ndarray,
        spring_forces: Dict[int, np.ndarray],
        episode_step: int,
        total_episodes: int
    ) -> np.ndarray:
        """Execute actions and compute rewards using reward calculator."""
        step_rewards = []
        
        for i, node in enumerate(self.nodes):
            # Frozen nodes don't move
            if node.id in self.frozen_nodes:
                step_rewards.append(0.0)
                continue
            
            action_id = actions[i] if actions is not None else 0
            old_pos = node.pos.copy()
            
            # Apply action
            node.apply_directional_action(action_id, dt=1.0)
            
            # Apply spring force
            if node.id in spring_forces:
                node.pos = node.pos + spring_forces[node.id] * 0.1
            
            # Check collision
            collided, _ = self.obstacle_manager.check_collision(
                node.pos, safety_margin=0.05
            )
            
            if collided:
                # Micro-WFC recovery
                node.pos = self._micro_wfc_escape(node, old_pos)
            
            # Compute reward using injected calculator
            loop_integrity = self.loop.calculate_integrity()
            
            reward = self.reward_calculator.compute_reward(
                node=node,
                action=action_id,
                old_pos=old_pos,
                new_pos=node.pos,
                collided=collided,
                loop_integrity=loop_integrity
            )
            
            step_rewards.append(reward)
        
        return np.array(step_rewards, dtype=np.float32)
    
    def _micro_wfc_escape(self, node: NodePositionND, old_pos: np.ndarray) -> np.ndarray:
        """Micro-WFC collision escape (domain-agnostic)."""
        best_pos = old_pos.copy()
        best_dist = 0.0
        
        search_radius = 0.04
        
        for attempt in range(16):
            angle = (attempt / 16) * 2 * np.pi
            
            if node.dimensions == 2:
                offset = np.array([np.cos(angle), np.sin(angle)]) * search_radius
            else:
                elevation = np.random.uniform(-search_radius, search_radius)
                offset = np.array([
                    np.cos(angle) * search_radius,
                    np.sin(angle) * search_radius,
                    elevation
                ])
            
            candidate = old_pos + offset
            
            coll_check, _ = self.obstacle_manager.check_collision(
                candidate, safety_margin=0.03
            )
            
            if not coll_check:
                min_dist = min(
                    np.linalg.norm(candidate - obs.pos) - obs.radius
                    for obs in self.obstacle_manager.obstacles
                )
                if min_dist > best_dist:
                    best_pos = candidate
                    best_dist = min_dist
        
        return best_pos
    
    def _store_experience_and_learn(
        self,
        node_features: np.ndarray,
        adj_mat: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        node_ids: List[int]
    ) -> Optional[float]:
        """Store experience and train GNN."""
        if self.mode != 'training':
            return None
        
        training_loss = None
        loop_integrity = self.loop.calculate_integrity()
        
        if self.last_state is not None:
            last_features, last_adj, last_actions = self.last_state
            
            self.gnn.memory.push(
                node_features=last_features,
                adj_matrix=last_adj,
                actions=last_actions,
                rewards=np.clip(rewards, -50.0, 50.0) / 10.0,
                next_node_features=node_features,
                next_adj_matrix=adj_mat,
                done=False,
                integrity=float(loop_integrity)
            )
            
            if len(self.gnn.memory) >= self.gnn.batch_size:
                training_loss = self.gnn.learn(node_ids=node_ids)
                self.training_losses.append({
                    'timestep': self.step_count,
                    'loss': float(training_loss)
                })
        
        self.last_state = (node_features, adj_mat, actions)
        
        # Update target network periodically
        if self.step_count % 100 == 0:
            self.gnn.update_target_network()
        
        return training_loss
    
    def _check_and_recover_if_needed(
        self,
        loop_integrity: float,
        step_rewards: np.ndarray,
        local_grids: List[np.ndarray]
    ):
        """Check WFC trigger and recover if needed."""
        current_coherence = np.clip(loop_integrity, 0.0, 1.0)
        
        self.wfc.assess_loop_coherence(
            current_coherence,
            self.nodes,
            loop_integrity
        )
        
        if self.wfc.needs_recovery() or loop_integrity < 0.5:
            recovery_info = self.wfc.collapse_and_reinitialize(
                nodes=self.nodes,
                local_grids=local_grids,
                ideal_dist=self.cfg['loop']['ideal_distance'],
                config=self.cfg
            )
            
            self.loop.repair_all_connections()
            
            self.wfc_trigger_events.append({
                'timestep': self.step_count,
                'coherence': current_coherence,
                'loop_integrity': loop_integrity,
                'recovery_info': recovery_info
            })
    
    def _record_metrics(
        self,
        step_rewards: np.ndarray,
        loop_integrity: float,
        training_loss: Optional[float]
    ):
        """Record performance metrics."""
        metrics = {
            'timestep': self.step_count,
            'episode': self.current_episode,
            'training_loss': float(training_loss) if training_loss else None,
            'avg_reward': float(np.mean(step_rewards)),
            'loop_integrity': loop_integrity,
            'coverage': self.map.get_coverage_percentage(),
            'num_active_nodes': len(self.get_active_nodes()),
            'num_frozen_nodes': len(self.frozen_nodes)
        }
        
        self.metrics_history.append(metrics)
    
    # =========================================================================
    # ACCESSORS (For BenchMARL)
    # =========================================================================
    
    def get_last_actions(self) -> np.ndarray:
        """Get last actions for BenchMARL adapter."""
        if self._last_actions is None:
            return np.zeros(len(self.nodes), dtype=np.int64)
        return self._last_actions.copy()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        coverage = self.map.get_coverage_percentage()
        loop_stats = self.loop.get_statistics()
        
        return {
            'step': self.step_count,
            'mode': self.mode,
            'coverage': coverage,
            'avg_reward': self.total_reward / max(1, self.step_count),
            'num_total_nodes': len(self.nodes),
            'num_active_nodes': len(self.get_active_nodes()),
            'num_frozen_nodes': len(self.frozen_nodes),
            'loop_integrity': loop_stats['current_integrity'],
            'total_loop_breaks': loop_stats['total_breaks_occurred'],
            'wfc_triggers': len(self.wfc_trigger_events)
        }
    
    def save_metrics(self, filepath: str):
        """Save metrics to JSON."""
        import json
        
        def convert_to_serializable(obj):
            if isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            else:
                return obj
        
        data = {
            'final_statistics': convert_to_serializable(self.get_statistics()),
            'metrics_timeseries': convert_to_serializable(self.metrics_history),
            'training_losses': convert_to_serializable(self.training_losses)
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"[Metrics] Saved to {filepath}")
