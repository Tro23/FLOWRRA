"""
agent.py - DOMAIN-AGNOSTIC VERSION

GNN agent with dynamic input sizing and clean architecture.

Key Changes:
1. Dynamic input dimension calculation
2. Removed domain-specific assumptions
3. Cleaner gradient masking for frozen nodes
4. Better error handling for variable batch sizes
"""

import math
import random
from collections import deque
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# GRAPH ATTENTION LAYER (Clean)
# =============================================================================

class GraphAttentionLayer(nn.Module):
    """Graph Attention layer - domain-agnostic."""
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_heads: int = 4,
        dropout: float = 0.1,
        concat: bool = True
    ):
        super().__init__()
        self.n_heads = n_heads
        self.concat = concat
        self.dropout = dropout
        self.out_features = out_features
        
        # Multi-head parameters
        self.W = nn.Parameter(torch.zeros(size=(in_features, n_heads * out_features)))
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.dropout_layer = nn.Dropout(dropout)
        
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.a)
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """Forward pass with proper attention aggregation."""
        B, N, F_in = x.shape
        
        # Linear transform
        h = torch.matmul(x, self.W)
        h = h.view(B, N, self.n_heads, self.out_features)
        
        # Attention scores
        h_i = h.unsqueeze(2).expand(B, N, N, self.n_heads, self.out_features)
        h_j = h.unsqueeze(1).expand(B, N, N, self.n_heads, self.out_features)
        
        concat_features = torch.cat([h_i, h_j], dim=-1)
        
        a_reshaped = self.a.view(1, 1, 1, 1, 2 * self.out_features, 1)
        
        e = torch.matmul(concat_features.unsqueeze(-2), a_reshaped).squeeze(-1).squeeze(-1)
        e = self.leakyrelu(e)
        
        # Mask
        mask = (adj.unsqueeze(-1) == 0).expand(B, N, N, self.n_heads)
        e = e.masked_fill(mask, float('-inf'))
        
        # Softmax
        alpha = F.softmax(e, dim=2)
        alpha = self.dropout_layer(alpha)
        
        # Aggregate
        alpha_transposed = alpha.permute(0, 3, 1, 2)
        h_transposed = h.permute(0, 2, 1, 3)
        
        h_prime = torch.matmul(alpha_transposed, h_transposed)
        h_prime = h_prime.permute(0, 2, 1, 3)
        
        if self.concat:
            return h_prime.reshape(B, N, self.n_heads * self.out_features)
        else:
            return h_prime.mean(dim=2)


# =============================================================================
# GNN POLICY (Dynamic Input Sizing)
# =============================================================================

class GNNPolicy(nn.Module):
    """
    Graph Neural Network with DYNAMIC input sizing.
    
    Automatically adapts to whatever feature dimension is provided.
    """
    
    def __init__(
        self,
        node_feature_dim: int,
        edge_feature_dim: int,
        action_size: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        n_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.node_feature_dim = node_feature_dim
        self.edge_feature_dim = edge_feature_dim
        self.action_size = action_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Flexible encoder - handles ANY feature dimension
        self.node_encoder = nn.Sequential(
            nn.Linear(node_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # GAT layers
        self.gat_layers = nn.ModuleList()
        for i in range(num_layers):
            in_dim = hidden_dim if i == 0 else hidden_dim * n_heads
            concat = i < num_layers - 1
            
            self.gat_layers.append(
                GraphAttentionLayer(
                    in_features=in_dim,
                    out_features=hidden_dim,
                    n_heads=n_heads,
                    dropout=dropout,
                    concat=concat
                )
            )
        
        # Final dimension calculation
        if num_layers == 0:
            final_dim = hidden_dim
        elif num_layers == 1 or not concat:
            final_dim = hidden_dim
        else:
            final_dim = hidden_dim * n_heads
        
        # Action head
        self.action_decoder = nn.Sequential(
            nn.Linear(final_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, action_size)
        )
        
        # Stability prediction head
        self.stability_head = nn.Sequential(
            nn.Linear(final_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(
        self,
        node_features: torch.Tensor,
        adj_matrix: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            node_features: [batch, num_nodes, feature_dim] - ANY feature_dim!
            adj_matrix: [batch, num_nodes, num_nodes]
        
        Returns:
            (q_values, stability_pred)
        """
        # Encode (handles dynamic input size)
        h = self.node_encoder(node_features)
        
        # Message passing
        for i, gat in enumerate(self.gat_layers):
            h = gat(h, adj_matrix)
            if i < len(self.gat_layers) - 1:
                h = F.elu(h)
        
        # Action Q-values
        q_values = self.action_decoder(h)
        
        # Stability prediction
        h_graph = torch.mean(h, dim=1)
        stability_pred = self.stability_head(h_graph)
        
        return q_values, stability_pred


# =============================================================================
# REPLAY BUFFER (Robust to variable batch sizes)
# =============================================================================

class GraphReplayBuffer:
    """Replay buffer with robust batching."""
    
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity
    
    def push(
        self,
        node_features: np.ndarray,
        adj_matrix: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_node_features: np.ndarray,
        next_adj_matrix: np.ndarray,
        done: bool,
        integrity: float
    ):
        """Store transition."""
        self.buffer.append((
            node_features.copy(),
            adj_matrix.copy(),
            actions.copy(),
            rewards.copy(),
            next_node_features.copy(),
            next_adj_matrix.copy(),
            bool(done),
            integrity
        ))
    
    def sample(self, batch_size: int) -> Optional[Tuple[torch.Tensor, ...]]:
        """
        Sample batch with uniform node counts.
        
        Returns None if insufficient valid samples.
        """
        # Find most common node count
        recent = list(self.buffer)[-min(500, len(self.buffer)):]
        node_counts = [exp[0].shape[0] for exp in recent]
        
        if not node_counts:
            return None
        
        from collections import Counter
        target_count = Counter(node_counts).most_common(1)[0][0]
        
        # Filter valid experiences
        valid = [exp for exp in self.buffer if exp[0].shape[0] == target_count]
        
        if len(valid) < batch_size:
            return None
        
        # Sample
        batch = random.sample(valid, batch_size)
        
        (node_feats, adjs, actions, rewards,
         next_node_feats, next_adjs, dones, integrity) = zip(*batch)
        
        # Convert to tensors
        node_feats_t = torch.from_numpy(np.array(node_feats, dtype=np.float32)).to(DEVICE)
        adjs_t = torch.from_numpy(np.array(adjs, dtype=np.float32)).to(DEVICE)
        actions_t = torch.from_numpy(np.array(actions, dtype=np.int64)).to(DEVICE)
        rewards_t = torch.from_numpy(np.array(rewards, dtype=np.float32)).to(DEVICE)
        next_node_feats_t = torch.from_numpy(np.array(next_node_feats, dtype=np.float32)).to(DEVICE)
        next_adjs_t = torch.from_numpy(np.array(next_adjs, dtype=np.float32)).to(DEVICE)
        dones_t = torch.from_numpy(np.array(dones, dtype=np.uint8)).to(DEVICE)
        integrity_t = torch.FloatTensor(np.array(integrity)).unsqueeze(-1).to(DEVICE)
        
        return (node_feats_t, adjs_t, actions_t, rewards_t,
                next_node_feats_t, next_adjs_t, dones_t, integrity_t)
    
    def __len__(self) -> int:
        return len(self.buffer)


# =============================================================================
# GNN AGENT (Clean)
# =============================================================================

class GNNAgent:
    """
    GNN-based RL agent - domain-agnostic.
    
    Adapts to any feature dimension automatically.
    """
    
    def __init__(
        self,
        node_feature_dim: int,
        edge_feature_dim: int,
        action_size: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        n_heads: int = 4,
        lr: float = 0.0003,
        gamma: float = 0.95,
        buffer_capacity: int = 15000,
        dropout: float = 0.1,
        seed: Optional[int] = None,
        stability_coef: float = 0.5
    ):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
        
        self.action_size = action_size
        self.gamma = gamma
        self.stability_coef = stability_coef
        self.steps_done = 0
        
        # Frozen nodes
        self.frozen_nodes: Set[int] = set()
        self.frozen_node_positions: Dict[int, np.ndarray] = {}
        
        # Networks
        self.policy_net = GNNPolicy(
            node_feature_dim=node_feature_dim,
            edge_feature_dim=edge_feature_dim,
            action_size=action_size,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            n_heads=n_heads,
            dropout=dropout
        ).to(DEVICE)
        
        self.target_net = GNNPolicy(
            node_feature_dim=node_feature_dim,
            edge_feature_dim=edge_feature_dim,
            action_size=action_size,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            n_heads=n_heads,
            dropout=dropout
        ).to(DEVICE)
        
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        
        # Replay
        self.memory = GraphReplayBuffer(buffer_capacity)
        self.batch_size = 32
        
        # Exploration
        self.epsilon = 1.0
        self.epsilon_decay = 0.9995
        self.epsilon_min = 0.05
    
    def epsilon_gaussian(
        self,
        t: int,
        total_episodes: int,
        eps_min: float = 0.05,
        eps_peak: float = 0.9,
        mu: Optional[float] = None,
        sigma: Optional[float] = None
    ) -> float:
        """Gaussian exploration schedule."""
        if mu is None:
            mu = total_episodes / 2.0
        if sigma is None:
            sigma = total_episodes / 6.0
        
        return eps_min + (eps_peak - eps_min) * math.exp(
            -((t - mu) ** 2) / (2 * sigma ** 2)
        )
    
    # =========================================================================
    # NODE FREEZING
    # =========================================================================
    
    def freeze_node(self, node_id: int, position: np.ndarray):
        """Freeze a node."""
        self.frozen_nodes.add(node_id)
        self.frozen_node_positions[node_id] = position.copy()
    
    def unfreeze_node(self, node_id: int):
        """Unfreeze a node."""
        if node_id in self.frozen_nodes:
            self.frozen_nodes.remove(node_id)
            if node_id in self.frozen_node_positions:
                del self.frozen_node_positions[node_id]
    
    def is_frozen(self, node_id: int) -> bool:
        return node_id in self.frozen_nodes
    
    def get_frozen_nodes(self) -> Set[int]:
        return self.frozen_nodes.copy()
    
    # =========================================================================
    # ACTION SELECTION
    # =========================================================================
    
    def choose_actions(
        self,
        node_features: np.ndarray,
        adj_matrix: np.ndarray,
        episode_number: int,
        total_episodes: int,
        node_ids: Optional[List[int]] = None,
        eps_min: float = 0.05,
        eps_peak: float = 0.9
    ) -> np.ndarray:
        """
        Choose actions with epsilon-greedy.
        
        Frozen nodes always return action 0 (no-op).
        """
        num_nodes = node_features.shape[0]
        
        if node_ids is None:
            node_ids = list(range(num_nodes))
        
        actions = np.zeros(num_nodes, dtype=np.int64)
        
        # Get epsilon
        epsilon = self.epsilon_gaussian(episode_number, total_episodes, eps_min, eps_peak)
        
        # Active nodes
        active_mask = np.array([nid not in self.frozen_nodes for nid in node_ids])
        active_indices = np.where(active_mask)[0]
        
        if len(active_indices) == 0:
            return actions
        
        # Epsilon-greedy
        if random.random() < epsilon:
            for idx in active_indices:
                actions[idx] = random.randrange(self.action_size)
        else:
            self.policy_net.eval()
            with torch.no_grad():
                node_feat_t = torch.from_numpy(node_features.astype(np.float32)).unsqueeze(0).to(DEVICE)
                adj_t = torch.from_numpy(adj_matrix.astype(np.float32)).unsqueeze(0).to(DEVICE)
                
                q_values, _ = self.policy_net(node_feat_t, adj_t)
                all_actions = q_values.argmax(dim=2).cpu().numpy().flatten()
                
                actions[active_indices] = all_actions[active_indices]
            
            self.policy_net.train()
        
        return actions
    
    # =========================================================================
    # LEARNING
    # =========================================================================
    
    def learn(self, node_ids: Optional[List[int]] = None) -> float:
        """Train with gradient masking for frozen nodes."""
        if len(self.memory) < self.batch_size:
            return 0.0
        
        batch = self.memory.sample(self.batch_size)
        if batch is None:
            return 0.0
        
        (node_feats, adjs, actions, rewards,
         next_node_feats, next_adjs, dones, integrity_target) = batch
        
        B, N, _ = node_feats.shape
        
        # Current Q-values
        q_values, curr_stability = self.policy_net(node_feats, adjs)
        actions_idx = actions.unsqueeze(-1)
        q_values_taken = q_values.gather(2, actions_idx).squeeze(-1)
        
        # Target Q-values
        with torch.no_grad():
            next_q_values, _ = self.target_net(next_node_feats, next_adjs)
            next_q_values_max, _ = next_q_values.max(dim=2)
            
            done_mask = dones.unsqueeze(1).expand(B, N).float()
            targets = rewards + (self.gamma * next_q_values_max * (1.0 - done_mask))
        
        # Losses
        q_loss = self.criterion(q_values_taken, targets)
        stability_loss = F.mse_loss(curr_stability.view(-1), integrity_target.view(-1))
        
        loss = q_loss + (self.stability_coef * stability_loss)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        
        # Mask frozen gradients
        if node_ids is not None and len(self.frozen_nodes) > 0:
            self._mask_frozen_gradients(node_ids)
        
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        return float(loss.item())
    
    def _mask_frozen_gradients(self, node_ids: List[int]):
        """Zero out gradients for frozen nodes."""
        if not self.frozen_nodes:
            return
        
        frozen_mask = torch.tensor(
            [nid in self.frozen_nodes for nid in node_ids],
            dtype=torch.bool,
            device=DEVICE
        )
        
        if not frozen_mask.any():
            return
        
        for name, param in self.policy_net.named_parameters():
            if param.grad is None:
                continue
            
            grad = param.grad
            grad_shape = grad.shape
            
            # Mask node-specific outputs
            if len(grad_shape) >= 2 and grad_shape[1] == len(node_ids):
                mask_shape = [1] * len(grad_shape)
                mask_shape[1] = len(node_ids)
                broadcast_mask = frozen_mask.view(*mask_shape)
                grad.masked_fill_(broadcast_mask, 0.0)
    
    def update_target_network(self):
        """Copy policy net to target net."""
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def save(self, path: str):
        """Save model."""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'frozen_nodes': self.frozen_nodes,
            'frozen_node_positions': self.frozen_node_positions
        }, path)
    
    def load(self, path: str):
        """Load model."""
        checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.frozen_nodes = checkpoint.get('frozen_nodes', set())
        self.frozen_node_positions = checkpoint.get('frozen_node_positions', {})


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def build_adjacency_matrix(nodes: List[Any], sensor_range: float) -> np.ndarray:
    """Build adjacency from sensor detections."""
    N = len(nodes)
    id_to_index = {node.id: i for i, node in enumerate(nodes)}
    
    adj = np.zeros((N, N), dtype=np.float32)
    
    for i, node_i in enumerate(nodes):
        detections = node_i.sense_nodes(nodes)
        for det in detections:
            node_id = det['id']
            if node_id in id_to_index:
                j = id_to_index[node_id]
                adj[i, j] = 1.0
    
    # Self-loops
    adj += np.eye(N, dtype=np.float32)
    
    return adj
