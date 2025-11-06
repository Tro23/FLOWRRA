"""
GNNAgent.py

Graph Neural Network agent for FLOWRRA swarm coordination.
Uses Graph Attention Networks (GAT) to learn distributed policies.

Key advantages over Q-learning:
- Permutation invariant (node order doesn't matter)
- Scales to variable numbers of nodes
- Natural graph structure from sensor networks
- Local computation with message passing
"""
import random
from collections import deque
from typing import Tuple, List, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============================================================================
# GRAPH ATTENTION LAYER
# =============================================================================

class GraphAttentionLayer(nn.Module):
    """
    Single Graph Attention layer (Veličković et al., 2018).
    
    Computes attention-weighted aggregation of neighbor features.
    """
    def __init__(self, in_features: int, out_features: int, 
                 n_heads: int = 4, dropout: float = 0.1, concat: bool = True):
        super().__init__()
        self.n_heads = n_heads
        self.concat = concat
        self.dropout = dropout
        
        # Multi-head attention
        self.W = nn.Parameter(torch.zeros(size=(in_features, n_heads * out_features)))
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.softmax = nn.Softmax(dim=1)
        self.dropout_layer = nn.Dropout(dropout)
        
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.a)
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Node features [batch, num_nodes, in_features]
            adj: Adjacency matrix [batch, num_nodes, num_nodes]
        
        Returns:
            Updated node features [batch, num_nodes, out_features * n_heads]
        """
        B, N, _ = x.shape
        
        # Linear transformation
        h = torch.matmul(x, self.W)  # [B, N, n_heads * out_features]
        h = h.view(B, N, self.n_heads, -1)  # [B, N, n_heads, out_features]
        
        # Compute attention scores
        h_i = h.unsqueeze(2).repeat(1, 1, N, 1, 1)  # [B, N, N, n_heads, out]
        h_j = h.unsqueeze(1).repeat(1, N, 1, 1, 1)  # [B, N, N, n_heads, out]
        
        concat = torch.cat([h_i, h_j], dim=-1)  # [B, N, N, n_heads, 2*out]
        
        # Attention mechanism
        e = self.leakyrelu(torch.matmul(concat, self.a.view(1, 1, 1, 1, -1, 1)).squeeze(-1))
        # [B, N, N, n_heads]
        
        # Mask non-existent edges
        mask = (adj == 0).unsqueeze(-1).repeat(1, 1, 1, self.n_heads)
        e = e.masked_fill(mask, float('-inf'))
        
        # Softmax attention weights
        alpha = self.softmax(e)  # [B, N, N, n_heads]
        alpha = self.dropout_layer(alpha)
        
        # Aggregate neighbor features
        h_prime = torch.matmul(alpha.transpose(2, 3), h_j.transpose(2, 3))
        # [B, N, n_heads, out]
        
        if self.concat:
            return h_prime.reshape(B, N, -1)  # Concatenate heads
        else:
            return h_prime.mean(dim=2)  # Average heads

# =============================================================================
# GRAPH NEURAL NETWORK
# =============================================================================

class GNNPolicy(nn.Module):
    """
    Graph Neural Network for distributed policy learning.
    
    Architecture:
    1. Node feature encoder
    2. Multiple GAT layers for message passing
    3. Per-node action decoder
    """
    def __init__(self, 
                 node_feature_dim: int,
                 edge_feature_dim: int,
                 action_size: int,
                 hidden_dim: int = 128,
                 num_layers: int = 3,
                 n_heads: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        
        self.node_feature_dim = node_feature_dim
        self.edge_feature_dim = edge_feature_dim
        self.action_size = action_size
        self.hidden_dim = hidden_dim
        
        # Node feature encoder
        self.node_encoder = nn.Sequential(
            nn.Linear(node_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Graph Attention layers
        self.gat_layers = nn.ModuleList([
            GraphAttentionLayer(
                in_features=hidden_dim if i == 0 else hidden_dim * n_heads,
                out_features=hidden_dim,
                n_heads=n_heads,
                dropout=dropout,
                concat=True if i < num_layers - 1 else False
            )
            for i in range(num_layers)
        ])
        
        # Action decoder (per-node Q-values)
        final_dim = hidden_dim if num_layers == 1 else hidden_dim * n_heads
        self.action_decoder = nn.Sequential(
            nn.Linear(final_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, action_size)
        )
    
    def forward(self, node_features: torch.Tensor, adj_matrix: torch.Tensor) -> torch.Tensor:
        """
        Args:
            node_features: [batch, num_nodes, node_feature_dim]
            adj_matrix: [batch, num_nodes, num_nodes] binary adjacency
        
        Returns:
            Q-values: [batch, num_nodes, action_size]
        """
        # Encode node features
        h = self.node_encoder(node_features)  # [B, N, hidden_dim]
        
        # Message passing through GAT layers
        for gat in self.gat_layers:
            h = gat(h, adj_matrix)
            h = F.elu(h)
        
        # Decode actions
        q_values = self.action_decoder(h)  # [B, N, action_size]
        
        return q_values

# =============================================================================
# REPLAY BUFFER
# =============================================================================

class GraphReplayBuffer:
    """
    Replay buffer for graph-structured experiences.
    
    Stores: (node_features, adjacency, actions, rewards, next_node_features, next_adjacency, done)
    """
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity
    
    def push(self, 
             node_features: np.ndarray,
             adj_matrix: np.ndarray,
             actions: np.ndarray,
             rewards: np.ndarray,
             next_node_features: np.ndarray,
             next_adj_matrix: np.ndarray,
             done: bool):
        """Store a transition."""
        self.buffer.append((
            node_features.copy(),
            adj_matrix.copy(),
            actions.copy(),
            rewards.copy(),
            next_node_features.copy(),
            next_adj_matrix.copy(),
            bool(done)
        ))
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """Sample a batch of transitions."""
        batch = random.sample(self.buffer, batch_size)
        
        node_feats, adjs, actions, rewards, next_node_feats, next_adjs, dones = zip(*batch)
        
        # Convert to tensors
        node_feats_t = torch.from_numpy(np.array(node_feats, dtype=np.float32)).to(DEVICE)
        adjs_t = torch.from_numpy(np.array(adjs, dtype=np.float32)).to(DEVICE)
        actions_t = torch.from_numpy(np.array(actions, dtype=np.int64)).to(DEVICE)
        rewards_t = torch.from_numpy(np.array(rewards, dtype=np.float32)).to(DEVICE)
        next_node_feats_t = torch.from_numpy(np.array(next_node_feats, dtype=np.float32)).to(DEVICE)
        next_adjs_t = torch.from_numpy(np.array(next_adjs, dtype=np.float32)).to(DEVICE)
        dones_t = torch.from_numpy(np.array(dones, dtype=np.uint8)).to(DEVICE)
        
        return node_feats_t, adjs_t, actions_t, rewards_t, next_node_feats_t, next_adjs_t, dones_t
    
    def __len__(self) -> int:
        return len(self.buffer)

# =============================================================================
# GNN AGENT
# =============================================================================

class GNNAgent:
    """
    GNN-based RL agent for FLOWRRA swarm control.
    
    Uses Graph Attention Networks to learn distributed policies that
    scale to arbitrary numbers of nodes.
    """
    def __init__(self,
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
                 seed: int = None):
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        
        self.action_size = action_size
        self.gamma = gamma
        
        # Policy and target networks
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
        
        # Replay buffer
        self.memory = GraphReplayBuffer(buffer_capacity)
        self.batch_size = 32
        
        # Exploration
        self.epsilon = 1.0
        self.epsilon_decay = 0.9995
        self.epsilon_min = 0.05
    
    def epsilon_gaussian(self, t: int, total_episodes: int, 
                        eps_min: float = 0.05, eps_peak: float = 0.8,
                        mu: float = None, sigma: float = None) -> float:
        """Gaussian-shaped exploration schedule."""
        if mu is None:
            mu = total_episodes / 2.0
        if sigma is None:
            sigma = total_episodes / 6.0
        
        return eps_min + (eps_peak - eps_min) * math.exp(-((t - mu) ** 2) / (2 * sigma ** 2))
    
    def choose_actions(self,
                      node_features: np.ndarray,
                      adj_matrix: np.ndarray,
                      episode_number: int,
                      total_episodes: int,
                      eps_min: float = 0.05,
                      eps_peak: float = 0.8) -> np.ndarray:
        """
        Choose actions for all nodes using epsilon-greedy.
        
        Args:
            node_features: [num_nodes, node_feature_dim]
            adj_matrix: [num_nodes, num_nodes]
            episode_number: Current episode
            total_episodes: Total training episodes
        
        Returns:
            actions: [num_nodes] integer actions
        """
        num_nodes = node_features.shape[0]
        epsilon = self.epsilon_gaussian(episode_number, total_episodes, eps_min, eps_peak)
        
        if random.random() < epsilon:
            # Random exploration
            return np.array([random.randrange(self.action_size) for _ in range(num_nodes)])
        else:
            # Greedy exploitation
            self.policy_net.eval()
            with torch.no_grad():
                node_feat_t = torch.from_numpy(node_features).float().unsqueeze(0).to(DEVICE)
                adj_t = torch.from_numpy(adj_matrix).float().unsqueeze(0).to(DEVICE)
                
                q_values = self.policy_net(node_feat_t, adj_t).squeeze(0)  # [num_nodes, action_size]
                actions = q_values.argmax(dim=1).cpu().numpy()
            
            self.policy_net.train()
            return actions
    
    def learn(self) -> float:
        """
        Perform one learning step using a batch from replay buffer.
        
        Returns:
            loss: Training loss value
        """
        if len(self.memory) < self.batch_size:
            return 0.0
        
        # Sample batch
        node_feats, adjs, actions, rewards, next_node_feats, next_adjs, dones = \
            self.memory.sample(self.batch_size)
        
        B, N, _ = node_feats.shape
        
        # Current Q-values
        q_values = self.policy_net(node_feats, adjs)  # [B, N, A]
        actions_idx = actions.unsqueeze(-1)  # [B, N, 1]
        q_values_taken = q_values.gather(2, actions_idx).squeeze(-1)  # [B, N]
        
        # Target Q-values
        with torch.no_grad():
            next_q_values = self.target_net(next_node_feats, next_adjs)  # [B, N, A]
            next_q_values_max, _ = next_q_values.max(dim=2)  # [B, N]
            
            targets = rewards + (self.gamma * next_q_values_max * (1.0 - dones.unsqueeze(1)))
        
        # Compute loss
        loss = self.criterion(q_values_taken, targets)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        return float(loss.item())
    
    def update_target_network(self):
        """Copy policy network weights to target network."""
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def save(self, path: str):
        """Save model weights."""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, path)
    
    def load(self, path: str):
        """Load model weights."""
        checkpoint = torch.load(path, map_location=DEVICE)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

# =============================================================================
# HELPER: BUILD GRAPH FROM DETECTIONS
# =============================================================================

def build_adjacency_matrix(nodes: List[Any], sensor_range: float) -> np.ndarray:
    """
    Builds adjacency matrix from node sensor detections.
    
    Args:
        nodes: List of NodePositionND objects
        sensor_range: Detection range
    
    Returns:
        adj_matrix: [num_nodes, num_nodes] binary adjacency
    """
    N = len(nodes)
    adj = np.zeros((N, N), dtype=np.float32)
    
    for i, node_i in enumerate(nodes):
        detections = node_i.sense_nodes(nodes)
        for det in detections:
            j = det['id']
            adj[i, j] = 1.0  # Edge exists
    
    # Add self-loops for message passing
    adj += np.eye(N, dtype=np.float32)
    
    return adj