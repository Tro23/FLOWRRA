"""
holon/r_gnn_agent.py

Recurrent Graph Neural Network Agent for temporal path memory.

Architecture: GAT (spatial) + LSTM (temporal) + Actor-Critic (policy)

This is Phase 2 - enables holons to remember stable routing patterns
and avoid boundary jitter through temporal coherence.
"""

import math
import random
from collections import deque
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GraphAttentionLayer(nn.Module):
    """
    Graph Attention layer for spatial aggregation.
    (Same as agent.py - reused for consistency)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_heads: int = 4,
        dropout: float = 0.1,
        concat: bool = True,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.concat = concat
        self.dropout = dropout
        self.out_features = out_features

        self.W = nn.Parameter(torch.zeros(size=(in_features, n_heads * out_features)))
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))

        self.leakyrelu = nn.LeakyReLU(0.2)
        self.dropout_layer = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.a)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        B, N, F_in = x.shape

        h = torch.matmul(x, self.W)
        h = h.view(B, N, self.n_heads, self.out_features)

        h_i = h.unsqueeze(2).expand(B, N, N, self.n_heads, self.out_features)
        h_j = h.unsqueeze(1).expand(B, N, N, self.n_heads, self.out_features)

        concat_features = torch.cat([h_i, h_j], dim=-1)
        a_reshaped = self.a.view(1, 1, 1, 1, 2 * self.out_features, 1)

        e = (
            torch.matmul(concat_features.unsqueeze(-2), a_reshaped)
            .squeeze(-1)
            .squeeze(-1)
        )
        e = self.leakyrelu(e)

        mask = (adj.unsqueeze(-1) == 0).expand(B, N, N, self.n_heads)
        e = e.masked_fill(mask, float("-inf"))

        alpha = F.softmax(e, dim=2)
        alpha = self.dropout_layer(alpha)

        alpha_transposed = alpha.permute(0, 3, 1, 2)
        h_transposed = h.permute(0, 2, 1, 3)

        h_prime = torch.matmul(alpha_transposed, h_transposed)
        h_prime = h_prime.permute(0, 2, 1, 3)

        if self.concat:
            return h_prime.reshape(B, N, self.n_heads * self.out_features)
        else:
            return h_prime.mean(dim=2)


class R_GNN_Policy(nn.Module):
    """
    Recurrent Graph Neural Network with temporal memory.

    Architecture:
    1. Node Encoder (MLP) - process raw features
    2. GAT Layers (spatial) - aggregate neighbor information
    3. LSTM Cell (temporal) - maintain path memory
    4. Action Decoder (MLP) - output Q-values/actions
    """

    def __init__(
        self,
        node_feature_dim: int,
        edge_feature_dim: int,
        action_size: int,
        hidden_dim: int = 128,
        num_gat_layers: int = 3,
        n_heads: int = 4,
        lstm_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.node_feature_dim = node_feature_dim
        self.action_size = action_size
        self.hidden_dim = hidden_dim
        self.num_gat_layers = num_gat_layers
        self.lstm_layers = lstm_layers

        # 1. Node feature encoder
        self.node_encoder = nn.Sequential(
            nn.Linear(node_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # 2. Graph Attention layers (spatial aggregation)
        self.gat_layers = nn.ModuleList()
        for i in range(num_gat_layers):
            if i == 0:
                in_dim = hidden_dim
            else:
                in_dim = hidden_dim * n_heads

            concat = i < num_gat_layers - 1

            self.gat_layers.append(
                GraphAttentionLayer(
                    in_features=in_dim,
                    out_features=hidden_dim,
                    n_heads=n_heads,
                    dropout=dropout,
                    concat=concat,
                )
            )

        # Determine GAT output dimension
        if num_gat_layers == 0:
            gat_out_dim = hidden_dim
        elif num_gat_layers == 1 or not concat:
            gat_out_dim = hidden_dim
        else:
            gat_out_dim = hidden_dim * n_heads

        # 3. LSTM cell (temporal memory)
        # Input: GAT output, Hidden: stores path history
        self.lstm = nn.LSTM(
            input_size=gat_out_dim,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        # 4. Action decoder (Q-values)
        self.action_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, action_size),
        )

    def forward(
        self,
        node_features: torch.Tensor,
        adj_matrix: torch.Tensor,
        hidden_state: Tuple[torch.Tensor, torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            node_features: [batch, num_nodes, node_feature_dim]
            adj_matrix: [batch, num_nodes, num_nodes]
            hidden_state: (h, c) tuple for LSTM, or None

        Returns:
            q_values: [batch, num_nodes, action_size]
            new_hidden_state: (h, c) tuple
        """
        B, N, _ = node_features.shape

        # 1. Encode node features
        h = self.node_encoder(node_features)  # [B, N, hidden_dim]

        # 2. GAT spatial aggregation
        for i, gat in enumerate(self.gat_layers):
            h = gat(h, adj_matrix)
            if i < len(self.gat_layers) - 1:
                h = F.elu(h)

        # 3. LSTM temporal integration
        # Reshape: [B, N, hidden] -> [B*N, 1, hidden] (treat each node as sequence of length 1)
        h_flat = h.reshape(B * N, 1, -1)

        # Initialize hidden state if not provided
        if hidden_state is None:
            h0 = torch.zeros(self.lstm_layers, B * N, self.hidden_dim).to(h.device)
            c0 = torch.zeros(self.lstm_layers, B * N, self.hidden_dim).to(h.device)
            hidden_state = (h0, c0)
        else:
            # Reshape hidden state: [lstm_layers, B, N, hidden] -> [lstm_layers, B*N, hidden]
            h0, c0 = hidden_state
            h0 = h0.reshape(self.lstm_layers, B * N, self.hidden_dim)
            c0 = c0.reshape(self.lstm_layers, B * N, self.hidden_dim)
            hidden_state = (h0, c0)

        # LSTM forward
        lstm_out, (h_new, c_new) = self.lstm(h_flat, hidden_state)

        # Reshape back: [B*N, 1, hidden] -> [B, N, hidden]
        lstm_out = lstm_out.reshape(B, N, self.hidden_dim)

        # Reshape hidden state: [lstm_layers, B*N, hidden] -> [lstm_layers, B, N, hidden]
        h_new = h_new.reshape(self.lstm_layers, B, N, self.hidden_dim)
        c_new = c_new.reshape(self.lstm_layers, B, N, self.hidden_dim)

        # 4. Decode actions
        q_values = self.action_decoder(lstm_out)  # [B, N, action_size]

        return q_values, (h_new, c_new)


class SequenceReplayBuffer:
    """
    Replay buffer that stores sequences for temporal learning.

    Unlike standard replay buffer, this maintains temporal coherence
    by storing short trajectories instead of single transitions.
    """

    def __init__(self, capacity: int, sequence_length: int = 4):
        self.buffer = deque(maxlen=capacity)
        self.sequence_length = sequence_length
        self.capacity = capacity

    def push_sequence(self, sequence: List[Dict]):
        """
        Store a sequence of transitions.

        Args:
            sequence: List of dicts, each containing:
                - node_features, adj_matrix, actions, rewards, etc.
        """
        if len(sequence) >= self.sequence_length:
            self.buffer.append(sequence[-self.sequence_length :])

    def sample(self, batch_size: int) -> List[List[Dict]]:
        """Sample batch of sequences."""
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    def __len__(self) -> int:
        return len(self.buffer)


class R_GNN_Agent:
    """
    Recurrent GNN agent with temporal path memory.

    Improvements over base GNN:
    - Remembers stable routing patterns
    - Reduces boundary jitter through temporal coherence
    - Better long-term planning
    """

    def __init__(
        self,
        node_feature_dim: int,
        edge_feature_dim: int,
        action_size: int,
        hidden_dim: int = 128,
        num_gat_layers: int = 3,
        n_heads: int = 4,
        lstm_layers: int = 1,
        lr: float = 0.0003,
        gamma: float = 0.95,
        buffer_capacity: int = 15000,
        sequence_length: int = 4,
        dropout: float = 0.1,
        seed: int = None,
    ):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        self.action_size = action_size
        self.gamma = gamma
        self.sequence_length = sequence_length

        # Policy and target networks
        self.policy_net = R_GNN_Policy(
            node_feature_dim=node_feature_dim,
            edge_feature_dim=edge_feature_dim,
            action_size=action_size,
            hidden_dim=hidden_dim,
            num_gat_layers=num_gat_layers,
            n_heads=n_heads,
            lstm_layers=lstm_layers,
            dropout=dropout,
        ).to(DEVICE)

        self.target_net = R_GNN_Policy(
            node_feature_dim=node_feature_dim,
            edge_feature_dim=edge_feature_dim,
            action_size=action_size,
            hidden_dim=hidden_dim,
            num_gat_layers=num_gat_layers,
            n_heads=n_heads,
            lstm_layers=lstm_layers,
            dropout=dropout,
        ).to(DEVICE)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

        # Sequence replay buffer
        self.memory = SequenceReplayBuffer(buffer_capacity, sequence_length)
        self.batch_size = 16  # Smaller batch for sequences

        # Hidden state tracking (per episode)
        self.hidden_state = None

        # Exploration
        self.epsilon = 1.0
        self.epsilon_decay = 0.9995
        self.epsilon_min = 0.05

    def reset_hidden_state(self):
        """Reset LSTM hidden state (call at episode start)."""
        self.hidden_state = None

    def choose_actions(
        self, node_features: np.ndarray, adj_matrix: np.ndarray, epsilon: float = None
    ) -> np.ndarray:
        """
        Choose actions using R-GNN with temporal memory.

        Args:
            node_features: [num_nodes, feature_dim]
            adj_matrix: [num_nodes, num_nodes]
            epsilon: Optional override for exploration rate

        Returns:
            actions: [num_nodes] array of action indices
        """
        if epsilon is None:
            epsilon = self.epsilon

        num_nodes = node_features.shape[0]

        if random.random() < epsilon:
            # Random exploration
            return np.array(
                [random.randrange(self.action_size) for _ in range(num_nodes)]
            )
        else:
            # Greedy exploitation with temporal memory
            self.policy_net.eval()
            with torch.no_grad():
                node_feat_t = (
                    torch.from_numpy(node_features.astype(np.float32))
                    .unsqueeze(0)
                    .to(DEVICE)
                )
                adj_t = (
                    torch.from_numpy(adj_matrix.astype(np.float32))
                    .unsqueeze(0)
                    .to(DEVICE)
                )

                q_values, self.hidden_state = self.policy_net(
                    node_feat_t, adj_t, self.hidden_state
                )
                actions = q_values.squeeze(0).argmax(dim=1).cpu().numpy()

            self.policy_net.train()
            return actions

    def learn(self) -> float:
        """
        Learn from sequence replay buffer.

        Returns:
            Training loss
        """
        if len(self.memory) < self.batch_size:
            return 0.0

        # Sample sequences
        sequences = self.memory.sample(self.batch_size)

        total_loss = 0.0

        for sequence in sequences:
            # Process sequence
            seq_loss = self._learn_from_sequence(sequence)
            total_loss += seq_loss

        avg_loss = total_loss / len(sequences)
        return float(avg_loss)

    def _learn_from_sequence(self, sequence: List[Dict]) -> float:
        """Learn from a single sequence."""
        # Extract data from sequence
        node_feats = [s["node_features"] for s in sequence]
        adjs = [s["adj_matrix"] for s in sequence]
        actions = [s["actions"] for s in sequence]
        rewards = [s["rewards"] for s in sequence]

        # Convert to tensors
        node_feats_t = torch.stack(
            [torch.from_numpy(f.astype(np.float32)) for f in node_feats]
        ).to(DEVICE)
        adjs_t = torch.stack([torch.from_numpy(a.astype(np.float32)) for a in adjs]).to(
            DEVICE
        )
        actions_t = torch.stack(
            [torch.from_numpy(a.astype(np.int64)) for a in actions]
        ).to(DEVICE)
        rewards_t = torch.stack(
            [torch.from_numpy(r.astype(np.float32)) for r in rewards]
        ).to(DEVICE)

        # Forward through sequence
        hidden = None
        losses = []

        for t in range(len(sequence)):
            # Current Q-values
            q_values, hidden = self.policy_net(
                node_feats_t[t : t + 1], adjs_t[t : t + 1], hidden
            )

            q_values_taken = (
                q_values.squeeze(0).gather(1, actions_t[t].unsqueeze(-1)).squeeze(-1)
            )

            # Target Q-values (if not last step)
            if t < len(sequence) - 1:
                with torch.no_grad():
                    next_q, _ = self.target_net(
                        node_feats_t[t + 1 : t + 2], adjs_t[t + 1 : t + 2], hidden
                    )
                    next_q_max = next_q.squeeze(0).max(dim=1)[0]
                    targets = rewards_t[t] + self.gamma * next_q_max
            else:
                targets = rewards_t[t]

            loss = self.criterion(q_values_taken, targets)
            losses.append(loss)

        # Backward through sequence
        total_loss = sum(losses) / len(losses)

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        return total_loss.item()

    def update_target_network(self):
        """Copy policy network to target network."""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, path: str):
        """Save model."""
        torch.save(
            {
                "policy_net": self.policy_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            path,
        )

    def load(self, path: str):
        """Load model."""
        checkpoint = torch.load(path, map_location=DEVICE)
        self.policy_net.load_state_dict(checkpoint["policy_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
