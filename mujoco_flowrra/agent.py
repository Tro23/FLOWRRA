"""
agent.py

Continuous Control Graph Neural Network for FLOWRRA.
Uses DDPG (Actor-Critic) architecture for 6D Trajectory output.
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
# GRAPH ATTENTION LAYER (Remains unchanged - your math was perfect!)
# =============================================================================
class GraphAttentionLayer(nn.Module):
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
        h = torch.matmul(x, self.W).view(B, N, self.n_heads, self.out_features)

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

        alpha = self.dropout_layer(F.softmax(e, dim=2))

        alpha_transposed = alpha.permute(0, 3, 1, 2)
        h_transposed = h.permute(0, 2, 1, 3)
        h_prime = torch.matmul(alpha_transposed, h_transposed).permute(0, 2, 1, 3)

        if self.concat:
            return h_prime.reshape(B, N, self.n_heads * self.out_features)
        else:
            return h_prime.mean(dim=2)


# =============================================================================
# THE ACTOR (The Pilot)
# =============================================================================
class GNNActor(nn.Module):
    """Outputs the 6D Trajectory [X, Y, Z, Vx, Vy, Vz]"""

    def __init__(
        self,
        node_feature_dim: int,
        action_size: int = 6,
        hidden_dim: int = 128,
        num_layers: int = 3,
        n_heads: int = 4,
    ):
        super().__init__()
        self.node_encoder = nn.Sequential(
            nn.Linear(node_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.gat_layers = nn.ModuleList()
        for i in range(num_layers):
            in_dim = hidden_dim if i == 0 else hidden_dim * n_heads
            concat = i < num_layers - 1
            self.gat_layers.append(
                GraphAttentionLayer(in_dim, hidden_dim, n_heads, concat=concat)
            )

        final_dim = (
            hidden_dim if (num_layers == 0 or not concat) else hidden_dim * n_heads
        )

        # Head 1: 6D Trajectory
        # Outputs values between -1 and 1 for the 6D vector
        self.action_decoder = nn.Sequential(
            nn.Linear(final_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_size),
            nn.Tanh(),
        )

        # HEAD 2: THE FLOWRRA SOUL (Loop Integrity Predictor)
        self.stability_head = nn.Sequential(
            nn.Linear(final_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),  # Bounds the prediction between 0.0 and 1.0
        )

    def forward(self, node_features: torch.Tensor, adj_matrix: torch.Tensor):
        h = self.node_encoder(node_features)
        for i, gat in enumerate(self.gat_layers):
            h = gat(h, adj_matrix)
            if i < len(self.gat_layers) - 1:
                h = F.elu(h)

        # 1. Output the Flight Vector
        actions = self.action_decoder(h)

        # 2. Output the Global Stability Prediction
        h_graph = torch.mean(h, dim=1)  # Pool all nodes together
        stability_pred = self.stability_head(h_graph)

        return actions, stability_pred


# =============================================================================
# THE CRITIC (The Instructor)
# =============================================================================
class GNNCritic(nn.Module):
    """Evaluates how good a specific 6D Trajectory is for the current state."""

    def __init__(
        self,
        node_feature_dim: int,
        action_size: int = 6,
        hidden_dim: int = 128,
        num_layers: int = 3,
        n_heads: int = 4,
    ):
        super().__init__()

        # Critic looks at BOTH the state AND the action the Actor chose
        self.node_encoder = nn.Sequential(
            nn.Linear(node_feature_dim + action_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.gat_layers = nn.ModuleList()
        for i in range(num_layers):
            in_dim = hidden_dim if i == 0 else hidden_dim * n_heads
            concat = i < num_layers - 1
            self.gat_layers.append(
                GraphAttentionLayer(in_dim, hidden_dim, n_heads, concat=concat)
            )

        final_dim = (
            hidden_dim if (num_layers == 0 or not concat) else hidden_dim * n_heads
        )

        # Outputs a single Q-Value (Expected Future Reward) per node
        self.q_decoder = nn.Sequential(
            nn.Linear(final_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )

    def forward(
        self,
        node_features: torch.Tensor,
        actions: torch.Tensor,
        adj_matrix: torch.Tensor,
    ):
        # Concatenate State and Action
        x = torch.cat([node_features, actions], dim=-1)
        h = self.node_encoder(x)

        for i, gat in enumerate(self.gat_layers):
            h = gat(h, adj_matrix)
            if i < len(self.gat_layers) - 1:
                h = F.elu(h)

        return self.q_decoder(h)


# =============================================================================
# REPLAY BUFFER (Upgraded for Continuous Actions)
# =============================================================================
class GraphReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, adj, action, reward, next_state, next_adj, done, integrity):
        self.buffer.append(
            (
                state.copy(),
                adj.copy(),
                action.copy(),
                reward.copy(),
                next_state.copy(),
                next_adj.copy(),
                bool(done),
                integrity,
            )
        )

    def sample(self, batch_size: int):
        if len(self.buffer) < batch_size:
            return None
        batch = random.sample(self.buffer, batch_size)

        states, adjs, actions, rewards, next_states, next_adjs, dones, integrities = (
            zip(*batch)
        )

        return (
            torch.FloatTensor(np.array(states)).to(DEVICE),
            torch.FloatTensor(np.array(adjs)).to(DEVICE),
            torch.FloatTensor(np.array(actions)).to(
                DEVICE
            ),  # Now FloatTensor for 6D continuous
            torch.FloatTensor(np.array(rewards)).unsqueeze(-1).to(DEVICE),
            torch.FloatTensor(np.array(next_states)).to(DEVICE),
            torch.FloatTensor(np.array(next_adjs)).to(DEVICE),
            torch.FloatTensor(np.array(dones)).view(-1, 1, 1).to(DEVICE),
            torch.FloatTensor(np.array(integrities)).unsqueeze(-1).to(DEVICE),
        )

    def __len__(self):
        return len(self.buffer)


# =============================================================================
# CONTINUOUS DDPG AGENT
# =============================================================================
class GNNAgent:
    def __init__(
        self,
        node_feature_dim: int,
        action_size: int = 6,
        hidden_dim: int = 128,
        lr_actor: float = 1e-4,
        lr_critic: float = 1e-3,
        gamma: float = 0.99,
        tau: float = 0.005,
        stability_coef: float = 0.5,
    ):
        self.action_size = action_size
        self.gamma = gamma
        self.tau = tau  # Polyak averaging rate for smooth target updates
        self.stability_coef = stability_coef  # stability coefficient.

        # Actor Networks
        self.actor = GNNActor(node_feature_dim, action_size, hidden_dim).to(DEVICE)
        self.actor_target = GNNActor(node_feature_dim, action_size, hidden_dim).to(
            DEVICE
        )
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)

        # Critic Networks
        self.critic = GNNCritic(node_feature_dim, action_size, hidden_dim).to(DEVICE)
        self.critic_target = GNNCritic(node_feature_dim, action_size, hidden_dim).to(
            DEVICE
        )
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.memory = GraphReplayBuffer(15000)
        self.batch_size = 32

    def choose_actions(
        self,
        node_features: np.ndarray,
        adj_matrix: np.ndarray,
        current_positions: np.ndarray,
        current_integrity: float,  # <--- NEW: Pass the real-time integrity here
        noise_scale: float = 0.1,
        max_movement: float = 2.0,
        max_velocity: float = 1.5,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        self.actor.eval()
        with torch.no_grad():
            feat_t = torch.FloatTensor(node_features).unsqueeze(0).to(DEVICE)
            adj_t = torch.FloatTensor(adj_matrix).unsqueeze(0).to(DEVICE)

            # FIXED: Unpack the tuple to catch the new stability prediction
            raw_action, _ = self.actor(feat_t, adj_t)
            raw_action = raw_action.squeeze(0).cpu().numpy()
        self.actor.train()

        # DYNAMIC SURVIVAL NOISE:
        # If integrity is 1.0, noise is normal. If integrity is 0.0, noise drops to 0.0.
        # Don't do random exploration when the swarm is dying!
        dynamic_noise_scale = noise_scale * current_integrity

        # Add Exploration Noise
        noise = np.random.normal(0, dynamic_noise_scale, size=raw_action.shape)
        action = np.clip(raw_action + noise, -1.0, 1.0)

        num_nodes = current_positions.shape[0]
        waypoints = np.zeros((num_nodes, 3), dtype=np.float32)
        target_vels = np.zeros((num_nodes, 3), dtype=np.float32)

        for i in range(num_nodes):
            # Scale coordinates and add to current position
            waypoints[i] = current_positions[i] + (action[i, 0:3] * max_movement)
            # Scale velocities purely
            target_vels[i] = action[i, 3:6] * max_velocity

        return (
            waypoints,
            target_vels,
            action,
        )  # Return raw action to save in memory buffer

    def learn(self):
        batch = self.memory.sample(self.batch_size)
        if batch is None:
            return 0.0, 0.0

        (
            states,
            adjs,
            actions,
            rewards,
            next_states,
            next_adjs,
            dones,
            integrity_targets,
        ) = batch

        # ----------------------------
        # 1. UPDATE CRITIC
        # ----------------------------
        with torch.no_grad():
            # Get next actions from target actor
            next_actions, _ = self.actor_target(next_states, next_adjs)
            # Get next Q values from target critic
            next_Q = self.critic_target(next_states, next_actions, next_adjs)
            # Compute target Q
            target_Q = rewards + (self.gamma * next_Q * (1 - dones))

        # Current expected Q
        current_Q = self.critic(states, actions, adjs)
        critic_loss = F.mse_loss(current_Q, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ----------------------------
        # 2. UPDATE ACTOR
        # ----------------------------
        # Predict actions for CURRENT states AND Stability
        predicted_actions, curr_stability = self.actor(states, adjs)
        # Base Actor Loss: Maximize the Critic's score
        base_actor_loss = -self.critic(states, predicted_actions, adjs).mean()

        # Auxiliary Loss: How accurate was the Actor's understanding of the swarm's integrity?
        stability_loss = F.mse_loss(curr_stability.view(-1), integrity_targets.view(-1))

        # Combine them! (self.stability_coef was 0.5 in your original code)
        actor_loss = base_actor_loss + (self.stability_coef * stability_loss)

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------------
        # 3. SOFT UPDATE TARGETS
        # ----------------------------
        for target_param, param in zip(
            self.actor_target.parameters(), self.actor.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data
            )

        for target_param, param in zip(
            self.critic_target.parameters(), self.critic.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data
            )

        return actor_loss.item(), critic_loss.item()


# =============================================================================
# HELPER: BUILD GRAPH FROM 3D POSITIONS
# =============================================================================
def build_adjacency_matrix(positions: np.ndarray, sensor_range: float) -> np.ndarray:
    """
    Builds adjacency matrix dynamically using fast NumPy broadcasting.
    Positions array should be shape [num_nodes, 3].
    """
    # num_nodes = positions.shape[0]

    # Calculate pairwise Euclidean distances using broadcasting
    # diff shape: [num_nodes, num_nodes, 3]
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]

    # dist matrix shape: [num_nodes, num_nodes]
    distances = np.linalg.norm(diff, axis=-1)

    # Create binary adjacency matrix based on sensor range
    adj_matrix = (distances <= sensor_range).astype(np.float32)

    # Ensure self-loops (diagonal is 1)
    np.fill_diagonal(adj_matrix, 1.0)

    return adj_matrix
