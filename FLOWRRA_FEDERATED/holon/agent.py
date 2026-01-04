"""
agent.py

Graph Neural Network agent for FLOWRRA swarm coordination.
Uses Graph Attention Networks (GAT) to learn distributed policies.

FIXES:
- Corrected GraphAttentionLayer tensor dimensions
- Fixed attention aggregation matmul operations
- Improved epsilon-greedy exploration
- Fixed state vector input handling
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
# GRAPH ATTENTION LAYER - FIXED
# =============================================================================


class GraphAttentionLayer(nn.Module):
    """
    Single Graph Attention layer (Veličković et al., 2018).

    FIXED: Corrected tensor operations for attention aggregation.
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

        # Multi-head attention parameters
        self.W = nn.Parameter(torch.zeros(size=(in_features, n_heads * out_features)))
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))

        self.leakyrelu = nn.LeakyReLU(0.2)
        self.dropout_layer = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.a)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        FIXED: Corrected attention aggregation.

        Args:
            x: Node features [batch, num_nodes, in_features]
            adj: Adjacency matrix [batch, num_nodes, num_nodes]

        Returns:
            Updated node features [batch, num_nodes, out_features * n_heads]
        """
        B, N, F_in = x.shape

        # Linear transformation: [B, N, n_heads * out_features]
        h = torch.matmul(x, self.W)
        h = h.view(B, N, self.n_heads, self.out_features)  # [B, N, n_heads, out]

        # Compute attention scores
        # For each head, we need to compute e_ij = LeakyReLU(a^T [Wh_i || Wh_j])

        # Expand for pairwise combinations
        h_i = h.unsqueeze(2).expand(
            B, N, N, self.n_heads, self.out_features
        )  # [B, N, N, n_heads, out]
        h_j = h.unsqueeze(1).expand(
            B, N, N, self.n_heads, self.out_features
        )  # [B, N, N, n_heads, out]

        # Concatenate [h_i || h_j]: [B, N, N, n_heads, 2*out]
        concat_features = torch.cat([h_i, h_j], dim=-1)

        # Reshape a for broadcasting: [2*out, 1] -> [1, 1, 1, 1, 2*out, 1]
        a_reshaped = self.a.view(1, 1, 1, 1, 2 * self.out_features, 1)

        # Compute attention coefficients: [B, N, N, n_heads]
        e = (
            torch.matmul(concat_features.unsqueeze(-2), a_reshaped)
            .squeeze(-1)
            .squeeze(-1)
        )
        e = self.leakyrelu(e)

        # Mask non-existent edges (where adj == 0)
        mask = (adj.unsqueeze(-1) == 0).expand(B, N, N, self.n_heads)
        e = e.masked_fill(mask, float("-inf"))

        # Softmax normalization along neighbors (dim=2)
        alpha = F.softmax(e, dim=2)  # [B, N, N, n_heads]
        alpha = self.dropout_layer(alpha)

        # Aggregate neighbor features
        # alpha: [B, N, N, n_heads]
        # h: [B, N, n_heads, out]
        # We want: h_prime[i] = sum_j alpha[i,j] * h[j]

        # Reshape for batch matrix multiply
        alpha_transposed = alpha.permute(0, 3, 1, 2)  # [B, n_heads, N, N]
        h_transposed = h.permute(0, 2, 1, 3)  # [B, n_heads, N, out]

        # Matrix multiply: [B, n_heads, N, N] @ [B, n_heads, N, out] -> [B, n_heads, N, out]
        h_prime = torch.matmul(alpha_transposed, h_transposed)

        # Reshape back: [B, n_heads, N, out] -> [B, N, n_heads, out]
        h_prime = h_prime.permute(0, 2, 1, 3)

        if self.concat:
            # Concatenate heads: [B, N, n_heads * out]
            return h_prime.reshape(B, N, self.n_heads * self.out_features)
        else:
            # Average heads: [B, N, out]
            return h_prime.mean(dim=2)


# =============================================================================
# GRAPH NEURAL NETWORK - FIXED
# =============================================================================


class GNNPolicy(nn.Module):
    """
    Graph Neural Network for distributed policy learning.

    FIXED: Properly handles variable input dimensions and GAT layer connections.
    """

    def __init__(
        self,
        node_feature_dim: int,
        edge_feature_dim: int,
        action_size: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        n_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.node_feature_dim = node_feature_dim
        self.edge_feature_dim = edge_feature_dim
        self.action_size = action_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Node feature encoder
        self.node_encoder = nn.Sequential(
            nn.Linear(node_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Graph Attention layers
        self.gat_layers = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                in_dim = hidden_dim
            else:
                in_dim = hidden_dim * n_heads

            # Last layer doesn't concatenate heads
            concat = i < num_layers - 1

            self.gat_layers.append(
                GraphAttentionLayer(
                    in_features=in_dim,
                    out_features=hidden_dim,
                    n_heads=n_heads,
                    dropout=dropout,
                    concat=concat,
                )
            )

        # Action decoder (per-node Q-values)
        if num_layers == 0:
            final_dim = hidden_dim
        elif num_layers == 1 or not concat:
            final_dim = hidden_dim
        else:
            final_dim = hidden_dim * n_heads

        # HEAD 1: ACTION POLICY
        self.action_decoder = nn.Sequential(
            nn.Linear(final_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, action_size),
        )

        # HEAD 2: STABILITY PREDICTOR
        # Predicts the scalar loop Integrity (0.0 to 1.0)
        # We Pool node features to get the graph-level prediction
        self.stability_head = nn.Sequential(
            nn.Linear(final_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),  # Integrity is [0, 1]
        )

    def forward(self, node_features: torch.Tensor, adj_matrix: torch.Tensor):
        """
        Args:
            node_features: [batch, num_nodes, node_feature_dim]
            adj_matrix: [batch, num_nodes, num_nodes] binary adjacency

        Returns:
            Q-values: [batch, num_nodes, action_size]
        """
        # Encode node features: [B, N, hidden_dim]
        h = self.node_encoder(node_features)

        # Message passing through GAT layers
        for i, gat in enumerate(self.gat_layers):
            h = gat(h, adj_matrix)
            # Apply activation (except possibly after last layer)
            if i < len(self.gat_layers) - 1:
                h = F.elu(h)

        # Decode actions: [B, N, action_size]
        q_values = self.action_decoder(h)

        # Decode Stability (Global)
        # Mean pool over nodes to get graph representation
        h_graph = torch.mean(h, dim=1)
        stability_pred = self.stability_head(h_graph)

        return q_values, stability_pred


# =============================================================================
# REPLAY BUFFER
# =============================================================================


class GraphReplayBuffer:
    """Replay buffer for graph-structured experiences."""

    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity

    def push(
        self,
        node_features: np.ndarray,
        adj_matrix: np.ndarray,
        actions: Optional[np.ndarray],
        rewards: np.ndarray,
        next_node_features: np.ndarray,
        next_adj_matrix: np.ndarray,
        done: bool,
        integrity: float,
    ):
        """Store a transition."""
        self.buffer.append(
            (
                node_features.copy(),
                adj_matrix.copy(),
                actions.copy(),
                rewards.copy(),
                next_node_features.copy(),
                next_adj_matrix.copy(),
                bool(done),
                integrity,
            )
        )

    def sample(self, batch_size: int) -> Optional[Tuple[torch.Tensor, ...]]:
        """
        Sample a batch of transitions with UNIFORM node counts.

        FIX: Filters out experiences with mismatched node counts to avoid shape errors.
        """
        # Find the most common node count in recent experiences
        recent_buffer = list(self.buffer)[-min(500, len(self.buffer)) :]
        node_counts = [exp[0].shape[0] for exp in recent_buffer]

        if not node_counts:
            raise ValueError("Buffer is empty!")

        # Use the most common node count (handles both 7 and 8 gracefully)
        from collections import Counter

        target_node_count = Counter(node_counts).most_common(1)[0][0]

        # Filter buffer to only include experiences with matching node count
        valid_experiences = [
            exp for exp in self.buffer if exp[0].shape[0] == target_node_count
        ]

        if len(valid_experiences) < batch_size:
            print(
                f"[BUFFER WARNING] Only {len(valid_experiences)} valid experiences "
                f"(need {batch_size}). Skipping training this step."
            )
            return None  # Signal to skip training

        # Sample from valid experiences only
        batch = random.sample(valid_experiences, batch_size)

        (
            node_feats,
            adjs,
            actions,
            rewards,
            next_node_feats,
            next_adjs,
            dones,
            integrity,
        ) = zip(*batch)

        # Convert to tensors (now all have same shape!)
        node_feats_t = torch.from_numpy(np.array(node_feats, dtype=np.float32)).to(
            DEVICE
        )
        adjs_t = torch.from_numpy(np.array(adjs, dtype=np.float32)).to(DEVICE)
        actions_t = torch.from_numpy(np.array(actions, dtype=np.int64)).to(DEVICE)
        rewards_t = torch.from_numpy(np.array(rewards, dtype=np.float32)).to(DEVICE)
        next_node_feats_t = torch.from_numpy(
            np.array(next_node_feats, dtype=np.float32)
        ).to(DEVICE)
        next_adjs_t = torch.from_numpy(np.array(next_adjs, dtype=np.float32)).to(DEVICE)
        dones_t = torch.from_numpy(np.array(dones, dtype=np.uint8)).to(DEVICE)
        integrity_target = (
            torch.FloatTensor(np.array(integrity)).unsqueeze(-1).to(DEVICE)
        )

        return (
            node_feats_t,
            adjs_t,
            actions_t,
            rewards_t,
            next_node_feats_t,
            next_adjs_t,
            dones_t,
            integrity_target,
        )

    def __len__(self) -> int:
        return len(self.buffer)


# =============================================================================
# GNN AGENT - FIXED
# =============================================================================


class GNNAgent:
    """
    GNN-based RL agent for FLOWRRA swarm control.

    FIXED: Improved exploration schedule and input handling.

    KEY FEATURE: Can freeze specific nodes, making them static landmarks
    while keeping them in the computational graph.
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
        stability_coef: float = 0.5,  # Weight for auxilary loss
    ):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        self.action_size = action_size
        self.gamma = gamma
        self.stability_coef = stability_coef
        self.steps_done = 0

        # NEW: Track frozen nodes
        self.frozen_nodes: Set[int] = set()
        self.frozen_node_positions: Dict[
            int, np.ndarray
        ] = {}  # Storing Frozen Positions
        self.node_lifetime_freeze_counts = {}

        # Policy and target networks
        self.policy_net = GNNPolicy(
            node_feature_dim=node_feature_dim,
            edge_feature_dim=edge_feature_dim,
            action_size=action_size,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            n_heads=n_heads,
            dropout=dropout,
        ).to(DEVICE)

        self.target_net = GNNPolicy(
            node_feature_dim=node_feature_dim,
            edge_feature_dim=edge_feature_dim,
            action_size=action_size,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            n_heads=n_heads,
            dropout=dropout,
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

    def epsilon_gaussian(
        self,
        t: int,
        total_episodes: int,
        eps_min: float = 0.05,
        eps_peak: float = 0.9,
        mu: Optional[float] = None,
        sigma: Optional[float] = None,
    ) -> float:
        """Gaussian-shaped exploration schedule."""
        if mu is None:
            mu = total_episodes / 2.0
        if sigma is None:
            sigma = total_episodes / 6.0

        return eps_min + (eps_peak - eps_min) * math.exp(
            -((t - mu) ** 2) / (2 * sigma**2)
        )

    ## NODE FREEZING - ADDED
    def freeze_node(self, node_id: int, position: np.ndarray):
        """
        Freeze a node - it becomes a static landmark.

        This node's:
        - Position is stored and becomes constant
        - Features still flow through the network (forward pass)
        - But gradients for its outputs are zeroed (no learning)

        Args:
            node_id: ID of node to freeze
            position: Final position to lock in
        """
        self.frozen_nodes.add(node_id)
        self.frozen_node_positions[node_id] = position.copy()

        # Increment lifetime count
        self.node_lifetime_freeze_counts[node_id] = (
            self.node_lifetime_freeze_counts.get(node_id, 0) + 1
        )

        print(f"[GNN] 🧊 Node {node_id} FROZEN at position {position}")
        print(f"[GNN] Total frozen nodes: {len(self.frozen_nodes)}")

    def unfreeze_node(self, node_id: int):
        """Unfreeze a node - it becomes active again."""
        if node_id in self.frozen_nodes:
            self.frozen_nodes.remove(node_id)
            if node_id in self.frozen_node_positions:
                del self.frozen_node_positions[node_id]
            print(f"[GNN] 🔥 Node {node_id} UNFROZEN")

    def is_frozen(self, node_id: int) -> bool:
        """Check if a node is frozen."""
        return node_id in self.frozen_nodes

    def get_frozen_nodes(self) -> Set[int]:
        """Get set of all frozen node IDs."""
        return self.frozen_nodes.copy()

    def choose_actions(
        self,
        node_features: np.ndarray,
        adj_matrix: np.ndarray,
        episode_number: int,
        total_episodes: int,
        node_ids: Optional[
            List[int]
        ] = None,  # NEW: Need to know which nodes are which.
        eps_min: float = 0.05,
        eps_peak: float = 0.9,
    ) -> Optional[np.ndarray]:
        """
        Choose actions for all nodes.

        Frozen nodes always return action=4 (no-op/stay still).
        Active nodes use epsilon-greedy.

        Args:
            node_features: [num_nodes, feature_dim]
            adj_matrix: [num_nodes, num_nodes]
            episode_number: Current episode
            total_episodes: Total episodes
            node_ids: List of node IDs (CRITICAL for knowing which are frozen)

        Returns:
            actions: [num_nodes] action indices
        """
        num_nodes = node_features.shape[0]

        # If no node IDs provided, assume sequential IDs
        if node_ids is None:
            node_ids = list(range(num_nodes))

        # Initialize with no-op action
        # For 2D: actions are 0-3 (left, right, up, down), so use action 0 as default
        # We'll override with actual actions for active nodes
        actions = np.zeros(num_nodes, dtype=np.int64)  # Start with all zeros

        # Get epsilon for this episode
        epsilon = self.epsilon_gaussian(
            episode_number, total_episodes, eps_min, eps_peak
        )

        # Identify active nodes
        active_mask = np.array(
            [node_id not in self.frozen_nodes for node_id in node_ids]
        )
        active_indices = np.where(active_mask)[0]

        if len(active_indices) == 0:
            # All nodes frozen!
            return actions

        # For active nodes: epsilon-greedy
        if random.random() < epsilon:
            # Random exploration for active nodes only
            for idx in active_indices:
                actions[idx] = random.randrange(self.action_size)
        else:
            # Greedy exploitation
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

                q_values, _ = self.policy_net(node_feat_t, adj_t)
                all_actions = q_values.argmax(dim=2).cpu().numpy().flatten()

                # Only update actions for active nodes
                actions[active_indices] = all_actions[active_indices]

            self.policy_net.train()

        return actions

    def learn(self, node_ids: Optional[List[int]] = None) -> float:
        """
        Perform one learning step with gradient masking for frozen nodes.

        KEY CHANGE: After computing loss, we zero out gradients for frozen nodes
        before calling optimizer.step().

        Args:
            node_ids: List of node IDs in the batch (needed for masking)

        Returns:
            loss value
        """
        if len(self.memory) < self.batch_size:
            return 0.0

        # Sample batch
        batch = self.memory.sample(self.batch_size)
        if batch is None:
            return 0.0

        # Sample batch
        (
            node_feats,
            adjs,
            actions,
            rewards,
            next_node_feats,
            next_adjs,
            dones,
            integrity_target,
        ) = batch

        B, N, _ = node_feats.shape

        # Current Q-values
        q_values, curr_stability = self.policy_net(node_feats, adjs)  # [B, N, A]
        actions_idx = actions.unsqueeze(-1)  # [B, N, 1]
        q_values_taken = q_values.gather(2, actions_idx).squeeze(-1)  # [B, N]

        # Target Q-values
        with torch.no_grad():
            next_q_values, _ = self.target_net(next_node_feats, next_adjs)  # [B, N, A]
            next_q_values_max, _ = next_q_values.max(dim=2)  # [B, N]

            # Broadcast done mask: [B] -> [B, N]
            done_mask = dones.unsqueeze(1).expand(B, N).float()
            targets = rewards + (self.gamma * next_q_values_max * (1.0 - done_mask))

        # Compute loss
        q_loss = self.criterion(q_values_taken, targets)

        # --- 2. Compute Stability Loss (Auxiliary) ---
        # Predict current integrity vs actual integrity
        stability_loss = F.mse_loss(curr_stability.view(-1), integrity_target.view(-1))

        # --- 3. Total Loss ---
        loss = q_loss + (self.stability_coef * stability_loss)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()

        # CRITICAL: Mask gradients for frozen nodes
        if node_ids is not None and len(self.frozen_nodes) > 0:
            self._mask_frozen_gradients(node_ids)

        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)

        # Update weights (frozen nodes' don't change)
        self.optimizer.step()

        return float(loss.item())

    def _mask_frozen_gradients(self, node_ids: List[int]):
        """
        Zero out gradients for frozen nodes.

        This is where the "crystallization" happens - frozen nodes
        can't learn anymore, their weights are locked.

        Strategy:
        - The GNN processes all nodes in a batch dimension [B, N, features]
        - During forward pass, frozen nodes participate normally
        - During backward pass, we zero their gradients in the batch dimension

        Args:
            node_ids: List of node IDs in current batch
        """
        if not self.frozen_nodes:
            return  # No frozen nodes, nothing to mask

        # Create frozen mask: True for frozen nodes, False for active
        # This maps batch positions to frozen status
        frozen_mask = torch.tensor(
            [node_id in self.frozen_nodes for node_id in node_ids],
            dtype=torch.bool,
            device=DEVICE,
        )

        if not frozen_mask.any():
            return  # No frozen nodes in this batch

        # ================================================================
        # GRADIENT MASKING LOGIC
        # ================================================================
        # The key insight: GAT layers process node features in dimension 1
        # Shape: [batch, num_nodes, features]
        # We need to zero gradients for frozen nodes across ALL parameters

        # Get all parameters that have gradients
        for name, param in self.policy_net.named_parameters():
            if param.grad is None:
                continue

            grad = param.grad
            grad_shape = grad.shape

            # ============================================================
            # CASE 1: Node-specific decoder outputs
            # These have shape [batch, num_nodes, action_size]
            # ============================================================
            if len(grad_shape) >= 2:
                # Check if second dimension matches number of nodes
                if grad_shape[1] == len(node_ids):
                    # This gradient has per-node outputs
                    # Zero out frozen node positions
                    # Shape: [B, N, ...] → mask dimension 1

                    # Create mask for broadcasting
                    mask_shape = [1] * len(grad_shape)
                    mask_shape[1] = len(node_ids)  # Match node dimension

                    # Reshape frozen_mask to broadcast correctly
                    broadcast_mask = frozen_mask.view(*mask_shape)

                    # Zero out frozen node gradients
                    # Active nodes keep their gradients, frozen nodes → 0
                    grad.masked_fill_(broadcast_mask, 0.0)

                    # Debug logging (can remove in production)
                    if "action_decoder" in name:
                        num_frozen_in_batch = frozen_mask.sum().item()
                        print(
                            f"[Gradient Mask] {name}: Zeroed {num_frozen_in_batch}/{len(node_ids)} node gradients"
                        )

    def update_target_network(self):
        """Copy policy network weights to target network."""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, path: str):
        """Save model weights."""
        torch.save(
            {
                "policy_net": self.policy_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "frozen_nodes": self.frozen_nodes,
                "frozen_node_positions": self.frozen_node_positions,
            },
            path,
        )

    def load(self, path: str):
        """Load model weights."""
        checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
        self.policy_net.load_state_dict(checkpoint["policy_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.frozen_nodes = checkpoint.get("frozen_nodes", set())
        self.frozen_node_positions = checkpoint.get("frozen_node_positions", {})


# =============================================================================
# HELPER: BUILD GRAPH FROM DETECTIONS
# =============================================================================


def build_adjacency_matrix(nodes: List[Any], sensor_range: float) -> np.ndarray:
    """
    Builds adjacency matrix from node sensor detections.

    Frozen nodes can still be detected by active nodes!
    They act as landmarks in the graph.

    Args:
        nodes: List of NodePositionND objects
        sensor_range: Detection range

    Returns:
        adj_matrix: [num_nodes, num_nodes] binary adjacency
    """
    N = len(nodes)
    id_to_index = {node.id: i for i, node in enumerate(nodes)}

    adj = np.zeros((N, N), dtype=np.float32)

    for i, node_i in enumerate(nodes):
        detections = node_i.sense_nodes(nodes)
        for det in detections:
            node_id = det["id"]
            if node_id in id_to_index:
                j = id_to_index[node_id]
                adj[i, j] = 1.0

    # Add self-loops
    adj += np.eye(N, dtype=np.float32)

    return adj
