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

        # ---- DECOMPOSED ADDITIVE ATTENTION -------------------------------
        # Mathematically identical to the previous formulation, but without ever
        # materialising the pairwise tensor.
        #
        # The old code built concat_features = [h_i || h_j] of shape
        # [B, N, N, n_heads, 2*out] and dotted it with `a`. At B=64, N=25,
        # n_heads=4, out=32 that single tensor is ~41 MB per layer before
        # autograd saves anything; at N=50 it is ~164 MB per layer, x3 layers,
        # x2 for the target net's forward. That is what made k=50 at batch 64
        # infeasible, and it was pure waste: since a = [a_src ; a_dst],
        #
        #     a^T [Wh_i || Wh_j] = a_src^T Wh_i + a_dst^T Wh_j
        #
        # so the score decomposes into two [B, N, n_heads] terms that broadcast
        # into [B, N, N, n_heads] additively. Peak memory drops by a factor of
        # 2*out (64x at out=32), and the scores are bit-comparable.
        a_src = self.a[: self.out_features].view(1, 1, 1, self.out_features)
        a_dst = self.a[self.out_features :].view(1, 1, 1, self.out_features)

        e_src = (h * a_src).sum(-1)  # [B, N, n_heads]  -- receiver term
        e_dst = (h * a_dst).sum(-1)  # [B, N, n_heads]  -- sender term

        # [B, N, 1, H] + [B, 1, N, H] -> [B, N, N, H]
        e = e_src.unsqueeze(2) + e_dst.unsqueeze(1)
        e = self.leakyrelu(e)

        # Mask non-existent edges (where adj == 0)
        mask = adj.unsqueeze(-1) == 0  # [B, N, N, 1], broadcasts over heads
        e = e.masked_fill(mask, float("-inf"))

        # Softmax normalization along neighbors (dim=2)
        alpha = F.softmax(e, dim=2)  # [B, N, N, n_heads]

        # NaN GUARD. A row of `adj` that is entirely zero makes every logit in
        # that row -inf, and softmax over all--inf is NaN, which then propagates
        # through the whole batch on the backward pass and silently destroys the
        # weights. _build_adjacency() always sets a self-loop so a real fleet
        # can never hit this, but PADDING rows added by GraphReplayBuffer.sample()
        # for variable fleet counts can -- and a caller passing a hand-built
        # adjacency might too. Zeroing the row is the correct behaviour: a node
        # with no neighbours aggregates nothing.
        alpha = torch.nan_to_num(alpha, nan=0.0)
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
        reward_heads: Optional[List[str]] = None,
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

        # DECOMPOSED Q-HEADS -- one per reward component.
        #
        # WHY, and this is the change that makes the rare behaviours learnable at
        # all. Previously nine reward sources (movement, arrival, collision,
        # warning, idle, overtime, pickup, handover, integrity) were SUMMED into a
        # single scalar per fleet before anything reached the buffer. The network
        # could not tell whether -3.0 came from a collision or from three hops of
        # moving away from the goal.
        #
        # Worse, the mass is wildly unbalanced. A 100-hop delivery earns ~300 from
        # movement alone against a one-off +25 for a package handover that occurs
        # on one fleet in one episode in six -- roughly 0.1% of an episode's total
        # reward. No amount of training extracts that from a summed scalar; the
        # signal is below the noise floor of the sum. That, mechanically, is why
        # resilience was never learned.
        #
        # With K heads each head does TD learning on ONE component. The rescue
        # head sees only pickup rewards: mostly zeros with an occasional +25.
        # Sparse, but CLEAN, and sparse-but-clean is a tractable learning problem
        # where 0.1% of a noisy sum is not.
        #
        # Q_total = sum_k w_k * Q_k. Because the head weights w_k are applied at
        # ACTION SELECTION and are independent of the reward scales used to train
        # each head, priority and learnability become separate knobs. Previously
        # they were the same one: turning up the collision penalty to make it
        # matter also made its TD targets larger and noisier.
        self.reward_heads = list(reward_heads) if reward_heads else ["total"]
        self.num_heads_q = len(self.reward_heads)

        self.action_decoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(final_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, action_size),
            )
            for _ in self.reward_heads
        ])

        # Head weights used ONLY for action selection, never for the loss. Kept as
        # a buffer (not a parameter) so they are saved with the checkpoint but not
        # learned -- they express what you want prioritised, which is a design
        # decision, not something to be optimised away.
        self.register_buffer(
            "head_weights",
            torch.ones(self.num_heads_q, dtype=torch.float32),
        )

        # HEAD 2: STABILITY PREDICTOR
        # Predicts the scalar loop Integrity (0.0 to 1.0)
        # We Pool node features to get the graph-level prediction
        # HEAD 3: RECOVERY INVOCATION (graph-level).
        # Q over {0 none, 1 spatial collapse, 2 temporal collapse} from the
        # mean-pooled node embedding. Graph-level and not per-fleet because a
        # temporal rewind restores MANY fleets at once -- it cannot be expressed
        # as one fleet's action, which is why the per-node action space could
        # never contain it. Tier 1 and Tier 3 need no invocation: they are an
        # ordinary move and an ordinary idle.
        self.recovery_head = nn.Sequential(
            nn.Linear(final_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),
        )

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

        # Decode actions per head: [B, N, K, action_size]
        q_per_head = torch.stack([dec(h) for dec in self.action_decoders], dim=2)

        # Weighted sum over heads for action selection: [B, N, action_size]
        w = self.head_weights.view(1, 1, -1, 1)
        q_values = (q_per_head * w).sum(dim=2)

        # Decode Stability (Global)
        # Mean pool over nodes to get graph representation
        h_graph = torch.mean(h, dim=1)
        stability_pred = self.stability_head(h_graph)
        self.last_recovery_q = self.recovery_head(h_graph)  # [B, 3]

        # q_per_head is returned so learn() can compute a separate TD target per
        # component; q_values is the weighted sum used to pick actions.
        return q_values, stability_pred, q_per_head


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
        active_mask: Optional[np.ndarray] = None,
        recovery_action: int = 0,
    ):
        """
        Store a transition.

        `rewards` may be either [N] (a single scalar per fleet, the old form) or
        [N, K] (one value per reward component per fleet, the decomposed form).
        A 1-D array is promoted to [N, 1] so a single-head configuration behaves
        exactly as before -- that matters for attribution: the first decomposed
        run should differ from the previous architecture ONLY because of the
        decomposition, not because the plumbing changed underneath it.

        active_mask: [N] float array, 1.0 for fleets that were ACTIVE at the time
        this transition was recorded, 0.0 for ones already parked at their goal.
        Stored per-transition rather than read from the live frozen set at learn()
        time, because the buffer mixes transitions from many different timesteps --
        applying today's frozen set to a transition recorded 400 steps ago would
        mask the wrong nodes. Defaults to all-active so an un-updated caller
        behaves exactly as before.
        """
        if active_mask is None:
            active_mask = np.ones(rewards.shape[0], dtype=np.float32)
        rewards = np.asarray(rewards, dtype=np.float32)
        if rewards.ndim == 1:
            rewards = rewards.reshape(-1, 1)

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
                np.asarray(active_mask, dtype=np.float32).copy(),
                int(recovery_action),
            )
        )

    @staticmethod
    def _pad_transition(exp, n_max: int):
        """
        Zero-pad one stored transition up to n_max fleets.

        Padding rows are inert everywhere that matters:
          * features / rewards / actions -> zeros, and
          * active_mask -> 0.0, so learn()'s masked Q-loss divides them out and
            they contribute exactly nothing to the gradient.

        The one row that is NOT zero is the adjacency diagonal. A padded fleet
        gets a SELF-LOOP. Without it its adjacency row is all zeros, every
        attention logit in that row becomes -inf, and softmax returns NaN --
        which back-propagates into the shared weights and corrupts the whole
        batch, not just the padding. (GraphAttentionLayer also nan_to_num's this
        as a second line of defence, but doing it correctly here is cheaper and
        keeps the attention distribution well-formed.)
        """
        (nf, adj, act, rew, nnf, nadj, done, integ, amask, rec) = exp
        n = nf.shape[0]
        if n == n_max:
            return exp
        pad = n_max - n

        def pad_rows(a):
            return np.pad(a, ((0, pad),) + ((0, 0),) * (a.ndim - 1))

        def pad_adj(a):
            out = np.zeros((n_max, n_max), dtype=a.dtype)
            out[:n, :n] = a
            # self-loops for the padding rows -- see docstring
            idx = np.arange(n, n_max)
            out[idx, idx] = 1.0
            return out

        return (
            pad_rows(nf),
            pad_adj(adj),
            pad_rows(act),
            pad_rows(rew),
            pad_rows(nnf),
            pad_adj(nadj),
            done,
            integ,
            pad_rows(amask),  # zeros -> excluded from the loss
            rec,              # graph-level scalar, no padding needed
        )

    def sample(self, batch_size: int) -> Optional[Tuple[torch.Tensor, ...]]:
        """
        Sample a batch of transitions, padding to the batch's largest fleet count.

        BUG THIS FIXES: this used to take the MODAL node count from the last 500
        entries and discard every transition that did not match it. That was
        adequate while every episode ran the same k, but it makes variable fleet
        counts silently impossible: training on k in 25..50 gives 26 distinct
        node counts, each holding roughly 1/26 of a 15k buffer, so the sampler
        would train on one rotating ~580-entry slice while the other 96% of
        collected experience sat unused and aged out. It also fails closed and
        LOUDLY at the start of any episode whose k is new -- "Only N valid
        experiences, skipping training" -- so the first episodes at each new
        fleet count learn nothing at all.

        Padding to the batch max instead means every transition is eligible for
        every batch regardless of k, which is what makes the variable-agent
        curriculum work.

        The FEATURE dimension still has to be homogeneous, and legitimately so:
        it is fixed by the ray count and affordance radius, not by the map or the
        fleet count, so a mismatch means the config changed mid-run and mixing
        the two would be wrong. Filtering on it (rather than assuming it) keeps
        that failure loud instead of letting numpy broadcast something wrong.
        """
        if len(self.buffer) < batch_size:
            return None

        from collections import Counter

        feat_dims = Counter(exp[0].shape[1] for exp in self.buffer)
        target_feat_dim = feat_dims.most_common(1)[0][0]

        if len(feat_dims) > 1:
            print(
                f"[BUFFER WARNING] Mixed feature dimensions in buffer {dict(feat_dims)}; "
                f"using {target_feat_dim}. This means the state vector changed "
                f"mid-run -- check CONFIG."
            )

        valid_experiences = [
            exp for exp in self.buffer if exp[0].shape[1] == target_feat_dim
        ]

        if len(valid_experiences) < batch_size:
            return None  # Signal to skip training

        raw_batch = random.sample(valid_experiences, batch_size)

        # Pad to the largest fleet count present IN THIS BATCH (not the global
        # max) so a batch that happens to be all-small stays small and cheap.
        n_max = max(exp[0].shape[0] for exp in raw_batch)
        batch = [self._pad_transition(exp, n_max) for exp in raw_batch]

        (
            node_feats,
            adjs,
            actions,
            rewards,
            next_node_feats,
            next_adjs,
            dones,
            integrity,
            active_masks,
            recovery_actions,
        ) = zip(*batch)

        # Convert to tensors (now all have same shape!)
        node_feats_t = torch.from_numpy(np.array(node_feats, dtype=np.float32)).to(
            DEVICE
        )
        adjs_t = torch.from_numpy(np.array(adjs, dtype=np.float32)).to(DEVICE)
        actions_t = torch.from_numpy(np.array(actions, dtype=np.int64)).to(DEVICE)
        # rewards is now [B, N, K] -- one column per reward component -- rather
        # than [B, N]. A 1-D reward is accepted and promoted to K=1 so older
        # checkpoints and any caller still pushing scalars keep working.
        rewards_np = np.array(rewards, dtype=np.float32)
        if rewards_np.ndim == 2:
            rewards_np = rewards_np[:, :, None]
        rewards_t = torch.from_numpy(rewards_np).to(DEVICE)
        next_node_feats_t = torch.from_numpy(
            np.array(next_node_feats, dtype=np.float32)
        ).to(DEVICE)
        next_adjs_t = torch.from_numpy(np.array(next_adjs, dtype=np.float32)).to(DEVICE)
        dones_t = torch.from_numpy(np.array(dones, dtype=np.uint8)).to(DEVICE)
        active_mask_t = torch.from_numpy(
            np.array(active_masks, dtype=np.float32)
        ).to(DEVICE)
        recovery_actions_t = torch.from_numpy(
            np.array(recovery_actions, dtype=np.int64)).to(DEVICE)
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
            active_mask_t,
            recovery_actions_t,
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
        reward_heads: Optional[List[str]] = None,
        head_weights: Optional[List[float]] = None,
        lr: float = 0.0003,
        gamma: float = 0.95,
        buffer_capacity: int = 15000,
        batch_size: int = 64,
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
            reward_heads=reward_heads,
        ).to(DEVICE)

        self.target_net = GNNPolicy(
            node_feature_dim=node_feature_dim,
            edge_feature_dim=edge_feature_dim,
            action_size=action_size,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            n_heads=n_heads,
            dropout=dropout,
            reward_heads=reward_heads,
        ).to(DEVICE)

        self.reward_heads = list(reward_heads) if reward_heads else ["total"]
        self.num_reward_heads = len(self.reward_heads)
        if head_weights is not None:
            if len(head_weights) != self.num_reward_heads:
                raise ValueError(
                    f"head_weights has {len(head_weights)} entries for "
                    f"{self.num_reward_heads} heads {self.reward_heads}")
            w = torch.tensor(head_weights, dtype=torch.float32, device=DEVICE)
            self.policy_net.head_weights.copy_(w)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Per-head TD loss from the last learn() call, for logging. Watching these
        # separately is the point of decomposing: a head stuck at zero means its
        # reward never fires or its state features cannot distinguish the
        # situation, and that is invisible in a summed loss.
        self.last_head_losses = {h: 0.0 for h in self.reward_heads}
        self.last_recovery_loss = 0.0
        # Set by the runner; see epsilon_gaussian's cold_start note.
        self.cold_start = False
        try:
            from config_warehouse import CONFIG as _C
            self.recovery_eps_scale = float(
                _C.get('recovery_policy', {}).get('exploration_scale', 1.0))
        except Exception:
            self.recovery_eps_scale = 1.0

        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

        # Replay buffer
        self.memory = GraphReplayBuffer(buffer_capacity)
        self.batch_size = batch_size

        # Populated by learn() -- see there for why these are exposed separately.
        self.last_q_loss = 0.0
        self.last_stability_loss = 0.0

    def choose_recovery(self) -> int:
        """
        Graph-level recovery decision: 0 none, 1 spatial collapse, 2 temporal.

        Reads the recovery Q-values cached by the most recent forward pass, so
        it costs nothing extra -- the orchestrator calls this once per step right
        after choose_actions. Explores on the same epsilon as the fleet actions:
        a decision taken a handful of times per episode needs exploration at
        least as much as the per-step ones, since it will otherwise sit at
        whatever the randomly-initialised network happens to prefer.
        """
        q = getattr(self.policy_net, "last_recovery_q", None)
        if q is None:
            return 0
        # Scaled epsilon -- see recovery_policy.exploration_scale. The recovery
        # head is a once-per-step 3-way decision; the fleet epsilon is sized for
        # a per-fleet 7-way one. Sharing it floods this head's own reward signal.
        eps = getattr(self, "_current_epsilon", 0.0) * getattr(
            self, "recovery_eps_scale", 1.0)
        if random.random() < eps:
            return random.randrange(3)
        return int(q[0].argmax().item())

    def reset_episode_state(self):
        """
        Clears PER-EPISODE state at the start of a new episode: which fleet
        IDs are currently frozen/parked, and their locked-in positions.

        BUG THIS FIXES: these were never reset between episodes, even though
        a fresh env (with its own correctly-reset frozen_nodes) gets
        constructed every episode in main_runner_warehouse.py's training
        loop. Since fleet IDs are REUSED across episodes (the same "1".."25"
        every time), any ID that was EVER frozen in ANY earlier episode
        stayed marked frozen here for the rest of the entire training run.
        Two consequences, both severe: choose_actions()'s active_mask would
        exclude that ID forever, forcing it to idle every step in every
        subsequent episode regardless of what actually happened THIS
        episode; and learn()'s _mask_frozen_gradients() would zero that ID's
        gradient for every replay-buffer sample referencing it, including
        samples from periods where it was legitimately active. Both effects
        compound as more IDs get "poisoned" over a long run.

        node_lifetime_freeze_counts is deliberately NOT cleared here -- that
        one is a genuine cross-episode statistic (how many times has this ID
        ever been frozen across all of training), not per-episode state.
        """
        self.frozen_nodes = set()
        self.frozen_node_positions = {}

    def epsilon_gaussian(
        self,
        t: int,
        total_episodes: int,
        eps_min: float = 0.00,
        eps_peak: float = 0.95,
        mu: Optional[float] = None,
        sigma: Optional[float] = None,
        cold_start: Optional[bool] = None,
    ) -> float:
        """
        Gaussian-shaped exploration schedule.

        This is now the SINGLE source of truth for eps_min/eps_peak -- see
        choose_actions() below, which used to specify its OWN separate eps_min=0.05,
        eps_peak=0.9 defaults and pass them in explicitly on every call, silently
        overriding whatever was set here. That meant this function's own defaults
        were dead: editing them here did nothing unless choose_actions's copy was
        *also* kept in sync by hand, which is exactly the kind of thing that drifts
        unnoticed. choose_actions no longer passes its own values, so this docstring
        and signature are the only place the schedule's shape lives now.

        USER TUNING (2026-08-24): narrower/lower peak, shifted earlier, versus the
        previous (eps_min=0.05, eps_peak=0.95, mu=0.5). Rationale: this is a
        goal-directed task (reach a fixed destination), not open-ended exploration
        of a state space -- so less time needs to be spent at very high randomness,
        and more of the back half of training should be low-noise policy
        consolidation instead of continued heavy exploration.
        """
        # COLD START: put the peak at episode 0 so the schedule opens at maximum
        # exploration and decays monotonically, instead of ramping up to a peak
        # 40-50% of the way in.
        #
        # WHY THIS MATTERS AND WHY THE DEFAULT IS WRONG FOR A FRESH RUN: with mu
        # at mid-run, epsilon at episode 1 of 300 is 0.050 -- effectively greedy.
        # "Exploit first" is only meaningful when there is a trained policy to
        # exploit. From a random initialisation, greedy means following an
        # arbitrary but DETERMINISTIC map, which is strictly worse than random
        # for state coverage: it is not just uninformative, it is correlated, so
        # the buffer fills with a narrow slice of trajectories the policy will
        # never revisit once it starts learning.
        #
        # Set cold_start=True whenever training from scratch. Leave it False when
        # resuming, where opening near-greedy on a real policy is the intended
        # exploit-explore-exploit shape.
        if cold_start is None:
            cold_start = getattr(self, "cold_start", False)
        if mu is None:
            mu = 0.0 if cold_start else total_episodes * 0.5
        if sigma is None:
            # Wider for cold start. With the peak at episode 0, sigma = T/6
            # puts epsilon below 0.02 by the halfway point and 0.000 by t=225 --
            # three quarters of the run at effectively zero exploration. T/3
            # spreads the decay across the whole schedule instead.
            sigma = (total_episodes / 3.0) if cold_start else (total_episodes / 6.0)

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
        eps_min: Optional[float] = None,
        eps_peak: Optional[float] = None,
        valid_action_masks: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        """
        Choose actions for all nodes.

        Frozen nodes always return action=0 (no-op/stay still).
        Active nodes use epsilon-greedy.

        Args:
            node_features: [num_nodes, feature_dim]
            adj_matrix: [num_nodes, num_nodes]
            episode_number: Current episode
            total_episodes: Total episodes
            node_ids: List of node IDs (CRITICAL for knowing which are frozen)
            eps_min, eps_peak: only pass these to override epsilon_gaussian's own
                defaults for this specific call; leave as None (the normal case) to
                use whatever schedule is configured there. See epsilon_gaussian()'s
                docstring for why this used to silently duplicate and override those
                defaults instead of deferring to them.
            valid_action_masks: [num_nodes, action_size] boolean, from each
                FleetNode's get_valid_action_mask(). Applied to BOTH random
                exploration and greedy exploitation -- not just exploration --
                since whether an edge exists in a direction is a fixed graph
                fact, not something meaningful for the network to learn through
                trial and error. On the real warehouse graph (average degree
                ~2.27 of 6 possible directions), unmasked exploration was
                landing on invalid edges roughly 70% of the time: a full step
                and the idle penalty spent, with nothing learned about actual
                navigation. None (the default) skips masking entirely, matching
                prior behavior exactly, for callers that don't have per-node
                validity available.

        Returns:
            actions: [num_nodes] action indices
        """
        num_nodes = node_features.shape[0]

        # If no node IDs provided, assume sequential IDs
        if node_ids is None:
            node_ids = list(range(num_nodes))

        # Initialize with no-op action
        actions = np.zeros(num_nodes, dtype=np.int64)  # Start with all zeros

        # Get epsilon for this episode. Only forward eps_min/eps_peak if the
        # caller actually provided an override -- otherwise let epsilon_gaussian
        # use its own defaults rather than re-specifying (and risking drifting
        # out of sync with) a second copy of them here.
        eps_kwargs = {}
        if eps_min is not None:
            eps_kwargs["eps_min"] = eps_min
        if eps_peak is not None:
            eps_kwargs["eps_peak"] = eps_peak
        epsilon = self.epsilon_gaussian(episode_number, total_episodes, **eps_kwargs)
        self._current_epsilon = epsilon

        # Identify active nodes
        active_mask = np.array(
            [node_id not in self.frozen_nodes for node_id in node_ids]
        )
        active_indices = np.where(active_mask)[0]

        if len(active_indices) == 0:
            # All nodes frozen!
            return actions

        # For active nodes: epsilon-greedy, decided PER FLEET.
        #
        # BUG THIS FIXES: this used to be a single `random.random() < epsilon`
        # for the entire fleet, so every step was all-greedy or all-random
        # together. Two consequences, both bad:
        #   1. Exploration was perfectly correlated across fleets. On an
        #      all-random step all 25 fleets jitter simultaneously, which is a
        #      joint action almost nothing like the ones the greedy policy will
        #      actually encounter -- so the transitions teach little about the
        #      single-fleet deviations Q-learning is trying to evaluate.
        #   2. Every transition in the buffer is drawn from one of two extreme
        #      joint policies, never the mixture in between, which is the regime
        #      the policy actually operates in.
        # Per-fleet draws give the intended interpretation of epsilon: each
        # fleet independently explores with probability epsilon, so on average
        # epsilon*k fleets deviate while the rest hold their greedy action.
        explore = np.zeros(num_nodes, dtype=bool)
        for idx in active_indices:
            if random.random() < epsilon:
                explore[idx] = True

        explore_indices = active_indices[explore[active_indices]]
        greedy_indices = active_indices[~explore[active_indices]]

        for idx in explore_indices:
            if valid_action_masks is not None:
                valid_for_node = np.where(valid_action_masks[idx])[0]
                # Idle is always valid (see get_valid_action_mask), so
                # valid_for_node is never empty -- there's always at least
                # one action to sample.
                actions[idx] = np.random.choice(valid_for_node)
            else:
                actions[idx] = random.randrange(self.action_size)

        # Only run the network if at least one fleet is actually exploiting --
        # with per-fleet epsilon that is almost always true, but skipping the
        # forward pass when it isn't costs nothing to check.
        if len(greedy_indices) > 0:
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

                q_values, _, _ = self.policy_net(node_feat_t, adj_t)
                q_values_np = q_values.squeeze(0).cpu().numpy()  # [num_nodes, action_size]

                if valid_action_masks is not None:
                    # Structurally-invalid actions can never win the argmax,
                    # for the same reason they're excluded from random
                    # exploration above -- there's nothing for the network to
                    # gain by being allowed to "choose" a direction with no
                    # edge, only a wasted step if it does.
                    q_values_np = np.where(valid_action_masks, q_values_np, -np.inf)

                all_actions = q_values_np.argmax(axis=1)

                # Only update actions for active nodes that are EXPLOITING this
                # step -- the exploring subset was already assigned above and
                # must not be overwritten.
                actions[greedy_indices] = all_actions[greedy_indices]

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
            active_mask,
            recovery_actions,
        ) = batch

        B, N, _ = node_feats.shape

        K = self.num_reward_heads

        # The buffer's reward width is set by core_warehouse.py from
        # CONFIG["reward_decomposition"]["heads"]; this network's head count came
        # from whatever the caller passed as reward_heads. They are the same list
        # in normal use, but nothing structurally forces that -- a runner
        # constructing GNNAgent without reward_heads gets K=1 while the
        # orchestrator happily pushes 5 columns.
        #
        # It does fail either way, but the raw error is
        # "size of tensor a (5) must match tensor b (2) at dimension 2", which
        # says nothing about the cause and sends you looking at the GAT. Naming
        # the mismatch turns a half-hour of confusion into a one-line fix.
        if rewards.shape[-1] != K:
            raise ValueError(
                f"Reward decomposition mismatch: the replay buffer holds "
                f"{rewards.shape[-1]} reward columns but this agent was built with "
                f"{K} head(s) {self.reward_heads}. The orchestrator takes its column "
                f"count from CONFIG['reward_decomposition']['heads'] -- pass that same "
                f"list as reward_heads= when constructing GNNAgent."
            )

        # Current Q, per head: q_per_head is [B, N, K, A]
        q_values, curr_stability, q_per_head = self.policy_net(node_feats, adjs)
        actions_idx = actions.unsqueeze(-1).unsqueeze(-1).expand(B, N, K, 1)
        q_taken_per_head = q_per_head.gather(3, actions_idx).squeeze(-1)  # [B, N, K]

        # Target, per head.
        #
        # The max is taken over the WEIGHTED SUM, not per head independently.
        # This matters and is easy to get wrong: taking argmax separately per head
        # would let each head bootstrap from a different next action, so the heads
        # would jointly evaluate a policy that no single agent ever follows and
        # their sum would not be the value of anything. Selecting one greedy action
        # from the combined Q and evaluating every head at THAT action keeps
        # sum_k Q_k a valid estimate of the return of the policy actually run.
        # (Hybrid Reward Architecture, van Seijen et al. 2017.)
        with torch.no_grad():
            next_q_sum, _, next_q_per_head = self.target_net(next_node_feats, next_adjs)
            next_a = next_q_sum.argmax(dim=2)                         # [B, N]
            gather_idx = next_a.unsqueeze(-1).unsqueeze(-1).expand(B, N, K, 1)
            next_q_taken = next_q_per_head.gather(3, gather_idx).squeeze(-1)  # [B,N,K]

            done_mask = dones.unsqueeze(1).unsqueeze(-1).expand(B, N, K).float()
            targets = rewards + (self.gamma * next_q_taken * (1.0 - done_mask))

        # Compute loss, counting ONLY fleets that were active in each transition.
        #
        # BUG THIS FIXES: _mask_frozen_gradients() below was intended to stop
        # parked fleets from training the network, but it never did anything. It
        # looped over policy_net.named_parameters() looking for a tensor whose
        # dim-1 equalled the fleet count -- but network PARAMETERS are shared
        # across nodes, shaped [in_features, out_features], with no node
        # dimension at all. Zero of the 18 parameter tensors ever matched, so the
        # condition was never true. Its own debug print ("[Gradient Mask] ...
        # Zeroed n/N node gradients") never once appeared in any training log.
        #
        # The consequence: every parked fleet kept contributing transitions with
        # reward 0.0 and an arbitrary action, from a position sitting exactly ON
        # a goal. With 8 of 25 fleets parked, roughly a third of every batch was
        # actively teaching the network that goal-adjacent states are worth ~0 --
        # working directly against the +100 mission_complete signal.
        #
        # Masking the per-element TD error is the correct place to do this: it
        # removes those samples from the loss entirely, so shared weights are
        # never updated toward them.
        # [B, N, K] elementwise TD error, masked by active fleets on [B, N].
        per_node_loss = F.smooth_l1_loss(q_taken_per_head, targets, reduction="none")
        mask3 = active_mask.unsqueeze(-1)  # [B, N, 1] broadcasts over heads
        mask_sum = active_mask.sum()
        if mask_sum > 0:
            # Per-head loss is recorded before summing so a head that never learns
            # is visible. A head pinned at 0.0 across an episode means either its
            # reward never fires or nothing in the state lets it tell the
            # situation apart -- both real bugs, and both invisible in a summed
            # loss. This is the diagnostic the previous architecture could not
            # provide.
            head_losses = (per_node_loss * mask3).sum(dim=(0, 1)) / mask_sum  # [K]
            q_loss = head_losses.sum()
            self.last_head_losses = {
                h: float(head_losses[k].item())
                for k, h in enumerate(self.reward_heads)
            }
        else:
            q_loss = per_node_loss.mean() * 0.0

        # --- 1b. RECOVERY HEAD TD LOSS ---
        #
        # WITHOUT THIS THE HEAD IS NEVER TRAINED. It was built, wired into
        # choose_recovery(), and charged an invocation cost in the reward -- but
        # given no gradient path of its own, so its output layer stayed at its
        # random initialisation and emitted a fixed argmax. Measured: recovery
        # invoked on 130-147 of every 150 steps for twelve straight episodes,
        # completely flat, while gradient agreement on the fleet actions climbed
        # 0.09 -> 0.57. The rest of the network was learning; this head could not.
        #
        # It is a graph-level decision, so its reward is the mean INTEGRITY
        # component across active fleets -- that is where the invocation cost and
        # the resolution bonus land. Standard DQN target on a 3-action space.
        rec_idx = self.reward_heads.index("integrity") if "integrity" in self.reward_heads else 0
        integ_r = rewards[:, :, rec_idx]                       # [B, N]
        denom = active_mask.sum(dim=1).clamp(min=1.0)
        graph_r = (integ_r * active_mask).sum(dim=1) / denom    # [B]

        q_rec = self.policy_net.last_recovery_q                # [B, 3]
        q_rec_taken = q_rec.gather(1, recovery_actions.unsqueeze(-1)).squeeze(-1)
        with torch.no_grad():
            _ = self.target_net(next_node_feats, next_adjs)
            next_rec_max = self.target_net.last_recovery_q.max(dim=1).values
            rec_target = graph_r + self.gamma * next_rec_max * (1.0 - dones.float())
        recovery_loss = F.smooth_l1_loss(q_rec_taken, rec_target)
        self.last_recovery_loss = float(recovery_loss.item())

        # --- 2. Compute Stability Loss (Auxiliary) ---
        # Predict current integrity vs actual integrity
        stability_loss = F.mse_loss(curr_stability.view(-1), integrity_target.view(-1))

        # --- 3. Total Loss ---
        loss = q_loss + recovery_loss + (self.stability_coef * stability_loss)

        # Expose both components separately (in addition to the combined float
        # still returned below, unchanged) so a caller can log them side by side --
        # e.g. to decide whether raising stability_coef is actually helping the
        # Q-loss converge, or just pulling the shared trunk toward the auxiliary
        # task at the main task's expense. Without seeing both, "stability_coef
        # higher or lower" is a guess; with them, it's a measurement.
        self.last_q_loss = float(q_loss.item())
        self.last_stability_loss = float(stability_loss.item())

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()

        # NOTE: _mask_frozen_gradients() is retained for reference but no longer
        # called -- it was a no-op (see the masked-loss comment above), and the
        # per-transition active_mask on the Q-loss now does the job correctly and
        # at the right point in the pipeline.

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