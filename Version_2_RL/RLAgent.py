# RLAgent.py
import random
from collections import deque
from typing import Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class QNetwork(nn.Module):
    def __init__(self, state_size: int, num_nodes: int, action_size: int, hidden: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(state_size, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, num_nodes * action_size)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
        self.maxlen = capacity

    def push(self, state: np.ndarray, actions: np.ndarray, rewards: np.ndarray, next_state: np.ndarray, done: bool):
        self.buffer.append((state.copy(), actions.copy(), rewards.copy(), next_state.copy(), bool(done)))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.stack(states), np.stack(actions), np.stack(rewards), np.stack(next_states), np.array(dones, dtype=np.float32))

    def __len__(self):
        return len(self.buffer)

    def prune_low_reward(self, threshold: float):
        kept = []
        for s, a, r, ns, d in list(self.buffer):
            if np.mean(r) >= threshold:
                kept.append((s, a, r, ns, d))
        self.buffer = deque(kept, maxlen=self.maxlen)


class SharedRLAgent:
    """
    Shared agent controlling all nodes. All nodes push experiences into one replay buffer.
    """
    def __init__(self, state_size: int, num_nodes: int, action_size: int,
                 lr: float = 1e-3, gamma: float = 0.99, buffer_capacity: int = 50000, seed: int = None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

        self.state_size = state_size
        self.num_nodes = num_nodes
        self.action_size = action_size
        self.gamma = gamma

        self.qnetwork = QNetwork(state_size, num_nodes, action_size).to(DEVICE)
        self.target_qnetwork = QNetwork(state_size, num_nodes, action_size).to(DEVICE)
        self.target_qnetwork.load_state_dict(self.qnetwork.state_dict())
        self.target_qnetwork.eval()

        self.optimizer = optim.Adam(self.qnetwork.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

        self.replay_buffer = ReplayBuffer(buffer_capacity)

    def choose_actions(self, state: np.ndarray, epsilon: float) -> np.ndarray:
        """Return an action per node for the given global state."""
        if np.random.rand() < epsilon:
            return np.array([random.randrange(self.action_size) for _ in range(self.num_nodes)], dtype=np.int64)

        state_t = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)  # (1, state_size)
        with torch.no_grad():
            qvals = self.qnetwork(state_t).view(self.num_nodes, self.action_size)  # (N, A)
            actions = qvals.argmax(dim=1).cpu().numpy().astype(np.int64)  # (N,)
        return actions

    def push_experience(self, state: np.ndarray, actions: np.ndarray, rewards: np.ndarray, next_state: np.ndarray, done: bool):
        self.replay_buffer.push(state, actions, rewards, next_state, done)

    # shortcut name used by Flowrra_RL
    def push(self, state, actions, rewards, next_state, done):
        self.push_experience(state, actions, rewards, next_state, done)

    def train_step(self, batch_size: int):
        if len(self.replay_buffer) < batch_size:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        B = states.shape[0]

        states_t = torch.FloatTensor(states).to(DEVICE)         # (B, state_size)
        next_states_t = torch.FloatTensor(next_states).to(DEVICE)
        actions_t = torch.LongTensor(actions).to(DEVICE)        # (B, N)
        rewards_t = torch.FloatTensor(rewards).to(DEVICE)      # (B, N)
        dones_t = torch.FloatTensor(dones).to(DEVICE)          # (B,)

        qvals = self.qnetwork(states_t).view(B, self.num_nodes, self.action_size)  # (B,N,A)
        actions_idx = actions_t.unsqueeze(-1)  # (B,N,1)
        qvals_taken = qvals.gather(2, actions_idx).squeeze(-1)  # (B,N)

        with torch.no_grad():
            next_qvals = self.target_qnetwork(next_states_t).view(B, self.num_nodes, self.action_size)  # (B,N,A)
            next_qvals_max, _ = next_qvals.max(dim=2)  # (B,N)

        targets = rewards_t + (self.gamma * next_qvals_max * (1.0 - dones_t).unsqueeze(1))  # (B,N)

        loss = self.criterion(qvals_taken, targets)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.qnetwork.parameters(), 1.0)
        self.optimizer.step()

        return float(loss.item())

    def update_target_network(self):
        self.target_qnetwork.load_state_dict(self.qnetwork.state_dict())

    def save(self, path: str):
        torch.save(self.qnetwork.state_dict(), path)

    def load(self, path: str, map_location=None):
        if map_location is None:
            map_location = DEVICE
        self.qnetwork.load_state_dict(torch.load(path, map_location=map_location))
        self.qnetwork.to(DEVICE)
        self.qnetwork.eval()
        self.target_qnetwork.load_state_dict(self.qnetwork.state_dict())
        self.target_qnetwork.eval()

    def retrocausal_prune(self, reward_threshold: float = 0.0):
        self.replay_buffer.prune_low_reward(reward_threshold)
