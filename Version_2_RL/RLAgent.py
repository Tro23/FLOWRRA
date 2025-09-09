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
    """
    The Q-Network that takes a flattened state vector and outputs a Q-value
    for each possible action for each node.
    """
    def __init__(self, state_size: int, num_nodes: int, action_size: int, hidden: int = 128):
        super().__init__()
        self.num_nodes = num_nodes
        self.action_size = action_size
        self.fc1 = nn.Linear(state_size, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, num_nodes * action_size)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

class ReplayBuffer:
    """
    A simple replay buffer to store experiences.
    """
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
        self.maxlen = capacity

    def push(self, state: np.ndarray, actions: np.ndarray, rewards: np.ndarray, next_state: np.ndarray, done: bool):
        self.buffer.append((state.copy(), actions.copy(), rewards.copy(), next_state.copy(), bool(done)))

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states_t = torch.from_numpy(np.array(states, dtype=np.float32)).to(DEVICE)
        actions_t = torch.from_numpy(np.array(actions, dtype=np.int64)).to(DEVICE)
        rewards_t = torch.from_numpy(np.array(rewards, dtype=np.float32)).to(DEVICE)
        next_states_t = torch.from_numpy(np.array(next_states, dtype=np.float32)).to(DEVICE)
        dones_t = torch.from_numpy(np.array(dones, dtype=np.uint8)).to(DEVICE)
        
        return states_t, actions_t, rewards_t, next_states_t, dones_t

    def __len__(self) -> int:
        return len(self.buffer)
    
class SharedRLAgent:
    """
    A single, shared RL agent that learns a joint policy for all nodes.
    
    This agent now handles a combined action space for both position and angle.
    """
    def __init__(self, state_size: int, num_nodes: int, action_size: int, lr: float, gamma: float, buffer_capacity: int, seed: int | None = None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        self.num_nodes = num_nodes
        self.action_size = action_size  # This now represents the combined action size (e.g., pos * angle)
        
        self.qnetwork = QNetwork(state_size, num_nodes, action_size).to(DEVICE)
        self.target_qnetwork = QNetwork(state_size, num_nodes, action_size).to(DEVICE)
        self.target_qnetwork.load_state_dict(self.qnetwork.state_dict())
        self.target_qnetwork.eval()
        
        self.optimizer = optim.Adam(self.qnetwork.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        
        self.gamma = gamma
        self.memory = ReplayBuffer(buffer_capacity)
        self.batch_size = 64
        self.epsilon = 1.0 # for exploration
        self.epsilon_decay = 0.9999
        self.epsilon_min = 0.01

    def choose_actions(self, state: np.ndarray, epsilon: float | None = None) -> np.ndarray:
        """
        Chooses actions for all nodes based on the current state and epsilon-greedy policy.
        
        Args:
            state (np.ndarray): The current flattened state vector.
            epsilon (float | None): The exploration rate. If None, uses internal decay.
            
        Returns:
            np.ndarray: A 1D array of chosen actions, one for each node.
        """
        if epsilon is None:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            epsilon = self.epsilon
            
        if random.random() < epsilon:
            # Exploration: choose random actions for all nodes
            return np.array([random.randrange(self.action_size) for _ in range(self.num_nodes)])
        else:
            # Exploitation: get Q-values and choose best action
            state_t = torch.from_numpy(state.copy()).float().unsqueeze(0).to(DEVICE)
            
            self.qnetwork.eval()
            with torch.no_grad():
                q_values = self.qnetwork(state_t).view(self.num_nodes, self.action_size)
            self.qnetwork.train()
            
            actions = q_values.argmax(dim=1).cpu().numpy()
            return actions

    def learn(self) -> float:
        """
        Performs one step of learning on a sampled batch from the replay buffer.
        
        Returns:
            float: The loss value for this learning step.
        """
        if len(self.memory) < self.batch_size:
            return 0.0
            
        states_t, actions_t, rewards_t, next_states_t, dones_t = self.memory.sample(self.batch_size)
        
        B = states_t.shape[0] # Batch size
        
        # Q-values for the actions taken
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
