import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

class QNetwork(nn.Module):
    """
    The neural network model for the Q-learning agent.
    It takes a processed state (sensor data) as input and outputs a Q-value
    for each possible action.
    """
    def __init__(self, state_size: int, action_size: int):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(116, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)
        self.relu = nn.ReLU()

    def forward(self, state):
        """
        Forward pass through the network.
        
        Args:
            state (torch.Tensor): The input state tensor.
        
        Returns:
            torch.Tensor: The Q-values for each action.
        """
        x = self.relu(self.fc1(state))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

class ReplayBuffer:
    """
    A simple replay buffer to store experiences for training.
    This helps to break the correlation between consecutive samples and
    improves learning stability.
    """
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """Adds a new experience to the buffer."""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        """Randomly samples a batch of experiences from the buffer."""
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        """Returns the current size of the buffer."""
        return len(self.buffer)


class RL_Agent:
    """
    Encapsulates the reinforcement learning logic for the nodes.
    This class manages the Q-network, optimizer, and training process.
    """
    def __init__(self, state_size: int, action_size: int, lr: float = 0.001, gamma: float = 0.99, replay_buffer_capacity: int = 10000):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma

        # Main Q-Network
        self.qnetwork = QNetwork(state_size, action_size)
        # Target Network for stable Q-updates
        self.target_qnetwork = QNetwork(state_size, action_size)
        self.target_qnetwork.load_state_dict(self.qnetwork.state_dict())
        self.target_qnetwork.eval() # Set target network to evaluation mode

        # Optimizer and Loss function
        self.optimizer = optim.Adam(self.qnetwork.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

        # Replay buffer
        self.replay_buffer = ReplayBuffer(replay_buffer_capacity)

    def choose_action(self, state: np.ndarray, epsilon: float) -> int:
        """
        Selects an action based on the epsilon-greedy strategy.
        
        Args:
            state (np.ndarray): The current state of the environment.
            epsilon (float): The current exploration rate.
            
        Returns:
            int: The chosen action index.
        """
        # Epsilon-greedy exploration vs. exploitation
        if np.random.rand() < epsilon:
            # Explore: choose a random action
            return random.randrange(self.action_size)
        else:
            # Exploit: choose the best action from the Q-network
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state)
                q_values = self.qnetwork(state_tensor.unsqueeze(0))
                return q_values.argmax().item()

    def train(self, batch_size: int):
        """
        Performs a single training step by sampling from the replay buffer.
        """
        if len(self.replay_buffer) < batch_size:
            return # Not enough data to train yet

        experiences = self.replay_buffer.sample(batch_size)
        
        states, actions, rewards, next_states, dones = zip(*experiences)

        # Convert to PyTorch tensors
        states = torch.FloatTensor(np.vstack(states))
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.vstack(next_states))
        dones = torch.FloatTensor(dones)

        # Compute Q-values for current states
        q_values = self.qnetwork(states).gather(1, actions.unsqueeze(1))

        # Compute max Q-values for next states from the target network
        with torch.no_grad():
            next_q_values = self.target_qnetwork(next_states).max(1)[0]
        
        # Compute the target Q-values using the Bellman equation
        target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))

        # Compute loss and perform backpropagation
        loss = self.criterion(q_values, target_q_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        """
        Updates the target network with the weights from the main network.
        """
        self.target_qnetwork.load_state_dict(self.qnetwork.state_dict())

    def save_model(self, path: str):
        """Saves the Q-network's state dictionary to a file."""
        torch.save(self.qnetwork.state_dict(), path)

    def load_model(self, path: str):
        """Loads the Q-network's state dictionary from a file."""
        self.qnetwork.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
        self.qnetwork.eval()
        self.target_qnetwork.load_state_dict(self.qnetwork.state_dict())
        self.target_qnetwork.eval()
