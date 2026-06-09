# Class DQN (nn.Module):
#   - Input:  state size (2 for x,y or flattened grid)
#   - Hidden: 2 layers (64 → 64 neurons, ReLU)
#   - Output: 4 Q-values (one per action)

# Class ReplayBuffer:
#   - store(state, action, reward, next_state, done)
#   - sample(batch_size) → random batch for training

# Class DQNAgent:
#   - select_action()   → epsilon-greedy
#   - train_step()      → sample buffer, compute loss, backprop
#   - update_target()   → copy weights to target network

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque


# ─────────────────────────────────────────
#  Neural Network: Deep Q-Network
# ─────────────────────────────────────────
class DQNNetwork(nn.Module):
    """
    3-layer fully connected network.
    Input:  state_size (4)
    Output: n_actions  (4) Q-values
    """
    def __init__(self, state_size, n_actions):
        super(DQNNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions)
        )

    def forward(self, x):
        return self.net(x)


# ─────────────────────────────────────────
#  Replay Buffer (Experience Replay)
# ─────────────────────────────────────────
class ReplayBuffer:
    """
    Stores (state, action, reward, next_state, done) transitions.
    Random sampling breaks correlation between consecutive samples.
    """
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(np.array(states)),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(dones)
        )

    def __len__(self):
        return len(self.buffer)


# ─────────────────────────────────────────
#  DQN Agent
# ─────────────────────────────────────────
class DQNAgent:
    def __init__(
        self,
        state_size=4,
        n_actions=4,
        lr=0.001,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        batch_size=64,
        buffer_capacity=10000,
        target_update_freq=10,
    ):
        self.state_size   = state_size
        self.n_actions    = n_actions
        self.gamma        = gamma
        self.epsilon      = epsilon_start
        self.epsilon_end  = epsilon_end
        self.epsilon_decay= epsilon_decay
        self.batch_size   = batch_size
        self.target_update_freq = target_update_freq
        self.steps_done   = 0

        # Online network (trained every step)
        self.policy_net = DQNNetwork(state_size, n_actions)
        # Target network (updated every N episodes, stabilizes training)
        self.target_net = DQNNetwork(state_size, n_actions)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn   = nn.MSELoss()
        self.replay_buffer = ReplayBuffer(buffer_capacity)

        self.losses = []

    def select_action(self, state):
        """Epsilon-greedy action selection."""
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)  # Explore
        else:
            with torch.no_grad():
                state_t = torch.FloatTensor(state).unsqueeze(0)
                q_values = self.policy_net(state_t)
                return q_values.argmax().item()  # Exploit

    def store(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)

    def train_step(self):
        """Sample a batch and do one gradient update."""
        if len(self.replay_buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        # Current Q-values for taken actions
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target Q-values using Bellman equation
        with torch.no_grad():
            max_next_q = self.target_net(next_states).max(1)[0]
            target_q   = rewards + self.gamma * max_next_q * (1 - dones)

        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        self.losses.append(loss.item())
        return loss.item()

    def decay_epsilon(self):
        """Decay exploration rate after each episode."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def update_target_network(self):
        """Copy policy network weights to target network."""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, path="models/dqn_model.pth"):
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'epsilon':    self.epsilon,
        }, path)
        print(f"Model saved → {path}")

    def load(self, path="models/dqn_model.pth"):
        checkpoint = torch.load(path, map_location='cpu')
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['policy_net'])
        self.epsilon = checkpoint['epsilon']
        print(f"Model loaded ← {path}")