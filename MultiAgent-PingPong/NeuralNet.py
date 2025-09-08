import torch
import torch.nn as nn
import numpy as np
import random
from collections import deque

# -----------------------------
# Dueling Double DQN Network
# -----------------------------
class DuelingDQN(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers=[256, 256, 128]):
        super(DuelingDQN, self).__init__()
        self.feature = nn.Sequential(
            nn.Linear(input_size, hidden_layers[0]),
            nn.ReLU(),
            nn.Linear(hidden_layers[0], hidden_layers[1]),
            nn.ReLU(),
        )
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_layers[1], hidden_layers[2]),
            nn.ReLU(),
            nn.Linear(hidden_layers[2], 1)
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_layers[1], hidden_layers[2]),
            nn.ReLU(),
            nn.Linear(hidden_layers[2], output_size)
        )

    def forward(self, x):
        x = self.feature(x)
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        return value + (advantage - advantage.mean(dim=1, keepdim=True))

# -----------------------------
# Prioritized Experience Replay
# -----------------------------
class PrioritizedReplayBuffer:
    def __init__(self, capacity=100000, alpha=0.6):
        self.capacity = capacity
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.alpha = alpha

    def push(self, state, action, reward, next_state, done):
        max_prio = self.priorities.max() if self.buffer else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]

        probs = prios ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)

        states, actions, rewards, next_states, dones = zip(*samples)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones),
            indices,
            weights
        )

    def update_priorities(self, indices, priorities):
        for idx, prio in zip(indices, priorities):
            self.priorities[idx] = prio

    def __len__(self):
        return len(self.buffer)


def train_dueling_ddqn(agent, target_agent, optimizer, replay_buffer, gamma=0.99, batch_size=1024, beta=0.4):
    if len(replay_buffer) < batch_size:
        return

    states, actions, rewards, next_states, dones, indices, weights = replay_buffer.sample(batch_size, beta)

    states = torch.FloatTensor(states)
    next_states = torch.FloatTensor(next_states)
    actions = torch.LongTensor(actions).unsqueeze(1)
    rewards = torch.FloatTensor(rewards)
    dones = torch.FloatTensor(dones)
    weights = torch.FloatTensor(weights)

    q_values = agent(states).gather(1, actions).squeeze(1)

    # Double DQN
    next_actions = agent(next_states).argmax(1, keepdim=True)
    next_q_values = target_agent(next_states).gather(1, next_actions).squeeze(1)

    targets = rewards + gamma * next_q_values * (1 - dones)

    loss = (q_values - targets.detach()).pow(2) * weights
    prios = loss + 1e-5
    loss = loss.mean()

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
    optimizer.step()

    replay_buffer.update_priorities(indices, prios.data.cpu().numpy())

