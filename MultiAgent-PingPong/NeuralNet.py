import torch
import torch.nn as nn
import numpy as np


# --- UPGRADED CONV1D DUELING DQN NETWORK ---
class ConvDuelingDQN(nn.Module):
    def __init__(self, input_channels, sequence_length, output_size):
        super(ConvDuelingDQN, self).__init__()

        # 1D Convolutional layers to process the time-series data of stacked frames
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=input_channels, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        # Calculate the size of the flattened convolutional output
        conv_output_size = self._get_conv_output_size(input_channels, sequence_length)

        # Dueling streams
        self.value_stream = nn.Sequential(
            nn.Linear(conv_output_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(conv_output_size, 256),
            nn.ReLU(),
            nn.Linear(256, output_size)
        )

    def _get_conv_output_size(self, input_channels, sequence_length):
        # Helper to calculate the output size of the conv layers
        with torch.no_grad():
            x = torch.zeros(1, input_channels, sequence_length)
            x = self.conv_layers(x)
            return x.flatten().shape[0]

    def forward(self, x):
        # Input x has shape: (batch_size, sequence_length, input_channels)
        # Conv1d expects: (batch_size, input_channels, sequence_length)
        x = x.permute(0, 2, 1)

        conv_out = self.conv_layers(x).flatten(start_dim=1)

        value = self.value_stream(conv_out)
        advantage = self.advantage_stream(conv_out)

        return value + (advantage - advantage.mean(dim=1, keepdim=True))


# --- PRIORITIZED EXPERIENCE REPLAY (No changes needed) ---
class PrioritizedReplayBuffer:
    # ... (This class remains the same as your previous version)
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
        if len(self.buffer) < batch_size: return None
        prios = self.priorities[:len(self.buffer)];
        probs = prios ** self.alpha;
        probs /= probs.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        total = len(self.buffer);
        weights = (total * probs[indices]) ** (-beta);
        weights /= weights.max()
        states, actions, rewards, next_states, dones = zip(*samples)
        return (np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones), indices,
                np.array(weights, dtype=np.float32))

    def update_priorities(self, indices, priorities):
        for idx, prio in zip(indices, priorities): self.priorities[idx] = prio

    def __len__(self):
        return len(self.buffer)


# --- TRAINING FUNCTION (Modified for ConvNet input shape) ---
def train_dueling_ddqn(agent, target_agent, optimizer, replay_buffer, gamma=0.99, batch_size=512, beta=0.4):
    sample = replay_buffer.sample(batch_size, beta)
    if sample is None: return

    states, actions, rewards, next_states, dones, indices, weights = sample

    # Reshape for Conv1D: (batch, seq_len, state_features)
    states = torch.FloatTensor(states)
    next_states = torch.FloatTensor(next_states)

    actions = torch.LongTensor(actions).unsqueeze(1)
    rewards = torch.FloatTensor(rewards).unsqueeze(1)
    dones = torch.FloatTensor(dones).unsqueeze(1)
    weights = torch.FloatTensor(weights).unsqueeze(1)

    q_values = agent(states).gather(1, actions)

    with torch.no_grad():
        next_actions = agent(next_states).argmax(1, keepdim=True)
        next_q_values = target_agent(next_states).gather(1, next_actions)
        targets = rewards + gamma * next_q_values * (1 - dones)

    td_errors = (q_values - targets).abs()
    loss = (td_errors.pow(2) * weights).mean()

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
    optimizer.step()

    replay_buffer.update_priorities(indices, td_errors.squeeze().detach().cpu().numpy())