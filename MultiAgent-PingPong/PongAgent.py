import torch
import numpy as np
import os
from NeuralNet import ConvDuelingDQN, PrioritizedReplayBuffer, train_dueling_ddqn


class PongAgent:
    def __init__(self, state_size=10, action_size=3, sequence_length=4, lr=0.0001, gamma=0.99, buffer_capacity=200000,
                 model_path=None, tau=0.002):
        self.action_size = action_size
        self.gamma = gamma
        self.tau = tau
        self.model_path = model_path
        self.sequence_length = sequence_length
        self.state_size = state_size
        self.agent = ConvDuelingDQN(input_channels=state_size, sequence_length=sequence_length, output_size=action_size)
        self.target_agent = ConvDuelingDQN(input_channels=state_size, sequence_length=sequence_length,
                                           output_size=action_size)
        self.target_agent.load_state_dict(self.agent.state_dict())
        self.optimizer = torch.optim.Adam(self.agent.parameters(), lr=lr)
        self.replay_buffer = PrioritizedReplayBuffer(capacity=buffer_capacity)
        self.epsilon = 1.0
        self.epsilon_min = 0.02
        self.epsilon_decay = 0.99995
        if model_path and os.path.exists(model_path):
            print(f"Loading model from {model_path}")
            try:
                self.load_model(model_path)
            except RuntimeError as e:
                print(f"Could not load model due to architecture mismatch. Starting fresh. Error: {e}")

    def choose_action(self, stacked_state):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_size)
        state_tensor = torch.FloatTensor(stacked_state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.agent(state_tensor)
        return q_values.argmax().item()

    def store_experience(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)

    def train(self, batch_size=512):
        train_dueling_ddqn(self.agent, self.target_agent, self.optimizer, self.replay_buffer, self.gamma, batch_size)
        self._soft_update()

    def _soft_update(self):
        for target_param, local_param in zip(self.target_agent.parameters(), self.agent.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

    def update_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save_model(self, path):
        torch.save(self.agent.state_dict(), path)

    def load_model(self, path):
        self.agent.load_state_dict(torch.load(path))
        self.target_agent.load_state_dict(self.agent.state_dict())