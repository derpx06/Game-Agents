import torch
import numpy as np
import os
from NeuralNet import DuelingDQN, PrioritizedReplayBuffer, train_dueling_ddqn

class PongAgent:
    def __init__(self, state_size=7, action_size=3, lr=0.0005, gamma=0.99, buffer_capacity=100000, model_path=None, tau=0.005):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.tau = tau
        self.model_path = model_path

        self.agent = DuelingDQN(state_size, action_size)
        self.target_agent = DuelingDQN(state_size, action_size)
        self.target_agent.load_state_dict(self.agent.state_dict())
        self.optimizer = torch.optim.Adam(self.agent.parameters(), lr=lr, weight_decay=1e-5)

        self.replay_buffer = PrioritizedReplayBuffer(capacity=buffer_capacity)

        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.999

        if model_path and os.path.exists(model_path):
            print(f"Loading model from {model_path}")
            self.load_model(model_path)

    def choose_action(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.agent(state)
        return q_values.argmax().item()

    def store_experience(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)

    def train(self, batch_size=1024):
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

