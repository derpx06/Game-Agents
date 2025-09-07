import random
import numpy as np
import torch
from collections import deque
from NeuralNet import Linear_QNet
from Trainer import QTrainer

class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.gamma = 0.9
        self.memory = deque(maxlen=100000)

        self.model = Linear_QNet(11, 256, 3)
        self.model.load()
        self.target_model = Linear_QNet(11, 256, 3)
        self.target_model.load_state_dict(self.model.state_dict())

        self.trainer = QTrainer(self.model, lr=0.001, gamma=self.gamma)

        self.target_update_freq = 100
        self.batch_size = 10000

    def get_state(self, game):
        return game.get_state()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) < self.batch_size:
            return None
        mini_sample = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        loss = self.trainer.train_step(states, actions, rewards, next_states, dones)

        if self.n_games % self.target_update_freq == 0:
            self.target_model.load_state_dict(self.model.state_dict())

        return loss

    def train_short_memory(self, state, action, reward, next_state, done):
        return self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # epsilon decay
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        final_move = [0, 0, 0]
        if random.random() < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float32)
            self.model.eval()
            with torch.no_grad():
                prediction = self.model(state0.unsqueeze(0))
            move = torch.argmax(prediction).item()
            final_move[move] = 1
            self.model.train()
        return final_move
