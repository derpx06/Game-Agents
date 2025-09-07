import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float32)
        next_state = torch.tensor(next_state, dtype=torch.float32)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float32)

        single_sample = len(state.shape) == 1
        if single_sample:
            state = state.unsqueeze(0)
            next_state = next_state.unsqueeze(0)
            action = action.unsqueeze(0)
            reward = reward.unsqueeze(0)
            done = (done,)

        if single_sample:
            self.model.eval()
        else:
            self.model.train()

        pred = self.model(state)
        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                next_state_idx = next_state[idx].unsqueeze(0) if len(next_state[idx].shape) == 1 else next_state[idx]
                self.model.eval()
                with torch.no_grad():
                    Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state_idx))
                if not single_sample:
                    self.model.train()
            target[idx][torch.argmax(action[idx]).item()] = Q_new

        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()
        return loss.item()