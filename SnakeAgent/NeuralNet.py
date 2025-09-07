import torch
import torch.nn as nn
import torch.nn.functional as F
import os

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size // 2)
        self.bn2 = nn.BatchNorm1d(hidden_size // 2)
        self.linear3 = nn.Linear(hidden_size // 2, hidden_size // 4)
        self.bn3 = nn.BatchNorm1d(hidden_size // 4)
        self.linear4 = nn.Linear(hidden_size // 4, output_size)
        self.dropout = nn.Dropout(0.3)

    def load(self, file_name='model.pth'):
        model_folder_path = './model'
        file_name = os.path.join(model_folder_path, file_name)
        if os.path.exists(file_name):
            self.load_state_dict(torch.load(file_name))
            print(f"✅ Loaded pretrained model from {file_name}")
        else:
            print("⚠️ No pretrained model found, starting fresh.")

    def forward(self, x):
        x = F.relu(self.bn1(self.linear1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.linear2(x)))
        x = self.dropout(x)
        x = F.relu(self.bn3(self.linear3(x)))
        x = self.linear4(x)
        return x

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)