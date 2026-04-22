import torch
import torch.nn as nn

class LSTMTabular(nn.Module):
    def __init__(self, input_dim=30, hidden_dim=30, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, (h, c) = self.lstm(x)
        return self.fc(h[-1])