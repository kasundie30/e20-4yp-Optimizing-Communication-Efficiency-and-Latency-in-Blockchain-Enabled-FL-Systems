import torch, torch.nn as nn

class LSTMTabular(nn.Module):
    """LSTM fraud classifier — Baabdullah et al. (2024) Table 4:
       input_dim=30, hidden_dim=30, num_layers=1
    """
    def __init__(self, input_dim=30, hidden_dim=30, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim,
                            num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x: (batch, seq=1, features)
        out, (h, _) = self.lstm(x)
        return self.fc(h[-1])   # (batch, 1)
