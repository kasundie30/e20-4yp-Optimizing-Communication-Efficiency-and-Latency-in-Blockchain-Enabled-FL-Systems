"""
fl-layer/model/FL_model.py
Cleaned LSTMTabular architecture ported from CCFD-FL-layer.

Architecture is intentionally unchanged so that saved .pt weight files
remain compatible.  Only cleanup applied: logging instead of print, and
no filesystem side-effects.
"""
import logging
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class LSTMTabular(nn.Module):
    """
    Single-layer LSTM for tabular fraud-detection data.

    Args:
        input_dim  : number of input features (default 30 for Kaggle dataset)
        hidden_dim : LSTM hidden state size (default 30)
        num_layers : stacked LSTM layers (default 1)

    Input shape  : (batch, timesteps, input_dim)   — caller adds the time dim
    Output shape : (batch, 1)
    """

    def __init__(self, input_dim: int = 30, hidden_dim: int = 30, num_layers: int = 1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_dim, 1)
        logger.debug(
            "LSTMTabular initialised: input_dim=%d hidden_dim=%d num_layers=%d",
            input_dim, hidden_dim, num_layers,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, timesteps, input_dim) → (batch, 1)"""
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1])
