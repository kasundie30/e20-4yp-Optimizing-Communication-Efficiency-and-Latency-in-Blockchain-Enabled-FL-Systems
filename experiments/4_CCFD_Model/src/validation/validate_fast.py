# src/validation/validate_fast.py
# Fast validation on a sample of data

from __future__ import annotations
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

from dataset import load_bank_dataset
from FL_model import LSTMTabular
from src.validation.metrics import fraud_metrics

@torch.no_grad()
def fast_validate_state_dict(
    state_dict: dict,
    node_id_for_data: str,
    data_root: str,
    hidden_dim: int = 30,
    num_layers: int = 1,
    fraction: float = 0.15,
    batch_size: int = 512,
    device: str = "cpu"
) -> dict:
    ds = load_bank_dataset(node_id_for_data, data_path=data_root)
    X, y = ds.tensors[0], ds.tensors[1].view(-1)

    n = len(X)
    m = max(1, int(n * fraction))
    idx = torch.randperm(n)[:m]

    Xs = X[idx]
    ys = y[idx]

    loader = DataLoader(TensorDataset(Xs, ys), batch_size=batch_size, shuffle=False)

    input_dim = X.shape[1]
    model = LSTMTabular(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    probs = []
    trues = []
    for xb, yb in loader:
        xb = xb.unsqueeze(1).to(device)
        logits = model(xb).view(-1)
        p = torch.sigmoid(logits).detach().cpu().numpy()
        probs.append(p)
        trues.append(yb.detach().cpu().numpy())

    y_prob = np.concatenate(probs)
    y_true = np.concatenate(trues).astype(int)
    return fraud_metrics(y_true, y_prob)