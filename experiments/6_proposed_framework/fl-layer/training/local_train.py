"""
fl-layer/training/local_train.py
Cleaned DP local training loop.

Changes vs. CCFD-FL-layer/local_train.py:
  - No filesystem I/O — model weights come in as state_dict, go out as state_dict
  - No env-var reads (BANK_ID, /data, /logs)
  - No CCFD-FL-layer imports
  - DP order verified: clip → noise → optimizer.step (was already correct)
  - Uses logging instead of print
  - Function signature: train_local(model, dataset, config) -> state_dict
"""
import logging
from copy import deepcopy
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)

# Default training hyper-parameters (override via config dict)
DEFAULT_CONFIG: Dict[str, Any] = {
    "local_epochs": 2,
    "lr": 1e-3,
    "batch_size": 256,
    "l2_norm_clip": 1.0,   # DP gradient clipping threshold
    "noise_multiplier": 0.05,
    "device": "cpu",
}


def train_local(
    model: nn.Module,
    dataset: TensorDataset,
    config: Dict[str, Any] | None = None,
) -> Dict[str, torch.Tensor]:
    """
    Run local DP training on `dataset` starting from `model`'s current weights.

    DP order (per-sample gradient DP approximation):
        1. loss.backward() — accumulate gradients
        2. clip_grad_norm_()  ← clip BEFORE noise
        3. add Gaussian noise to each gradient
        4. optimizer.step()   ← update AFTER clip+noise

    Args:
        model   : LSTMTabular (or any nn.Module) with initial weights
        dataset : TensorDataset with tensors (X: float32, y: float32 col)
        config  : optional dict overriding DEFAULT_CONFIG keys

    Returns:
        state_dict (dict[str, Tensor]) — trained weights, keys match input model
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    device = torch.device(cfg["device"])

    # Work on a deep copy so caller's model is unaffected
    model = deepcopy(model).to(device)

    X_all, y_all = dataset.tensors[0], dataset.tensors[1]

    # Class weighting to handle fraud imbalance
    pos = (y_all == 1).sum().item()
    neg = (y_all == 0).sum().item()
    pos_weight = torch.tensor([neg / max(pos, 1)], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    loader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=cfg["lr"])

    model.train()
    for epoch in range(cfg["local_epochs"]):
        epoch_loss = 0.0
        batches = 0
        for x, y in loader:
            # Add time dimension: (batch, features) → (batch, 1, features)
            x = x.unsqueeze(1).to(device)
            y = y.to(device)

            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()

            # DP Step 1: clip gradients (BEFORE noise)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=cfg["l2_norm_clip"]
            )

            # DP Step 2: add calibrated Gaussian noise (AFTER clip)
            with torch.no_grad():
                for p in model.parameters():
                    if p.grad is not None:
                        noise = torch.randn_like(p.grad) * (
                            cfg["l2_norm_clip"] * cfg["noise_multiplier"]
                        )
                        p.grad.add_(noise)

            # DP Step 3: optimizer step (AFTER clip+noise)
            optimizer.step()

            epoch_loss += loss.item()
            batches += 1

        logger.debug("Epoch %d/%d — avg loss %.6f", epoch + 1, cfg["local_epochs"], epoch_loss / max(batches, 1))

    return model.state_dict()
