"""
fl-layer/validation/validate_fast.py
PR-AUC evaluator — returns a float score, no threshold logic.

Changes vs. CCFD-FL-layer/src/validation/validate_fast.py:
  - Function accepts (model, dataset) directly instead of loading from disk.
  - Returns float PR-AUC instead of a metrics dict.
  - Threshold logic removed — callers decide what to do with the score.
  - No imports from CCFD-FL-layer or src/.
"""
from __future__ import annotations

import logging
from typing import Union, Dict

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    dataset: TensorDataset,
    sample_fraction: float = 0.15,
    batch_size: int = 512,
    device: Union[str, torch.device] = "cpu",
) -> Dict[str, float]:
    """
    Evaluate a model on a random fraction of the dataset and return a dict of metrics.

    Args:
        model           : any nn.Module producing (batch, 1) logits
        dataset         : TensorDataset (X_float32, y_float32_col)
        sample_fraction : fraction of dataset to evaluate on (default 0.15)
        batch_size      : batch size for inference
        device          : torch device string or object

    Returns:
        Dict containing: 'pr_auc', 'roc_auc', 'f1'
    """
    device = torch.device(device)
    model = model.to(device).eval()

    X, y = dataset.tensors[0], dataset.tensors[1].view(-1)
    n = len(X)
    m = max(1, int(n * sample_fraction))
    idx = torch.randperm(n)[:m]

    sample_ds = TensorDataset(X[idx], y[idx])
    loader = DataLoader(sample_ds, batch_size=batch_size, shuffle=False)

    probs, trues = [], []
    for xb, yb in loader:
        xb = xb.unsqueeze(1).to(device)          # add time dim
        logits = model(xb).view(-1)
        probs.append(torch.sigmoid(logits).cpu().numpy())
        trues.append(yb.cpu().numpy())

    y_prob = np.concatenate(probs)
    y_true = np.concatenate(trues).astype(int)

    pr_auc = float(average_precision_score(y_true, y_prob))
    
    # Needs at least one positive and one negative for roc_auc
    if len(np.unique(y_true)) > 1:
        roc_auc = float(roc_auc_score(y_true, y_prob))
    else:
        roc_auc = 0.5
        
    y_pred = (y_prob >= 0.5).astype(int)
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    
    from sklearn.metrics import precision_score, recall_score
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))

    metrics = {
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "f1": f1,
        "precision": precision,
        "recall": recall
    }

    logger.debug(
        "evaluate_model: n_samples=%d fraction=%.2f metrics=%s",
        m, sample_fraction, metrics,
    )
    return metrics
