# src/validation/metrics.py
# Metrics calculation

from __future__ import annotations
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_fscore_support

def pr_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return float(average_precision_score(y_true, y_prob))

def fraud_metrics(y_true: np.ndarray, y_prob: np.ndarray, thr: float = 0.5) -> dict:
    y_pred = (y_prob >= thr).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    return {
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "prauc": pr_auc(y_true, y_prob),
    }