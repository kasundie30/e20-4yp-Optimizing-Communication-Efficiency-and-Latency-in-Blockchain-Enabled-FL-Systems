"""
metrics_utils.py — All 8 evaluation metrics for Aljunaid et al. (2025).

Metrics:
  1. PR-AUC          (Average Precision)
  2. ROC-AUC
  3. F1 Score        (binary, threshold=0.5)
  4. Precision
  5. Recall
  6. Comm MB         (total model bytes uploaded + downloaded per round)
  7. E2E sec         (max client training time + server aggregation time)

Communication model (sklearn):
  upload   = Σ serialised local model sizes (one per client)
  download = serialised global model size × NUM_CLIENTS
  total    = upload + download
"""
import warnings
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_fscore_support
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config


def predict_proba_sklearn(model, X: np.ndarray) -> np.ndarray:
    """Return P(fraud=1) for any sklearn estimator."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    # Decision function fallback (e.g. LinearSVC without calibration)
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        # Normalise to [0, 1] via sigmoid
        return 1.0 / (1.0 + np.exp(-scores))
    raise ValueError(f"Model {type(model).__name__} has no predict_proba or decision_function")


def classification_metrics(y_true: np.ndarray, y_prob: np.ndarray,
                            threshold: float = config.THRESHOLD) -> dict:
    """Compute PR-AUC, ROC-AUC, F1, Precision, Recall."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prauc  = float(average_precision_score(y_true, y_prob))
        rocauc = float(roc_auc_score(y_true, y_prob))
        y_pred = (y_prob >= threshold).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", zero_division=0)
    return {
        "prauc":     prauc,
        "rocauc":    rocauc,
        "f1":        float(f1),
        "precision": float(p),
        "recall":    float(r),
    }


def communication_overhead_mb(upload_bytes_list: list, global_bytes: int,
                               num_clients: int) -> float:
    """
    Total communication cost in MB for one FL round.
    upload   = Σ client model sizes
    download = global model × num_clients (broadcast)
    """
    total_bytes = sum(upload_bytes_list) + global_bytes * num_clients
    return total_bytes / (1024 ** 2)


def end_to_end_latency_sec(client_times: list, agg_time: float) -> float:
    """
    E2E = max(parallel client training times) + aggregation time.
    Clients train in parallel (simulated sequentially), so we take max.
    """
    return max(client_times) + agg_time
