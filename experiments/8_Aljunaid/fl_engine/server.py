"""
fl_engine/server.py — Global server for Aljunaid et al. (2025).

Paper Eq. 2 — Best-model selection aggregation:
    W* = argmax_{Wi} A(Wi, Vi)

The server evaluates each local model on a shared validation dataset and
selects the best-performing model as the new global model. If none exceeds
the performance threshold τ, it requests additional training.

This contrasts with FedAvg (used in Baabdullah) — here NO weight averaging
is performed; we simply select the winning local model.
"""
import io
import time
import copy
import joblib
import numpy as np
from sklearn.metrics import accuracy_score
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class BestModelServer:
    """
    Best-model FL server implementing paper Eq. 2.
    Maintains the current global model and evaluates candidates each round.
    """
    def __init__(self, X_val: np.ndarray, y_val: np.ndarray):
        """
        Parameters
        ----------
        X_val, y_val : Validation set held by the server.
                       In practice this is the global test set
                       (consistent with the paper's evaluation description).
        """
        self.X_val        = X_val
        self.y_val        = y_val
        self.global_model = None   # Updated each round

    def aggregate(self, candidates):
        """
        Select the best local model.

        Parameters
        ----------
        candidates : list of (model, n_samples, upload_bytes)

        Returns
        -------
        best_model    : selected global model
        best_acc      : float, accuracy on validation set
        agg_sec       : float, time spent in selection
        global_bytes  : int, serialised global model size (download cost)
        """
        t0 = time.perf_counter()
        best_model, best_acc, best_idx = None, -1.0, -1

        for i, (model, n, _) in enumerate(candidates):
            y_pred = model.predict(self.X_val)
            acc    = float(accuracy_score(self.y_val, y_pred))
            if acc > best_acc:
                best_acc, best_model, best_idx = acc, model, i

        self.global_model = copy.deepcopy(best_model)
        agg_sec = time.perf_counter() - t0

        # Serialise global model to measure download cost
        buf = io.BytesIO()
        joblib.dump(self.global_model, buf)
        global_bytes = buf.tell()

        return self.global_model, best_acc, agg_sec, global_bytes, best_idx

    def get_global_model(self):
        return copy.deepcopy(self.global_model) if self.global_model else None

    def save(self, path: str):
        joblib.dump(self.global_model, path)
        print(f"[SERVER] Global model saved → {path}")
