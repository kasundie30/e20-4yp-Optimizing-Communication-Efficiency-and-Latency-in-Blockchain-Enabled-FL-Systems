"""
fl_engine/client.py — Local bank client for Aljunaid et al. (2025).

Each client:
  1. Receives the current global model (serialised bytes via joblib)
  2. Fine-tunes a fresh GBM on its local shard
  3. Returns (model, n_samples, elapsed_sec, upload_bytes)

The global model received from the server is used as a warm-start hint
for GBM (warm_start strategy: init with global parameters where possible).
For sklearn estimators we simply train from scratch each round — this is
consistent with the paper's description where local banks train independently.
"""
import time
import copy
import io
import joblib
import numpy as np
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from models.ml_models import build_model


class LocalClient:
    def __init__(self, client_id: str, X_train: np.ndarray, y_train: np.ndarray,
                 model_name: str = "GBM"):
        self.client_id   = client_id
        self.model_name  = model_name
        self.X_train     = X_train
        self.y_train     = y_train
        self.n_samples   = len(X_train)

    def train_one_round(self, global_model=None):
        """
        Train locally. If global_model is provided and is a warm-start-capable
        GBM, use warm_start to extend it; otherwise train fresh.

        Returns
        -------
        model        : fitted sklearn estimator
        n_samples    : int
        elapsed_sec  : float
        upload_bytes : int  — serialised model size (communication cost)
        """
        # Build or clone the global model for local training
        if global_model is not None and self.model_name == "GBM":
            # Warm-start: clone global GBM and continue fitting
            model = copy.deepcopy(global_model)
            model.set_params(warm_start=True,
                             n_estimators=model.n_estimators + 20)
        else:
            model = build_model(self.model_name)

        t0 = time.perf_counter()
        model.fit(self.X_train, self.y_train)
        elapsed = time.perf_counter() - t0

        # Measure upload cost (serialised model bytes)
        buf = io.BytesIO()
        joblib.dump(model, buf)
        upload_bytes = buf.tell()

        return model, self.n_samples, elapsed, upload_bytes
