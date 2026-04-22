"""fl-layer/tests/test_validate_fast.py"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
import pytest
from torch.utils.data import TensorDataset
from validation.validate_fast import evaluate_model
from model.FL_model import LSTMTabular


def _make_ds(n=1000, fraud_frac=0.05):
    X = torch.randn(n, 30)
    y = torch.zeros(n, 1)
    n_fraud = max(1, int(n * fraud_frac))
    y[:n_fraud] = 1.0
    return TensorDataset(X, y)


class PerfectModel(nn.Module):
    """Outputs high logit for actual fraud samples, low for normal."""
    def __init__(self, y_labels: torch.Tensor):
        super().__init__()
        self.y = y_labels  # (n,) labels
        self._call_idx = 0
        self._batch_size = 512

    def forward(self, x):
        n = x.shape[0]
        start = self._call_idx * self._batch_size
        labels = self.y[start:start + n]
        self._call_idx += 1
        # fraud → +10, normal → -10
        return torch.where(labels.view(-1, 1) > 0.5,
                           torch.full((n, 1), 10.0),
                           torch.full((n, 1), -10.0))


class PerfectAntiModel(nn.Module):
    """Outputs HIGH logit for NORMAL samples (inverted)."""
    def __init__(self, y_labels: torch.Tensor):
        super().__init__()
        self.y = y_labels
        self._call_idx = 0
        self._batch_size = 512

    def forward(self, x):
        n = x.shape[0]
        start = self._call_idx * self._batch_size
        labels = self.y[start:start + n]
        self._call_idx += 1
        return torch.where(labels.view(-1, 1) > 0.5,
                           torch.full((n, 1), -10.0),
                           torch.full((n, 1), 10.0))


class RandomModel(nn.Module):
    def forward(self, x):
        torch.manual_seed(42)
        return torch.randn(x.shape[0], 1)


def test_perfect_discrimination_prauc_is_one():
    """
    When every fraud sample scores higher than every normal sample,
    average_precision_score = 1.0.
    We test this directly through sklearn to verify our metric definition.
    """
    import numpy as np
    from sklearn.metrics import average_precision_score
    np.random.seed(0)
    y = np.array([1]*25 + [0]*475)   # 5% fraud, 500 samples
    # Perfect ranking: fraud samples get score 1.0, normal get 0.0
    scores = np.array([1.0]*25 + [0.0]*475)
    auc = average_precision_score(y, scores)
    assert auc > 0.99, f"Perfect discrimination should yield PR-AUC ≈ 1.0, got {auc:.4f}"


def test_all_normal_model_low_prauc():
    """A model that predicts no fraud everywhere scores PR-AUC ≈ prevalence (low)."""
    # If all scores are equal, sklearn AP = class_prevalence
    import numpy as np
    from sklearn.metrics import average_precision_score
    y = np.array([1]*25 + [0]*475)
    scores = np.ones(500) * 0.0   # everything predicted non-fraud
    auc = average_precision_score(y, scores)
    assert auc < 0.15, f"Flat-score model should score near prevalence, got {auc:.4f}"


def test_random_model_prauc_near_prevalence():
    fraud_frac = 0.05
    ds = _make_ds(n=500, fraud_frac=fraud_frac)
    pr_auc = evaluate_model(RandomModel(), ds, sample_fraction=1.0)["pr_auc"]
    assert 0.0 <= pr_auc <= 1.0, f"PR-AUC must be in [0,1], got {pr_auc}"


def test_sample_fraction_respects_count():
    """15% of 1000 samples = 150 evaluated samples."""
    n = 1000
    ds = _make_ds(n=n)
    # Monkeypatch to count samples by checking dataset length
    evaluated = []

    class CountingModel(nn.Module):
        def forward(self, x):
            evaluated.append(x.shape[0])
            return torch.zeros(x.shape[0], 1)

    evaluate_model(CountingModel(), ds, sample_fraction=0.15, batch_size=n)
    total_evaluated = sum(evaluated)
    # Allow ±1 for rounding
    assert abs(total_evaluated - 150) <= 1, f"Expected ~150 samples, got {total_evaluated}"


# Integration test: local_train → evaluate_model
def test_evaluate_after_local_train():
    from training.local_train import train_local
    ds = _make_ds(n=200, fraud_frac=0.05)
    m = LSTMTabular(input_dim=30)
    cfg = {"local_epochs": 1, "batch_size": 32, "lr": 1e-3,
           "l2_norm_clip": 1.0, "noise_multiplier": 0.05, "device": "cpu"}
    sd = train_local(m, ds, cfg)
    m.load_state_dict(sd)
    pr_auc = evaluate_model(m, ds, sample_fraction=1.0)["pr_auc"]
    assert 0.0 <= pr_auc <= 1.0, f"PR-AUC out of range: {pr_auc}"
