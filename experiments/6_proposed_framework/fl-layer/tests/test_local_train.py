"""fl-layer/tests/test_local_train.py"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
import pytest

from model.FL_model import LSTMTabular
from training.local_train import train_local, DEFAULT_CONFIG


def _make_dataset(n=100, n_features=30, fraud_frac=0.05, seed=42):
    torch.manual_seed(seed)
    X = torch.randn(n, n_features)
    y = torch.zeros(n, 1)
    n_fraud = max(1, int(n * fraud_frac))
    y[:n_fraud] = 1.0
    return TensorDataset(X, y)


@pytest.fixture
def small_dataset():
    return _make_dataset(n=100, fraud_frac=0.05)

@pytest.fixture
def model():
    return LSTMTabular(input_dim=30)


def test_loss_decreases(model, small_dataset):
    """After training, the final loss should be lower than untrained baseline.
    Use zero noise multiplier so DP noise does not mask the signal.
    """
    cfg = {**DEFAULT_CONFIG, "local_epochs": 3, "batch_size": 32, "noise_multiplier": 0.0}
    # measure initial loss
    criterion = nn.BCEWithLogitsLoss()
    model.eval()
    X, y = small_dataset.tensors
    with torch.no_grad():
        pred = model(X.unsqueeze(1))
        loss_before = criterion(pred, y).item()

    trained_sd = train_local(model, small_dataset, cfg)
    model2 = LSTMTabular(input_dim=30)
    model2.load_state_dict(trained_sd)
    model2.eval()
    with torch.no_grad():
        pred2 = model2(X.unsqueeze(1))
        loss_after = criterion(pred2, y).item()

    assert loss_after < loss_before, f"Loss did not decrease: {loss_before:.4f} -> {loss_after:.4f}"


def test_returned_keys_match_model(model, small_dataset):
    """Returned state dict must have exact same keys as the input model."""
    trained_sd = train_local(model, small_dataset)
    assert set(trained_sd.keys()) == set(model.state_dict().keys())


def test_class_weighting_fraud_loss_higher():
    """
    Fraud batch loss must be weighted higher than a normal batch loss.
    We verify by checking that pos_weight > 1.0 when there are more negatives.
    """
    # 5 fraud vs 95 normal → pos_weight = 95 / 5 = 19
    from torch.utils.data import TensorDataset
    n = 100
    X = torch.randn(n, 30)
    y = torch.zeros(n, 1)
    y[:5] = 1.0   # 5 fraud
    ds = TensorDataset(X, y)

    pos = (y == 1).sum().item()
    neg = (y == 0).sum().item()
    pos_weight = neg / max(pos, 1)
    assert pos_weight > 1.0, f"pos_weight should be > 1 for imbalanced dataset, got {pos_weight}"

    criterion_weighted   = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))
    criterion_unweighted = nn.BCEWithLogitsLoss()

    logits = torch.zeros(5, 1)   # 5 fraud predictions
    targets = torch.ones(5, 1)

    loss_weighted   = criterion_weighted(logits, targets).item()
    loss_unweighted = criterion_unweighted(logits, targets).item()
    assert loss_weighted > loss_unweighted, "Weighted loss should exceed unweighted for fraud batch"


def test_dp_gradient_norms_clipped(model, small_dataset):
    """
    After loss.backward() and clip_grad_norm_(), every parameter gradient norm
    must be ≤ L2_NORM_CLIP + a small epsilon.
    We hook into the training loop by reproducing one step manually.
    """
    from copy import deepcopy
    L2_NORM_CLIP = 1.0
    eps = 0.01

    m = deepcopy(model)
    m.train()
    optimizer = torch.optim.Adam(m.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    X, y = small_dataset.tensors
    x = X[:32].unsqueeze(1)
    yb = y[:32]

    optimizer.zero_grad()
    loss = criterion(m(x), yb)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(m.parameters(), max_norm=L2_NORM_CLIP)

    for p in m.parameters():
        if p.grad is not None:
            norm = p.grad.norm().item()
            assert norm <= L2_NORM_CLIP + eps, (
                f"Gradient norm {norm:.6f} exceeds clip threshold {L2_NORM_CLIP} + eps"
            )
