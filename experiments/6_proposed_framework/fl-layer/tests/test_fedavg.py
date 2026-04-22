"""fl-layer/tests/test_fedavg.py"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import pytest
from aggregation.fedavg import fedavg, ModelKeyMismatchError, ZeroSamplesError


def _make_sd(val: float):
    """State dict where all weights are a constant tensor."""
    return {
        "lstm.weight_ih_l0": torch.full((120, 30), val),
        "lstm.weight_hh_l0": torch.full((120, 30), val),
        "lstm.bias_ih_l0":   torch.full((120,), val),
        "lstm.bias_hh_l0":   torch.full((120,), val),
        "fc.weight":         torch.full((1, 30),  val),
        "fc.bias":           torch.full((1,),      val),
    }


def test_numerical_correctness_three_models():
    """Three models with known weights and sample counts → manual weighted avg."""
    sd1 = _make_sd(1.0)
    sd2 = _make_sd(2.0)
    sd3 = _make_sd(3.0)
    # samples: 100, 200, 300 → weights: 1/6, 2/6, 3/6 → expected avg: 1/6+4/6+9/6 = 14/6 ≈ 2.333
    expected = (100 * 1.0 + 200 * 2.0 + 300 * 3.0) / 600
    result = fedavg([(sd1, 100), (sd2, 200), (sd3, 300)])
    for k in result:
        avg_val = result[k].float().mean().item()
        assert abs(avg_val - expected) < 1e-5, f"Key {k}: expected {expected:.6f}, got {avg_val:.6f}"


def test_single_model_unchanged():
    """Single model input must return identical weights."""
    sd = _make_sd(7.5)
    result = fedavg([(sd, 50)])
    for k in result:
        assert torch.allclose(result[k].float(), sd[k].float(), atol=1e-6)


def test_equal_sample_sizes_simple_average():
    """Two models with equal sample sizes produce a simple average."""
    sd1 = _make_sd(0.0)
    sd2 = _make_sd(4.0)
    result = fedavg([(sd1, 100), (sd2, 100)])
    for k in result:
        assert abs(result[k].float().mean().item() - 2.0) < 1e-5


def test_mismatched_keys_raises_error():
    """Models with different keys must raise ModelKeyMismatchError."""
    sd1 = {"a": torch.tensor(1.0), "b": torch.tensor(2.0)}
    sd2 = {"a": torch.tensor(1.0), "c": torch.tensor(3.0)}  # 'c' instead of 'b'
    with pytest.raises(ModelKeyMismatchError):
        fedavg([(sd1, 10), (sd2, 10)])


def test_zero_samples_raises_error():
    """Zero sample count must raise ZeroSamplesError."""
    sd = _make_sd(1.0)
    with pytest.raises(ZeroSamplesError):
        fedavg([(sd, 0)])


# ----- Integration test: local_train → fedavg -----

def test_fedavg_integration_with_local_train():
    """local_train × 2 → fedavg → loads into LSTMTabular → valid predictions."""
    from torch.utils.data import TensorDataset
    from model.FL_model import LSTMTabular
    from training.local_train import train_local

    def _ds(seed):
        torch.manual_seed(seed)
        X = torch.randn(60, 30)
        y = torch.zeros(60, 1)
        y[:5] = 1.0
        return TensorDataset(X, y)

    m = LSTMTabular(input_dim=30)
    cfg = {"local_epochs": 1, "batch_size": 30, "lr": 1e-3,
           "l2_norm_clip": 1.0, "noise_multiplier": 0.05, "device": "cpu"}

    sd1 = train_local(m, _ds(0), cfg)
    sd2 = train_local(m, _ds(1), cfg)

    avg_sd = fedavg([(sd1, 60), (sd2, 60)])

    m2 = LSTMTabular(input_dim=30)
    m2.load_state_dict(avg_sd)
    m2.eval()
    X_test = torch.randn(4, 1, 30)
    with torch.no_grad():
        out = m2(X_test)
    assert out.shape == (4, 1), f"Unexpected output shape: {out.shape}"
