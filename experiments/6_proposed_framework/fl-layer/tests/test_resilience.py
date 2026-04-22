"""fl-layer/tests/test_resilience.py — covers both deadline_collect and backup_logic."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import time
import torch
import pytest
from resilience.deadline_collect import wait_for_submissions
from resilience.backup_logic import blend_with_global


# ========================
# deadline_collect tests
# ========================

def test_deadline_returns_partial_when_slow():
    """2/3 arrive immediately, 3rd never arrives → 2 submissions at deadline."""
    submissions = ["A", "B"]   # 3rd never added

    def collect_fn():
        return list(submissions)

    # Short 1s deadline for test speed
    result = wait_for_submissions(3, collect_fn, deadline_sec=1.0, poll_interval=0.2)
    assert len(result) == 2


def test_early_return_when_all_arrive():
    """All 3 arrive immediately → function returns fast without waiting full deadline."""
    subs = ["A", "B", "C"]
    start = time.time()
    result = wait_for_submissions(3, lambda: subs, deadline_sec=10.0, poll_interval=0.1)
    elapsed = time.time() - start
    assert len(result) == 3
    assert elapsed < 2.0, f"Should return early; took {elapsed:.2f}s"


def test_empty_submissions_on_deadline():
    """0 submissions → empty list returned at deadline (no hang)."""
    result = wait_for_submissions(3, lambda: [], deadline_sec=0.5, poll_interval=0.1)
    assert result == []


# ========================
# backup_logic tests
# ========================

def _sd(val: float):
    return {
        "fc.weight": torch.full((1, 30), val, dtype=torch.float32),
        "fc.bias":   torch.full((1,),     val, dtype=torch.float32),
    }


def test_blend_beta_half_exact_midpoint():
    """beta=0.5 → result = exact midpoint of brand and global."""
    brand  = _sd(0.0)
    global_ = _sd(4.0)
    result = blend_with_global(brand, global_, beta=0.5)
    for k in result:
        val = result[k].mean().item()
        assert abs(val - 2.0) < 1e-5, f"Expected 2.0, got {val}"


def test_blend_beta_zero_equals_brand():
    """beta=0.0 → output == brand model."""
    brand  = _sd(3.0)
    global_ = _sd(9.0)
    result = blend_with_global(brand, global_, beta=0.0)
    for k in result:
        assert torch.allclose(result[k], brand[k], atol=1e-5)


def test_blend_beta_one_equals_global():
    """beta=1.0 → output == global model."""
    brand  = _sd(1.0)
    global_ = _sd(5.0)
    result = blend_with_global(brand, global_, beta=1.0)
    for k in result:
        assert torch.allclose(result[k].float(), global_[k].float(), atol=1e-5)


def test_blend_output_has_same_keys():
    brand  = _sd(1.0)
    global_ = _sd(2.0)
    result = blend_with_global(brand, global_, beta=0.3)
    assert set(result.keys()) == set(brand.keys())


# Integration: local_train × 2 → blend → loads into LSTMTabular → valid predictions
def test_blend_integration_with_local_train():
    from torch.utils.data import TensorDataset
    from model.FL_model import LSTMTabular
    from training.local_train import train_local

    def _ds(seed):
        torch.manual_seed(seed)
        X = torch.randn(60, 30); y = torch.zeros(60, 1); y[:3] = 1.0
        return TensorDataset(X, y)

    cfg = {"local_epochs": 1, "batch_size": 30, "lr": 1e-3,
           "l2_norm_clip": 1.0, "noise_multiplier": 0.05, "device": "cpu"}
    m = LSTMTabular()
    sd1 = train_local(m, _ds(0), cfg)
    sd2 = train_local(m, _ds(1), cfg)

    blended = blend_with_global(sd1, sd2, beta=0.3)
    m2 = LSTMTabular()
    m2.load_state_dict(blended)
    m2.eval()
    with torch.no_grad():
        out = m2(torch.randn(4, 1, 30))
    assert out.shape == (4, 1)
