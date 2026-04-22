"""
fl-layer/tests/test_integration.py
Capstone Phase 5 integration test — full FL pipeline with no blockchain.

Scenario:
  1. Three non-IID synthetic branch datasets (different class distributions)
  2. local_train × 3 → state_dicts + sample counts
  3. fedavg → brand model
  4. evaluate_model → PR-AUC ∈ [0, 1]
  5. blend_with_global (brand vs randomly init global) → blended model
  6. Load blended model into LSTMTabular → valid predictions
  7. wait_for_submissions → all 3 collected within deadline
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from torch.utils.data import TensorDataset

from model.FL_model import LSTMTabular
from training.local_train import train_local
from aggregation.fedavg import fedavg
from validation.validate_fast import evaluate_model
from resilience.backup_logic import blend_with_global
from resilience.deadline_collect import wait_for_submissions


def _make_noniid_dataset(seed: int, n: int = 150, fraud_frac: float = 0.05, n_features: int = 30):
    """Small synthetic dataset with slightly different distributions per seed."""
    torch.manual_seed(seed)
    X = torch.randn(n, n_features) + (seed * 0.1)   # slight distribution shift
    y = torch.zeros(n, 1)
    n_fraud = max(1, int(n * fraud_frac))
    y[:n_fraud] = 1.0
    return TensorDataset(X, y)


TRAIN_CFG = {
    "local_epochs": 1,
    "batch_size": 30,
    "lr": 1e-3,
    "l2_norm_clip": 1.0,
    "noise_multiplier": 0.05,
    "device": "cpu",
}


def test_full_fl_pipeline_no_blockchain():
    """Full pipeline: 3 branches → train → fedavg → validate → blend → predict → collect."""

    # Step 1: Three non-IID branch datasets
    datasets = [
        _make_noniid_dataset(seed=0, n=120, fraud_frac=0.03),
        _make_noniid_dataset(seed=1, n=150, fraud_frac=0.06),
        _make_noniid_dataset(seed=2, n=200, fraud_frac=0.02),
    ]

    # Step 2: Local training on each branch
    base_model = LSTMTabular(input_dim=30)
    updates = []
    for i, ds in enumerate(datasets):
        sd = train_local(base_model, ds, TRAIN_CFG)
        n_samples = len(ds.tensors[0])
        updates.append((sd, n_samples))

    assert len(updates) == 3
    for sd, n in updates:
        assert set(sd.keys()) == set(base_model.state_dict().keys())
        assert n > 0

    # Step 3: FedAvg → brand model
    brand_sd = fedavg(updates)
    brand_model = LSTMTabular(input_dim=30)
    brand_model.load_state_dict(brand_sd)

    # Step 4: Evaluate brand model → PR-AUC in [0, 1]
    # Use first dataset as validation set
    pr_auc = evaluate_model(brand_model, datasets[0], sample_fraction=0.5)["pr_auc"]
    assert 0.0 <= pr_auc <= 1.0, f"PR-AUC out of range: {pr_auc}"

    # Step 5: Blend brand with a random global model (simulating backup scenario)
    global_model = LSTMTabular(input_dim=30)    # randomly initialized = "old global"
    blended_sd = blend_with_global(
        brand_model=brand_sd,
        global_model=global_model.state_dict(),
        beta=0.3,
    )

    # Step 6: Load blended model → valid predictions
    blended_model = LSTMTabular(input_dim=30)
    blended_model.load_state_dict(blended_sd)
    blended_model.eval()
    with torch.no_grad():
        x_test = torch.randn(8, 1, 30)
        out = blended_model(x_test)
    assert out.shape == (8, 1), f"Unexpected shape: {out.shape}"

    # Step 7: wait_for_submissions with preloaded updates (all 3 available immediately)
    all_updates = list(updates)   # already collected
    collected = wait_for_submissions(
        expected_count=3,
        collect_fn=lambda: all_updates,
        deadline_sec=5.0,
        poll_interval=0.1,
    )
    assert len(collected) == 3, f"Expected 3 submissions, got {len(collected)}"
    print(f"\n✅ Full FL pipeline integration test passed. PR-AUC={pr_auc:.4f}")
