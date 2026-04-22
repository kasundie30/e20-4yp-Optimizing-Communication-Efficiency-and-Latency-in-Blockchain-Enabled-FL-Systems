"""
fl-integration/tests/test_real_data_e2e.py
Integration test that verifies the full FL pipeline using REAL CCFD CSV data.

- Loads actual train_ready.csv from data/splits/fl_clients/BankA
- Trains one local round using real local_train()
- Evaluates with real evaluate_model() (NO monkeypatch)
- Asserts metrics are genuine values (not the fake 0.85 constant)
- Skips gracefully if the CSV files don't exist (so CI doesn't break on a fresh clone)
"""
import sys, os
import pytest
import torch
from torch.utils.data import TensorDataset

# Path setup
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "fl-layer"))

DATA_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "data", "splits", "fl_clients")
)
BANKA_CSV = os.path.join(DATA_ROOT, "BankA", "train_ready.csv")
TEST_CSV  = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "data", "splits", "test", "global_test.csv")
)

real_data_present = pytest.mark.skipif(
    not os.path.exists(BANKA_CSV),
    reason=f"Real data not found at {BANKA_CSV}. Run data/prepare_fl_splits.py first.",
)

TRAIN_CFG = {
    "local_epochs": 1,
    "batch_size":   256,
    "lr":           1e-3,
    "l2_norm_clip": 1.0,
    "noise_multiplier": 0.0,   # zero noise keeps unit tests deterministic
    "device":       "cpu",
}
# Real data: 29 features (V1-V28 + Amount; Time was dropped in preprocessing)
MODEL_CFG = {"input_dim": 29, "hidden_dim": 30, "num_layers": 1}


# ── Helpers ─────────────────────────────────────────────────────────────────

def _load_bank(bank_id: str, max_rows: int = 2000) -> TensorDataset:
    """Load a small slice of a bank's real CSV for fast testing."""
    from model.dataset import load_bank_dataset
    ds = load_bank_dataset(bank_id=bank_id, data_path=DATA_ROOT)
    # Trim to max_rows for speed
    X, y = ds.tensors
    if len(X) > max_rows:
        X, y = X[:max_rows], y[:max_rows]
    return TensorDataset(X, y)


# ── Tests ────────────────────────────────────────────────────────────────────

@real_data_present
def test_real_data_loads_correctly():
    """CSV loads without error and has correct shape (30 features + 1 label)."""
    ds = _load_bank("BankA")
    X, y = ds.tensors
    assert X.shape[1] == 29, f"Expected 29 features, got {X.shape[1]}"
    assert y.shape[1] == 1,  f"Expected y shape (N,1), got {y.shape}"
    assert len(X) > 0,       "Dataset is empty"


@real_data_present
def test_real_data_all_banks_load():
    """All three bank CSVs load and have 30 features."""
    from model.dataset import load_bank_dataset
    for bank in ["BankA", "BankB", "BankC"]:
        ds = load_bank_dataset(bank_id=bank, data_path=DATA_ROOT)
        X, y = ds.tensors
        assert X.shape[1] == 29, f"[{bank}] Expected 29 features, got {X.shape[1]}"
        assert len(X) > 0,       f"[{bank}] Dataset is empty"


@real_data_present
def test_local_training_on_real_data():
    """train_local() runs to completion on real data without errors."""
    from model.FL_model import LSTMTabular
    from training.local_train import train_local

    ds    = _load_bank("BankA")
    model = LSTMTabular(**MODEL_CFG)
    sd    = train_local(model, ds, TRAIN_CFG)

    assert isinstance(sd, dict), "train_local() must return a state_dict"
    assert "lstm.weight_ih_l0" in sd
    assert "fc.weight"         in sd


@real_data_present
def test_evaluate_model_returns_real_metrics():
    """
    evaluate_model() produces genuine values — NOT the hardcoded 0.85 fake.
    This test fails if someone re-adds the monkeypatch.
    """
    from model.FL_model import LSTMTabular
    from training.local_train import train_local
    from validation.validate_fast import evaluate_model

    ds    = _load_bank("BankC")   # Use BankC — it has higher fraud rate (2.33%)
    model = LSTMTabular(**MODEL_CFG)
    sd    = train_local(model, ds, TRAIN_CFG)
    model.load_state_dict(sd)

    metrics = evaluate_model(model, ds, sample_fraction=1.0)

    # Metrics must be proper floats in [0, 1]
    for key in ["pr_auc", "roc_auc", "f1", "precision", "recall"]:
        assert key in metrics, f"Missing metric: {key}"
        val = metrics[key]
        assert isinstance(val, float),         f"{key} is not a float (got {type(val)})"
        assert 0.0 <= val <= 1.0,              f"{key}={val:.4f} outside [0,1]"

    # The fake monkeypatch always returned exactly 0.85 for all metrics.
    # After real training, at least one metric must differ from 0.85.
    all_fake = all(abs(metrics[k] - 0.85) < 1e-6 for k in ["pr_auc", "f1", "roc_auc"])
    assert not all_fake, (
        "All metrics are exactly 0.85 — the evaluate_model monkeypatch may still be active!\n"
        f"metrics={metrics}"
    )


@real_data_present
def test_input_dim_matches_real_data():
    """
    Confirm model input_dim=29 matches the real dataset (29 features: V1-V28 + Amount).
    Time column was dropped in preprocessing. A model with input_dim=30 would fail.
    """
    from model.FL_model import LSTMTabular

    ds    = _load_bank("BankB")
    X, _  = ds.tensors
    assert X.shape[1] == 29, f"Real data must have 29 features, got {X.shape[1]}"

    # Model with correct input_dim=29 must forward-pass without error
    model    = LSTMTabular(input_dim=29, hidden_dim=30, num_layers=1)
    x_sample = X[:4].unsqueeze(1)     # (batch=4, timestep=1, features=29)
    out      = model(x_sample)
    assert out.shape == (4, 1), f"Unexpected output shape: {out.shape}"


@real_data_present
@pytest.mark.skipif(not os.path.exists(TEST_CSV), reason="global_test.csv not found")
def test_global_test_set_loads():
    """Global test CSV loads and has the expected schema."""
    import pandas as pd
    df = pd.read_csv(TEST_CSV)
    assert df.shape[1] == 30, f"Expected 30 columns (29 features + Class), got {df.shape[1]}"
    assert "Class" in df.columns
    assert len(df) > 100, "Global test set too small"


@real_data_present
def test_full_round_with_real_data():
    """
    End-to-end: load real data → train → FedAvg → evaluate.
    Uses mock IPFS + mock API. No blockchain needed.
    """
    from unittest.mock import MagicMock
    from api_client import APIClient, APIError, compute_sha256
    from model.FL_model import LSTMTabular
    from training.local_train import train_local
    from hq_agent import HQAgent

    # Mock out blockchain/IPFS
    store = {}
    def mock_upload(data):
        cid = "Qm" + compute_sha256(data)[:10]
        store[cid] = data
        return cid
    def mock_download(cid):
        return store.get(cid, b"")

    mock_api = MagicMock(spec=APIClient)
    mock_api.get_global_model.side_effect = APIError(404, "No prior global")
    mock_api.submit_update.return_value   = {"status": "success", "tx_id": "tx-test"}

    # Load real BankC (smallest, fastest)
    ds    = _load_bank("BankC", max_rows=500)
    agent = HQAgent(
        bank_id="BankC",
        client=mock_api,
        ipfs_upload=mock_upload,
        ipfs_download=mock_download,
        val_dataset=ds,
        val_threshold=0.0,
        model_cfg=MODEL_CFG,
    )

    # Train locally
    model = LSTMTabular(**MODEL_CFG)
    sd    = train_local(model, ds, TRAIN_CFG)

    # Run HQ round
    result = agent.run_round(round_num=1, branch_updates=[(sd, len(ds))])

    assert "val_score" in result
    val_score = result["val_score"]
    assert isinstance(val_score, float)
    assert 0.0 <= val_score <= 1.0

    # Must have uploaded something to mock IPFS
    if result.get("submitted"):
        assert result["model_cid"] in store, "Model CID not found in mock IPFS"

    # Critically — val_score should NOT be exactly 0.85 (that's the monkeypatch value)
    assert abs(val_score - 0.85) > 1e-6 or not result.get("submitted"), (
        f"val_score is exactly 0.85 — monkeypatch may still be active!"
    )
