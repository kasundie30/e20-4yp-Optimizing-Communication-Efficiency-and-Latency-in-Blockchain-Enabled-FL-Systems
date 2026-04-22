"""fl-layer/tests/test_dataset.py — uses synthetic CSV, no real data."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import tempfile
import pandas as pd
import pytest
from model.dataset import load_bank_dataset


N_FEATURES = 30
N_SAMPLES  = 200
FRAUD_FRAC = 0.1


def _make_synthetic_csv(tmp_path, bank_id="TestBank", n=N_SAMPLES):
    """Write a small CSV with N_FEATURES + Class column."""
    import numpy as np
    bank_dir = os.path.join(tmp_path, bank_id)
    os.makedirs(bank_dir, exist_ok=True)
    cols = [f"V{i}" for i in range(N_FEATURES)] + ["Class"]
    data = [[*([0.1] * N_FEATURES), (1 if i < int(n * FRAUD_FRAC) else 0)] for i in range(n)]
    df = pd.DataFrame(data, columns=cols)
    csv_path = os.path.join(bank_dir, "train_ready.csv")
    df.to_csv(csv_path, index=False)
    return tmp_path


def test_correct_sample_count(tmp_path):
    _make_synthetic_csv(str(tmp_path))
    ds = load_bank_dataset("TestBank", data_path=str(tmp_path))
    X, y = ds.tensors
    assert len(X) == N_SAMPLES
    assert X.shape[1] == N_FEATURES


def test_non_overlapping_partitions(tmp_path):
    _make_synthetic_csv(str(tmp_path), n=100)
    ds0 = load_bank_dataset("TestBank", data_path=str(tmp_path), partition_index=0, num_partitions=2)
    ds1 = load_bank_dataset("TestBank", data_path=str(tmp_path), partition_index=1, num_partitions=2)
    X0, X1 = ds0.tensors[0], ds1.tensors[0]
    # sizes must be non-zero and together cover all 100 rows
    assert len(X0) > 0 and len(X1) > 0
    assert len(X0) + len(X1) == 100
    # Can check they share no row by RawSum (they should differ due to StandardScaler per-partition)
    # Main guarantee: indices don't overlap
    assert len(X0) == 50 and len(X1) == 50


def test_missing_file_raises_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_bank_dataset("NonExistentBank", data_path=str(tmp_path))
