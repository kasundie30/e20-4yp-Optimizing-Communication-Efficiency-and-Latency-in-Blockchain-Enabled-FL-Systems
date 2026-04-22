"""fl-integration/tests/test_hq_agent.py — unit tests for HQAgent (mocked IPFS + API)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "fl-layer"))

import io
import hashlib
from unittest.mock import MagicMock, patch
from typing import List, Tuple, Dict

import torch
from torch.utils.data import TensorDataset
import pytest

from api_client import APIError, compute_sha256
from model.FL_model import LSTMTabular
from hq_agent import HQAgent


# -------------------------------------------------------
# Helpers
# -------------------------------------------------------

def _make_val_dataset(n=200, fraud_frac=0.05) -> TensorDataset:
    torch.manual_seed(0)
    X = torch.randn(n, 30)
    y = torch.zeros(n, 1)
    y[:max(1, int(n * fraud_frac))] = 1.0
    return TensorDataset(X, y)


def _make_branch_updates(n_branches=3, n_samples=100) -> List[Tuple[Dict, int]]:
    m = LSTMTabular(input_dim=30)
    return [(m.state_dict(), n_samples)] * n_branches


def _make_agent(mock_api=None, ipfs_up=None, ipfs_dn=None):
    if mock_api is None:
        mock_api = MagicMock()
        mock_api.submit_update.return_value = {"status": "success", "tx_id": "tx-test"}
        mock_api.get_global_model.side_effect = APIError(404, "Not found")

    val_ds = _make_val_dataset()

    def _upload(b: bytes) -> str:
        return "QmFakeIPFS" + hashlib.md5(b).hexdigest()[:8]

    def _download(cid: str) -> bytes:
        # Return serialized random LSTMTabular weights
        m = LSTMTabular(input_dim=30)
        buf = io.BytesIO()
        torch.save(m.state_dict(), buf)
        return buf.getvalue()

    return HQAgent(
        bank_id="BankA",
        client=mock_api,
        ipfs_upload=ipfs_up or _upload,
        ipfs_download=ipfs_dn or _download,
        val_dataset=val_ds,
        val_threshold=0.0,
    )


# -------------------------------------------------------
# Tests: fetch_global_model
# -------------------------------------------------------

def test_fetch_global_model_round1_returns_none():
    agent = _make_agent()
    result = agent.fetch_global_model(round_num=1)
    assert result is None


def test_fetch_global_model_404_returns_none():
    mock_api = MagicMock()
    mock_api.get_global_model.side_effect = APIError(404, "Not found")
    agent = _make_agent(mock_api=mock_api)
    result = agent.fetch_global_model(round_num=3)
    assert result is None


def test_fetch_global_model_hash_mismatch_raises():
    """If blockchain hash ≠ IPFS data hash, must raise ValueError."""
    model = LSTMTabular(input_dim=30)
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    data = buf.getvalue()
    real_hash = compute_sha256(data)
    tampered_hash = "0000" + real_hash[4:]

    mock_api = MagicMock()
    mock_api.get_global_model.return_value = {
        "round": 2, "global_cid": "QmFake", "global_hash": tampered_hash
    }

    def _bad_download(cid):
        return data   # hash will NOT match tampered_hash

    agent = _make_agent(mock_api=mock_api, ipfs_dn=_bad_download)
    with pytest.raises(ValueError, match="hash mismatch"):
        agent.fetch_global_model(round_num=3)


# -------------------------------------------------------
# Tests: run_round  (6.2 — FedAvg output to IPFS + submit)
# -------------------------------------------------------

def test_run_round_submits_and_returns_dict():
    agent = _make_agent()
    updates = _make_branch_updates(n_branches=3)
    result = agent.run_round(round_num=1, branch_updates=updates)
    assert "val_score" in result
    assert result["submitted"] is True
    assert result["model_cid"] is not None
    assert result["model_hash"] is not None


def test_run_round_calls_submit_update_with_correct_bank():
    mock_api = MagicMock()
    mock_api.submit_update.return_value = {"status": "success", "tx_id": "tx-x"}
    mock_api.get_global_model.side_effect = APIError(404, "not found")

    agent = _make_agent(mock_api=mock_api)
    updates = _make_branch_updates()
    agent.run_round(round_num=1, branch_updates=updates)

    call_kwargs = mock_api.submit_update.call_args.kwargs
    assert call_kwargs["bank_id"] == "BankA"
    assert call_kwargs["round_num"] == 1
    assert isinstance(call_kwargs["val_score"], float)


def test_run_round_empty_updates_raises():
    agent = _make_agent()
    with pytest.raises(ValueError, match="No branch updates"):
        agent.run_round(round_num=1, branch_updates=[])


def test_val_threshold_skip_submit_when_score_below():
    """val_score below threshold → submitted=False, no API call."""
    mock_api = MagicMock()
    mock_api.get_global_model.side_effect = APIError(404, "not found")

    agent = _make_agent(mock_api=mock_api)
    agent.val_threshold = 100.0   # impossibly high threshold

    updates = _make_branch_updates()
    result = agent.run_round(round_num=1, branch_updates=updates)

    assert result["submitted"] is False
    mock_api.submit_update.assert_not_called()

# -------------------------------------------------------
# Tests: verify_peer_updates (CBFT Phase 2)
# -------------------------------------------------------

def test_verify_peer_updates_success():
    mock_api = MagicMock()
    mock_api.get_cluster_update.return_value = {
        "modelCID": "QmValid", "modelHash": compute_sha256(b"fake_data")
    }
    
    _cache = {}
    def _ipfs_download(cid):
        if cid not in _cache:
            m = LSTMTabular(input_dim=30)
            buf = io.BytesIO()
            torch.save(m.state_dict(), buf)
            _cache[cid] = buf.getvalue()
        return _cache[cid]
        
    mock_api.get_cluster_update.side_effect = lambda target, round_num: {
        "modelCID": "Qm" + target, "modelHash": compute_sha256(_ipfs_download("Qm" + target))
    }

    agent = _make_agent(mock_api=mock_api, ipfs_dn=_ipfs_download)
    peers = ["BankA", "BankB", "BankC"]
    
    results = agent.verify_peer_updates(round_num=1, peers=peers)
    
    # Excludes self
    assert "BankA" not in results
    assert "BankB" in results
    assert "BankC" in results
    
    # With threshold 0.0, both should be verified
    assert results["BankB"] is True
    assert results["BankC"] is True
    
    # Verify API calls
    assert mock_api.submit_verification.call_count == 2
    mock_api.submit_verification.assert_any_call("BankA", "BankB", 1, True)
    mock_api.submit_verification.assert_any_call("BankA", "BankC", 1, True)

def test_verify_peer_updates_hash_mismatch():
    mock_api = MagicMock()
    mock_api.get_cluster_update.return_value = {
        "modelCID": "QmFake", "modelHash": "wrong_hash"
    }
    
    def _ipfs_download(cid):
        return b"some_bytes"
        
    agent = _make_agent(mock_api=mock_api, ipfs_dn=_ipfs_download)
    results = agent.verify_peer_updates(round_num=1, peers=["BankA", "BankB"])
    
    assert results["BankB"] is False
    mock_api.submit_verification.assert_called_once_with("BankA", "BankB", 1, False)

# -------------------------------------------------------
# Tests: commit_peer_updates (CBFT Phase 3)
# -------------------------------------------------------

def test_commit_peer_updates():
    mock_api = MagicMock()
    # BankB has quorum, BankC does not
    mock_api.check_verify_quorum.side_effect = lambda target, r: target == "BankB"
    
    agent = _make_agent(mock_api=mock_api)
    committed = agent.commit_peer_updates(round_num=1, peers=["BankA", "BankB", "BankC"])
    
    assert "BankB" in committed
    assert "BankC" not in committed
    mock_api.submit_commit.assert_called_once_with("BankA", "BankB", 1)
