"""fl-integration/tests/test_global_aggregator.py — unit tests for GlobalAggregator."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "fl-layer"))

import io
import hashlib
from unittest.mock import MagicMock
from typing import Dict

import torch
import pytest

from api_client import APIError, compute_sha256
from model.FL_model import LSTMTabular
from global_aggregator import GlobalAggregator


# -------------------------------------------------------
# Shared helpers
# -------------------------------------------------------

def _serialize(sd: dict) -> bytes:
    buf = io.BytesIO()
    torch.save(sd, buf)
    return buf.getvalue()


def _make_real_bank_data(bank_ids):
    """
    Serialise a real LSTMTabular for each bank, store the bytes and hash.
    Returns:
        bank_bytes  : {bank_id: bytes}
        bank_updates: {bank_id: {"model_cid", "model_hash", "num_samples"}}
    """
    bank_bytes: Dict[str, bytes] = {}
    bank_updates: Dict[str, dict] = {}
    for i, bid in enumerate(bank_ids):
        torch.manual_seed(i * 42)
        m = LSTMTabular(input_dim=30)
        data = _serialize(m.state_dict())
        bank_bytes[bid] = data
        bank_updates[bid] = {
            "model_cid":   f"Qm{bid}Fake",
            "model_hash":  compute_sha256(data),   # matches real data
            "num_samples": 100,
        }
    return bank_bytes, bank_updates


def _make_aggregator(accepted_banks, trust_scores=None, with_bad_hash=None):
    """
    Build a GlobalAggregator where _download returns the EXACT bytes
    used to compute the hash in bank_updates.

    Args:
        accepted_banks : list of bank IDs to include
        trust_scores   : optional {bank_id: float} override
        with_bad_hash  : optional bank_id whose stored hash is corrupted
    """
    bank_bytes, bank_updates = _make_real_bank_data(accepted_banks)

    # Optionally corrupt one bank's stored hash
    if with_bad_hash:
        bank_updates[with_bad_hash]["model_hash"] = "0" * 64

    mock_api = MagicMock()
    mock_api.check_consensus.return_value = accepted_banks
    mock_api.get_trust_scores.return_value = (
        trust_scores or {b: 1.0 for b in accepted_banks}
    )
    mock_api.store_global_model.return_value = {"status": "success", "tx_id": "tx-agg"}

    def _download(cid: str) -> bytes:
        for bid, rec in bank_updates.items():
            if rec["model_cid"] == cid:
                return bank_bytes[bid]
        return b""   # should not happen in normal tests

    _cid_store = {}
    def _upload(data: bytes) -> str:
        cid = "QmGlobal" + hashlib.md5(data).hexdigest()[:8]
        _cid_store[cid] = data
        return cid

    agg = GlobalAggregator(
        client=mock_api,
        ipfs_download=_download,
        ipfs_upload=_upload,
        poll_interval=0.01,
        consensus_timeout=0.05,
    )
    return agg, mock_api, bank_updates


# -------------------------------------------------------
# Tests: trust-weighted aggregation (6.3)
# -------------------------------------------------------

def test_aggregate_round_returns_global_cid():
    agg, mock_api, bu = _make_aggregator(["BankA", "BankB"])
    result = agg.aggregate_round(round_num=5, accepted_banks=["BankA", "BankB"], bank_updates=bu)
    assert result is not None
    assert result["global_cid"].startswith("QmGlobal")
    assert len(result["global_hash"]) == 64


def test_aggregate_round_calls_store_global_model():
    agg, mock_api, bu = _make_aggregator(["BankA", "BankB"])
    agg.aggregate_round(round_num=5, accepted_banks=["BankA", "BankB"], bank_updates=bu)
    mock_api.store_global_model.assert_called_once()
    call_kwargs = mock_api.store_global_model.call_args.kwargs
    assert call_kwargs["round_num"] == 5
    assert "global_cid" in call_kwargs
    assert "global_hash" in call_kwargs


def test_trust_weighted_avg_correct():
    """Effective weight = trust_score × num_samples; BankA (trust=4) > BankC (trust=2)."""
    trust = {"BankA": 4.0, "BankB": 3.0, "BankC": 2.0}
    agg, _, bu = _make_aggregator(["BankA", "BankB", "BankC"], trust_scores=trust)
    result = agg.aggregate_round(
        round_num=3, accepted_banks=["BankA", "BankB", "BankC"], bank_updates=bu
    )
    assert result is not None
    assert result["weights"]["BankA"] > result["weights"]["BankC"]


def test_hash_mismatch_excludes_bank():
    """BankB has corrupted stored hash → excluded; BankA alone must still succeed."""
    agg, _, bu = _make_aggregator(["BankA", "BankB"], with_bad_hash="BankB")
    result = agg.aggregate_round(
        round_num=2, accepted_banks=["BankA", "BankB"], bank_updates=bu
    )
    # BankA alone should be enough for aggregation
    assert result is not None
    assert "BankB" not in result["weights"]
    assert "BankA" in result["weights"]


def test_empty_accepted_banks_returns_none():
    agg, _, bu = _make_aggregator(["BankA"])
    result = agg.aggregate_round(round_num=1, accepted_banks=[], bank_updates=bu)
    assert result is None
