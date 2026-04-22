"""
test_byzantine_attack.py
Simulates a Byzantine attack (Sub-task 9.3) where a malicious HQ submits poisoned weights.
Honest HQs should evaluate the model, see a low PR-AUC, and cast `verified=False`.
"""
import io
import torch
from unittest.mock import MagicMock
import pytest

from hq_agent import HQAgent
from api_client import compute_sha256
from model.FL_model import LSTMTabular
from tests.test_hq_agent import _make_val_dataset


def test_byzantine_attack_rejected():
    # 1. Setup Honest Bank A and Malicious Bank B
    val_ds_A = _make_val_dataset(n=200, fraud_frac=0.05)
    
    mock_api = MagicMock()
    # Bank B proposed a "poisoned" model
    # We simulate "poisoned" by returning flat zeros for weights (horrible performance)
    poisoned_model = LSTMTabular(input_dim=30)
    for p in poisoned_model.parameters():
        p.data.fill_(0.0)
        
    buf = io.BytesIO()
    torch.save(poisoned_model.state_dict(), buf)
    poisoned_bytes = buf.getvalue()
    poison_hash = compute_sha256(poisoned_bytes)
    
    mock_api.get_cluster_update.side_effect = lambda target, r: {
        "modelCID": "QmPoison", "modelHash": poison_hash
    } if target == "BankB" else {"modelCID": "QmGood", "modelHash": "good"}
    
    def ipfs_down(cid):
        if cid == "QmPoison":
            return poisoned_bytes
        return b""
        
    agent_A = HQAgent(
        bank_id="BankA",
        client=mock_api,
        ipfs_upload=lambda b: "QmOut",
        ipfs_download=ipfs_down,
        val_dataset=val_ds_A,
        val_threshold=0.5  # Requires at least 0.5 PR-AUC
    )
    
    # 2. Bank A runs CBFT Phase 2 (Verify) on Bank B's update
    results = agent_A.verify_peer_updates(round_num=1, peers=["BankA", "BankB"])
    
    # 3. Verification should fail due to terrible val_score
    assert "BankB" in results
    assert results["BankB"] is False, "Bank A should reject the poisoned model"
    
    # Check that API was called with verified=False
    mock_api.submit_verification.assert_called_once_with("BankA", "BankB", 1, False)
