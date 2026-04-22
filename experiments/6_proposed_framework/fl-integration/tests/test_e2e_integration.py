"""
fl-integration/tests/test_e2e_integration.py
E2E integration test for the Phase 6 FL-Blockchain integration layer.

Scenario:
  1. Initialize 3 mock HQAgents and 1 RoundCoordinator.
  2. The collect_fn generates synthetic branch datasets and performs local training.
  3. The RoundCoordinator collects the 3 branch updates.
  4. Each HQAgent runs its round (FedAvg, blend, evaluate), uploads to mock IPFS, and submits to mock API.
  5. The GlobalAggregator (running on BankA) waits for consensus, downloads the models, performs trust-weighted aggregation, and outputs a final global model.
  6. Assertions verify the data flow and presence of expected outputs.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "fl-layer"))

import io
import time
import hashlib
from typing import Dict, List, Tuple
from unittest.mock import MagicMock

import torch
from torch.utils.data import TensorDataset
import pytest

from api_client import APIClient, compute_sha256
from model.FL_model import LSTMTabular
from training.local_train import train_local
from hq_agent import HQAgent
from global_aggregator import GlobalAggregator
from round_coordinator import RoundCoordinator, RoundConfig, RoundResult


# --- Mock Datasets and Training ---

def _make_noniid_dataset(seed: int, n: int = 100, fraud_frac: float = 0.05):
    torch.manual_seed(seed)
    X = torch.randn(n, 30) + (seed * 0.1)
    y = torch.zeros(n, 1)
    n_fraud = max(1, int(n * fraud_frac))
    y[:n_fraud] = 1.0
    return TensorDataset(X, y)

TRAIN_CFG = {
    "local_epochs": 1,
    "batch_size": 32,
    "lr": 1e-3,
    "l2_norm_clip": 1.0,
    "noise_multiplier": 0.0,
    "device": "cpu",
}

def _collect_fn_generator():
    """Simulates branches completing training and making their state_dicts available."""
    datasets = [
        _make_noniid_dataset(0, 100),
        _make_noniid_dataset(1, 120),
        _make_noniid_dataset(2, 110),
    ]
    base_model = LSTMTabular(input_dim=30)
    updates = []
    for ds in datasets:
        sd = train_local(base_model, ds, TRAIN_CFG)
        updates.append((sd, len(ds)))
    return updates


# --- Mock Infrastructure ---

class MockIPFS:
    def __init__(self):
        self.store = {}
    
    def upload(self, data: bytes) -> str:
        cid = "Qm" + compute_sha256(data)[:10]
        self.store[cid] = data
        return cid
    
    def download(self, cid: str) -> bytes:
        return self.store.get(cid, b"")


def build_mock_api(bank_ids=None):
    mock_api = MagicMock(spec=APIClient)
    # Default behaviors
    mock_api.get_global_model.return_value = {
        "round": 0, "global_cid": "QmInit", "global_hash": "0000"
    }
    # We will simulate missing global model for round 1
    mock_api.get_global_model.side_effect = Exception("Not Found via 404")
    
    # Store global model tracking
    mock_api.store_global_model.return_value = {"status": "success", "tx_id": "tx-global"}
    
    # Submit update tracking
    mock_api.submit_update.return_value = {"status": "success", "tx_id": "tx-update"}
    
    # Consensus
    accepted = bank_ids or ["BankA", "BankB", "BankC"]
    mock_api.check_consensus.return_value = accepted
    mock_api.get_trust_scores.return_value = {b: 1.0 for b in accepted}
    
    return mock_api


# --- E2E Test ---

def test_full_round_e2e_integration():
    """
    Test a full FL round using the Integration Layer components.
    """
    bank_ids = ["BankA", "BankB", "BankC"]
    
    mock_ipfs = MockIPFS()
    mock_api = build_mock_api(bank_ids)
    
    # Need to properly mock single APIError 404 for global model
    # HQ agent expects APIError
    from api_client import APIError
    mock_api.get_global_model.side_effect = APIError(404, "Not Found")

    # 1. Initialize HQ Agents
    val_dataset = _make_noniid_dataset(99, 150)
    hq_agents = {}
    for bid in bank_ids:
        agent = HQAgent(
            bank_id=bid,
            client=mock_api,
            ipfs_upload=mock_ipfs.upload,
            ipfs_download=mock_ipfs.download,
            val_dataset=val_dataset,
            val_threshold=0.0, # Accept all to ensure test flow
        )
        hq_agents[bid] = agent
        
    # 2. Initialize Global Aggregator
    global_agg = GlobalAggregator(
        client=mock_api,
        ipfs_download=mock_ipfs.download,
        ipfs_upload=mock_ipfs.upload,
        poll_interval=0.01,
        consensus_timeout=0.1,
    )

    # 3. Round Coordinator Config
    cfg = RoundConfig(
        round_num=1,
        bank_ids=bank_ids,
        deadline_sec=2.0,
        poll_interval=0.1,
        is_aggregator=True,
        aggregator_bank_id="BankA"
    )

    coordinator = RoundCoordinator(
        config=cfg,
        collect_fn=_collect_fn_generator,
        hq_agents=hq_agents,
        global_agg=global_agg
    )

    # 4. RUN FULL ROUND
    result: RoundResult = coordinator.run()

    # 5. Assertions
    assert result.round_num == 1
    assert result.collected == 3
    assert len(result.submitted_banks) == 3
    assert result.timed_out is False
    
    # Verify Global Aggregation output
    assert result.global_cid is not None
    assert result.global_cid.startswith("Qm")
    assert result.global_hash is not None
    
    # Verify IPFS Usage
    # Because all 3 HQAgents received the same branch updates, FedAvg produces
    # identical state_dicts for all 3, resulting in 1 unique CID uploaded
    # by the branches, and 1 CID for the global merged model.
    assert len(mock_ipfs.store) >= 1
    assert result.global_cid in mock_ipfs.store
    
    # Verify API Interactions
    assert mock_api.submit_update.call_count == 3
    mock_api.store_global_model.assert_called_once()
