"""
tests/integration/test_live_network.py — Integration tests against the live Fabric network.

Requires: network.sh up, chaincode deployed, IPFS running.
Run with: python -m pytest tests/integration/ -v
"""
import json
import sys
import os

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from main import app

client = TestClient(app)


def test_health_live():
    """Basic health check — server is up."""
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_trust_scores_live():
    """Query trust scores from live ledger. Expects all 3 banks."""
    resp = client.get("/trust-scores")
    assert resp.status_code == 200
    scores = resp.json()["scores"]
    for bank in ["BankA", "BankB", "BankC"]:
        assert bank in scores, f"Missing bank {bank} in trust scores"
        assert isinstance(scores[bank], (int, float))


def test_submit_update_live():
    """Submit a test model update with val_score ≥ 0.7 to the live chain."""
    resp = client.post("/submit-update", json={
        "bank_id": "BankA",
        "round": 99,
        "model_cid": "QmIntegrationTestCID",
        "model_hash": "cafecafe" * 8,
        "val_score": 0.91
    })
    assert resp.status_code == 200, resp.text
    assert resp.json()["status"] == "success"


def test_full_cbft_flow_live():
    """
    End-to-end CBFT flow: submit → verify (2 banks) → commit (2 banks) → check consensus.
    Uses round 96 (fresh round to avoid stale state from prior runs).
    """
    import time
    ROUND = 96

    # Phase 1: BankA submits cluster update
    r = client.post("/submit-update", json={
        "bank_id": "BankA",
        "round": ROUND,
        "model_cid": "QmE2ECID96",
        "model_hash": "deadbeef" * 8,
        "val_score": 0.92
    })
    assert r.status_code == 200, r.text
    time.sleep(3)  # wait for tx to commit across all peers

    # Phase 2: BankB verifies BankA's update
    r = client.post("/submit-verification", json={
        "verifier_id": "BankB",
        "target_bank_id": "BankA",
        "round": ROUND,
        "verified": True
    })
    assert r.status_code == 200, r.text
    time.sleep(3)  # ensure verification is committed before BankC also verifies

    # Phase 2: BankC verifies BankA's update
    r = client.post("/submit-verification", json={
        "verifier_id": "BankC",
        "target_bank_id": "BankA",
        "round": ROUND,
        "verified": True
    })
    assert r.status_code == 200, r.text
    time.sleep(5)  # long wait: ensure both verifications fully committed before commits

    # Phase 3: BankB commits
    r = client.post("/submit-commit", json={
        "committer_id": "BankB",
        "target_bank_id": "BankA",
        "round": ROUND
    })
    assert r.status_code == 200, r.text
    time.sleep(3)

    # Phase 3: BankC commits
    r = client.post("/submit-commit", json={
        "committer_id": "BankC",
        "target_bank_id": "BankA",
        "round": ROUND
    })
    assert r.status_code == 200, r.text
    time.sleep(3)

    # Check consensus: BankA should be accepted
    r = client.get(f"/check-consensus/{ROUND}")
    assert r.status_code == 200, r.text
    accepted = r.json()["accepted_banks"]
    assert "BankA" in accepted, f"Expected BankA to be accepted; got {accepted}"


def test_store_and_get_global_model_live():
    """Store a global model then retrieve it successfully."""
    ROUND = 97
    r = client.post("/store-global-model", json={
        "round": ROUND,
        "global_cid": "QmGlobalModelCID",
        "global_hash": "feedface" * 8
    })
    assert r.status_code == 200, r.text

    r = client.get(f"/global-model/{ROUND}")
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["global_cid"] == "QmGlobalModelCID"
    assert data["round"] == ROUND


def test_global_model_not_found_live():
    """Requesting a non-existent round should yield 404."""
    r = client.get("/global-model/9999")
    assert r.status_code == 404
