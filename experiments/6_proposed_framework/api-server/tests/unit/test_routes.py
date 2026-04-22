"""
tests/unit/test_routes.py — Unit tests for FastAPI routes with mocked Fabric client.

Uses pytest + httpx TestClient to test all routes without hitting the live network.
"""
import json
import sys
import os
from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient

# Ensure api-server is in path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from main import app

client = TestClient(app)


# ---------------------------------------------------------------------------
#  Helper mock values
# ---------------------------------------------------------------------------
MOCK_INVOKE_RESULT = {"status": "success", "tx_id": "abc123tx", "output": ""}
MOCK_TRUST_SCORES = json.dumps({"BankA": 1.0, "BankB": 0.9, "BankC": 1.1})
MOCK_CONSENSUS = json.dumps(["BankA", "BankC"])
MOCK_GLOBAL_MODEL = json.dumps({"round": 1, "globalCID": "QmXXX", "globalHash": "aabbcc"})


# ---------------------------------------------------------------------------
#  /health
# ---------------------------------------------------------------------------

def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


# ---------------------------------------------------------------------------
#  POST /submit-update
# ---------------------------------------------------------------------------

def test_submit_update_valid():
    with patch("main.fc.invoke", return_value=MOCK_INVOKE_RESULT):
        resp = client.post("/submit-update", json={
            "bank_id": "BankA",
            "round": 1,
            "model_cid": "QmTestCID",
            "model_hash": "a" * 64,
            "val_score": 0.88
        })
    assert resp.status_code == 200
    assert resp.json()["status"] == "success"
    assert resp.json()["tx_id"] == "abc123tx"


def test_submit_update_missing_field_returns_422():
    resp = client.post("/submit-update", json={
        "bank_id": "BankA",
        "round": 1,
        # Missing model_cid, model_hash, val_score
    })
    assert resp.status_code == 422


def test_submit_update_invalid_bank_returns_422():
    resp = client.post("/submit-update", json={
        "bank_id": "BankZ",
        "round": 1,
        "model_cid": "QmCID",
        "model_hash": "a" * 64,
        "val_score": 0.88
    })
    assert resp.status_code == 422


def test_submit_update_fabric_down_returns_503():
    with patch("main.fc.invoke", side_effect=Exception("peer timeout")):
        resp = client.post("/submit-update", json={
            "bank_id": "BankA",
            "round": 1,
            "model_cid": "QmTest",
            "model_hash": "a" * 64,
            "val_score": 0.88
        })
    assert resp.status_code == 503


# ---------------------------------------------------------------------------
#  POST /submit-verification
# ---------------------------------------------------------------------------

def test_submit_verification_valid():
    with patch("main.fc.invoke", return_value=MOCK_INVOKE_RESULT):
        resp = client.post("/submit-verification", json={
            "verifier_id": "BankB",
            "target_bank_id": "BankA",
            "round": 1,
            "verified": True
        })
    assert resp.status_code == 200
    assert resp.json()["status"] == "success"


def test_self_verification_returns_403():
    resp = client.post("/submit-verification", json={
        "verifier_id": "BankA",
        "target_bank_id": "BankA",
        "round": 1,
        "verified": True
    })
    assert resp.status_code == 403
    assert "Self-verification" in resp.json()["detail"]


# ---------------------------------------------------------------------------
#  POST /submit-commit
# ---------------------------------------------------------------------------

def test_submit_commit_valid():
    with patch("main.fc.invoke", return_value=MOCK_INVOKE_RESULT):
        resp = client.post("/submit-commit", json={
            "committer_id": "BankB",
            "target_bank_id": "BankA",
            "round": 1
        })
    assert resp.status_code == 200


def test_self_commit_returns_403():
    resp = client.post("/submit-commit", json={
        "committer_id": "BankA",
        "target_bank_id": "BankA",
        "round": 1
    })
    assert resp.status_code == 403


# ---------------------------------------------------------------------------
#  POST /update-trust-score
# ---------------------------------------------------------------------------

def test_update_trust_score_valid():
    with patch("main.fc.invoke", return_value=MOCK_INVOKE_RESULT):
        resp = client.post("/update-trust-score", json={
            "bank_id": "BankA",
            "delta": 0.1
        })
    assert resp.status_code == 200


def test_update_trust_score_invalid_bank():
    resp = client.post("/update-trust-score", json={
        "bank_id": "BankXYZ",
        "delta": -0.2
    })
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
#  POST /store-global-model
# ---------------------------------------------------------------------------

def test_store_global_model_valid():
    with patch("main.fc.invoke", return_value=MOCK_INVOKE_RESULT):
        resp = client.post("/store-global-model", json={
            "round": 1,
            "global_cid": "QmGlobal",
            "global_hash": "b" * 64
        })
    assert resp.status_code == 200


# ---------------------------------------------------------------------------
#  GET /trust-scores
# ---------------------------------------------------------------------------

def test_get_trust_scores():
    with patch("main.fc.query", return_value=MOCK_TRUST_SCORES):
        resp = client.get("/trust-scores")
    assert resp.status_code == 200
    data = resp.json()
    assert data["scores"]["BankA"] == 1.0
    assert data["scores"]["BankB"] == 0.9


# ---------------------------------------------------------------------------
#  GET /check-consensus/{round}
# ---------------------------------------------------------------------------

def test_check_consensus():
    with patch("main.fc.query", return_value=MOCK_CONSENSUS):
        resp = client.get("/check-consensus/1")
    assert resp.status_code == 200
    data = resp.json()
    assert "BankA" in data["accepted_banks"]
    assert data["round"] == 1


# ---------------------------------------------------------------------------
#  GET /global-model/{round}
# ---------------------------------------------------------------------------

def test_get_global_model():
    with patch("main.fc.query", return_value=MOCK_GLOBAL_MODEL):
        resp = client.get("/global-model/1")
    assert resp.status_code == 200
    data = resp.json()
    assert data["global_cid"] == "QmXXX"
    assert data["round"] == 1


def test_get_global_model_not_found():
    import fabric_client
    with patch("main.fc.query", side_effect=fabric_client.FabricError("no model found for round 99")):
        resp = client.get("/global-model/99")
    assert resp.status_code == 404
