"""fl-integration/tests/test_api_client.py — tests APIClient with mocked requests."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
from unittest.mock import MagicMock, patch
import pytest
import requests

from api_client import APIClient, APIError, compute_sha256


def _mock_response(status_code: int, body: dict):
    r = MagicMock()
    r.status_code = status_code
    r.ok = (200 <= status_code < 300)
    r.json.return_value = body
    r.text = json.dumps(body)
    return r


@pytest.fixture
def client():
    return APIClient(base_url="http://fakehost:8000", timeout=5.0, max_retries=1)


def test_health_ok(client):
    with patch.object(client._session, "request",
                      return_value=_mock_response(200, {"status": "ok"})) as m:
        result = client.health()
    assert result["status"] == "ok"


def test_get_trust_scores_returns_dict(client):
    payload = {"scores": {"BankA": 1.0, "BankB": 0.8, "BankC": 0.9}}
    with patch.object(client._session, "request", return_value=_mock_response(200, payload)):
        scores = client.get_trust_scores()
    assert scores["BankA"] == 1.0
    assert scores["BankB"] == 0.8


def test_get_global_model_ok(client):
    payload = {"round": 5, "global_cid": "Qmtest", "global_hash": "abc123"}
    with patch.object(client._session, "request", return_value=_mock_response(200, payload)):
        result = client.get_global_model(5)
    assert result["global_cid"] == "Qmtest"


def test_get_global_model_404_raises_APIError(client):
    err_resp = _mock_response(404, {"detail": "Not found"})
    with patch.object(client._session, "request", return_value=err_resp):
        with pytest.raises(APIError) as exc:
            client.get_global_model(999)
    assert exc.value.status_code == 404


def test_submit_update_sends_correct_payload(client):
    success = {"status": "success", "message": "ok", "tx_id": "tx-abc"}
    with patch.object(client._session, "request", return_value=_mock_response(200, success)) as mock_req:
        client.submit_update("BankA", 10, "Qmcid", "hash123", 0.85)
    _, kwargs = mock_req.call_args
    payload = kwargs["json"]
    assert payload["bank_id"] == "BankA"
    assert payload["round"] == 10
    assert payload["val_score"] == 0.85


def test_check_consensus_returns_list(client):
    resp = {"accepted_banks": ["BankA", "BankB"], "round": 7}
    with patch.object(client._session, "request", return_value=_mock_response(200, resp)):
        accepted = client.check_consensus(7)
    assert "BankA" in accepted
    assert "BankB" in accepted


def test_store_global_model_ok(client):
    success = {"status": "success", "message": "stored", "tx_id": "tx-xyz"}
    with patch.object(client._session, "request", return_value=_mock_response(200, success)):
        result = client.store_global_model(3, "QmGlobal", "ghash")
    assert result["status"] == "success"


def test_compute_sha256_deterministic():
    import hashlib
    data = b"hello world"
    expected = hashlib.sha256(data).hexdigest()
    h1 = compute_sha256(data)
    h2 = compute_sha256(data)
    assert h1 == h2 == expected
    # length check
    assert len(h1) == 64


def test_503_raises_after_single_retry():
    """With max_retries=1, a 503 should raise APIError after one attempt."""
    c = APIClient(base_url="http://fakehost:8000", timeout=5.0, max_retries=1, retry_delay=0)
    resp_503 = _mock_response(503, {"detail": "Service Unavailable"})
    with patch.object(c._session, "request", return_value=resp_503):
        with pytest.raises(APIError) as exc:
            c.health()
    assert exc.value.status_code == 503
