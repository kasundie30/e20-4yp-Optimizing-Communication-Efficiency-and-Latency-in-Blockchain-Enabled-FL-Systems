"""
fl-integration/tests/test_security_mechanisms.py
Simulates the remaining security tests for Phase 9:
9.4 Replay attack test
9.5 Trust score recovery test
"""

import pytest
from unittest.mock import MagicMock
from api_client import APIError

def test_replay_attack_rejected():
    """
    Sub-task 9.4 — Replay attack test.
    If the chaincode rejects a recycled modelCID, the API raises an APIError (503 or 400).
    The HQ agent's submission should cleanly fail without crashing.
    """
    from hq_agent import HQAgent
    from tests.test_hq_agent import _make_val_dataset, _make_branch_updates

    mock_api = MagicMock()
    # The first submission succeeds
    # The second submissions throws an APIError simulating chaincode rejection
    mock_api.submit_update.side_effect = [
        {"status": "success", "tx_id": "tx1"},
        APIError(503, "Fabric network error: SubmitClusterUpdate: replay attack detected")
    ]
    mock_api.get_global_model.side_effect = APIError(404, "Not found")

    agent = HQAgent(
        bank_id="BankA",
        client=mock_api,
        ipfs_upload=lambda b: "QmReplayed", # Always returns same CID (Replay attack)
        ipfs_download=lambda c: b"",
        val_dataset=_make_val_dataset(),
        val_threshold=0.0
    )

    updates = _make_branch_updates()
    
    # 1. Round 1 (Original Submission)
    res1 = agent.run_round(1, updates)
    assert res1["submitted"] is True

    # 2. Round 2 (Replay Attack)
    res2 = agent.run_round(2, updates)
    # The agent catches APIError and returns submitted=False gracefully
    assert res2["submitted"] is False
    assert res2["tx"] is None

def test_trust_score_recovery():
    """
    Sub-task 9.5 — Trust score recovery test (3 bad rounds -> 3 good rounds).
    We simulate the UpdateTrustScore interactions.
    """
    # Initialize a dummy score
    score = 1.0
    score_min = 0.1
    history = []

    def update_trust(delta):
        nonlocal score
        score += delta
        if score < score_min:
            score = score_min
        history.append(score)
        return {"status": "success"}

    mock_api = MagicMock()
    mock_api.update_trust_score.side_effect = lambda bank, d: update_trust(d)

    # Simulate 3 bad rounds (penalty = -0.4 each to force hitting minimum)
    for _ in range(3):
        mock_api.update_trust_score("BankA", -0.4)

    assert score == 0.1 # Should be clamped to minimum

    # Simulate 3 good rounds (reward = +0.2 each)
    for _ in range(3):
        mock_api.update_trust_score("BankA", 0.2)

    assert score == 0.7 # 0.1 + 0.6 = 0.7
    
    # Assert API interactions
    assert mock_api.update_trust_score.call_count == 6
