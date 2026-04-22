"""
Tests for Round Zero Initialization Script (Sub-task 7.2)
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "fl-layer"))

import io
import torch
import pytest
from unittest.mock import MagicMock, patch

from scripts.init_round_zero import (
    generate_initial_model,
    upload_initial_model_to_ipfs,
    register_round_zero
)
from model.FL_model import LSTMTabular
from api_client import APIError


def test_generate_initial_model_is_deterministic_and_valid():
    """Confirms seeding produces identical weights and valid models."""
    seed = 42
    bytes1, hash1 = generate_initial_model(seed)
    bytes2, hash2 = generate_initial_model(seed)
    
    # Determinism
    assert bytes1 == bytes2
    assert hash1 == hash2
    
    # Hash format
    assert len(hash1) == 64
    assert all(c in "0123456789abcdef" for c in hash1)
    
    # Deserialization
    buf = io.BytesIO(bytes1)
    state_dict = torch.load(buf, map_location="cpu", weights_only=True)
    
    model = LSTMTabular(input_dim=30, hidden_dim=30, num_layers=1)
    # Should load without errors (keys match exactly)
    model.load_state_dict(state_dict)


@patch("scripts.init_round_zero.requests.post")
def test_upload_initial_model_to_ipfs(mock_post):
    """Confirm the IPFS utility is called with the exact bytes."""
    mock_post.return_value.json.return_value = {"Hash": "QmTestCID123"}
    test_bytes = b"fake_model_data"
    
    cid = upload_initial_model_to_ipfs(test_bytes, "http://fake-ipfs:5001")
    
    assert cid == "QmTestCID123"
    mock_post.assert_called_once_with(
        "http://fake-ipfs:5001/api/v0/add",
        files={'file': ('model.pt', test_bytes)}
    )


def test_register_round_zero_success():
    """Confirm POST call is made with round=0, CID, and hash."""
    mock_client = MagicMock()
    
    register_round_zero(mock_client, "Qm123", "abcde")
    
    mock_client.store_global_model.assert_called_once_with(
        round_num=0, global_cid="Qm123", global_hash="abcde"
    )


@patch("scripts.init_round_zero.time.sleep", return_value=None)
def test_register_round_zero_retries_on_503(mock_sleep):
    """Confirm 503 causes retries before failing or succeeding."""
    mock_client = MagicMock()
    
    # Fail twice with 503, succeed on third attempt
    mock_client.store_global_model.side_effect = [
        APIError(503, "Service Unavailable"),
        APIError(503, "Service Unavailable"),
        None  # Success
    ]
    
    register_round_zero(mock_client, "Qm123", "abcde")
    
    assert mock_client.store_global_model.call_count == 3
    assert mock_sleep.call_count == 2


@patch("scripts.init_round_zero.time.sleep", return_value=None)
def test_register_round_zero_fails_after_max_retries(mock_sleep):
    """Confirm it raises a final APIError if all retries fail with 503."""
    mock_client = MagicMock()
    mock_client.store_global_model.side_effect = APIError(503, "Service Unavailable")
    
    with pytest.raises(APIError, match="Failed to register Round 0 after multiple retries"):
        register_round_zero(mock_client, "Qm123", "abcde")
    
    assert mock_client.store_global_model.call_count == 3
