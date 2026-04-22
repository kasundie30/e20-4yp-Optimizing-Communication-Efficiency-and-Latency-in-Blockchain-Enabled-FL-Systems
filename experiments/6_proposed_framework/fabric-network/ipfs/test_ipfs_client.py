import pytest
import time
import uuid
import random

from ipfs_client import IPFSClient, IntegrityError

@pytest.fixture
def ipfs() -> IPFSClient:
    return IPFSClient()

def test_upload_idempotent(ipfs: IPFSClient):
    """Test that uploading the same content twice results in the same CID and Hash."""
    weights = {"layer1": [1.0, 2.0, 3.0], "layer2": [4.0, 5.0, 6.0]}
    
    cid1, hash1 = ipfs.upload_model(weights)
    cid2, hash2 = ipfs.upload_model(weights)
    
    assert cid1 == cid2
    assert hash1 == hash2

def test_roundtrip_integrity(ipfs: IPFSClient):
    """Test that downloaded content exactly matches what was uploaded."""
    weights = {"unique_id": str(uuid.uuid4()), "data": [0.1, 0.2, 0.3]}
    
    cid, hsh = ipfs.upload_model(weights)
    downloaded = ipfs.download_and_verify(cid, hsh)
    
    assert downloaded == weights

def test_hash_mismatch_raises(ipfs: IPFSClient):
    """Test that providing an incorrect hash raises IntegrityError."""
    weights = {"test": "data"}
    cid, hsh = ipfs.upload_model(weights)
    
    wrong_hash = "0" * 64
    with pytest.raises(IntegrityError) as exc_info:
        ipfs.download_and_verify(cid, wrong_hash)
        
    assert "Hash mismatch" in str(exc_info.value)

def test_performance(ipfs: IPFSClient):
    """Test that uploading and downloading a ~10MB model takes < 5s."""
    # Generate ~10MB of dummy float data (~1M floats * 8-10 bytes string rep ≈ 10MB JSON)
    dummy_weights = {"bias": 1.0, "weights": [random.random() for _ in range(1_000_000)]}

    
    start_time = time.time()
    
    # 1. Upload
    cid, hsh = ipfs.upload_model(dummy_weights)
    
    # 2. Download
    downloaded = ipfs.download_and_verify(cid, hsh)
    
    end_time = time.time()
    
    duration = end_time - start_time
    # Note: 10MB JSON stringification and IPFS roundtrip is fast locally but parsing is heavy
    # We just ensure it completes successfully and enforce a soft 5s limit for local loops.
    assert duration < 5.0, f"Performance test failed: took {duration:.2f}s, expected < 5s"
    assert downloaded["bias"] == 1.0
