import hashlib
import requests
import json
from base64 import b64encode
import time

IPFS_API_URL = "http://127.0.0.1:5001/api/v0"

class IntegrityError(Exception):
    pass

class IPFSClient:
    def __init__(self, api_url=IPFS_API_URL):
        self.api_url = api_url

    def compute_hash(self, content: bytes) -> str:
        """Compute SHA-256 hash of the content."""
        return hashlib.sha256(content).hexdigest()

    def upload_model(self, weights: dict) -> tuple[str, str]:
        """
        Uploads dictionary weights as a JSON string to IPFS.
        Returns the CID and SHA256 hash of the content.
        """
        content = json.dumps(weights, sort_keys=True).encode('utf-8')
        model_hash = self.compute_hash(content)

        files = {
            'file': ('model.json', content)
        }
        
        # IPFS API for adding a file
        response = requests.post(f"{self.api_url}/add", files=files)
        response.raise_for_status()
        
        result = response.json()
        cid = result['Hash']
        return cid, model_hash

    def download_and_verify(self, cid: str, expected_hash: str) -> dict:
        """
        Downloads content from IPFS by CID and verifies it against the expected SHA256 hash.
        Returns the parsed dictionary.
        Raises IntegrityError if the hash does not match.
        """
        params = {'arg': cid}
        # IPFS API for fetching a file
        response = requests.post(f"{self.api_url}/cat", params=params)
        response.raise_for_status()
        
        content = response.content
        actual_hash = self.compute_hash(content)
        
        if actual_hash != expected_hash:
            raise IntegrityError(f"Hash mismatch for CID {cid}. Expected: {expected_hash}, Actual: {actual_hash}")
            
        return json.loads(content.decode('utf-8'))
