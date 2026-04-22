#!/usr/bin/env python3
"""
fl-integration/scripts/init_round_zero.py
Bootstraps the HCFL network by generating a deterministically seeded Random Model (Round 0),
uploading it to IPFS, and recording it on the blockchain ledger via the API server.
"""
import sys, os
import argparse
import io
import time
import logging

import torch

import requests

# Add workspace root and fl-integration to sys_path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "fl-layer")))

from config.config_loader import load_config
from api_client import APIClient, APIError, compute_sha256
from model.FL_model import LSTMTabular


logger = logging.getLogger(__name__)


def generate_initial_model(seed: int, model_cfg: dict = None) -> tuple[bytes, str]:
    """
    Generates a deterministic random layout for the global model and returns bytes & sha256 hash.
    """
    cfg = model_cfg or {"input_dim": 30, "hidden_dim": 30, "num_layers": 1}
    
    # Force deterministic parameter initialization
    torch.manual_seed(seed)
    
    model = LSTMTabular(**cfg)
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    model_bytes = buf.getvalue()
    
    model_hash = compute_sha256(model_bytes)
    return model_bytes, model_hash


def upload_initial_model_to_ipfs(model_bytes: bytes, ipfs_api_url: str) -> str:
    """Uploads model to IPFS via HTTP API."""
    logger.info("Uploading Round 0 model to IPFS at %s", ipfs_api_url)
    files = {'file': ('model.pt', model_bytes)}
    response = requests.post(f"{ipfs_api_url}/api/v0/add", files=files)
    response.raise_for_status()
    cid = response.json()['Hash']
    logger.info("Upload successful → CID: %s", cid)
    return cid


def register_round_zero(client: APIClient, cid: str, model_hash: str):
    """Submits the Round 0 global model directly to the designated Phase 6 Store Global Model API."""
    logger.info("Registering Round 0 on Blockchain via API...")
    max_retries = 3
    for attempt in range(max_retries):
        try:
            client.store_global_model(round_num=0, global_cid=cid, global_hash=model_hash)
            logger.info("Successfully registered Round 0 model!")
            return
        except APIError as e:
            if e.status_code == 503:
                logger.warning("API Server 503 Unavailable. Retrying (%d/%d)...", attempt + 1, max_retries)
                time.sleep(2)
            else:
                logger.error("Failed to register Round 0: %s", e)
                raise
    raise APIError(503, "Failed to register Round 0 after multiple retries.")


def main():
    parser = argparse.ArgumentParser(description="Bootstrap the Blockchain FL system with Round 0")
    parser.add_argument("--config", type=str, default=None, help="Path to fl_config.yaml")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for model deterministic init")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Load configuration
    try:
        if args.config:
            config = load_config(args.config)
        else:
            config = load_config()
    except Exception as e:
        logger.error("Failed to load configuration: %s", e)
        sys.exit(1)

    try:
        # Generate random bytes + hash
        model_bytes, model_hash = generate_initial_model(seed=args.seed)
        logger.info("Generated Round 0 Model (seed=%d): hash=%s...", args.seed, model_hash[:16])
        
        # Upload
        cid = upload_initial_model_to_ipfs(model_bytes, config.ipfs.api_url)
        
        # Register
        client = APIClient(base_url=config.blockchain.api_url)
        register_round_zero(client, cid, model_hash)
        
        logger.info("\n=== Round 0 Bootstrap Complete ===")
        logger.info("Registered CID: %s", cid)
        
    except Exception as e:
        logger.error("Bootstrap failed: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
