"""
fl-integration/global_aggregator.py
Global Aggregator — runs on the aggregating bank (BankA by default).

Responsibilities (6.3):
  1. Poll blockchain until CheckConsensus returns a non-empty accepted_banks list
  2. Download each accepted bank's update from IPFS
  3. Get trust scores from blockchain
  4. Compute trust-weighted FedAvg: weight = trust_score * num_samples
  5. Serialize and upload aggregated global model to IPFS
  6. Call POST /store-global-model to record the CID on the ledger

Design: Injectable ipfs_download so unit tests avoid real IPFS;
        Injectable api_client so unit tests avoid live Fabric.
"""
from __future__ import annotations

import io
import logging
import time
from typing import Callable, Dict, List, Optional, Tuple

import torch

from api_client import APIClient, APIError, compute_sha256
from config.config_loader import load_config

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "fl-layer"))

from model.FL_model import LSTMTabular
from aggregation.fedavg import fedavg, StateDict

logger = logging.getLogger(__name__)

IpfsDownloadFn = Callable[[str], bytes]
IpfsUploadFn   = Callable[[bytes], str]


class GlobalAggregator:
    """
    Trust-weighted global aggregation service.

    Args:
        client          : APIClient
        ipfs_download   : callable(cid) -> bytes
        ipfs_upload     : callable(bytes) -> cid
        poll_interval   : seconds between consensus polls
        consensus_timeout : max seconds to wait for consensus
        model_cfg       : {"input_dim", "hidden_dim", "num_layers"}
    """

    def __init__(
        self,
        client: APIClient,
        ipfs_download: IpfsDownloadFn,
        ipfs_upload: IpfsUploadFn,
        poll_interval: float = 5.0,
        consensus_timeout: float | None = None,
        model_cfg: dict | None = None,
    ):
        self.config = load_config()
        self.client = client
        self.ipfs_download = ipfs_download
        self.ipfs_upload = ipfs_upload
        self.poll_interval = poll_interval
        self.consensus_timeout = consensus_timeout if consensus_timeout is not None else self.config.blockchain.round_timeout_seconds
        self.model_cfg = model_cfg or {"input_dim": 30, "hidden_dim": 30, "num_layers": 1}

    # ------------------------------------------------------------------
    # 6.3 — Main aggregation pipeline
    # ------------------------------------------------------------------

    def wait_for_consensus(self, round_num: int) -> List[str]:
        """
        Poll GET /check-consensus/{round_num} until at least one bank
        appears in the accepted list or timeout expires.

        Returns:
            List of accepted bank IDs (may be empty on timeout).
        """
        start = time.time()
        logger.info("Waiting for consensus (timeout=%.0fs) ...", self.consensus_timeout, extra={"round_num": round_num, "component": "global_aggregator"})

        while True:
            accepted = self.client.check_consensus(round_num)
            if accepted:
                logger.info("Consensus achieved: %s (%.1fs elapsed)", accepted, time.time() - start, extra={"round_num": round_num, "component": "global_aggregator"})
                return accepted

            elapsed = time.time() - start
            if elapsed >= self.consensus_timeout:
                logger.warning("Consensus timeout after %.1fs — proceeding with 0 banks.", elapsed, extra={"round_num": round_num, "component": "global_aggregator"})
                return []

            logger.debug("No consensus yet — retrying in %.1fs", self.poll_interval, extra={"round_num": round_num, "component": "global_aggregator"})
            time.sleep(self.poll_interval)

    def aggregate_round(
        self,
        round_num: int,
        accepted_banks: List[str],
        bank_updates: Dict[str, dict],
    ) -> Optional[dict]:
        """
        Compute trust-weighted FedAvg and store to blockchain.

        Args:
            round_num     : current FL round
            accepted_banks: banks that achieved CBFT consensus
            bank_updates  : {bank_id: {"model_cid": str, "model_hash": str, "num_samples": int}}

        Returns:
            {"global_cid": str, "global_hash": str, "weights": dict} or None if no valid updates.
        """
        if not accepted_banks:
            logger.warning("aggregate_round: accepted_banks is empty — nothing to aggregate.", extra={"round_num": round_num, "component": "global_aggregator"})
            return None

        # Fetch trust scores from ledger
        try:
            trust_scores: Dict[str, float] = self.client.get_trust_scores()
        except APIError as e:
            logger.error("Failed to fetch trust scores: %s — using uniform weights.", e, extra={"round_num": round_num, "component": "global_aggregator"})
            trust_scores = {b: 1.0 for b in accepted_banks}

        # Download and verify each accepted bank's model
        model_updates: List[Tuple[StateDict, int]] = []
        effective_weights: Dict[str, float] = {}

        for bank_id in accepted_banks:
            if bank_id not in bank_updates:
                logger.warning("No update record for accepted bank %s — skipping.", bank_id, extra={"round_num": round_num, "component": "global_aggregator"})
                continue

            record = bank_updates[bank_id]
            cid         = record["model_cid"]
            stored_hash = record["model_hash"]
            num_samples = record.get("num_samples", 100)

            logger.debug("Downloading %s model CID=%s", bank_id, cid, extra={"round_num": round_num, "component": "global_aggregator"})
            data = self.ipfs_download(cid)
            actual_hash = compute_sha256(data)

            if actual_hash != stored_hash:
                logger.error(
                    "Hash mismatch for %s CID=%s — EXCLUDING from aggregation. "
                    "Expected=%s  Got=%s", bank_id, cid, stored_hash, actual_hash,
                    extra={"round_num": round_num, "component": "global_aggregator"}
                )
                continue

            buf = io.BytesIO(data)
            sd = torch.load(buf, map_location="cpu")

            trust = trust_scores.get(bank_id, 1.0)
            effective_weight = trust * num_samples
            effective_weights[bank_id] = effective_weight
            model_updates.append((sd, int(max(1, effective_weight * 1000))))

        if not model_updates:
            logger.error("No valid model updates after hash verification — aborting aggregation.", extra={"round_num": round_num, "component": "global_aggregator"})
            return None

        logger.info("Trust-weighted FedAvg over %d models. Effective weights: %s",
                    len(model_updates), {k: f"{v:.3f}" for k, v in effective_weights.items()},
                    extra={"round_num": round_num, "component": "global_aggregator"})

        global_sd = fedavg(model_updates)

        # Serialize and upload
        buf = io.BytesIO()
        torch.save(global_sd, buf)
        global_bytes = buf.getvalue()
        global_hash  = compute_sha256(global_bytes)
        global_cid   = self.ipfs_upload(global_bytes)

        logger.info("Global model uploaded — CID=%s hash=%s", global_cid, global_hash[:16] + "...", extra={"round_num": round_num, "component": "global_aggregator"})

        # Store on ledger
        try:
            self.client.store_global_model(
                round_num=round_num,
                global_cid=global_cid,
                global_hash=global_hash
            )
            logger.info("Global model stored on ledger.", extra={"round_num": round_num, "component": "global_aggregator"})
        except APIError as e:
            logger.error("Failed to store global model on ledger: %s", e, extra={"round_num": round_num, "component": "global_aggregator"})

        return {
            "global_cid": global_cid,
            "global_hash": global_hash,
            "weights": effective_weights,
        }

    def run_full_aggregation(
        self,
        round_num: int,
        bank_updates: Dict[str, dict],
    ) -> Optional[dict]:
        """
        High-level entry point: wait for consensus, then aggregate.

        Args:
            round_num    : current round
            bank_updates : {bank_id: {"model_cid", "model_hash", "num_samples"}}
        """
        accepted = self.wait_for_consensus(round_num)
        return self.aggregate_round(round_num, accepted, bank_updates)
