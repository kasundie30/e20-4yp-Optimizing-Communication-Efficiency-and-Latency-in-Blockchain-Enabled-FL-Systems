"""
fl-integration/hq_agent.py
HQ Agent — the FL pipeline driver for a single bank.

Responsibilities (one FL round):
  Step 1: Fetch previous global model CID from blockchain (GET /global-model/{round-1})
  Step 2: Download model weights from IPFS
  Step 3: Run local FedAvg over branch models (from deadline_collect output)
  Step 4: Compute SHA-256 of weights, upload updated model to IPFS
  Step 5: Evaluate merged model → PR-AUC val_score
  Step 6: Submit to blockchain (POST /submit-update)

The HQ agent does NOT run local_train itself — it receives already-trained
branch state_dicts from the round_coordinator.

Design: All external I/O (IPFS upload/download, API calls) is injected as
callable arguments so the unit tests can replace them with stubs without
needing a live network.
"""
from __future__ import annotations

import hashlib
import io
import logging
from typing import Callable, Dict, List, Optional, Tuple

import torch
from torch.utils.data import TensorDataset

from api_client import APIClient, APIError, compute_sha256
from global_aggregator import GlobalAggregator
from config.config_loader import load_config

# fl-layer imports (pure, no blockchain)
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "fl-layer"))

from model.FL_model import LSTMTabular
from aggregation.fedavg import fedavg
from validation.validate_fast import evaluate_model

logger = logging.getLogger(__name__)

StateDict = Dict[str, torch.Tensor]

# -----------------------------------------------------------
# Type aliases for injectable I/O callables
# -----------------------------------------------------------
# ipfs_upload_fn(bytes) -> str    (returns CID)
# ipfs_download_fn(str) -> bytes  (takes CID, returns bytes)
IpfsUploadFn   = Callable[[bytes], str]
IpfsDownloadFn = Callable[[str], bytes]


class HQAgent:
    """
    Orchestrates a single FL training round for one bank's HQ peer.

    Args:
        bank_id        : e.g. "BankA"
        client         : APIClient pointing at the FastAPI server
        ipfs_upload    : callable(bytes) -> cid   (injectable)
        ipfs_download  : callable(cid)  -> bytes  (injectable)
        val_dataset    : TensorDataset used for PR-AUC evaluation
        model_cfg      : dict of {"input_dim", "hidden_dim", "num_layers"}
        val_threshold  : minimum PR-AUC to submit (default 0.0 — always submit)
    """

    def __init__(
        self,
        bank_id: str,
        client: APIClient,
        ipfs_upload: IpfsUploadFn,
        ipfs_download: IpfsDownloadFn,
        val_dataset: TensorDataset,
        model_cfg: dict | None = None,
        val_threshold: float | None = None,
    ):
        self.bank_id = bank_id
        self.client = client
        self.ipfs_upload = ipfs_upload
        self.ipfs_download = ipfs_download
        self.val_dataset = val_dataset
        self.model_cfg = model_cfg or {"input_dim": 30, "hidden_dim": 30, "num_layers": 1}
        
        # Load from unified config
        self.config = load_config()
        self.val_threshold = val_threshold if val_threshold is not None else self.config.fl.validation_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_global_model(self, round_num: int) -> Optional[StateDict]:
        """
        6.1 — Fetch global model for round_num from blockchain + IPFS.

        Returns:
            state_dict if found and verified, None for round 1 (no prior global).

        Raises:
            ValueError if hash verification fails.
        """
        if round_num <= 1:
            logger.info("Round %d — no prior global model; using random init.", round_num, extra={"bank_id": self.bank_id, "round_num": round_num})
            return None

        try:
            record = self.client.get_global_model(round_num - 1)
        except APIError as e:
            if e.status_code == 404:
                logger.info("No global model recorded for round %d; fresh start.", round_num - 1, extra={"bank_id": self.bank_id, "round_num": round_num})
                return None
            raise

        cid        = record["global_cid"]
        stored_hash = record["global_hash"]

        logger.info("Downloading global model CID=%s from IPFS", cid, extra={"bank_id": self.bank_id, "round_num": round_num})
        data = self.ipfs_download(cid)

        actual_hash = compute_sha256(data)
        if actual_hash != stored_hash:
            raise ValueError(
                f"[{self.bank_id}] Global model hash mismatch for CID={cid}. "
                f"Expected={stored_hash}  Got={actual_hash}"
            )

        buf = io.BytesIO(data)
        state_dict = torch.load(buf, map_location="cpu")
        logger.info("Global model verified and loaded (round %d).", round_num - 1, extra={"bank_id": self.bank_id, "round_num": round_num})
        return state_dict

    def run_round(
        self,
        round_num: int,
        branch_updates: List[Tuple[StateDict, int]],
    ) -> dict:
        """
        6.2 — Run a full HQ FL round.

        Args:
            round_num      : current round number (1-indexed)
            branch_updates : list of (state_dict, num_samples) from branches

        Returns:
            dict with keys: val_score, model_cid, model_hash, submitted (bool), tx

        Note: hash mismatch in fetch_global_model raises ValueError and
        aborts the round (no submission).
        """
        logger.info("=== Round %d start — %d branch update(s) ===",
                    round_num, len(branch_updates), extra={"bank_id": self.bank_id, "round_num": round_num})

        # Step 1–2: Fetch prior global model (may be None for round 1)
        global_sd = self.fetch_global_model(round_num)

        # Step 3: FedAvg over branch updates
        if not branch_updates:
            raise ValueError(f"[{self.bank_id}] No branch updates for round {round_num}")

        avg_sd = fedavg(branch_updates)

        # Optional: blend with global model if available
        if global_sd is not None:
            from sys import path as _p
            _fl = os.path.join(os.path.dirname(__file__), "..", "fl-layer")
            if _fl not in _p:
                _p.insert(0, _fl)
            from resilience.backup_logic import blend_with_global
            beta = self.config.fl.backup_beta
            avg_sd = blend_with_global(avg_sd, global_sd, beta=beta)
            logger.debug("Blended FedAvg output with global (beta=%.2f).", beta, extra={"bank_id": self.bank_id, "round_num": round_num})

        # Step 4: Evaluate → PR-AUC
        model = LSTMTabular(**self.model_cfg)
        model.load_state_dict(avg_sd)
        metrics = evaluate_model(model, self.val_dataset, sample_fraction=0.15)
        val_score = metrics["pr_auc"]
        logger.info("Round %d Evaluation: PR-AUC=%.4f, ROC-AUC=%.4f, F1=%.4f (threshold=%.4f)",
                    round_num, val_score, metrics["roc_auc"], metrics["f1"], self.val_threshold, 
                    extra={"bank_id": self.bank_id, "round_num": round_num, **metrics})

        if val_score < self.val_threshold:
            logger.warning("val_score below threshold — skipping submission.", extra={"bank_id": self.bank_id, "round_num": round_num})
            return {"val_score": val_score, "model_cid": None, "model_hash": None,
                    "submitted": False, "tx": None}

        # Step 5: Serialize weights → bytes → CID
        buf = io.BytesIO()
        torch.save(avg_sd, buf)
        weight_bytes = buf.getvalue()
        model_hash = compute_sha256(weight_bytes)
        model_cid  = self.ipfs_upload(weight_bytes)
        logger.info("Uploaded CID=%s hash=%s", model_cid, model_hash[:16] + "...", extra={"bank_id": self.bank_id, "round_num": round_num})

        # Step 6: Submit to blockchain
        try:
            tx = self.client.submit_update(
                bank_id=self.bank_id,
                round_num=round_num,
                model_cid=model_cid,
                model_hash=model_hash,
                val_score=val_score,
            )
            logger.info("Submitted update — tx_id=%s", tx.get("tx_id"), extra={"bank_id": self.bank_id, "round_num": round_num})
            return {"val_score": val_score, "model_cid": model_cid,
                    "model_hash": model_hash, "submitted": True, "tx": tx}
        except APIError as e:
            logger.error("submit_update failed: %s", e, extra={"bank_id": self.bank_id, "round_num": round_num})
            return {"val_score": val_score, "model_cid": model_cid,
                    "model_hash": model_hash, "submitted": False, "tx": None}

    def verify_peer_updates(self, round_num: int, peers: List[str]) -> Dict[str, bool]:
        """
        CBFT Phase 2: Fetch peer updates, evaluate, and submit verification vote.
        """
        results = {}
        for target in peers:
            if target == self.bank_id:
                continue
            
            try:
                update = self.client.get_cluster_update(target, round_num)
            except APIError as e:
                # Target hasn't submitted yet or failed
                continue
                
            # If the update is a placeholder for backup activation
            if update.get("backupActive") and not update.get("modelCID"):
                continue

            model_cid = update["modelCID"]
            model_hash = update["modelHash"]
            
            # Download and verify hash
            try:
                data = self.ipfs_download(model_cid)
            except Exception as e:
                logger.warning("Failed to download model CID=%s for %s", model_cid, target, extra={"bank_id": self.bank_id, "round_num": round_num})
                continue

            if compute_sha256(data) != model_hash:
                logger.warning("Hash mismatch for %s update. Expected %s", target, model_hash, extra={"bank_id": self.bank_id, "round_num": round_num})
                self.client.submit_verification(self.bank_id, target, round_num, False)
                results[target] = False
                continue
                
            buf = io.BytesIO(data)
            try:
                peer_sd = torch.load(buf, map_location="cpu", weights_only=True)
            except Exception:
                peer_sd = torch.load(buf, map_location="cpu") # fallback for older torch
            
            # Evaluate
            model = LSTMTabular(**self.model_cfg)
            model.load_state_dict(peer_sd)
            metrics = evaluate_model(model, self.val_dataset, sample_fraction=0.15)
            
            verified = metrics["pr_auc"] >= self.val_threshold
            logger.info("Verifying %s round %d: PR-AUC=%.4f -> %s", target, round_num, metrics["pr_auc"], verified, extra={"bank_id": self.bank_id, "round_num": round_num})
            
            try:
                self.client.submit_verification(self.bank_id, target, round_num, verified)
                results[target] = verified
            except APIError as e:
                logger.warning("Failed to submit verification for %s: %s", target, e, extra={"bank_id": self.bank_id, "round_num": round_num})
            
        return results

    def commit_peer_updates(self, round_num: int, peers: List[str]) -> List[str]:
        """
        CBFT Phase 3: Check verify quorum for peers and submit commit.
        """
        committed = []
        for target in peers:
            if target == self.bank_id:
                continue
            
            try:
                has_quorum = self.client.check_verify_quorum(target, round_num)
                if has_quorum:
                    self.client.submit_commit(self.bank_id, target, round_num)
                    committed.append(target)
                    logger.info("Committed to %s round %d", target, round_num, extra={"bank_id": self.bank_id, "round_num": round_num})
            except APIError as e:
                pass
                
        return committed

