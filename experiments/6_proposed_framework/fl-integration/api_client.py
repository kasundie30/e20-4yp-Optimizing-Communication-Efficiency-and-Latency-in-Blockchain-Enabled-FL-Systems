"""
fl-integration/api_client.py
Thin HTTP client wrapping the FastAPI server from Phase 4.

Responsibilities:
  - Abstract all REST calls to a single object (APIClient)
  - Raise APIError on non-2xx responses
  - Retry on 503 (Fabric network hiccup)
  - No Fabric / IPFS / chaincode knowledge here

All methods are synchronous (requests library) to keep the FL pipeline
simple. Phase 7+ can swap in an async version without changing callers.
"""
from __future__ import annotations

import hashlib
import io
import logging
import time
from typing import Dict, List, Optional

import requests

logger = logging.getLogger(__name__)


class APIError(Exception):
    """Raised when the API returns an unexpected HTTP status."""
    def __init__(self, status_code: int, detail: str):
        super().__init__(f"HTTP {status_code}: {detail}")
        self.status_code = status_code
        self.detail = detail


class APIClient:
    """
    Synchronous client for the HCFL FastAPI server.

    Args:
        base_url     : e.g. "http://localhost:8000"
        timeout      : per-request timeout in seconds
        max_retries  : number of retries on 503 ServiceUnavailable
        retry_delay  : seconds between retries
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        timeout: float = 30.0,
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._session = requests.Session()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _request(self, method: str, path: str, **kwargs) -> dict:
        url = f"{self.base_url}{path}"
        last_exc: Optional[Exception] = None

        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self._session.request(
                    method, url, timeout=self.timeout, **kwargs
                )
                if resp.status_code == 503 and attempt < self.max_retries:
                    logger.warning("503 on %s — retry %d/%d in %.1fs",
                                   path, attempt, self.max_retries, self.retry_delay)
                    time.sleep(self.retry_delay)
                    continue
                if not resp.ok:
                    try:
                        detail = resp.json().get("detail", resp.text)
                    except Exception:
                        detail = resp.text
                    raise APIError(resp.status_code, detail)
                return resp.json()
            except APIError:
                raise
            except requests.RequestException as e:
                last_exc = e
                logger.warning("Request error on %s attempt %d: %s", path, attempt, e)
                time.sleep(self.retry_delay)

        raise APIError(0, f"All retries exhausted for {path}: {last_exc}")

    # ------------------------------------------------------------------
    # READ endpoints (6.1 — global model fetch)
    # ------------------------------------------------------------------

    def health(self) -> dict:
        return self._request("GET", "/health")

    def get_trust_scores(self) -> Dict[str, float]:
        """Returns {BankA: score, BankB: score, BankC: score}."""
        resp = self._request("GET", "/trust-scores")
        return resp["scores"]

    def get_global_model(self, round_num: int) -> dict:
        """
        Fetch global model record for `round_num`.

        Returns:
            {"round": int, "global_cid": str, "global_hash": str}
        Raises:
            APIError(404) if no global model stored for that round.
        """
        return self._request("GET", f"/global-model/{round_num}")

    def get_latest_round(self) -> int:
        """Return the highest completed round number from the ledger."""
        resp = self._request("GET", "/latest-round")
        return int(resp["latest_round"])

    def check_consensus(self, round_num: int) -> List[str]:
        """Returns the list of banks that achieved CBFT consensus for `round_num`."""
        resp = self._request("GET", f"/check-consensus/{round_num}")
        return resp["accepted_banks"]

    def get_cluster_update(self, bank_id: str, round_num: int) -> dict:
        """Fetch the ClusterUpdate record proposed by a bank for a round."""
        return self._request("GET", f"/cluster-update/{bank_id}/{round_num}")

    def check_verify_quorum(self, bank_id: str, round_num: int) -> bool:
        """Check if a bank has received enough verification votes to proceed to commit."""
        resp = self._request("GET", f"/verify-quorum/{bank_id}/{round_num}")
        return resp["has_quorum"]

    # ------------------------------------------------------------------
    # WRITE endpoints (6.2 / 6.3 — submit update, store global)
    # ------------------------------------------------------------------

    def submit_update(
        self,
        bank_id: str,
        round_num: int,
        model_cid: str,
        model_hash: str,
        val_score: float,
    ) -> dict:
        """POST /submit-update — CBFT Phase 1: submit local cluster model."""
        payload = {
            "bank_id": bank_id,
            "round": round_num,
            "model_cid": model_cid,
            "model_hash": model_hash,
            "val_score": val_score,
        }
        return self._request("POST", "/submit-update", json=payload)

    def submit_verification(
        self,
        verifier_id: str,
        target_bank_id: str,
        round_num: int,
        verified: bool,
    ) -> dict:
        """POST /submit-verification — CBFT Phase 2: cast verification vote."""
        payload = {
            "verifier_id": verifier_id,
            "target_bank_id": target_bank_id,
            "round": round_num,
            "verified": verified,
        }
        return self._request("POST", "/submit-verification", json=payload)

    def submit_commit(
        self,
        committer_id: str,
        target_bank_id: str,
        round_num: int,
    ) -> dict:
        """POST /submit-commit — CBFT Phase 3: commit to accepting the update."""
        payload = {
            "committer_id": committer_id,
            "target_bank_id": target_bank_id,
            "round": round_num,
        }
        return self._request("POST", "/submit-commit", json=payload)

    def store_global_model(
        self,
        round_num: int,
        global_cid: str,
        global_hash: str,
    ) -> dict:
        """POST /store-global-model — persist the aggregated global model CID."""
        payload = {
            "round": round_num,
            "global_cid": global_cid,
            "global_hash": global_hash,
        }
        return self._request("POST", "/store-global-model", json=payload)

    def update_trust_score(self, bank_id: str, delta: float) -> dict:
        """POST /update-trust-score — reward (+) or penalise (-) a bank."""
        payload = {"bank_id": bank_id, "delta": delta}
        return self._request("POST", "/update-trust-score", json=payload)

    def close(self):
        self._session.close()


# ------------------------------------------------------------------
# IPFS helpers (thin wrappers — abstract IPFS for callers)
# ------------------------------------------------------------------

def compute_sha256(data: bytes) -> str:
    """Return lowercase hex SHA-256 of the byte string."""
    return hashlib.sha256(data).hexdigest()
