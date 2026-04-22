"""
main.py — FastAPI REST server for the HCFL Fabric Blockchain Layer.

Routes (8 endpoints + /health):
  POST /submit-update          — SubmitClusterUpdate
  POST /submit-verification    — SubmitVerification (CBFT Phase 2)
  POST /submit-commit          — SubmitCommit (CBFT Phase 3)
  POST /update-trust-score     — UpdateTrustScore
  POST /store-global-model     — StoreGlobalModel
  GET  /trust-scores           — GetTrustScores
  GET  /check-consensus/{round} — CheckConsensus
  GET  /global-model/{round}   — GetGlobalModel
  GET  /health                 — Health check
"""
import json
import logging
import sys
import time

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import fabric_client as fc
from models import (
    ConsensusResponse,
    GlobalModelResponse,
    StoreGlobalModelRequest,
    SubmitCommitRequest,
    SubmitUpdateRequest,
    SubmitVerificationRequest,
    TransactionResponse,
    TrustScoresResponse,
    UpdateTrustScoreRequest,
)

# ---------------------------------------------------------------------------
#  Logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("api_server")

# ---------------------------------------------------------------------------
#  FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="HCFL Blockchain API",
    description="REST interface between the FL pipeline and the Hyperledger Fabric network.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
#  Middleware: structured request logging
# ---------------------------------------------------------------------------

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration_ms = (time.time() - start) * 1000
    logger.info(
        "method=%s path=%s status=%d duration=%.1fms",
        request.method, request.url.path, response.status_code, duration_ms,
    )
    return response


# ---------------------------------------------------------------------------
#  Helper
# ---------------------------------------------------------------------------

def _handle_fabric_error(e: Exception, ctx: str) -> None:
    logger.error("[%s] FabricError: %s", ctx, str(e))
    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail=f"Fabric network error: {str(e)}"
    )


# ---------------------------------------------------------------------------
#  Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "ok", "service": "HCFL Blockchain API"}


@app.post("/submit-update", response_model=TransactionResponse, status_code=status.HTTP_200_OK)
def submit_update(req: SubmitUpdateRequest):
    """
    Submit a cluster-level model update (CBFT Phase 1).
    Validation: valScore ≥ 0.7 enforced by chaincode.
    """
    logger.info("submit-update bank=%s round=%d cid=%s val_score=%.4f",
                req.bank_id, req.round, req.model_cid, req.val_score)
    try:
        result = fc.invoke(
            req.bank_id,
            "SubmitClusterUpdate",
            [req.bank_id, str(req.round), req.model_cid, req.model_hash, str(req.val_score)],
        )
        return TransactionResponse(
            status="success",
            message=f"Update submitted for {req.bank_id} round {req.round}",
            tx_id=result.get("tx_id"),
        )
    except Exception as e:
        _handle_fabric_error(e, "submit-update")


@app.post("/submit-verification", response_model=TransactionResponse)
def submit_verification(req: SubmitVerificationRequest):
    """
    Cast a verification vote (CBFT Phase 2 – Prepare).
    Identity enforcement: a bank cannot verify itself.
    """
    if req.verifier_id == req.target_bank_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Self-verification is not allowed."
        )
    logger.info("submit-verification verifier=%s target=%s round=%d verified=%s",
                req.verifier_id, req.target_bank_id, req.round, req.verified)
    try:
        result = fc.invoke(
            req.verifier_id,
            "SubmitVerification",
            [req.verifier_id, req.target_bank_id, str(req.round), str(req.verified).lower()],
            # endorse on all 3 peers so state is committed globally
        )
        return TransactionResponse(
            status="success",
            message=f"{req.verifier_id} verified {req.target_bank_id} round {req.round}",
            tx_id=result.get("tx_id"),
        )
    except Exception as e:
        _handle_fabric_error(e, "submit-verification")


@app.post("/submit-commit", response_model=TransactionResponse)
def submit_commit(req: SubmitCommitRequest):
    """Commit to accepting the target bank's update (CBFT Phase 3 – Commit)."""
    if req.committer_id == req.target_bank_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="A bank cannot commit to its own update."
        )
    logger.info("submit-commit committer=%s target=%s round=%d",
                req.committer_id, req.target_bank_id, req.round)
    try:
        result = fc.invoke(
            req.committer_id,
            "SubmitCommit",
            [req.committer_id, req.target_bank_id, str(req.round)],
            # endorse on all 3 peers so they all see the verification votes
        )
        return TransactionResponse(
            status="success",
            message=f"{req.committer_id} committed to {req.target_bank_id} round {req.round}",
            tx_id=result.get("tx_id"),
        )
    except Exception as e:
        _handle_fabric_error(e, "submit-commit")


@app.post("/update-trust-score", response_model=TransactionResponse)
def update_trust_score(req: UpdateTrustScoreRequest):
    """Update trust score for a bank (reward: +α, penalty: −β)."""
    logger.info("update-trust-score bank=%s delta=%.4f", req.bank_id, req.delta)
    try:
        result = fc.invoke(
            req.bank_id,
            "UpdateTrustScore",
            [req.bank_id, f"{req.delta:.4f}"],
        )
        return TransactionResponse(
            status="success",
            message=f"Trust score updated for {req.bank_id} by {req.delta:+.4f}",
            tx_id=result.get("tx_id"),
        )
    except Exception as e:
        _handle_fabric_error(e, "update-trust-score")


@app.post("/store-global-model", response_model=TransactionResponse)
def store_global_model(req: StoreGlobalModelRequest):
    """Store the trust-weighted global model CID for the given round."""
    logger.info("store-global-model round=%d cid=%s", req.round, req.global_cid)
    try:
        result = fc.invoke(
            "BankA",  # Global aggregator is always BankA (or use env DEFAULT_BANK)
            "StoreGlobalModel",
            [str(req.round), req.global_cid, req.global_hash],
        )
        return TransactionResponse(
            status="success",
            message=f"Global model stored for round {req.round}",
            tx_id=result.get("tx_id"),
        )
    except Exception as e:
        _handle_fabric_error(e, "store-global-model")


@app.get("/trust-scores", response_model=TrustScoresResponse)
def get_trust_scores():
    """Return all current trust scores for BankA, BankB, BankC."""
    try:
        raw = fc.query("BankA", "GetTrustScores", [])
        scores = json.loads(raw)
        return TrustScoresResponse(scores=scores)
    except Exception as e:
        _handle_fabric_error(e, "trust-scores")


@app.get("/check-consensus/{round_num}", response_model=ConsensusResponse)
def check_consensus(round_num: int):
    """
    Check which banks have achieved CBFT consensus (verify + commit quorum) for this round.
    """
    try:
        raw = fc.query("BankA", "CheckConsensus", [str(round_num)])
        accepted = json.loads(raw)
        return ConsensusResponse(accepted_banks=accepted, round=round_num)
    except Exception as e:
        _handle_fabric_error(e, "check-consensus")


@app.get("/global-model/{round_num}", response_model=GlobalModelResponse)
def get_global_model(round_num: int):
    """Retrieve the stored global model record (CID + hash) for the given round."""
    try:
        raw = fc.query("BankA", "GetGlobalModel", [str(round_num)])
        data = json.loads(raw)
        return GlobalModelResponse(
            round=data["round"],
            global_cid=data["globalCID"],
            global_hash=data["globalHash"],
        )
    except fc.FabricError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        _handle_fabric_error(e, "get-global-model")


@app.get("/latest-round")
def get_latest_round():
    """Return the highest completed round number on the ledger."""
    try:
        raw = fc.query("BankA", "GetLatestRound", [])
        return {"latest_round": int(raw)}
    except Exception as e:
        _handle_fabric_error(e, "get-latest-round")


@app.get("/cluster-update/{bank_id}/{round_num}")
def get_cluster_update(bank_id: str, round_num: int):
    """Retrieve the cluster update proposed by a bank for the given round."""
    try:
        raw = fc.query("BankA", "GetClusterUpdate", [bank_id, str(round_num)])
        return json.loads(raw)
    except fc.FabricError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        _handle_fabric_error(e, "get-cluster-update")


@app.get("/verify-quorum/{bank_id}/{round_num}")
def check_verify_quorum(bank_id: str, round_num: int):
    """Check if the target bank has achieved VerifyQuorum for the given round."""
    try:
        raw = fc.query("BankA", "CheckVerifyQuorum", [bank_id, str(round_num)])
        has_quorum = json.loads(raw)
        return {"bank_id": bank_id, "round": round_num, "has_quorum": bool(has_quorum)}
    except Exception as e:
        _handle_fabric_error(e, "check-verify-quorum")

