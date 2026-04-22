"""
models.py — Pydantic request/response schemas for the API server.
"""
from pydantic import BaseModel, Field, field_validator
from typing import Optional


class SubmitUpdateRequest(BaseModel):
    bank_id: str = Field(..., description="Bank identifier, e.g. 'BankA'")
    round: int = Field(..., ge=1, description="FL training round (>= 1)")
    model_cid: str = Field(..., description="IPFS content identifier of the cluster model")
    model_hash: str = Field(..., description="SHA-256 hex digest of the model file")
    val_score: float = Field(..., ge=0.0, le=1.0, description="Validation metric (F1 / AUC-ROC) ∈ [0, 1]")

    @field_validator("bank_id")
    @classmethod
    def bank_id_must_be_valid(cls, v: str) -> str:
        if v not in ("BankA", "BankB", "BankC"):
            raise ValueError(f"bank_id must be one of BankA, BankB, BankC; got '{v}'")
        return v

    @field_validator("model_cid")
    @classmethod
    def cid_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("model_cid must not be empty")
        return v


class SubmitVerificationRequest(BaseModel):
    verifier_id: str = Field(..., description="Bank casting the verification vote")
    target_bank_id: str = Field(..., description="Bank whose update is being verified")
    round: int = Field(..., ge=1)
    verified: bool = Field(..., description="True = approve, False = reject")

    @field_validator("verifier_id", "target_bank_id")
    @classmethod
    def bank_must_be_valid(cls, v: str) -> str:
        if v not in ("BankA", "BankB", "BankC"):
            raise ValueError(f"bank_id must be one of BankA, BankB, BankC; got '{v}'")
        return v


class SubmitCommitRequest(BaseModel):
    committer_id: str = Field(..., description="Bank submitting the commit")
    target_bank_id: str = Field(..., description="Bank whose update is being committed")
    round: int = Field(..., ge=1)

    @field_validator("committer_id", "target_bank_id")
    @classmethod
    def bank_must_be_valid(cls, v: str) -> str:
        if v not in ("BankA", "BankB", "BankC"):
            raise ValueError(f"bank_id must be one of BankA, BankB, BankC; got '{v}'")
        return v


class UpdateTrustScoreRequest(BaseModel):
    bank_id: str = Field(..., description="Bank whose trust score is updated")
    delta: float = Field(..., description="Signed delta: +α for reward, −β for penalty")

    @field_validator("bank_id")
    @classmethod
    def bank_must_be_valid(cls, v: str) -> str:
        if v not in ("BankA", "BankB", "BankC"):
            raise ValueError(f"bank_id must be one of BankA, BankB, BankC; got '{v}'")
        return v


class StoreGlobalModelRequest(BaseModel):
    round: int = Field(..., ge=0)
    global_cid: str = Field(..., description="IPFS CID of the global model")
    global_hash: str = Field(..., description="SHA-256 hash of the global model")


class TransactionResponse(BaseModel):
    status: str
    message: str
    tx_id: Optional[str] = None


class TrustScoresResponse(BaseModel):
    scores: dict

    model_config = {"json_schema_extra": {"example": {"scores": {"BankA": 1.0, "BankB": 0.8, "BankC": 1.1}}}}


class ConsensusResponse(BaseModel):
    accepted_banks: list[str]
    round: int


class GlobalModelResponse(BaseModel):
    round: int
    global_cid: str
    global_hash: str
