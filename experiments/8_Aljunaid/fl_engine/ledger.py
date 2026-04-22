"""
fl_engine/ledger.py — Simulated blockchain ledger for Aljunaid et al. (2025).

Each FL round appends one immutable block containing:
  - model hash (SHA-256 of serialised model parameters)
  - communication overhead (MB)
  - E2E latency (seconds)
  - round evaluation metrics
  - previous block hash (chain linkage)

verify_chain() proves immutability of the audit trail.
"""
import json
import hashlib
import time
import io
import joblib
from pathlib import Path

GENESIS = "0" * 64


def _hash_model(model) -> str:
    """SHA-256 hash of joblib-serialised model."""
    buf = io.BytesIO()
    joblib.dump(model, buf)
    return hashlib.sha256(buf.getvalue()).hexdigest()


def _hash_block(blk: dict) -> str:
    payload = {k: v for k, v in blk.items() if k != "block_hash"}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


class Ledger:
    def __init__(self, path: str):
        self.path   = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._chain = []
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                if line.strip():
                    self._chain.append(json.loads(line))

    @property
    def prev_hash(self) -> str:
        return self._chain[-1]["block_hash"] if self._chain else GENESIS

    def append_block(self, round_num: int, model, comm_mb: float,
                     e2e_sec: float, metrics: dict, best_client_idx: int) -> dict:
        blk = {
            "index":           len(self._chain),
            "timestamp":       time.time(),
            "round":           round_num,
            "prev_hash":       self.prev_hash,
            "model_hash":      _hash_model(model),
            "best_client":     best_client_idx,
            "comm_mb":         round(comm_mb, 6),
            "e2e_sec":         round(e2e_sec, 4),
            "metrics":         {k: round(float(v), 6) for k, v in metrics.items()
                                if isinstance(v, (int, float))},
        }
        blk["block_hash"] = _hash_block(blk)
        self._chain.append(blk)
        with open(self.path, "a") as f:
            f.write(json.dumps(blk) + "\n")
        return blk

    def verify_chain(self) -> bool:
        for i, blk in enumerate(self._chain):
            if _hash_block(blk) != blk["block_hash"]:
                return False
            expected = GENESIS if i == 0 else self._chain[i - 1]["block_hash"]
            if blk["prev_hash"] != expected:
                return False
        return True

    def __len__(self) -> int:
        return len(self._chain)
