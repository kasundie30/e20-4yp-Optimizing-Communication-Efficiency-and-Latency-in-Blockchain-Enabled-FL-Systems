"""
Simulated Blockchain: SHA-256 hash-chained ledger.
Each FL round appends one block. verify_chain() proves immutability.
"""
import json, hashlib, time, struct
from pathlib import Path
import torch

GENESIS = "0" * 64

def _hash_sd(sd):
    h = hashlib.sha256()
    for k in sorted(sd.keys()):
        v = sd[k]
        h.update(k.encode())
        h.update(v.detach().cpu().float().numpy().tobytes() if torch.is_tensor(v)
                 else struct.pack("d", float(v)))
    return h.hexdigest()

def _hash_block(blk):
    payload = {k: v for k, v in blk.items() if k != "block_hash"}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

class Ledger:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._chain = []
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                if line.strip():
                    self._chain.append(json.loads(line))

    @property
    def prev_hash(self):
        return self._chain[-1]["block_hash"] if self._chain else GENESIS

    def append_block(self, round_num, state_dict, comm_mb, e2e_sec, metrics):
        blk = {
            "index":      len(self._chain),
            "timestamp":  time.time(),
            "round":      round_num,
            "prev_hash":  self.prev_hash,
            "model_hash": _hash_sd(state_dict),
            "comm_mb":    round(comm_mb, 6),
            "e2e_sec":    round(e2e_sec, 4),
            "metrics":    {k: round(float(v), 6) for k, v in metrics.items()
                           if isinstance(v, (int, float))},
        }
        blk["block_hash"] = _hash_block(blk)
        self._chain.append(blk)
        with open(self.path, "a") as f:
            f.write(json.dumps(blk) + "\n")
        return blk

    def verify_chain(self):
        for i, blk in enumerate(self._chain):
            if _hash_block(blk) != blk["block_hash"]:
                return False
            expected = GENESIS if i == 0 else self._chain[i-1]["block_hash"]
            if blk["prev_hash"] != expected:
                return False
        return True

    def __len__(self):
        return len(self._chain)
