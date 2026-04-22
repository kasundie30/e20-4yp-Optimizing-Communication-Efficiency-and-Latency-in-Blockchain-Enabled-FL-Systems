# src/utils/ledger.py
# Ledger for recording model states and events

from __future__ import annotations
import os, json, time, hashlib
import torch

def hash_state_dict(sd: dict) -> str:
    h = hashlib.sha256()
    for k in sorted(sd.keys()):
        v = sd[k]
        h.update(k.encode())
        if torch.is_tensor(v):
            h.update(v.detach().cpu().numpy().tobytes())
        else:
            h.update(str(v).encode())
    return h.hexdigest()

def append_record(path: str, record: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    record["ts"] = time.time()
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")