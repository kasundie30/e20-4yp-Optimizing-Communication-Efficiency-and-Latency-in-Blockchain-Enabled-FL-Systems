# src/resilience/backup_logic.py
# Backup recovery logic (Phase 2)

from __future__ import annotations
import os
import torch
from typing import Dict

from src.aggregation.fedavg import load_state_dict

def blend_with_prev_global(round_dir: str, brand_sd: Dict[str, torch.Tensor], beta: float = 0.7) -> Dict[str, torch.Tensor]:
    """
    w_alt = beta*w_prev_global + (1-beta)*w_brand
    """
    prev_path = os.path.join(round_dir, "global_model.pt")
    prev_sd = load_state_dict(prev_path)

    out = {}
    for k in brand_sd.keys():
        a = prev_sd[k]
        b = brand_sd[k]
        if torch.is_tensor(a) and torch.is_tensor(b) and torch.is_floating_point(a) and torch.is_floating_point(b):
            out[k] = beta * a + (1.0 - beta) * b
        else:
            out[k] = a
    return out