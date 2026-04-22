# src/aggregation/fedavg.py
# Standard FedAvg implementation

from __future__ import annotations
from typing import Dict, List, Optional
import torch


StateDict = Dict[str, torch.Tensor]


def _is_float_tensor(x: torch.Tensor) -> bool:
    return torch.is_floating_point(x)


def fedavg_state_dicts(
    state_dicts: List[StateDict],
    weights: Optional[List[float]] = None
) -> StateDict:
    """
    FedAvg over a list of PyTorch state_dicts.

    - If weights is None: uniform average.
    - If weights provided: weighted average (must be same length).

    Handles non-float tensors (e.g., BatchNorm num_batches_tracked) by copying from first model.
    """
    if not state_dicts:
        raise ValueError("fedavg_state_dicts: state_dicts list is empty")

    n = len(state_dicts)

    if weights is None:
        weights = [1.0 / n] * n
    else:
        if len(weights) != n:
            raise ValueError("fedavg_state_dicts: weights length != number of models")
        s = sum(weights)
        if s <= 0:
            raise ValueError("fedavg_state_dicts: sum(weights) must be > 0")
        weights = [w / s for w in weights]  # normalize

    # Initialize averaged dict
    avg: StateDict = {}

    # Use keys from first dict
    keys = state_dicts[0].keys()

    for k in keys:
        v0 = state_dicts[0][k]

        # If not a tensor (rare), just copy
        if not isinstance(v0, torch.Tensor):
            avg[k] = v0
            continue

        # Non-float tensors should not be averaged (copy from first model)
        if not _is_float_tensor(v0):
            avg[k] = v0.clone() if hasattr(v0, "clone") else v0
            continue

        # Weighted sum for float tensors
        acc = torch.zeros_like(v0, dtype=torch.float32, device="cpu")

        for sd, w in zip(state_dicts, weights):
            v = sd[k].detach().to("cpu", dtype=torch.float32)
            acc += w * v

        # Cast back to original dtype
        avg[k] = acc.to(dtype=v0.dtype)

    return avg


def load_state_dict(path: str) -> StateDict:
    """
    Loads a state_dict saved by torch.save(model.state_dict(), path).
    Always maps to CPU to avoid GPU dependency during aggregation.
    """
    obj = torch.load(path, map_location="cpu")
    if not isinstance(obj, dict):
        raise ValueError(f"Expected state_dict (dict) in {path}, got {type(obj)}")
    return obj


def save_state_dict(state_dict: StateDict, path: str) -> None:
    """
    Saves a state_dict to disk.
    """
    torch.save(state_dict, path)