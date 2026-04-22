"""
fl-layer/resilience/backup_logic.py
Pure beta-blend backup model function.

Changes vs. CCFD-FL-layer/src/resilience/backup_logic.py:
  - Removed filesystem reads (no round_dir, no global_model.pt loading).
  - Both state dicts are passed in as arguments (pure function).
  - Note on blending convention: beta * global_model + (1-beta) * brand_model
    (beta=0 → brand only, beta=1 → global only).
"""
from __future__ import annotations

import logging
from typing import Dict

import torch

logger = logging.getLogger(__name__)

StateDict = Dict[str, torch.Tensor]


def blend_with_global(
    brand_model: StateDict,
    global_model: StateDict,
    beta: float = 0.3,
) -> StateDict:
    """
    Blend brand_model and global_model:  beta * global + (1 - beta) * brand

    Args:
        brand_model  : state dict from local FedAvg (the brand aggregation)
        global_model : state dict of the previous global model (from IPFS)
        beta         : blending coefficient ∈ [0, 1]
                       beta=0.0 → pure brand model
                       beta=1.0 → pure global model
                       beta=0.3 → 30% global, 70% brand (plan default)

    Returns:
        Blended state dict with same keys as brand_model / global_model

    Raises:
        ValueError if beta is outside [0, 1]
    """
    if not (0.0 <= beta <= 1.0):
        raise ValueError(f"beta must be in [0, 1], got {beta}")

    out: StateDict = {}
    for k in brand_model.keys():
        a = global_model[k]
        b = brand_model[k]
        if (
            isinstance(a, torch.Tensor)
            and isinstance(b, torch.Tensor)
            and torch.is_floating_point(a)
            and torch.is_floating_point(b)
        ):
            out[k] = beta * a.float() + (1.0 - beta) * b.float()
            out[k] = out[k].to(dtype=b.dtype)
        else:
            out[k] = b  # non-float: keep from brand
    logger.debug("blend_with_global: beta=%.3f keys=%d", beta, len(out))
    return out
