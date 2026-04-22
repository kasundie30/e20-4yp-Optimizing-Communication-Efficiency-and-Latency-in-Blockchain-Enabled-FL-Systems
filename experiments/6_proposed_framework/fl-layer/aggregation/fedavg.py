"""
fl-layer/aggregation/fedavg.py
Pure FedAvg function — no file I/O, no config reads, no side effects.

Changes vs. CCFD-FL-layer/src/aggregation/fedavg.py:
  - Signature uses (model_updates: list[tuple[state_dict, int]]) -> state_dict
    to bundle weights with sample counts (cleaner than separate lists).
  - Added explicit ModelKeyMismatchError and ZeroSamplesError.
  - Removed load_state_dict / save_state_dict helpers (filesystem).
  - Kept float-only averaging logic unchanged (numerically identical).
"""
from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import torch

logger = logging.getLogger(__name__)

StateDict = Dict[str, torch.Tensor]


class ModelKeyMismatchError(ValueError):
    """Raised when model state_dicts have different keys."""


class ZeroSamplesError(ValueError):
    """Raised when a model contributes zero samples (would cause div-by-zero)."""


def fedavg(
    model_updates: List[Tuple[StateDict, int]],
) -> StateDict:
    """
    Weighted FedAvg over a list of (state_dict, num_samples) tuples.

    - Weights are proportional to num_samples.
    - Non-float tensors (e.g. LSTM num_batches_tracked) are copied from first.
    - Raises ModelKeyMismatchError if any model has different keys.
    - Raises ZeroSamplesError if any model reports 0 samples.

    Args:
        model_updates: list of (state_dict, num_samples)

    Returns:
        Averaged state_dict
    """
    if not model_updates:
        raise ValueError("fedavg: model_updates list is empty")

    # Validate and extract
    state_dicts: List[StateDict] = []
    sample_counts: List[int] = []
    reference_keys = set(model_updates[0][0].keys())

    for i, (sd, n) in enumerate(model_updates):
        if n <= 0:
            raise ZeroSamplesError(
                f"fedavg: model at index {i} has zero or negative sample count ({n})"
            )
        if set(sd.keys()) != reference_keys:
            raise ModelKeyMismatchError(
                f"fedavg: model at index {i} has different keys than model at index 0.\n"
                f"  Expected: {sorted(reference_keys)}\n"
                f"  Got:      {sorted(sd.keys())}"
            )
        state_dicts.append(sd)
        sample_counts.append(n)

    total = sum(sample_counts)
    weights = [n / total for n in sample_counts]

    logger.debug(
        "fedavg: %d models, total_samples=%d, weights=%s",
        len(state_dicts), total, [f"{w:.3f}" for w in weights],
    )

    avg: StateDict = {}
    for k in state_dicts[0].keys():
        v0 = state_dicts[0][k]

        if not isinstance(v0, torch.Tensor) or not torch.is_floating_point(v0):
            # Non-float tensors: copy from first model
            avg[k] = v0.clone() if isinstance(v0, torch.Tensor) else v0
            continue

        acc = torch.zeros_like(v0, dtype=torch.float32)
        for sd, w in zip(state_dicts, weights):
            acc += w * sd[k].detach().float()

        avg[k] = acc.to(dtype=v0.dtype)

    return avg
