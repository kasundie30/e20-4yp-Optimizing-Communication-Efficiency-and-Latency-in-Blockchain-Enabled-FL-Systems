# src/resilience/deadline_collect.py
# Waits up to N seconds for branch models

from __future__ import annotations
import os
import time
from typing import List, Tuple, Optional, Dict

from src.aggregation.fedavg import load_state_dict
from src.clustering.ids import local_model_filename

def _candidate_paths(round_dir: str, filename: str) -> List[str]:
    return [
        os.path.join(round_dir, filename),
        os.path.join(round_dir, "branches", filename),
    ]

def find_model_file(round_dir: str, branch_id: str) -> Optional[str]:
    fname = local_model_filename(branch_id)
    for p in _candidate_paths(round_dir, fname):
        if os.path.exists(p):
            return p
    return None

def collect_until_deadline(
    round_dir: str,
    branch_ids: List[str],
    deadline_sec: int,
    poll_interval: float = 1.0
) -> Tuple[List[str], List[Dict]]:
    """
    Wait up to deadline_sec and collect whatever branch local models appear.
    Returns: (arrived_branch_ids, state_dicts)
    """
    start = time.time()
    arrived: dict[str, Dict] = {}

    while True:
        for bid in branch_ids:
            if bid in arrived:
                continue
            p = find_model_file(round_dir, bid)
            if p:
                arrived[bid] = load_state_dict(p)

        if time.time() - start >= deadline_sec:
            break

        time.sleep(poll_interval)

    arrived_ids = list(arrived.keys())
    return arrived_ids, [arrived[i] for i in arrived_ids]