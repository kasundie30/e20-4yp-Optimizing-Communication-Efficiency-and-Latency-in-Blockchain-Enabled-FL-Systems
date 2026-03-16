# src/utils/score_store.py
# Score storage for brand reputation

from __future__ import annotations
import os, json
from typing import Dict, List

def load_scores(path: str, brands: List[str], init_score: float) -> Dict[str, float]:
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {b: float(init_score) for b in brands}

def save_scores(path: str, scores: Dict[str, float]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(scores, f, indent=2)

def update_score(scores: Dict[str, float], brand: str, ok: bool, reward: float, penalty: float, floor: float, سق_max: float) -> None:
    s = float(scores.get(brand, 1.0))
    s = s + reward if ok else s - penalty
    s = max(floor, min(سق_max, s))
    scores[brand] = s