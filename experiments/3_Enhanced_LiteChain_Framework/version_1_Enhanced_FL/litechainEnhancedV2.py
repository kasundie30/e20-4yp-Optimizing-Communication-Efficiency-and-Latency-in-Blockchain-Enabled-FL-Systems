# ============================================================
# LiteChain Full (Algorithms 1–6) + PARTIAL PARTICIPATION
# + LATENCY OPTIMIZATIONS (added on top of your partial-participation version)
#
# Added Techniques (on top of your LiteChain Full + partial participation):
#   (T1) Straggler-aware client selection (fast clients prioritized + exploration)
#   (T2) Deadline-based aggregation (select/accept limited updates per round)
#   (T3) Two-stage committee verification (cheap pre-check + expensive check only for top candidates)
#   (T4) Adaptive verification set size schedule (bigger early, smaller later)
#
# Output metrics aligned to your baseline/enhanced format:
#   [Round xx] Val PR-AUC=... F1(tuned)=... | Test PR-AUC=... F1(tuned)=... | Blocks=... K=... Comm=...MB E2E=...s
#
# ============================================================

import os, time, math, json, hashlib, random, hmac
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional, Deque
from collections import deque

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_recall_fscore_support, confusion_matrix
)

import wandb
import kagglehub


# -----------------------------
# 0) Safe W&B wrapper
# -----------------------------
class WB:
    """
    Safe wrapper around W&B so the script never crashes if W&B is unavailable.
    - WANDB_MODE=online/offline/disabled supported.
    """
    def __init__(self):
        self.enabled = False
        self.run = None

    def init(self, **kwargs):
        mode = os.environ.get("WANDB_MODE", "").lower().strip()

        if mode in {"disabled", "disable"} or os.environ.get("DISABLE_WANDB", "0") == "1":
            print("[W&B] Disabled.")
            self.enabled = False
            self.run = None
            return None

        if mode == "offline":
            kwargs.setdefault("mode", "offline")
        elif mode == "online":
            kwargs.setdefault("mode", "online")

        try:
            self.run = wandb.init(**kwargs)
            self.enabled = True
            return self.run
        except Exception as e:
            print(f"[W&B] init failed; continuing without W&B. Reason: {e}")
            self.enabled = False
            self.run = None
            return None

    def log(self, data: dict):
        if self.enabled and self.run is not None:
            try:
                wandb.log(data)
            except Exception as e:
                print(f"[W&B] log failed (ignored): {e}")

    def summary_set(self, key: str, value):
        if self.enabled and self.run is not None:
            try:
                self.run.summary[key] = value
            except Exception as e:
                print(f"[W&B] summary set failed (ignored): {e}")

    def finish(self):
        if self.enabled and self.run is not None:
            try:
                self.run.finish()
            except Exception as e:
                print(f"[W&B] finish failed (ignored): {e}")


# -----------------------------
# 1) Reproducibility + device selection
# -----------------------------
RANDOM_STATE = int(os.environ.get("SEED", "42"))

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(RANDOM_STATE)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------
# 2) Hyperparameters and Environment Config
# -----------------------------
PROJECT = os.environ.get("WANDB_PROJECT", "fl-fraud-litechain")
GROUP   = os.environ.get("WANDB_GROUP", "litechain")
RUN_NAME = os.environ.get("WANDB_RUN_NAME", f"litechain_full_partial_latencyplus_{int(time.time())}")
ENTITY  = os.environ.get("WANDB_ENTITY", None)

# FL setup
NUM_CLIENTS   = int(os.environ.get("NUM_CLIENTS", "10"))
NUM_ROUNDS    = int(os.environ.get("NUM_ROUNDS", "20"))
LOCAL_EPOCHS  = int(os.environ.get("LOCAL_EPOCHS", "1"))
BATCH_SIZE    = int(os.environ.get("BATCH_SIZE", "512"))
LR            = float(os.environ.get("LR", "1e-3"))
WEIGHT_DECAY  = float(os.environ.get("WEIGHT_DECAY", "1e-5"))

# LSTM
HIDDEN_DIM = int(os.environ.get("HIDDEN_DIM", "64"))
NUM_LAYERS = int(os.environ.get("NUM_LAYERS", "1"))

# DP-like (clip + noise)
L2_NORM_CLIP     = float(os.environ.get("DP_CLIP", "1.0"))
NOISE_MULTIPLIER = float(os.environ.get("DP_NOISE", "0.5"))

# Verification thresholds (Alg3/4)
PR_AUC_MIN = float(os.environ.get("PR_AUC_MIN", "0.10"))
ACC_MIN    = float(os.environ.get("ACC_MIN", "0.90"))

# Outlier rejection (MAD)
NORM_MAD_Z = float(os.environ.get("NORM_MAD_Z", "3.5"))

# Communication compression simulation
USE_QUANT_COMM = os.environ.get("USE_QUANT_COMM", "1") == "1"

# Global default threshold (for “default” metrics); tuned threshold used for reporting
GLOBAL_DEFAULT_THRESHOLD = 0.5

# Algorithm 5 / 6: intercluster staleness + window aggregation
INTERCLUSTER_EVERY = int(os.environ.get("INTERCLUSTER_EVERY", "1"))
STALE_Q = float(os.environ.get("STALE_Q", "1.0"))
CHI_WINDOW = int(os.environ.get("CHI_WINDOW", "3"))

# Algorithm 1/2: dynamic clustering
START_SINGLETONS = os.environ.get("START_SINGLETONS", "0") == "1"
OPTIMIZE_EVERY = int(os.environ.get("OPTIMIZE_EVERY", "1"))
MAX_OPT_SLOTS  = int(os.environ.get("MAX_OPT_SLOTS", "200"))
NEIGHBOR_TOP_M = int(os.environ.get("NEIGHBOR_TOP_M", "4"))

MIN_CLUSTER_SIZE = int(os.environ.get("MIN_CLUSTER_SIZE", "2"))
MAX_CLUSTER_SIZE = int(os.environ.get("MAX_CLUSTER_SIZE", "9999"))
COST_BIG = float(os.environ.get("COST_BIG", "1e9"))

# Latency proxy weights used by Alg1 objective
LAT_VERIFY_COST = float(os.environ.get("LAT_VERIFY_COST", "1e6"))
LAT_COMM_WEIGHT = float(os.environ.get("LAT_COMM_WEIGHT", "1.0"))

# Security proxy weights
SEC_W_SIZE     = float(os.environ.get("SEC_W_SIZE", "1.0"))
SEC_W_QUORUM   = float(os.environ.get("SEC_W_QUORUM", "1.0"))
SEC_W_REPUT    = float(os.environ.get("SEC_W_REPUT", "1.0"))
SEC_W_QUALITY  = float(os.environ.get("SEC_W_QUALITY", "1.0"))

# Committee re-election (Alg6)
REELECT_EVERY = int(os.environ.get("REELECT_EVERY", "1"))
COMMITTEE_W_REPUT = float(os.environ.get("COMMITTEE_W_REPUT", "1.0"))
COMMITTEE_W_SIZE  = float(os.environ.get("COMMITTEE_W_SIZE", "0.2"))

# Byzantine simulation (optional)
BYZANTINE_FRAC = float(os.environ.get("BYZANTINE_FRAC", "0.0"))

# --------- Partial participation (your technique) ----------
PARTICIPATION_RATE = float(os.environ.get("PARTICIPATION_RATE", "0.5"))   # fraction of clients per cluster per round
MIN_PARTICIPANTS   = int(os.environ.get("MIN_PARTICIPANTS", "1"))         # min clients per cluster per round

# --------- Added latency techniques (new knobs) ----------
# (T1) Straggler-aware selection
FAST_SELECT_FRAC = float(os.environ.get("FAST_SELECT_FRAC", "0.7"))        # % of selected clients from fastest pool
EXPLORE_FRAC     = float(os.environ.get("EXPLORE_FRAC", "0.3"))            # % from random exploration pool
LAT_EMA_BETA     = float(os.environ.get("LAT_EMA_BETA", "0.85"))           # EMA smoothing for per-client latency

# (T2) Deadline-based aggregation (approximation)
# We limit how many client updates we fully verify/consider per cluster per round.
MAX_UPDATES_PER_CLUSTER = int(os.environ.get("MAX_UPDATES_PER_CLUSTER", "9999"))

# (T3) Two-stage verification
STAGE1_SIZE = int(os.environ.get("STAGE1_SIZE", "2000"))                  # cheap subset size
STAGE2_MAX  = int(os.environ.get("STAGE2_MAX", "4"))                      # do expensive verify only for top-N candidates
STAGE1_PR_AUC_MIN = float(os.environ.get("STAGE1_PR_AUC_MIN", "0.02"))     # loose pre-filter (avoid rejecting early learning)

# (T4) Adaptive stage2 verification size schedule
STAGE2_SIZE_MAX = int(os.environ.get("STAGE2_SIZE_MAX", "15000"))          # early rounds
STAGE2_SIZE_MIN = int(os.environ.get("STAGE2_SIZE_MIN", "4000"))           # later rounds
STAGE2_DECAY    = float(os.environ.get("STAGE2_DECAY", "0.12"))            # exp decay rate

# Per-round threshold tuning (for your “F1(tuned)” logs)
THRESH_MIN = float(os.environ.get("THRESH_MIN", "0.01"))
THRESH_MAX = float(os.environ.get("THRESH_MAX", "0.99"))
THRESH_STEPS = int(os.environ.get("THRESH_STEPS", "50"))


# -----------------------------
# CBFT quorum
# -----------------------------
def cbft_quorum(k: int) -> int:
    return int(math.ceil((2 * k + 1) / 3))


# ============================================================
# 3) Helpers: hashing + deltas + averaging + quantization
# ============================================================

def state_dict_to_bytes(sd: Dict[str, torch.Tensor]) -> bytes:
    m = bytearray()
    for k in sorted(sd.keys()):
        t = sd[k].detach().cpu().contiguous()
        m.extend(k.encode("utf-8"))
        m.extend(t.numpy().tobytes())
    return bytes(m)

def hash_state_dict(sd: Dict[str, torch.Tensor]) -> str:
    return hashlib.sha256(state_dict_to_bytes(sd)).hexdigest()

def hash_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def clone_state_dict(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in sd.items()}

def delta_state_dict(new_sd: Dict[str, torch.Tensor], old_sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: (new_sd[k] - old_sd[k]) for k in new_sd.keys()}

def apply_delta(base_sd: Dict[str, torch.Tensor], delta: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: (base_sd[k] + delta[k]) for k in base_sd.keys()}

def weighted_average_state_dict(sds: List[Dict[str, torch.Tensor]], weights: List[float]) -> Dict[str, torch.Tensor]:
    wsum = float(np.sum(weights)) + 1e-12
    out = {}
    for k in sds[0].keys():
        acc = None
        for sd, w in zip(sds, weights):
            if acc is None:
                acc = sd[k].detach().clone() * (w / wsum)
            else:
                acc += sd[k].detach() * (w / wsum)
        out[k] = acc
    return out

def weighted_average_deltas(deltas: List[Dict[str, torch.Tensor]], weights: List[float]) -> Dict[str, torch.Tensor]:
    wsum = float(np.sum(weights)) + 1e-12
    out = {}
    for k in deltas[0].keys():
        acc = None
        for d, w in zip(deltas, weights):
            if acc is None:
                acc = d[k].detach().clone() * (w / wsum)
            else:
                acc += d[k].detach() * (w / wsum)
        out[k] = acc
    return out

def update_l2_norm(delta: Dict[str, torch.Tensor]) -> float:
    s = 0.0
    for v in delta.values():
        vv = v.detach().cpu()
        s += float(torch.sum(vv * vv))
    return float(np.sqrt(s))

def quantize_delta_int8(delta: Dict[str, torch.Tensor]) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    """
    Simulated communication compression:
    - Per-tensor symmetric int8 quantization (scale per tensor).
    - Useful to reduce comm volume in logs; it also reduces effective bandwidth requirements.
    """
    q = {}
    scales = {}
    for k, t in delta.items():
        a = t.detach().cpu().numpy()
        max_abs = float(np.max(np.abs(a))) + 1e-12
        scale = max_abs / 127.0
        qi = np.clip(np.round(a / scale), -127, 127).astype(np.int8)
        q[k] = qi
        scales[k] = scale
    return q, scales

def dequantize_delta_int8(q: Dict[str, np.ndarray], scales: Dict[str, float], device: torch.device) -> Dict[str, torch.Tensor]:
    out = {}
    for k, qi in q.items():
        out[k] = torch.tensor(qi.astype(np.float32) * scales[k], device=device)
    return out

def estimate_comm_bytes_for_delta(delta: Dict[str, torch.Tensor]) -> int:
    return sum(int(t.numel()) * 4 for t in delta.values())

def estimate_comm_bytes_for_quant_delta(q: Dict[str, np.ndarray], scales: Dict[str, float]) -> int:
    total = 0
    for _, qi in q.items():
        total += qi.size * 1   # int8
        total += 8            # scale overhead (approx)
    return total

def estimate_chain_record_bytes(record: Dict[str, Any]) -> int:
    return len(json.dumps(record, sort_keys=True).encode("utf-8"))


# ============================================================
# 4) Metrics
# ============================================================
def compute_metrics(y_true, y_prob, threshold=0.5) -> Dict[str, Any]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    acc = float((y_pred == y_true).mean())

    roc_auc = None
    pr_auc = None
    if len(np.unique(y_true)) > 1:
        roc_auc = float(roc_auc_score(y_true, y_prob))
        pr_auc = float(average_precision_score(y_true, y_prob))

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    return {
        "acc": acc,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }

@torch.no_grad()
def predict_proba(model: nn.Module, X: np.ndarray, batch_size: int = 4096) -> np.ndarray:
    model.eval()
    probs = []
    n = len(X)
    for i in range(0, n, batch_size):
        bx = torch.tensor(X[i:i+batch_size], dtype=torch.float32).unsqueeze(1).to(DEVICE)
        logits = model(bx)
        p = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)
        probs.append(p)
    return np.concatenate(probs, axis=0)

@torch.no_grad()
def evaluate_loss(model: nn.Module, X: np.ndarray, y: np.ndarray, pos_weight: float, batch_size: int = 4096) -> float:
    model.eval()
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], dtype=torch.float32, device=DEVICE))
    losses = []
    n = len(X)
    for i in range(0, n, batch_size):
        bx = torch.tensor(X[i:i+batch_size], dtype=torch.float32).unsqueeze(1).to(DEVICE)
        by = torch.tensor(y[i:i+batch_size], dtype=torch.float32).view(-1, 1).to(DEVICE)
        logits = model(bx)
        loss = crit(logits, by).detach().cpu().item()
        losses.append(loss)
    return float(np.mean(losses))

def tune_threshold_for_f1(y_true: np.ndarray, y_prob: np.ndarray,
                          tmin: float, tmax: float, steps: int) -> Tuple[float, float]:
    """
    Purpose:
      Match your baseline/enhanced reporting style:
      pick threshold that maximizes validation F1 over a grid, then report F1(tuned).
    """
    thresholds = np.linspace(tmin, tmax, steps)
    best_t, best_f1 = 0.5, -1.0
    for t in thresholds:
        m = compute_metrics(y_true, y_prob, threshold=float(t))
        if m["f1"] > best_f1:
            best_f1 = float(m["f1"])
            best_t = float(t)
    return best_t, best_f1


# ============================================================
# 5) Model: LSTMTabular
# ============================================================
class LSTMTabular(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, (h, c) = self.lstm(x)
        last_h = h[-1]
        logits = self.fc(last_h)
        return logits


# ============================================================
# 6) Local training with DP-like clip + Gaussian noise
# ============================================================
def local_train_private(
    model: nn.Module,
    client_x: np.ndarray,
    client_y: np.ndarray,
    pos_weight: float,
    epochs: int,
    lr: float,
    weight_decay: float,
    batch_size: int,
    l2_norm_clip: float,
    noise_multiplier: float,
    device: torch.device
) -> Tuple[nn.Module, float]:
    """
    Purpose:
      Local update step (Alg3 local training):
      - Optimize BCEWithLogits with pos_weight for imbalance.
      - Clip gradients and add Gaussian noise (DP-like).
    Returns:
      (trained model, wall-clock local training time)
    """
    model.train()
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], dtype=torch.float32, device=device))
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    x = torch.tensor(client_x, dtype=torch.float32).unsqueeze(1)
    y = torch.tensor(client_y, dtype=torch.float32).view(-1, 1)
    dl = DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=True)

    t0 = time.time()
    for _ in range(epochs):
        for bx, by in dl:
            bx, by = bx.to(device), by.to(device)
            opt.zero_grad()
            logits = model(bx)
            loss = crit(logits, by)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=l2_norm_clip)

            for p in model.parameters():
                if p.grad is not None:
                    noise = torch.randn_like(p.grad) * (l2_norm_clip * noise_multiplier)
                    p.grad.add_(noise)

            opt.step()

    return model, (time.time() - t0)


# ============================================================
# 7) LiteChain ledger + block payload
# ============================================================
@dataclass
class Block:
    height: int
    round_id: int
    cluster_id: int
    proposer: int
    model_hash: str
    metrics: Dict[str, float]
    accepted_client_ids: List[int]
    prev_hash: str
    timestamp: float
    block_hash: str

class LiteChainLedger:
    """
    Purpose:
      Minimal ledger tracking block hashes + preventing replay/duplicates.
      Mirrors the “LiteChain” behavior in your integrated implementation.
    """
    def __init__(self):
        self.blocks: List[Block] = []
        self.seen_model_hashes = set()

    def last_hash(self) -> str:
        return "GENESIS" if not self.blocks else self.blocks[-1].block_hash

    def append_block(self, blk: Block):
        self.blocks.append(blk)
        self.seen_model_hashes.add(blk.model_hash)

    def latest_block_for_cluster(self, cluster_id: int) -> Optional[Block]:
        for b in reversed(self.blocks):
            if b.cluster_id == cluster_id:
                return b
        return None

def make_block_payload(
    height: int,
    round_id: int,
    cluster_id: int,
    proposer: int,
    model_hash: str,
    metrics: Dict[str, float],
    accepted_client_ids: List[int],
    prev_hash: str,
    timestamp: float
) -> Tuple[Dict[str, Any], str]:
    payload = {
        "height": height,
        "round_id": round_id,
        "cluster_id": cluster_id,
        "proposer": proposer,
        "model_hash": model_hash,
        "metrics": metrics,
        "accepted_client_ids": accepted_client_ids,
        "prev_hash": prev_hash,
        "timestamp": timestamp,
    }
    b = json.dumps(payload, sort_keys=True).encode("utf-8")
    return payload, hash_bytes(b)


# ============================================================
# Signatures (Alg3 verification step)
# We simulate signatures using HMAC-SHA256 with per-client secret keys.
# ============================================================
def sign_hash(client_secret: bytes, model_hash: str) -> str:
    return hmac.new(client_secret, model_hash.encode("utf-8"), hashlib.sha256).hexdigest()

def verify_signature(client_secret: bytes, model_hash: str, signature: str) -> bool:
    expected = sign_hash(client_secret, model_hash)
    return hmac.compare_digest(expected, signature)


# ============================================================
# Algorithm 4: CBFT consensus (verify + commit phases)
# ============================================================
def cbft_consensus_algorithm4(
    ledger: LiteChainLedger,
    payload: Dict[str, Any],
    block_hash: str,
    committee_members: List[int],
    pr_auc_min: float,
    acc_min: float,
    byzantine_frac: float = 0.0
) -> Tuple[bool, Dict[str, Any]]:
    """
    Purpose:
      Implements Algorithm 4 (CBFT):
        - verify-phase: check duplicate + thresholds (PR-AUC + ACC)
        - commit-phase: quorum commit
      If committed: append block to ledger.
    """
    k = len(committee_members)
    quorum = cbft_quorum(k)

    # Duplicate protection
    if payload["model_hash"] in ledger.seen_model_hashes:
        return False, {"reason": "duplicate_model_hash", "verify_msgs": 0, "commit_msgs": 0, "quorum": quorum}

    # Byzantine simulation: some committee members won't vote
    n_byz = int(round(byzantine_frac * k))
    byz_set = set(random.sample(committee_members, k=n_byz)) if n_byz > 0 else set()

    # Verify thresholds
    pr_auc = float(payload["metrics"].get("pr_auc", -1.0))
    acc = float(payload["metrics"].get("acc", -1.0))
    thresholds_ok = (pr_auc >= pr_auc_min) and (acc >= acc_min)
    if not thresholds_ok:
        return False, {"reason": "threshold_fail", "pr_auc": pr_auc, "acc": acc, "verify_msgs": 0, "commit_msgs": 0, "quorum": quorum}

    # Verify-phase votes
    verify_msgs = sum(1 for m in committee_members if m not in byz_set)
    if verify_msgs < quorum:
        return False, {"reason": "not_enough_verify_msgs", "verify_msgs": verify_msgs, "commit_msgs": 0, "quorum": quorum}

    # Commit-phase votes
    commit_msgs = sum(1 for m in committee_members if m not in byz_set)
    if commit_msgs < quorum:
        return False, {"reason": "not_enough_commit_msgs", "verify_msgs": verify_msgs, "commit_msgs": commit_msgs, "quorum": quorum}

    blk = Block(
        height=payload["height"],
        round_id=payload["round_id"],
        cluster_id=payload["cluster_id"],
        proposer=payload["proposer"],
        model_hash=payload["model_hash"],
        metrics=payload["metrics"],
        accepted_client_ids=payload["accepted_client_ids"],
        prev_hash=payload["prev_hash"],
        timestamp=payload["timestamp"],
        block_hash=block_hash
    )
    ledger.append_block(blk)
    return True, {"reason": "committed", "verify_msgs": verify_msgs, "commit_msgs": commit_msgs, "quorum": quorum}


# ============================================================
# Algorithm 5: staleness-aware weights
# ============================================================
def staleness_weight_alg5(current_round: int, model_round: int, q: float) -> float:
    st = max(0, current_round - model_round)
    return float((st + 1.0) ** (-q))


# ============================================================
# Reputation (Algorithm 6)
# ============================================================
@dataclass
class ClientReputation:
    participated: int = 0
    accepted: int = 0
    rejected: int = 0
    pr_auc_sum: float = 0.0
    acc_sum: float = 0.0
    norm_sum: float = 0.0

    @property
    def acceptance_rate(self) -> float:
        return float(self.accepted / max(1, self.participated))

    @property
    def avg_pr_auc(self) -> float:
        return float(self.pr_auc_sum / max(1, self.participated))

    @property
    def avg_acc(self) -> float:
        return float(self.acc_sum / max(1, self.participated))

    @property
    def avg_norm(self) -> float:
        return float(self.norm_sum / max(1, self.participated))

def reputation_score(rep: ClientReputation) -> float:
    """
    Purpose:
      Algorithm 6 reputation proxy:
        - acceptance_rate (quality + compliance)
        - avg PR-AUC, avg ACC
        - stability: penalize large update norms
    """
    ar = rep.acceptance_rate
    pr = max(0.0, min(1.0, rep.avg_pr_auc))
    ac = max(0.0, min(1.0, rep.avg_acc))
    norm_penalty = 1.0 / (1.0 + rep.avg_norm)
    score = 0.4*ar + 0.3*pr + 0.2*ac + 0.1*norm_penalty
    return float(max(0.0, min(1.0, score)))


# ============================================================
# Algorithm 1/2: Distributed network optimization components (your existing proxies)
# ============================================================
def compute_cluster_latency_proxy(members: List[int], est_delta_bytes: int, committee_size: int) -> float:
    comm = LAT_COMM_WEIGHT * float(len(members) * est_delta_bytes)
    verify = LAT_VERIFY_COST * float(len(members))
    c2c = LAT_COMM_WEIGHT * float(max(0, committee_size - 1) * 1200.0)
    return float(comm + verify + c2c + 1e-9)

def compute_cluster_security_proxy(members: List[int], committee_size: int, avg_reputation: float, last_quality: float) -> float:
    size_term = SEC_W_SIZE * math.log(1.0 + len(members))
    quorum_term = SEC_W_QUORUM * float(cbft_quorum(max(1, committee_size)))
    reput_term = SEC_W_REPUT * float(avg_reputation)
    qual_term = SEC_W_QUALITY * float(max(0.0, last_quality))
    return float(size_term + quorum_term + reput_term + qual_term + 1e-9)

def cluster_cost_constraints(members: List[int], latency: float) -> float:
    if len(members) < MIN_CLUSTER_SIZE:
        return COST_BIG
    if len(members) > MAX_CLUSTER_SIZE:
        return COST_BIG
    if not np.isfinite(latency):
        return COST_BIG
    return 0.0

def cluster_value_alg1(
    members: List[int],
    est_delta_bytes: int,
    committee_size: int,
    last_quality: float,
    reputations: Dict[int, float],
) -> float:
    avg_rep = float(np.mean([reputations.get(i, 1.0) for i in members])) if members else 0.0
    T = compute_cluster_latency_proxy(members, est_delta_bytes, committee_size)
    S = compute_cluster_security_proxy(members, committee_size, avg_rep, last_quality)
    u = S / T
    c = cluster_cost_constraints(members, T)
    return float(u - c)

def marginal_contribution_alg1(
    i: int,
    members: List[int],
    est_delta_bytes: int,
    committee_size: int,
    last_quality: float,
    reputations: Dict[int, float],
) -> float:
    if i not in members:
        v_with = cluster_value_alg1(members + [i], est_delta_bytes, committee_size, last_quality, reputations)
        v_wo = cluster_value_alg1(members, est_delta_bytes, committee_size, last_quality, reputations)
        return float(v_with - v_wo)
    members_wo = [x for x in members if x != i]
    v_with = cluster_value_alg1(members, est_delta_bytes, committee_size, last_quality, reputations)
    v_wo = cluster_value_alg1(members_wo, est_delta_bytes, committee_size, last_quality, reputations)
    return float(v_with - v_wo)

def switch_gain_alg1(
    i: int,
    k_from: int,
    k_to: int,
    clusters: Dict[int, List[int]],
    est_delta_bytes: int,
    committee_size: int,
    last_quality_by_cluster: Dict[int, float],
    reputations: Dict[int, float],
) -> float:
    from_members = clusters[k_from]
    to_members = clusters[k_to]
    if i not in from_members or i in to_members:
        return -1e18

    r_from = marginal_contribution_alg1(
        i, from_members, est_delta_bytes, committee_size,
        last_quality_by_cluster.get(k_from, 0.0), reputations
    )
    r_to = marginal_contribution_alg1(
        i, to_members + [i], est_delta_bytes, committee_size,
        last_quality_by_cluster.get(k_to, 0.0), reputations
    )
    return float(r_to - r_from)

def membership_map(clusters: Dict[int, List[int]]) -> Dict[int, int]:
    m = {}
    for k, ids in clusters.items():
        for cid in ids:
            m[cid] = k
    return m

def neighbor_clusters_for_device(i: int, clusters: Dict[int, List[int]], k_from: int, est_delta_bytes: int, committee_size: int) -> List[int]:
    scores = []
    for k in clusters.keys():
        if k == k_from:
            continue
        members = clusters[k] + [i]
        lat = compute_cluster_latency_proxy(members, est_delta_bytes, committee_size)
        scores.append((lat, k))
    scores.sort(key=lambda x: x[0])
    return [k for _, k in scores[:max(1, NEIGHBOR_TOP_M)]]

def algorithm2_update_preference_list(
    i: int,
    k_from: int,
    clusters: Dict[int, List[int]],
    last_quality_by_cluster: Dict[int, float],
    reputations: Dict[int, float],
    est_delta_bytes: int,
    committee_size: int
) -> List[Tuple[float, int]]:
    neigh = neighbor_clusters_for_device(i, clusters, k_from, est_delta_bytes, committee_size)
    prefs = []
    for k_to in neigh:
        g = switch_gain_alg1(i, k_from, k_to, clusters, est_delta_bytes, committee_size, last_quality_by_cluster, reputations)
        if g > 0.0:
            prefs.append((g, k_to))
    prefs.sort(key=lambda x: x[0], reverse=True)
    return prefs

def algorithm1_distributed_network_optimization(
    clusters_init: Dict[int, List[int]],
    num_clients: int,
    est_delta_bytes: int,
    committee_size: int,
    last_quality_by_cluster: Dict[int, float],
    reputations: Dict[int, float],
    max_slots: int,
) -> Dict[int, List[int]]:
    cluster_state = {k: "available" for k in clusters_init.keys()}
    visited = {i: 0 for i in range(num_clients)}
    prev_gain = {i: -1e18 for i in range(num_clients)}
    clusters = {k: list(v) for k, v in clusters_init.items()}
    last_accept: Dict[int, Tuple[int, int, float]] = {}

    for _t in range(max_slots):
        mem = membership_map(clusters)
        i = min(visited.keys(), key=lambda x: visited[x])
        visited[i] += 1
        k_from = mem[i]

        prefs = algorithm2_update_preference_list(
            i, k_from, clusters, last_quality_by_cluster, reputations, est_delta_bytes, committee_size
        )

        if not prefs:
            stable = True
            mem = membership_map(clusters)
            for ii in range(num_clients):
                kf = mem[ii]
                neigh = neighbor_clusters_for_device(ii, clusters, kf, est_delta_bytes, committee_size)
                for kt in neigh:
                    g = switch_gain_alg1(ii, kf, kt, clusters, est_delta_bytes, committee_size, last_quality_by_cluster, reputations)
                    if g > 0.0:
                        stable = False
                        break
                if not stable:
                    break
            if stable:
                break
            else:
                continue

        proposed_ops = []
        for (g, k_to) in prefs[:1]:
            if g > prev_gain[i] and cluster_state.get(k_to, "available") == "available":
                proposed_ops.append((g, i, k_from, k_to))

        if not proposed_ops:
            continue

        g, i, k_from, k_to = proposed_ops[0]

        if k_to in last_accept:
            old_i, old_from, old_g = last_accept[k_to]
            if g > old_g:
                if old_i in clusters[k_to]:
                    clusters[k_to].remove(old_i)
                clusters[old_from].append(old_i)
                last_accept.pop(k_to, None)

        if i in clusters[k_from]:
            clusters[k_from].remove(i)
        clusters[k_to].append(i)

        cluster_state[k_to] = "occupied"
        prev_gain[i] = g
        last_accept[k_to] = (i, k_from, g)

        empty = [k for k, ids in clusters.items() if len(ids) == 0]
        for k in empty:
            clusters.pop(k, None)
            cluster_state.pop(k, None)
            last_quality_by_cluster.pop(k, None)

        for k in list(cluster_state.keys()):
            cluster_state[k] = "available"

    new_keys = sorted(clusters.keys())
    remap = {old: new for new, old in enumerate(new_keys)}
    clusters2 = {remap[old]: clusters[old] for old in new_keys}
    return clusters2


# ============================================================
# Committee election (Algorithm 6 step 18)
# ============================================================
def elect_committee_member_alg6(cluster_client_ids: List[int], client_sizes: List[int], reputations: Dict[int, float]) -> int:
    if not cluster_client_ids:
        return 0
    sizes = np.array([client_sizes[c] for c in cluster_client_ids], dtype=float)
    smin, smax = float(np.min(sizes)), float(np.max(sizes))
    best = cluster_client_ids[0]
    best_score = -1e18
    for cid in cluster_client_ids:
        rep = float(reputations.get(cid, 1.0))
        sz = float(client_sizes[cid])
        norm_sz = 0.0 if smax <= smin else (sz - smin) / (smax - smin)
        score = COMMITTEE_W_REPUT * rep + COMMITTEE_W_SIZE * norm_sz
        if score > best_score:
            best_score = score
            best = cid
    return best


# ============================================================
# Latency optimizations helpers (NEW)
# ============================================================
def stage2_size_for_round(round_idx_1based: int) -> int:
    """
    (T4) Adaptive verification set size schedule:
      Start large early, decay to a smaller size later.
    """
    r = float(round_idx_1based)
    size = STAGE2_SIZE_MIN + (STAGE2_SIZE_MAX - STAGE2_SIZE_MIN) * math.exp(-STAGE2_DECAY * (r - 1.0))
    return int(max(STAGE2_SIZE_MIN, min(STAGE2_SIZE_MAX, round(size))))

def pick_subset_indices(n: int, k: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if k >= n:
        return np.arange(n)
    return rng.choice(n, size=k, replace=False)

def make_verif_subset(X: np.ndarray, y: np.ndarray, cluster_id: int, size: int, salt: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Purpose:
      Build deterministic-ish subsets per cluster for verification.
      - cluster_id changes which slice you see
      - salt changes per round (avoid overfitting to a fixed subset)
    """
    n = len(X)
    if n <= size:
        return X, y
    # pseudo-deterministic window start
    start = (cluster_id * 997 + salt * 7919) % max(1, n - size)
    return X[start:start+size], y[start:start+size]

def update_ema(old: float, new: float, beta: float) -> float:
    if old is None or not np.isfinite(old):
        return float(new)
    return float(beta * old + (1.0 - beta) * new)

def select_clients_straggler_aware(
    client_ids: List[int],
    k_select: int,
    client_latency_ema: Dict[int, float],
    round_seed: int
) -> List[int]:
    """
    (T1) Straggler-aware selection:
      - pick FAST_SELECT_FRAC from lowest-latency clients
      - pick EXPLORE_FRAC random from remaining (exploration/fairness)
    """
    if k_select >= len(client_ids):
        return list(client_ids)

    rng = np.random.default_rng(round_seed)

    scored = []
    for cid in client_ids:
        lat = float(client_latency_ema.get(cid, 1.0))
        scored.append((lat, cid))
    scored.sort(key=lambda x: x[0])  # small latency first

    n_fast = int(round(k_select * FAST_SELECT_FRAC))
    n_fast = max(0, min(k_select, n_fast))
    n_rand = k_select - n_fast

    fast_pool = [cid for _, cid in scored[:max(1, len(scored))]]
    selected_fast = fast_pool[:n_fast]

    remaining = [cid for cid in client_ids if cid not in selected_fast]
    if n_rand > 0:
        selected_rand = rng.choice(remaining, size=min(n_rand, len(remaining)), replace=False).tolist()
    else:
        selected_rand = []

    out = selected_fast + selected_rand
    rng.shuffle(out)
    return out

def deadline_cap_updates(selected_clients: List[int]) -> List[int]:
    """
    (T2) Deadline-based aggregation approximation:
      You can cap the number of updates processed per cluster per round.
      This acts like a deadline under a parallel world where only the first
      MAX_UPDATES_PER_CLUSTER arrivals are included.
    """
    if MAX_UPDATES_PER_CLUSTER <= 0:
        return []
    if len(selected_clients) <= MAX_UPDATES_PER_CLUSTER:
        return selected_clients
    return selected_clients[:MAX_UPDATES_PER_CLUSTER]


# ============================================================
# MAIN
# ============================================================

wb = WB()
config_dict = {
    # base
    "num_clients": NUM_CLIENTS,
    "num_rounds": NUM_ROUNDS,
    "local_epochs": LOCAL_EPOCHS,
    "batch_size": BATCH_SIZE,
    "lr": LR,
    "weight_decay": WEIGHT_DECAY,
    "hidden_dim": HIDDEN_DIM,
    "num_layers": NUM_LAYERS,
    "dp_clip": L2_NORM_CLIP,
    "dp_noise_multiplier": NOISE_MULTIPLIER,
    "pr_auc_min": PR_AUC_MIN,
    "acc_min": ACC_MIN,
    "use_quant_comm": USE_QUANT_COMM,
    "intercluster_every": INTERCLUSTER_EVERY,
    "stale_q": STALE_Q,
    "chi_window": CHI_WINDOW,
    "optimize_every": OPTIMIZE_EVERY,
    "start_singletons": START_SINGLETONS,
    "min_cluster_size": MIN_CLUSTER_SIZE,
    "max_cluster_size": MAX_CLUSTER_SIZE,
    "seed": RANDOM_STATE,
    "device": str(DEVICE),

    # partial participation
    "participation_rate": PARTICIPATION_RATE,
    "min_participants": MIN_PARTICIPANTS,

    # latency techniques
    "fast_select_frac": FAST_SELECT_FRAC,
    "explore_frac": EXPLORE_FRAC,
    "lat_ema_beta": LAT_EMA_BETA,
    "max_updates_per_cluster": MAX_UPDATES_PER_CLUSTER,
    "stage1_size": STAGE1_SIZE,
    "stage2_max": STAGE2_MAX,
    "stage1_pr_auc_min": STAGE1_PR_AUC_MIN,
    "stage2_size_max": STAGE2_SIZE_MAX,
    "stage2_size_min": STAGE2_SIZE_MIN,
    "stage2_decay": STAGE2_DECAY,

    # tuned f1 reporting
    "thresh_min": THRESH_MIN,
    "thresh_max": THRESH_MAX,
    "thresh_steps": THRESH_STEPS,
}
run = wb.init(
    project=PROJECT,
    group=GROUP,
    name=RUN_NAME,
    config=config_dict,
    **({"entity": ENTITY} if ENTITY else {})
)

# ---- Load dataset ----
print("Downloading dataset via kagglehub...")
dataset_path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
csv_path = os.path.join(dataset_path, "creditcard.csv")

df = pd.read_csv(csv_path).drop_duplicates().fillna(0)
y = df["Class"].values.astype(int)
X_df = df.drop(columns=["Class"])
time_col = X_df["Time"].values.astype(float)

# Split
X_train_df, X_temp_df, y_train, y_temp, t_train, t_temp = train_test_split(
    X_df, y, time_col, test_size=0.30, random_state=RANDOM_STATE, stratify=y
)
X_val_df, X_test_df, y_val, y_test, t_val, t_test = train_test_split(
    X_temp_df, y_temp, t_temp, test_size=0.50, random_state=RANDOM_STATE, stratify=y_temp
)

# Scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_df.values)
X_val   = scaler.transform(X_val_df.values)
X_test  = scaler.transform(X_test_df.values)

# pos_weight
neg = float(np.sum(y_train == 0))
pos = float(np.sum(y_train == 1))
pos_weight = (neg / max(pos, 1.0))

# ---- Client split (time-sorted for drift) ----
order = np.argsort(t_train)
X_train_sorted = X_train[order]
y_train_sorted = y_train[order]

client_data   = np.array_split(X_train_sorted, NUM_CLIENTS)
client_labels = np.array_split(y_train_sorted, NUM_CLIENTS)
client_sizes  = [len(cd) for cd in client_data]

# ---- Per-client secret keys (signature simulation) ----
client_secrets: Dict[int, bytes] = {i: os.urandom(32) for i in range(NUM_CLIENTS)}

# ---- Model factory ----
input_dim = X_train.shape[1]
def new_model() -> nn.Module:
    return LSTMTabular(input_dim=input_dim, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS).to(DEVICE)

# Estimate delta bytes for Alg1 latency proxy
tmpm = new_model()
param_count = sum(p.numel() for p in tmpm.parameters())
EST_DELTA_BYTES = param_count * (1 if USE_QUANT_COMM else 4)

# ---- Ledger + histories ----
ledger = LiteChainLedger()
cluster_history: Dict[int, Deque[Dict[str, torch.Tensor]]] = {}

# ---- Reputation store ----
rep_stats: Dict[int, ClientReputation] = {i: ClientReputation() for i in range(NUM_CLIENTS)}
reputations: Dict[int, float] = {i: 1.0 for i in range(NUM_CLIENTS)}

# ---- Latency EMA per client (NEW: for straggler-aware selection) ----
# Start all clients with the same “expected latency”
client_latency_ema: Dict[int, float] = {i: 1.0 for i in range(NUM_CLIENTS)}

# ---- Initial clusters ----
if START_SINGLETONS:
    clusters = {i: [i] for i in range(NUM_CLIENTS)}
else:
    init_k = int(os.environ.get("INIT_CLUSTERS", "2"))
    init_k = max(1, min(init_k, NUM_CLIENTS))
    clusters = {k: [] for k in range(init_k)}
    chunk = int(math.ceil(NUM_CLIENTS / init_k))
    clients = list(range(NUM_CLIENTS))
    for k in range(init_k):
        clusters[k] = clients[k*chunk:(k+1)*chunk]
        if len(clusters[k]) == 0 and clients:
            clusters[k] = [clients[-1]]

last_quality_by_cluster = {k: 0.0 for k in clusters.keys()}
committee = {k: elect_committee_member_alg6(clusters[k], client_sizes, reputations) for k in clusters}

# cluster models (start from fresh model per cluster)
cluster_models: Dict[int, Dict[str, torch.Tensor]] = {k: clone_state_dict(new_model().state_dict()) for k in clusters}
client_models = [new_model() for _ in range(NUM_CLIENTS)]

for k in clusters.keys():
    cluster_history[k] = deque([clone_state_dict(cluster_models[k])], maxlen=max(1, CHI_WINDOW))


# ============================================================
# Training (Rounds)
# ============================================================
metrics_rows = []

for r in range(NUM_ROUNDS):
    round_start = time.time()
    round_id = r + 1

    # ---------- Algorithm 1/2: dynamic clustering optimization ----------
    if OPTIMIZE_EVERY > 0 and ((r == 0) or ((r + 1) % max(1, OPTIMIZE_EVERY) == 0)):
        clusters = algorithm1_distributed_network_optimization(
            clusters_init=clusters,
            num_clients=NUM_CLIENTS,
            est_delta_bytes=EST_DELTA_BYTES,
            committee_size=len(committee),
            last_quality_by_cluster=last_quality_by_cluster,
            reputations=reputations,
            max_slots=MAX_OPT_SLOTS
        )
        # re-align models/history/committee after cluster changes
        new_cluster_models = {}
        new_history = {}
        new_committee = {}
        for k, members in clusters.items():
            if k in cluster_models:
                new_cluster_models[k] = clone_state_dict(cluster_models[k])
                new_history[k] = cluster_history.get(k, deque([clone_state_dict(cluster_models[k])], maxlen=max(1, CHI_WINDOW)))
            else:
                new_cluster_models[k] = clone_state_dict(new_model().state_dict())
                new_history[k] = deque([clone_state_dict(new_cluster_models[k])], maxlen=max(1, CHI_WINDOW))
            new_committee[k] = elect_committee_member_alg6(members, client_sizes, reputations)

        cluster_models = new_cluster_models
        cluster_history = new_history
        committee = new_committee
        wb.log({"round": round_id, "optimizer/K": int(len(clusters)), "optimizer/clusters": str(clusters)})

    # ---------- Algorithm 6 step 18: committee re-election ----------
    if (round_id) % max(1, REELECT_EVERY) == 0:
        committee = {k: elect_committee_member_alg6(clusters[k], client_sizes, reputations) for k in clusters}
        wb.log({"round": round_id, "committee": str(committee)})

    # ---- comm/time tracking ----
    comm_client_to_committee = 0
    comm_committee_to_committee = 0
    comm_chain_bytes = 0

    t_local_train = 0.0
    t_cluster_verify = 0.0
    t_chain_consensus = 0.0
    t_intercluster = 0.0

    proposed_blocks: List[Tuple[int, Dict[str, Any], str, Dict[str, torch.Tensor]]] = []

    # Adaptive stage2 verification size for this round (T4)
    stage2_size = stage2_size_for_round(round_id)

    # ============================================================
    # Algorithm 3: Intra-cluster training (partial participation + latency optimizations)
    # ============================================================
    for k, cluster_client_ids in clusters.items():
        base_sd = clone_state_dict(cluster_models[k])
        committee_id = committee[k]

        # --- PARTIAL PARTICIPATION: decide how many clients participate this round ---
        k_select = int(math.ceil(len(cluster_client_ids) * PARTICIPATION_RATE))
        k_select = max(MIN_PARTICIPANTS, min(len(cluster_client_ids), k_select))

        # (T1) straggler-aware selection (fast clients prioritized)
        selected_clients = select_clients_straggler_aware(
            client_ids=cluster_client_ids,
            k_select=k_select,
            client_latency_ema=client_latency_ema,
            round_seed=RANDOM_STATE + 1000 * round_id + 17 * k
        )

        # (T2) deadline-like cap (process only first N updates)
        selected_clients = deadline_cap_updates(selected_clients)

        if len(selected_clients) == 0:
            continue

        # --- Verification subsets ---
        # Two-stage verification:
        #   Stage 1: cheap approximate check on small subset
        #   Stage 2: expensive check on larger subset, only for top candidates
        Xc_s1, yc_s1 = make_verif_subset(X_val, y_val, cluster_id=k, size=STAGE1_SIZE, salt=round_id)
        Xc_s2, yc_s2 = make_verif_subset(X_val, y_val, cluster_id=k, size=stage2_size, salt=round_id)

        # Collect candidate info
        client_updates = []
        client_norms = []
        client_weights = []
        client_sig = []
        client_stage1 = []   # stage1 metrics
        client_stage2 = [None] * len(selected_clients)  # stage2 metrics or None

        # ----- Train each selected device -----
        for idx, cid in enumerate(selected_clients):
            # load cluster base
            client_models[cid].load_state_dict(base_sd)

            # local training
            m, dt = local_train_private(
                model=client_models[cid],
                client_x=client_data[cid],
                client_y=client_labels[cid],
                pos_weight=pos_weight,
                epochs=LOCAL_EPOCHS,
                lr=LR,
                weight_decay=WEIGHT_DECAY,
                batch_size=BATCH_SIZE,
                l2_norm_clip=L2_NORM_CLIP,
                noise_multiplier=NOISE_MULTIPLIER,
                device=DEVICE
            )
            t_local_train += dt

            # delta
            new_sd = clone_state_dict(m.state_dict())
            d = delta_state_dict(new_sd, base_sd)

            # comm + decompress
            if USE_QUANT_COMM:
                qd, scales = quantize_delta_int8(d)
                comm_client_to_committee += estimate_comm_bytes_for_quant_delta(qd, scales)
                d_recv = dequantize_delta_int8(qd, scales, DEVICE)
            else:
                comm_client_to_committee += estimate_comm_bytes_for_delta(d)
                d_recv = d

            # candidate state = base + received delta
            cand_sd = apply_delta(base_sd, d_recv)
            cand_hash = hash_state_dict(cand_sd)
            sig = sign_hash(client_secrets[cid], cand_hash)

            # Stage 1 verify (cheap) — PR-AUC/ACC on small subset
            t0 = time.time()
            tmp_model = new_model()
            tmp_model.load_state_dict(cand_sd)
            p1 = predict_proba(tmp_model, Xc_s1)
            met1 = compute_metrics(yc_s1, p1, threshold=GLOBAL_DEFAULT_THRESHOLD)
            t_cluster_verify += (time.time() - t0)

            # update stats for Alg6
            norm = update_l2_norm(d_recv)
            rep_stats[cid].participated += 1
            rep_stats[cid].pr_auc_sum += float(met1["pr_auc"] or 0.0)
            rep_stats[cid].acc_sum += float(met1["acc"])
            rep_stats[cid].norm_sum += float(norm)

            # update latency EMA (NEW: straggler-aware)
            # We combine local train + stage1 verify as "observed latency"
            observed_lat = float(dt) + float(0.0)  # stage1 verify time already in t_cluster_verify; we approximate as negligible per client
            client_latency_ema[cid] = update_ema(client_latency_ema.get(cid, 1.0), observed_lat, LAT_EMA_BETA)

            client_updates.append(d_recv)
            client_norms.append(norm)
            client_weights.append(client_sizes[cid])
            client_sig.append((cand_hash, sig))
            client_stage1.append(met1)

        # --- Outlier cutoff (MAD on norms) ---
        norms = np.asarray(client_norms, dtype=float)
        med = float(np.median(norms))
        mad = float(np.median(np.abs(norms - med))) + 1e-12
        cutoff = med + NORM_MAD_Z * mad

        # --- Two-stage verification decision ---
        # Stage 1 pre-filter: signature + (very loose) PR-AUC + ACC + norm cutoff
        stage1_pass = []
        for i_idx, cid in enumerate(selected_clients):
            cand_hash, sig = client_sig[i_idx]
            sig_ok = verify_signature(client_secrets[cid], cand_hash, sig)

            pr1 = client_stage1[i_idx].get("pr_auc")
            pr1 = -1.0 if pr1 is None else float(pr1)
            ac1 = float(client_stage1[i_idx]["acc"])

            if sig_ok and (pr1 >= STAGE1_PR_AUC_MIN) and (ac1 >= 0.5) and (client_norms[i_idx] <= cutoff):
                stage1_pass.append(i_idx)

        # Stage 2: run expensive verification only for top candidates by stage1 PR-AUC
        # This is the key latency reducer (T3).
        if len(stage1_pass) > 0:
            ranked = sorted(
                stage1_pass,
                key=lambda i: float(client_stage1[i].get("pr_auc") or -1.0),
                reverse=True
            )
            stage2_idx = ranked[:max(1, min(STAGE2_MAX, len(ranked)))]
        else:
            stage2_idx = []

        for i_idx in stage2_idx:
            cid = selected_clients[i_idx]
            # build candidate state again
            cand_sd = apply_delta(base_sd, client_updates[i_idx])

            t0 = time.time()
            tmp_model = new_model()
            tmp_model.load_state_dict(cand_sd)
            p2 = predict_proba(tmp_model, Xc_s2)
            met2 = compute_metrics(yc_s2, p2, threshold=GLOBAL_DEFAULT_THRESHOLD)
            t_cluster_verify += (time.time() - t0)

            client_stage2[i_idx] = met2

        # Final accept rule for Alg3:
        # Use stage2 metrics when available; else stage1 metrics.
        accepted_idx = []
        accepted_client_ids = []

        for i_idx, cid in enumerate(selected_clients):
            met = client_stage2[i_idx] if client_stage2[i_idx] is not None else client_stage1[i_idx]
            pr = met.get("pr_auc")
            pr_val = -1.0 if pr is None else float(pr)
            acc_val = float(met["acc"])

            cand_hash, sig = client_sig[i_idx]
            sig_ok = verify_signature(client_secrets[cid], cand_hash, sig)

            # Final thresholds (paper-aligned style): PR-AUC + ACC + norm + signature
            if sig_ok and (pr_val >= PR_AUC_MIN) and (acc_val >= ACC_MIN) and (client_norms[i_idx] <= cutoff):
                accepted_idx.append(i_idx)
                accepted_client_ids.append(cid)
                rep_stats[cid].accepted += 1
            else:
                rep_stats[cid].rejected += 1

        # fallback: if none accepted, accept best stage2 (or stage1) PR-AUC with valid signature
        if len(accepted_idx) == 0:
            best_i = None
            best_pr = -1e18
            for i_idx, cid in enumerate(selected_clients):
                met = client_stage2[i_idx] if client_stage2[i_idx] is not None else client_stage1[i_idx]
                pr_val = float(met.get("pr_auc") or -1.0)
                cand_hash, sig = client_sig[i_idx]
                if verify_signature(client_secrets[cid], cand_hash, sig) and pr_val > best_pr:
                    best_pr = pr_val
                    best_i = i_idx
            if best_i is None:
                best_i = 0
            accepted_idx = [best_i]
            accepted_client_ids = [selected_clients[best_i]]
            rep_stats[selected_clients[best_i]].accepted += 1

        # Update reputations (Alg6)
        for cid in selected_clients:
            reputations[cid] = reputation_score(rep_stats[cid])

        accept_rate = len(accepted_idx) / max(1, len(selected_clients))

        # ----- Aggregate verified deltas (Alg3 FedAvg, weighted by size * reputation) -----
        acc_deltas = [client_updates[i] for i in accepted_idx]
        acc_wts = [client_weights[i] * max(1e-3, reputations[selected_clients[i]]) for i in accepted_idx]
        agg_delta = weighted_average_deltas(acc_deltas, acc_wts)

        candidate_cluster_sd = apply_delta(base_sd, agg_delta)

        # Evaluate candidate on full validation set for block metrics + Alg1 quality
        tmp_model = new_model()
        tmp_model.load_state_dict(candidate_cluster_sd)
        val_probs_cluster = predict_proba(tmp_model, X_val)
        val_m_cluster = compute_metrics(y_val, val_probs_cluster, threshold=GLOBAL_DEFAULT_THRESHOLD)
        last_quality_by_cluster[k] = float(val_m_cluster["pr_auc"] or 0.0)

        # ----- Algorithm 6 window aggregation (χ) -----
        if k not in cluster_history:
            cluster_history[k] = deque([], maxlen=max(1, CHI_WINDOW))
        cluster_history[k].append(clone_state_dict(candidate_cluster_sd))

        hist_list = list(cluster_history[k])
        recency_wts = [(idx + 1) for idx in range(len(hist_list))]
        windowed_sd = weighted_average_state_dict(hist_list, recency_wts)

        # ----- Create block payload (Alg3) -----
        model_hash = hash_state_dict(windowed_sd)
        payload, blk_hash = make_block_payload(
            height=len(ledger.blocks) + len(proposed_blocks) + 1,
            round_id=round_id,
            cluster_id=k,
            proposer=committee_id,
            model_hash=model_hash,
            metrics={
                "pr_auc": float(val_m_cluster["pr_auc"] or -1.0),
                "roc_auc": float(val_m_cluster["roc_auc"] or -1.0),
                "acc": float(val_m_cluster["acc"]),
                "f1": float(val_m_cluster["f1"]),
                "precision": float(val_m_cluster["precision"]),
                "recall": float(val_m_cluster["recall"]),
            },
            accepted_client_ids=accepted_client_ids,
            prev_hash=ledger.last_hash(),
            timestamp=time.time()
        )

        proposed_blocks.append((k, payload, blk_hash, windowed_sd))

        # committee-to-committee broadcast estimate
        payload_bytes = estimate_chain_record_bytes(payload)
        k_comm = len(committee)
        comm_committee_to_committee += (payload_bytes + 256) * max(0, (k_comm - 1))

        wb.log({
            "round": round_id,
            f"cluster/{k}/selected_clients": int(len(selected_clients)),
            f"cluster/{k}/accepted_clients": int(len(accepted_client_ids)),
            f"cluster/{k}/accept_rate": float(accept_rate),
            f"cluster/{k}/norm_median": float(med),
            f"cluster/{k}/norm_cutoff": float(cutoff),
            f"cluster/{k}/val_pr_auc_default": float(val_m_cluster["pr_auc"] or -1.0),
            f"cluster/{k}/val_acc_default": float(val_m_cluster["acc"]),
            f"cluster/{k}/stage2_size": int(stage2_size),
            f"cluster/{k}/stage2_used": int(len(stage2_idx)),
        })

    # ============================================================
    # Algorithm 4: CBFT consensus
    # ============================================================
    committee_members = list(set(committee.values()))
    committed_this_round = 0

    t0 = time.time()
    for (k, payload, blk_hash, new_cluster_sd) in proposed_blocks:
        ok, info = cbft_consensus_algorithm4(
            ledger=ledger,
            payload=payload,
            block_hash=blk_hash,
            committee_members=committee_members,
            pr_auc_min=PR_AUC_MIN,
            acc_min=ACC_MIN,
            byzantine_frac=BYZANTINE_FRAC
        )
        comm_chain_bytes += estimate_chain_record_bytes(payload)

        if ok:
            cluster_models[k] = clone_state_dict(new_cluster_sd)
            committed_this_round += 1

        wb.log({
            "round": round_id,
            f"chain/{k}/committed": int(ok),
            f"chain/{k}/verify_msgs": int(info.get("verify_msgs", 0)),
            f"chain/{k}/commit_msgs": int(info.get("commit_msgs", 0)),
            f"chain/{k}/quorum": int(info.get("quorum", cbft_quorum(len(committee_members)))),
        })
    t_chain_consensus += (time.time() - t0)

    # ============================================================
    # Algorithm 5: Inter-cluster staleness-aware aggregation
    # ============================================================
    if (round_id) % max(1, INTERCLUSTER_EVERY) == 0:
        t0 = time.time()
        latest_round = {}
        latest_state = {}
        for k in clusters.keys():
            blk = ledger.latest_block_for_cluster(k)
            latest_round[k] = 0 if blk is None else int(blk.round_id)
            latest_state[k] = clone_state_dict(cluster_models[k])

        mixed = {}
        for k in clusters.keys():
            sds, wts = [], []
            for j in clusters.keys():
                w = staleness_weight_alg5(current_round=round_id, model_round=latest_round[j], q=STALE_Q)
                sds.append(latest_state[j])
                wts.append(w)
            mixed[k] = weighted_average_state_dict(sds, wts)

        for k in clusters.keys():
            cluster_models[k] = clone_state_dict(mixed[k])

        t_intercluster += (time.time() - t0)

    # ============================================================
    # Global evaluation (aligned with baseline/enhanced reporting)
    # ============================================================
    cluster_sizes = []
    cluster_sds = []
    for k, client_ids in clusters.items():
        cluster_sizes.append(sum(client_sizes[cid] for cid in client_ids))
        cluster_sds.append(cluster_models[k])

    global_sd = weighted_average_state_dict(cluster_sds, cluster_sizes)
    global_model = new_model()
    global_model.load_state_dict(global_sd)

    val_probs = predict_proba(global_model, X_val)
    test_probs = predict_proba(global_model, X_test)

    # tuned threshold each round
    best_t, best_f1_val = tune_threshold_for_f1(y_val, val_probs, THRESH_MIN, THRESH_MAX, THRESH_STEPS)
    val_m_tuned = compute_metrics(y_val, val_probs, threshold=best_t)
    test_m_tuned = compute_metrics(y_test, test_probs, threshold=best_t)

    # PR-AUC values
    val_pr_auc = float(val_m_tuned["pr_auc"] or -1.0)
    test_pr_auc = float(test_m_tuned["pr_auc"] or -1.0)

    # other metrics (for W&B)
    test_roc = float(test_m_tuned["roc_auc"] or -1.0)
    test_acc = float(test_m_tuned["acc"])
    val_acc = float(val_m_tuned["acc"])

    round_time = time.time() - round_start

    total_comm_mb = (comm_client_to_committee + comm_committee_to_committee + comm_chain_bytes) / (1024**2)

    wb.log({
        "round": round_id,
        "val/pr_auc": float(val_pr_auc),
        "val/f1_tuned": float(val_m_tuned["f1"]),
        "val/acc_tuned": float(val_acc),
        "test/pr_auc": float(test_pr_auc),
        "test/f1_tuned": float(test_m_tuned["f1"]),
        "test/acc_tuned": float(test_acc),
        "test/roc_auc": float(test_roc),
        "best_threshold/value": float(best_t),

        "litechain/blocks_total": int(len(ledger.blocks)),
        "litechain/committed_blocks_this_round": int(committed_this_round),
        "litechain/K": int(len(clusters)),

        "comm/client_to_committee_MB": float(comm_client_to_committee / (1024**2)),
        "comm/committee_to_committee_MB": float(comm_committee_to_committee / (1024**2)),
        "comm/chain_MB": float(comm_chain_bytes / (1024**2)),
        "comm/total_MB": float(total_comm_mb),

        "time/e2e_round_sec": float(round_time),
        "time/local_train_sec": float(t_local_train),
        "time/cluster_verify_sec": float(t_cluster_verify),
        "time/chain_consensus_sec": float(t_chain_consensus),
        "time/intercluster_sec": float(t_intercluster),
    })

    # Save per-round row for CSV
    metrics_rows.append({
        "round": round_id,
        "val_pr_auc": val_pr_auc,
        "val_f1_tuned": float(val_m_tuned["f1"]),
        "test_pr_auc": test_pr_auc,
        "test_f1_tuned": float(test_m_tuned["f1"]),
        "blocks": int(len(ledger.blocks)),
        "K": int(len(clusters)),
        "comm_total_mb": float(total_comm_mb),
        "e2e_sec": float(round_time),
        "best_threshold": float(best_t),
        "test_roc_auc": float(test_roc),
        "test_acc": float(test_acc),
    })

    print(
        f"[Round {round_id:02d}] "
        f"Val PR-AUC={val_pr_auc:.4f} F1(tuned)={val_m_tuned['f1']:.4f} | "
        f"Test PR-AUC={test_pr_auc:.4f} F1(tuned)={test_m_tuned['f1']:.4f} | "
        f"Blocks={len(ledger.blocks)} K={len(clusters)} "
        f"Comm={total_comm_mb:.3f}MB E2E={round_time:.3f}s"
    )

# ============================================================
# Save metrics CSV
# ============================================================
out_csv = os.environ.get("OUT_CSV", "litechain_metrics_full.csv")
pd.DataFrame(metrics_rows).to_csv(out_csv, index=False)
print(f"Saved {out_csv}")

# ============================================================
# Final summary (use last round’s tuned threshold)
# ============================================================
final_probs_val = predict_proba(global_model, X_val)
final_probs_test = predict_proba(global_model, X_test)
final_best_t, final_best_f1_val = tune_threshold_for_f1(y_val, final_probs_val, THRESH_MIN, THRESH_MAX, THRESH_STEPS)

final_test_loss = evaluate_loss(global_model, X_test, y_test, pos_weight=pos_weight)
final_test_m = compute_metrics(y_test, final_probs_test, threshold=final_best_t)

wb.summary_set("best_threshold/value", float(final_best_t))
wb.summary_set("best_threshold/val_f1", float(final_best_f1_val))
wb.summary_set("final/test_loss", float(final_test_loss))
wb.summary_set("final/test_f1", float(final_test_m["f1"]))
wb.summary_set("final/test_acc", float(final_test_m["acc"]))
wb.summary_set("final/test_roc_auc", float(final_test_m["roc_auc"] or -1.0))
wb.summary_set("final/test_pr_auc", float(final_test_m["pr_auc"] or -1.0))
wb.summary_set("final/litechain_blocks_total", int(len(ledger.blocks)))
wb.summary_set("final/K", int(len(clusters)))

# Optional W&B plots
if wb.enabled and wb.run is not None:
    wb.log({
        "confusion_matrix": wandb.plot.confusion_matrix(
            y_true=y_test.astype(int).tolist(),
            preds=(final_probs_test >= final_best_t).astype(int).tolist(),
            class_names=["legit", "fraud"]
        )
    })
    probas_2col = np.stack([1.0 - final_probs_test, final_probs_test], axis=1)
    try:
        wb.log({"roc_test": wandb.plot.roc_curve(y_test.astype(int), probas_2col, labels=["legit", "fraud"])})
    except Exception as e:
        print("ROC plot skipped:", e)
    try:
        wb.log({"pr_test": wandb.plot.pr_curve(y_test.astype(int), probas_2col, labels=["legit", "fraud"])})
    except Exception as e:
        print("PR plot skipped:", e)

print("\n=== FINAL TEST METRICS (tuned threshold) ===")
print(f"Best threshold (VAL): {final_best_t:.3f} | VAL F1: {final_best_f1_val:.4f}")
print(f"Test Loss : {final_test_loss:.4f}")
print(f"ACC       : {final_test_m['acc']:.4f}")
print(f"F1        : {final_test_m['f1']:.4f}")
print(f"ROC-AUC   : {final_test_m['roc_auc']}")
print(f"PR-AUC    : {final_test_m['pr_auc']}")

wb.finish()
print("\nDone.")
