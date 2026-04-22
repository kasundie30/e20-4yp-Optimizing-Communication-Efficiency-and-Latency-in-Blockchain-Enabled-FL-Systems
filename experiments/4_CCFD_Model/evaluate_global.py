"""
evaluate_global.py
------------------
Evaluates the final global FL model and reports ALL 7 comparison metrics:

  Classification metrics (on combined data from all branches):
    1. F1 Score
    2. PR-AUC   (Precision-Recall Area Under Curve)
    3. ROC-AUC  (Receiver Operating Characteristic AUC)
    4. Precision
    5. Recall

  FL-system metrics (derived from round artefacts):
    6. Comm / round  — Communication Overhead per Round (MB)
    7. E2E / round   — End-to-End Latency per Round (seconds)

Usage:
    python3 evaluate_global.py
    python3 evaluate_global.py --model_path shared/round_0005/global_model.pt
    python3 evaluate_global.py --rounds 5 --shared_root shared --threshold 0.5
"""

import argparse
import os
import warnings
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score

from FL_model import LSTMTabular
from dataset import load_bank_dataset
from src.validation.metrics import fraud_metrics
from src.clustering.topology_loader import load_topology
from src.clustering.ids import local_model_filename, global_model_filename


# ── Helpers ────────────────────────────────────────────────────────────────────

def file_size_mb(path: str) -> float:
    """Return file size in megabytes, or 0 if not found."""
    try:
        return os.path.getsize(path) / (1024 ** 2)
    except FileNotFoundError:
        return 0.0


def compute_comm_overhead(shared_root: str, num_rounds: int, all_branch_ids: list) -> dict:
    """
    Communication Overhead per Round (MB).

    In each FL round:
      - UPLOAD:   each branch sends its local model to the aggregator
                  = sum of all branch_local_model.pt file sizes
      - DOWNLOAD: the server broadcasts the new global model back to all branches
                  = global_model.pt size × number of branches

    Returns per-round breakdown and average.
    """
    upload_per_round = []
    download_per_round = []
    total_per_round = []

    for r in range(1, num_rounds + 1):
        round_dir = os.path.join(shared_root, f"round_{r:04d}")

        # Upload: all branch local models uploaded to aggregator
        upload_mb = sum(
            file_size_mb(os.path.join(round_dir, local_model_filename(bid)))
            for bid in all_branch_ids
        )

        # Download: global model sent back to every branch
        global_mb = file_size_mb(os.path.join(round_dir, global_model_filename()))
        download_mb = global_mb * len(all_branch_ids)

        upload_per_round.append(upload_mb)
        download_per_round.append(download_mb)
        total_per_round.append(upload_mb + download_mb)

    return {
        "upload_mb_per_round":   upload_per_round,
        "download_mb_per_round": download_per_round,
        "total_mb_per_round":    total_per_round,
        "avg_upload_mb":         float(np.mean(upload_per_round)),
        "avg_download_mb":       float(np.mean(download_per_round)),
        "avg_total_comm_mb":     float(np.mean(total_per_round)),
    }


def compute_e2e_latency(shared_root: str, num_rounds: int) -> dict:
    """
    End-to-End Latency per Round (seconds).

    Measured as the elapsed time between:
      - Round start : modification time of the INPUT global_model.pt placed into the
                      round folder at the start (same file that branch containers load)
      - Round end   : modification time of the OUTPUT global_model.pt written by
                      global_aggregate.py at the end of the round

    Because global_aggregate OVERWRITES the same filename, we use the mtime of the
    first branch local model as the "start" proxy (it appears right after docker up),
    and the mtime of global_model.pt as the "end".
    """
    latencies = []

    for r in range(1, num_rounds + 1):
        round_dir = os.path.join(shared_root, f"round_{r:04d}")
        global_pt = os.path.join(round_dir, global_model_filename())

        # Collect mtime of all branch local model files — the earliest is round start
        branch_mtimes = []
        for fname in os.listdir(round_dir):
            if fname.endswith("_local_model.pt"):
                branch_mtimes.append(os.path.getmtime(os.path.join(round_dir, fname)))

        if not branch_mtimes or not os.path.exists(global_pt):
            latencies.append(None)
            continue

        round_start = min(branch_mtimes)   # first branch finished writing
        round_end   = os.path.getmtime(global_pt)  # global model written
        latencies.append(round_end - round_start)

    valid = [x for x in latencies if x is not None and x > 0]
    return {
        "latency_sec_per_round": latencies,
        "avg_e2e_sec":           float(np.mean(valid)) if valid else float("nan"),
        "min_e2e_sec":           float(np.min(valid))  if valid else float("nan"),
        "max_e2e_sec":           float(np.max(valid))  if valid else float("nan"),
    }


# ── Main evaluation ────────────────────────────────────────────────────────────

def evaluate_global_model(
    model_path: str,
    topology_path: str,
    data_root: str,
    shared_root: str,
    num_rounds: int,
    hidden_dim: int = 30,
    num_layers: int = 1,
    batch_size: int = 512,
    threshold: float = 0.5,
    device: str = "cpu",
):
    topo = load_topology(topology_path)
    all_brand_ids   = list(topo.brand_to_branches.keys())
    all_branch_ids  = [b for branches in topo.brand_to_branches.values() for b in branches]

    print(f"\n{'='*65}")
    print(f"  FL GLOBAL MODEL — FULL METRICS EVALUATION")
    print(f"{'='*65}")
    print(f"  Model      : {model_path}")
    print(f"  Branches   : {len(all_branch_ids)}  ({', '.join(all_branch_ids)})")
    print(f"  Data root  : {data_root}")
    print(f"  Threshold  : {threshold}")
    print(f"{'='*65}\n")

    # ── 1. Load all branch data ────────────────────────────────────────────────
    all_X, all_y = [], []
    skipped = []

    for branch_id in all_branch_ids:
        try:
            ds = load_bank_dataset(branch_id, data_path=data_root)
            X, y = ds.tensors[0], ds.tensors[1].view(-1)
            all_X.append(X)
            all_y.append(y)
            fraud  = int(y.sum().item())
            normal = len(y) - fraud
            print(f"  ✔ {branch_id:25s}  {len(X):>7,} samples  "
                  f"(fraud={fraud:>6,} / normal={normal:>7,})")
        except FileNotFoundError as e:
            print(f"  ✘ {branch_id:25s}  SKIPPED — {e}")
            skipped.append(branch_id)

    if not all_X:
        raise RuntimeError("No branch data could be loaded.")

    X_all = torch.cat(all_X, dim=0)
    y_all = torch.cat(all_y, dim=0)

    total        = len(X_all)
    total_fraud  = int(y_all.sum().item())
    total_normal = total - total_fraud

    print(f"\n  Combined : {total:,} samples  "
          f"(fraud={total_fraud:,} [{100*total_fraud/total:.3f}%] / "
          f"normal={total_normal:,})")

    # ── 2. Load model & run inference ─────────────────────────────────────────
    input_dim = X_all.shape[1]
    model = LSTMTabular(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    loader = DataLoader(TensorDataset(X_all, y_all), batch_size=batch_size, shuffle=False)
    probs_list, trues_list = [], []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.unsqueeze(1).to(device)
            logits = model(xb).view(-1)
            p = torch.sigmoid(logits).cpu().numpy()
            probs_list.append(p)
            trues_list.append(yb.cpu().numpy())

    y_prob = np.concatenate(probs_list)
    y_true = np.concatenate(trues_list).astype(int)

    # ── 3. Classification metrics ──────────────────────────────────────────────
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cls_metrics = fraud_metrics(y_true, y_prob, thr=threshold)
        roc_auc     = float(roc_auc_score(y_true, y_prob))

    # ── 4. FL system metrics ───────────────────────────────────────────────────
    comm = compute_comm_overhead(shared_root, num_rounds, all_branch_ids)
    e2e  = compute_e2e_latency(shared_root, num_rounds)

    # ── 5. Per-branch breakdown ────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  PER-BRANCH CLASSIFICATION BREAKDOWN")
    print(f"{'='*65}")
    print(f"  {'Branch':<25}  {'PR-AUC':>7}  {'ROC-AUC':>7}  {'P':>6}  {'R':>6}  {'F1':>6}")
    print(f"  {'-'*25}  {'-'*7}  {'-'*7}  {'-'*6}  {'-'*6}  {'-'*6}")

    offset = 0
    branch_data_sizes = [len(x) for x in all_X]
    for i, branch_id in enumerate(all_branch_ids):
        if branch_id in skipped:
            continue
        n   = branch_data_sizes[i]
        yp  = y_prob[offset:offset+n]
        yt  = y_true[offset:offset+n]
        offset += n
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            bm  = fraud_metrics(yt, yp, thr=threshold)
            try:
                broc = float(roc_auc_score(yt, yp))
            except ValueError:
                broc = float("nan")
        print(f"  {branch_id:<25}  {bm['prauc']:>7.4f}  {broc:>7.4f}  "
              f"{bm['precision']:>6.3f}  {bm['recall']:>6.3f}  {bm['f1']:>6.3f}")

    # ── 6. Per-round FL metrics ────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  PER-ROUND FL SYSTEM METRICS")
    print(f"{'='*65}")
    print(f"  {'Round':<8}  {'Upload (MB)':>11}  {'Download (MB)':>13}  "
          f"{'Total Comm (MB)':>15}  {'E2E (sec)':>9}")
    print(f"  {'-'*8}  {'-'*11}  {'-'*13}  {'-'*15}  {'-'*9}")

    latencies = e2e["latency_sec_per_round"]
    for r in range(num_rounds):
        lat_str = f"{latencies[r]:.1f}" if latencies[r] is not None else "N/A"
        print(f"  {r+1:<8}  {comm['upload_mb_per_round'][r]:>11.4f}  "
              f"{comm['download_mb_per_round'][r]:>13.4f}  "
              f"{comm['total_mb_per_round'][r]:>15.4f}  {lat_str:>9}")

    # ── 7. Final summary ───────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  FINAL MODEL COMPARISON METRICS  (Round {num_rounds})")
    print(f"{'='*65}")
    print(f"\n  ── Classification ──────────────────────────────────────")
    print(f"  F1 Score    : {cls_metrics['f1']:.6f}")
    print(f"  PR-AUC      : {cls_metrics['prauc']:.6f}")
    print(f"  ROC-AUC     : {roc_auc:.6f}")
    print(f"  Precision   : {cls_metrics['precision']:.6f}")
    print(f"  Recall      : {cls_metrics['recall']:.6f}")
    print(f"\n  ── FL System ───────────────────────────────────────────")
    print(f"  Comm/round  : {comm['avg_total_comm_mb']:.4f} MB "
          f"(↑ upload {comm['avg_upload_mb']:.4f} MB  "
          f"↓ download {comm['avg_download_mb']:.4f} MB)")
    print(f"  E2E/round   : {e2e['avg_e2e_sec']:.2f} sec "
          f"(min={e2e['min_e2e_sec']:.1f}s  max={e2e['max_e2e_sec']:.1f}s)")
    print(f"\n{'='*65}\n")

    return {
        "f1":            cls_metrics["f1"],
        "prauc":         cls_metrics["prauc"],
        "rocauc":        roc_auc,
        "precision":     cls_metrics["precision"],
        "recall":        cls_metrics["recall"],
        "comm_mb_round": comm["avg_total_comm_mb"],
        "e2e_sec_round": e2e["avg_e2e_sec"],
    }


def main():
    ap = argparse.ArgumentParser(description="Evaluate final global FL model — 7 metrics")
    ap.add_argument("--model_path",  default="shared/round_0005/global_model.pt",
                    help="Path to the final global_model.pt")
    ap.add_argument("--topology",    default="config/topology.yaml")
    ap.add_argument("--data_root",   default="data/processed/3_local_silo_balancing")
    ap.add_argument("--shared_root", default="shared",
                    help="Root directory containing round_XXXX folders")
    ap.add_argument("--rounds",      type=int, default=5,
                    help="Number of FL rounds that were run")
    ap.add_argument("--threshold",   type=float, default=0.5,
                    help="Decision threshold for P/R/F1 (default: 0.5)")
    ap.add_argument("--device",      default="cpu")
    args = ap.parse_args()

    evaluate_global_model(
        model_path=args.model_path,
        topology_path=args.topology,
        data_root=args.data_root,
        shared_root=args.shared_root,
        num_rounds=args.rounds,
        threshold=args.threshold,
        device=args.device,
    )


if __name__ == "__main__":
    main()
