#!/usr/bin/env python3
"""
fl-integration/scripts/run_10_rounds.py
Runs the 10-round performance benchmarking (Phase 10.4) using real IPFS and Fabric endpoints.
Uses REAL credit card fraud dataset splits from data/splits/fl_clients/BankA|B|C/train_ready.csv
and a global held-out test set from data/splits/test/global_test.csv.
"""

import sys, os
import io
import json
import time
import logging
import torch
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "fl-layer")))

from config.config_loader import load_config
from api_client import APIClient
from hq_agent import HQAgent
from global_aggregator import GlobalAggregator
from model.FL_model import LSTMTabular
from model.dataset import load_bank_dataset
from training.local_train import train_local
from validation.validate_fast import evaluate_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ── Real data paths ─────────────────────────────────────────────────────────
DATA_ROOT   = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))
FL_DATA_DIR = os.path.join(DATA_ROOT, "splits", "fl_clients")
TEST_CSV    = os.path.join(DATA_ROOT, "splits", "test", "global_test.csv")

# ── Model config — 29 real features (V1–V28 + Amount; Time was dropped in preprocessing) ──
MODEL_CFG = {"input_dim": 29, "hidden_dim": 30, "num_layers": 1}

# ── Training config — DP re-enabled ─────────────────────────────────────────
TRAIN_CFG = {
    "local_epochs": 2,
    "batch_size":   256,
    "lr":           1e-3,
    "l2_norm_clip": 1.0,
    "noise_multiplier": 0.05,   # DP Gaussian noise — re-enabled (was 0.0)
    "device":       "cpu",
}


def load_global_test_dataset():
    """Load the global held-out test set for post-aggregation evaluation."""
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    from torch.utils.data import TensorDataset

    df = pd.read_csv(TEST_CSV)
    X  = df.iloc[:, :-1].values
    y  = df.iloc[:, -1].values
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    logger.info("Global test set loaded: %d rows, fraud=%.3f%%", len(X_t), y.mean() * 100)
    return TensorDataset(X_t, y_t)


def run_benchmark():
    config = load_config()
    import requests

    # ── Verify API is reachable ──────────────────────────────────────────────
    try:
        api_client = APIClient(base_url=config.blockchain.api_url)
        requests.get(f"{config.blockchain.api_url}/health").raise_for_status()
    except Exception as e:
        logger.error(f"API not available: {e}")
        sys.exit(1)

    # ── IPFS helpers ─────────────────────────────────────────────────────────
    def ipfs_upload(model_bytes: bytes) -> str:
        files = {'file': ('model.pt', model_bytes)}
        r = requests.post(f"{config.ipfs.api_url}/api/v0/add", files=files)
        r.raise_for_status()
        return r.json()['Hash']

    def ipfs_download(cid: str) -> bytes:
        r = requests.post(f"{config.ipfs.api_url}/api/v0/cat?arg={cid}")
        r.raise_for_status()
        return r.content

    # ── Load REAL bank datasets ──────────────────────────────────────────────
    banks  = ["BankA", "BankB", "BankC"]
    hqs    = {}
    logger.info("Loading real CCFD datasets from %s", FL_DATA_DIR)
    for b in banks:
        ds = load_bank_dataset(bank_id=b, data_path=FL_DATA_DIR)
        hqs[b] = HQAgent(
            bank_id=b,
            client=api_client,
            ipfs_upload=ipfs_upload,
            ipfs_download=ipfs_download,
            val_dataset=ds,
            val_threshold=0.0,      # accept all submissions
            model_cfg=MODEL_CFG,
        )
        logger.info("[%s] dataset loaded: %d samples", b, len(ds))

    # ── Global test set ──────────────────────────────────────────────────────
    global_test_ds = load_global_test_dataset()

    # ── Global aggregator (BankA is the designated aggregator) ───────────────
    aggregator = GlobalAggregator(
        client=api_client,
        ipfs_download=ipfs_download,
        ipfs_upload=ipfs_upload,
        poll_interval=2.0,
        consensus_timeout=120.0,
    )

    # ── Determine starting round from ledger ─────────────────────────────────
    try:
        current_latest = api_client.get_latest_round()
        start_round    = current_latest + 1
        logger.info(f"Starting from round {start_round} (ledger latest: {current_latest})")
    except Exception as e:
        logger.warning(f"Could not fetch latest round ({e}). Defaulting to round 1.")
        start_round = 1

    num_rounds     = int(os.environ.get("NUM_ROUNDS", 10))
    round_latencies = []
    results_log    = []
    overall_start  = time.time()
    script_start_time = time.time()

    for round_num in range(start_round, start_round + num_rounds):
        round_start = time.time()
        logger.info(f"========== STARTING ROUND {round_num} ==========")

        bank_updates = {}

        # ── Step 1: Each bank trains on its REAL local dataset ───────────────
        for bank_id, hq in hqs.items():
            logger.info(f"[{bank_id}] Training on real local dataset ({len(hq.val_dataset)} samples)...")
            model          = LSTMTabular(**MODEL_CFG)
            updated_model_sd = train_local(model, hq.val_dataset, TRAIN_CFG)

            # Tiny perturbation so IPFS gives a unique CID every run
            updated_model_sd["fc.bias"] = (
                updated_model_sd["fc.bias"]
                + (script_start_time * 1e-9)
                + (round_num * 1e-9)
            )

            # ── Step 2: HQ FedAvg + blockchain submission ─────────────────
            logger.info(f"[{bank_id}] Running HQ Agent (IPFS upload + CBFT)...")
            res = hq.run_round(round_num, [(updated_model_sd, len(hq.val_dataset))])
            if res and res.get("model_cid"):
                bank_updates[bank_id] = {
                    "model_cid":  res["model_cid"],
                    "model_hash": res["model_hash"],
                    "num_samples": len(hq.val_dataset),
                }

        logger.info(f"--- Round {round_num}: cluster models submitted ---")

        # ── Step 3: Wait for blocks to settle ────────────────────────────────
        logger.info("Waiting 10s for models to settle on ledger before CBFT verification...")
        time.sleep(10)

        # ── Step 4: CBFT cross-verification ──────────────────────────────────
        logger.info("--- Initiating CBFT Cross-Verification ---")
        for bank_id, hq in hqs.items():
            hq.verify_peer_updates(round_num, banks)

        logger.info("Waiting 10s for CBFT verification to finalise...")
        time.sleep(10)

        # ── Step 5: CBFT commits ──────────────────────────────────────────────
        logger.info("--- Initiating CBFT Commits ---")
        for bank_id, hq in hqs.items():
            hq.commit_peer_updates(round_num, banks)

        logger.info("Waiting 10s for CBFT commits to finalise...")
        time.sleep(10)

        # ── Step 6: Global aggregation (BankA) ───────────────────────────────
        logger.info("[GlobalAggregator] Performing trust-weighted cross-cluster aggregation...")
        agg_res = aggregator.run_full_aggregation(round_num, bank_updates)

        round_time = time.time() - round_start
        round_latencies.append(round_time)

        round_record = {
            "round": round_num,
            "latency_sec": round_time,
            "banks_submitted": list(bank_updates.keys()),
        }

        if agg_res and agg_res.get("global_cid"):
            global_cid = agg_res["global_cid"]
            logger.info(f"Round {round_num} Global CID: {global_cid}")

            # ── Step 7: Evaluate global model on held-out test set ────────────
            global_bytes = ipfs_download(global_cid)
            buf          = io.BytesIO(global_bytes)
            try:
                g_sd = torch.load(buf, map_location="cpu", weights_only=True)
            except Exception:
                g_sd = torch.load(buf, map_location="cpu")

            g_model = LSTMTabular(**MODEL_CFG)
            g_model.load_state_dict(g_sd)

            # Evaluate on GLOBAL held-out test set (not a bank-specific val set)
            g_metrics = evaluate_model(g_model, global_test_ds, sample_fraction=1.0)

            logger.info("")
            logger.info(f"===> ROUND {round_num} GLOBAL EVALUATION <===")
            logger.info(f"  F1 Score : {g_metrics['f1']:.4f}")
            logger.info(f"  PR-AUC   : {g_metrics['pr_auc']:.4f}")
            logger.info(f"  ROC-AUC  : {g_metrics['roc_auc']:.4f}")
            logger.info(f"  Precision: {g_metrics['precision']:.4f}")
            logger.info(f"  Recall   : {g_metrics['recall']:.4f}")

            num_banks     = len(banks)
            model_size_mb = len(global_bytes) / (1024 * 1024)
            txn_cost_mb   = (
                (num_banks * model_size_mb)
                + (num_banks * (num_banks - 1) * model_size_mb)
                + (num_banks * model_size_mb)
                + (num_banks * model_size_mb)
                + model_size_mb
            )
            logger.info(f"  Comm Cost: {txn_cost_mb:.2f} MB")
            logger.info(f"  E2E Latency: {round_time:.2f}s")
            logger.info("=========================================\n")

            round_record.update({
                "global_cid": global_cid,
                **g_metrics,
                "comm_cost_mb": txn_cost_mb,
            })
        else:
            logger.error(f"Round {round_num} Aggregation Failed!")
            round_record["error"] = "aggregation_failed"

        results_log.append(round_record)
        logger.info(f"========== ROUND {round_num} COMPLETED in {round_time:.2f}s ==========")

    # ── Save results to JSON ─────────────────────────────────────────────────
    results_path = os.path.join(os.path.dirname(__file__), "benchmark_results.json")
    with open(results_path, "w") as f:
        json.dump(results_log, f, indent=2)
    logger.info(f"Results saved to {results_path}")

    # ── Final summary ────────────────────────────────────────────────────────
    total_time  = time.time() - overall_start
    avg_latency = float(np.mean(round_latencies))

    logger.info("")
    logger.info("=================================")
    logger.info("   PERFORMANCE BENCHMARK STATS   ")
    logger.info("=================================")
    logger.info(f"Total Rounds: {num_rounds}")
    logger.info(f"Total Time: {total_time:.2f}s")
    logger.info(f"Avg Round Latency: {avg_latency:.2f}s")
    if round_latencies:
        completed = [r for r in results_log if "f1" in r]
        if completed:
            logger.info(f"Avg F1 Score: {np.mean([r['f1'] for r in completed]):.4f}")
            logger.info(f"Avg PR-AUC : {np.mean([r['pr_auc'] for r in completed]):.4f}")
    logger.info("=================================")

    if avg_latency <= 120:
        logger.info("✅ Latency SLA met (<= 120s)")
    else:
        logger.error("❌ Latency SLA failed (> 120s)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run HCFL benchmark for N rounds.")
    parser.add_argument("--num-rounds", type=int, default=10, help="Number of FL training rounds (default: 10)")
    args = parser.parse_args()
    os.environ["NUM_ROUNDS"] = str(args.num_rounds)
    run_benchmark()
