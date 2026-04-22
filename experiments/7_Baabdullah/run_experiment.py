"""
run_experiment.py — Main FL orchestrator for Baabdullah et al. (2024) baseline.

Runs 10 FL rounds:
  broadcast → local train → FedAvg → ledger block → evaluate
"""
import os, sys, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import config
from data_prep import load_and_prepare
from fl_engine.client import LocalClient
from fl_engine.server import FedAvgServer
from fl_engine.ledger import Ledger
from metrics_utils import (predict_proba, classification_metrics,
                            communication_overhead_mb, end_to_end_latency_sec)

os.makedirs(config.OUTPUT_DIR, exist_ok=True)
LEDGER_PATH      = os.path.join(config.OUTPUT_DIR, "ledger.jsonl")
FINAL_MODEL_PATH = os.path.join(config.OUTPUT_DIR, "final_model.pt")
METRICS_PATH     = os.path.join(config.OUTPUT_DIR, "round_metrics.json")


def main():
    print("\n" + "="*66)
    print("  Baabdullah et al. (2024) — FL + Blockchain CCFD Baseline")
    print("  LSTM + ADAM | 3 fog nodes | 10 FL rounds | SMOTE")
    print("="*66)

    # 1 — Data
    clients_data, X_test, y_test = load_and_prepare()
    print(f"\n[INFO] Test: {len(X_test):,} rows (fraud={int((y_test==1).sum()):,})\n")

    # 2 — Initialise FL
    server  = FedAvgServer()
    clients = [LocalClient(f"bank_{i+1}", X, y) for i, (X, y) in enumerate(clients_data)]
    ledger  = Ledger(LEDGER_PATH)

    # 3 — FL rounds
    records = []
    print(f"  {'Rnd':>3}  {'Comm MB':>9}  {'E2E sec':>8}  {'F1':>7}  {'PR-AUC':>7}  {'ROC-AUC':>8}  {'Prec':>6}  {'Recall':>7}")
    print(f"  {'---':>3}  {'-'*9}  {'-'*8}  {'-'*7}  {'-'*7}  {'-'*8}  {'-'*6}  {'-'*7}")

    for rnd in range(1, config.FL_ROUNDS + 1):
        global_sd = server.get_global_state_dict()

        # Local training (simulate parallel execution sequentially)
        updates, client_times, upload_bytes = [], [], []
        for client in clients:
            local_sd, n, t, b = client.train_one_round(global_sd)
            updates.append((local_sd, n))
            client_times.append(t)
            upload_bytes.append(b)

        # Aggregation
        new_sd, agg_sec, global_bytes = server.aggregate(updates)

        # Metrics
        comm_mb = communication_overhead_mb(upload_bytes, global_bytes, config.NUM_CLIENTS)
        e2e_sec = end_to_end_latency_sec(client_times, agg_sec)
        y_prob  = predict_proba(new_sd, X_test)
        m       = classification_metrics(y_test, y_prob)

        # Ledger
        block = ledger.append_block(rnd, new_sd, comm_mb, e2e_sec, m)

        rec = {"round": rnd, "comm_mb": comm_mb, "e2e_sec": e2e_sec,
               "model_hash": block["model_hash"], **m}
        records.append(rec)

        print(f"  {rnd:>3}  {comm_mb:>9.4f}  {e2e_sec:>8.2f}  "
              f"{m['f1']:>7.4f}  {m['prauc']:>7.4f}  {m['rocauc']:>8.4f}  "
              f"{m['precision']:>6.4f}  {m['recall']:>7.4f}")

    # 4 — Save outputs
    server.save(FINAL_MODEL_PATH)
    with open(METRICS_PATH, "w") as f:
        json.dump(records, f, indent=2)
    print(f"\n[INFO] round_metrics.json -> {METRICS_PATH}")
    print(f"[INFO] ledger.jsonl       -> {LEDGER_PATH}")

    # 5 — Final summary
    final     = records[-1]
    comm_vals = [r["comm_mb"] for r in records]
    e2e_vals  = [r["e2e_sec"] for r in records]
    chain_ok  = ledger.verify_chain()

    print(f"\n{'='*66}")
    print(f"  FINAL RESULTS — Round {config.FL_ROUNDS}")
    print(f"{'='*66}")
    print(f"  PR-AUC      : {final['prauc']:.6f}")
    print(f"  ROC-AUC     : {final['rocauc']:.6f}")
    print(f"  F1 Score    : {final['f1']:.6f}")
    print(f"  Precision   : {final['precision']:.6f}")
    print(f"  Recall      : {final['recall']:.6f}")
    print(f"  Comm/round  : {np.mean(comm_vals):.4f} MB  (min={np.min(comm_vals):.4f}  max={np.max(comm_vals):.4f})")
    print(f"  E2E/round   : {np.mean(e2e_vals):.2f} sec  (min={np.min(e2e_vals):.2f}  max={np.max(e2e_vals):.2f})")
    print(f"  Blockchain  : {'VERIFIED' if chain_ok else 'INVALID'}  ({len(ledger)} blocks)")
    print(f"{'='*66}\n")

    out = {**final, "avg_comm_mb": float(np.mean(comm_vals)),
           "avg_e2e_sec": float(np.mean(e2e_vals)), "chain_verified": chain_ok}
    with open(os.path.join(config.OUTPUT_DIR, "final_metrics.json"), "w") as f:
        json.dump(out, f, indent=2)
    return out

if __name__ == "__main__":
    main()
