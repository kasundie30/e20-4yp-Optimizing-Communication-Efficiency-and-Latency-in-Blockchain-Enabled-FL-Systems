"""
evaluate.py — Standalone evaluator for the saved final model.
Usage: python evaluate.py
"""
import os, sys, json
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import config
from data_prep import load_and_prepare
from metrics_utils import predict_proba, classification_metrics
from fl_engine.ledger import Ledger

def main():
    model_path = os.path.join(config.OUTPUT_DIR, "final_model.pt")
    if not os.path.exists(model_path):
        print("[ERROR] Run run_experiment.py first."); return

    state_dict = torch.load(model_path, map_location="cpu")
    _, X_test, y_test = load_and_prepare()

    y_prob = predict_proba(state_dict, X_test)
    m = classification_metrics(y_test, y_prob)

    rounds_path = os.path.join(config.OUTPUT_DIR, "round_metrics.json")
    comm_vals, e2e_vals, rounds = [], [], []
    if os.path.exists(rounds_path):
        with open(rounds_path) as f: rounds = json.load(f)
        comm_vals = [r["comm_mb"] for r in rounds]
        e2e_vals  = [r["e2e_sec"] for r in rounds]

    ledger_path = os.path.join(config.OUTPUT_DIR, "ledger.jsonl")
    chain_ok, n_blocks = False, 0
    if os.path.exists(ledger_path):
        led = Ledger(ledger_path); chain_ok = led.verify_chain(); n_blocks = len(led)

    print(f"\n{'='*66}")
    print("  BAABDULLAH et al. (2024) — EVALUATION REPORT")
    print(f"{'='*66}")
    print(f"  PR-AUC      : {m['prauc']:.6f}")
    print(f"  ROC-AUC     : {m['rocauc']:.6f}")
    print(f"  F1 Score    : {m['f1']:.6f}")
    print(f"  Precision   : {m['precision']:.6f}")
    print(f"  Recall      : {m['recall']:.6f}")
    if comm_vals:
        print(f"  Comm/round  : {np.mean(comm_vals):.4f} MB")
        print(f"  E2E/round   : {np.mean(e2e_vals):.2f} sec")
        print(f"\n  Per-round breakdown:")
        print(f"  {'Rnd':>4}  {'Comm MB':>9}  {'E2E sec':>8}  {'F1':>7}  {'PR-AUC':>7}  {'ROC-AUC':>8}")
        for r in rounds:
            print(f"  {r['round']:>4}  {r['comm_mb']:>9.4f}  {r['e2e_sec']:>8.2f}  "
                  f"{r['f1']:>7.4f}  {r['prauc']:>7.4f}  {r['rocauc']:>8.4f}")
    print(f"\n  Blockchain  : {'VERIFIED' if chain_ok else 'INVALID'}  ({n_blocks} blocks)")
    print(f"{'='*66}\n")
if __name__ == "__main__":
    main()
