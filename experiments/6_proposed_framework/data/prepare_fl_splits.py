#!/usr/bin/env python3
"""
data/prepare_fl_splits.py
One-time script: maps existing Dirichlet-split bank silos into the
 splits/fl_clients/BankA / BankB / BankC folder structure expected by
 load_bank_dataset(), and creates a global held-out test set.

Mapping (chosen to give roughly equal sample counts while preserving non-IID):
  BankA ← bank_2 + bank_3     (large silos)
  BankB ← bank_4 + bank_5 + bank_7   (medium silos)
  BankC ← bank_8 + bank_9    (smaller silos)
  bank_1, bank_6, bank_10 → held aside for the global test set

Input CSVs  : data/processed/2_bank_silos/bank_N/local_data.csv
               (columns: V1-V28, Amount, Class — 30 cols total)
Output CSVs :
  data/splits/fl_clients/BankA/train_ready.csv
  data/splits/fl_clients/BankB/train_ready.csv
  data/splits/fl_clients/BankC/train_ready.csv
  data/splits/test/global_test.csv
"""
import os
import sys
import pandas as pd
import numpy as np

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
SILO_DIR     = os.path.join(SCRIPT_DIR, "processed", "2_bank_silos")
FL_SPLIT_DIR = os.path.join(SCRIPT_DIR, "splits", "fl_clients")
TEST_DIR     = os.path.join(SCRIPT_DIR, "splits", "test")

# ── Silo → bank mapping ───────────────────────────────────────────────────────
# Excludes bank_1 (185 rows after header, very small) and bank_10 (58 rows).
# bank_6 (522 rows) is too small for useful training; fold into test.
BANK_MAP = {
    "BankA": ["bank_2", "bank_3"],
    "BankB": ["bank_4", "bank_5", "bank_7"],
    "BankC": ["bank_8", "bank_9"],
}
TEST_SILOS = ["bank_1", "bank_6", "bank_10"]

# ── Hold-out fraction from each training bank for global test ─────────────────
HOLDOUT_FRAC = 0.15
RANDOM_SEED  = 42


def load_silo(silo_name: str) -> pd.DataFrame:
    path = os.path.join(SILO_DIR, silo_name, "local_data.csv")
    if not os.path.exists(path):
        print(f"  [WARN] {path} not found — skipping")
        return pd.DataFrame()
    df = pd.read_csv(path)
    print(f"  Loaded {silo_name}: {len(df)} rows, fraud={df['Class'].sum()} ({df['Class'].mean()*100:.2f}%)")
    return df


def save_csv(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"  Saved {len(df)} rows → {path}")


def main():
    print("=" * 60)
    print("  prepare_fl_splits.py — Creating real FL client splits")
    print("=" * 60)

    rng = np.random.default_rng(RANDOM_SEED)
    test_frames = []

    # ── Process each bank ──────────────────────────────────────────────────────
    for bank_id, silos in BANK_MAP.items():
        print(f"\n[{bank_id}] Loading silos: {silos}")
        frames = []
        for silo in silos:
            df = load_silo(silo)
            if not df.empty:
                frames.append(df)

        if not frames:
            print(f"  [ERROR] No data found for {bank_id} — aborting")
            sys.exit(1)

        combined = pd.concat(frames, ignore_index=True)
        # Shuffle to mix silos
        combined = combined.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

        # Heuristic validation: ensure at least some fraud rows
        n_fraud = combined["Class"].sum()
        if n_fraud < 6:
            print(f"  [WARN] {bank_id} has only {n_fraud} fraud rows — metrics will be unreliable")

        # Split off holdout for global test
        n_holdout = max(1, int(len(combined) * HOLDOUT_FRAC))
        holdout   = combined.iloc[:n_holdout]
        train     = combined.iloc[n_holdout:].reset_index(drop=True)

        test_frames.append(holdout)

        out_path = os.path.join(FL_SPLIT_DIR, bank_id, "train_ready.csv")
        save_csv(train, out_path)

        print(f"  [{bank_id}] train={len(train)} rows, fraud={train['Class'].sum()} ({train['Class'].mean()*100:.2f}%)")
        print(f"  [{bank_id}] holdout={len(holdout)} rows (added to global test)")

    # ── Build global test set ─────────────────────────────────────────────────
    print(f"\n[Global Test] Loading small silos: {TEST_SILOS}")
    for silo in TEST_SILOS:
        df = load_silo(silo)
        if not df.empty:
            test_frames.append(df)

    global_test = pd.concat(test_frames, ignore_index=True)
    global_test = global_test.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    test_path   = os.path.join(TEST_DIR, "global_test.csv")
    save_csv(global_test, test_path)

    print(f"\n  [Global Test] {len(global_test)} rows, fraud={global_test['Class'].sum()} ({global_test['Class'].mean()*100:.2f}%)")

    # ── Final summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)
    for bank_id in BANK_MAP:
        path = os.path.join(FL_SPLIT_DIR, bank_id, "train_ready.csv")
        df   = pd.read_csv(path)
        print(f"  {bank_id}: {len(df)} rows | {df.shape[1]} cols | fraud {df['Class'].mean()*100:.3f}%")
    test_df = pd.read_csv(test_path)
    print(f"  global_test: {len(test_df)} rows | {test_df.shape[1]} cols | fraud {test_df['Class'].mean()*100:.3f}%")
    print("=" * 60)
    print("  Done! All splits ready.")


if __name__ == "__main__":
    main()
