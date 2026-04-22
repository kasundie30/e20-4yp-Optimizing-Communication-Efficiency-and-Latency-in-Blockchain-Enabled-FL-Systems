"""
Data preparation following Baabdullah et al. (2024) §4.3:
  1. Load European CC Fraud dataset
  2. 40% stratified sample
  3. Scale Time & Amount; keep all 30 features
  4. 70/30 stratified train-test split
  5. Apply SMOTE on training set only
  6. Partition into 3 equal client shards (IID)
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

def load_and_prepare():
    print("[DATA] Loading dataset ...")
    df = pd.read_csv(config.DATA_PATH)
    print(f"[DATA] Full dataset: {len(df):,} rows")

    # 40% stratified sample
    df_s, _ = train_test_split(df, train_size=config.SAMPLE_FRAC,
                                stratify=df["Class"], random_state=config.RANDOM_SEED)
    df_s = df_s.reset_index(drop=True)
    print(f"[DATA] 40% sample : {len(df_s):,}  (fraud={int(df_s.Class.sum()):,})")

    # Features: all 30 (V1-V28 + Time + Amount); scale Time & Amount
    feat_cols = [c for c in df_s.columns if c != "Class"]
    X = df_s[feat_cols].values.astype("float32")
    y = df_s["Class"].values.astype("int64")
    scaler = StandardScaler()
    idx = [feat_cols.index("Time"), feat_cols.index("Amount")]
    X[:, idx] = scaler.fit_transform(X[:, idx])

    # 70/30 split
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, train_size=config.TRAIN_RATIO,
                                               stratify=y, random_state=config.RANDOM_SEED)
    print(f"[DATA] Train: {len(X_tr):,}  Test: {len(X_te):,}")

    # SMOTE on training only
    print("[DATA] Applying SMOTE ...")
    X_tr, y_tr = SMOTE(random_state=config.RANDOM_SEED).fit_resample(X_tr, y_tr)
    print(f"[DATA] After SMOTE: {len(X_tr):,}  (fraud={int((y_tr==1).sum()):,} / normal={int((y_tr==0).sum()):,})")

    # Partition into NUM_CLIENTS shards
    rng = np.random.default_rng(config.RANDOM_SEED)
    idx = rng.permutation(len(X_tr))
    X_tr, y_tr = X_tr[idx].astype("float32"), y_tr[idx].astype("int64")
    splits = np.array_split(np.arange(len(X_tr)), config.NUM_CLIENTS)
    clients = []
    for i, s in enumerate(splits):
        Xi, yi = X_tr[s], y_tr[s]
        print(f"[DATA] Client {i+1}: {len(Xi):,}  (fraud={int((yi==1).sum()):,})")
        clients.append((Xi, yi))
    return clients, X_te.astype("float32"), y_te.astype("int64")
