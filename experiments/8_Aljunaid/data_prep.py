"""
data_prep.py — Dataset loading and preprocessing for Aljunaid et al. (2025).

Steps (Paper §5.1):
  1. Load European CC Fraud (creditcard.csv)
  2. IQR outlier removal on Amount & Time
  3. StandardScaler on all features
  4. 70/30 stratified train-test split
  5. SMOTE on training set (class imbalance)
  6. Partition training set into NUM_CLIENTS IID shards
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config


def _iqr_clip(df, cols):
    """Remove samples where any of `cols` lies outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR]."""
    mask = pd.Series(True, index=df.index)
    for c in cols:
        q1, q3 = df[c].quantile(0.25), df[c].quantile(0.75)
        iqr = q3 - q1
        mask &= df[c].between(q1 - 1.5 * iqr, q3 + 1.5 * iqr)
    return df[mask].reset_index(drop=True)


def load_and_prepare():
    """
    Returns
    -------
    clients_data : list of (X_train, y_train) per client
    X_test       : float32 numpy array
    y_test       : int64 numpy array
    feature_names: list[str]
    """
    print("[DATA] Loading creditcard.csv ...")
    df = pd.read_csv(config.DATA_PATH)
    print(f"[DATA] Raw dataset  : {len(df):,} rows  (fraud={int(df.Class.sum()):,})")

    # ── Paper §5.1: IQR outlier removal on Amount & Time ──────────────────────
    if config.IQR_CLIP:
        df = _iqr_clip(df, ["Amount", "Time"])
        print(f"[DATA] After IQR    : {len(df):,} rows  (fraud={int(df.Class.sum()):,})")

    # Features: V1-V28 + Time + Amount (30 features); label: Class
    feat_cols = [c for c in df.columns if c != "Class"]
    X = df[feat_cols].values.astype("float32")
    y = df["Class"].values.astype("int64")

    # ── StandardScaler (all features, paper §5.1: feature scaling) ────────────
    scaler = StandardScaler()
    X = scaler.fit_transform(X).astype("float32")

    # ── 70/30 stratified split ─────────────────────────────────────────────────
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, train_size=config.TRAIN_RATIO,
        stratify=y, random_state=config.RANDOM_SEED
    )
    print(f"[DATA] Train        : {len(X_tr):,}  Test: {len(X_te):,}")

    # ── SMOTE on training only ─────────────────────────────────────────────────
    if config.USE_SMOTE:
        print("[DATA] Applying SMOTE ...")
        X_tr, y_tr = SMOTE(random_state=config.RANDOM_SEED).fit_resample(X_tr, y_tr)
        print(f"[DATA] After SMOTE  : {len(X_tr):,}  "
              f"(fraud={int((y_tr==1).sum()):,}  normal={int((y_tr==0).sum()):,})")

    # ── Partition into NUM_CLIENTS IID shards ─────────────────────────────────
    rng = np.random.default_rng(config.RANDOM_SEED)
    idx = rng.permutation(len(X_tr))
    X_tr = X_tr[idx].astype("float32")
    y_tr = y_tr[idx].astype("int64")
    splits = np.array_split(np.arange(len(X_tr)), config.NUM_CLIENTS)
    clients_data = []
    for i, s in enumerate(splits):
        Xi, yi = X_tr[s], y_tr[s]
        print(f"[DATA] Client {i+1}     : {len(Xi):,}  (fraud={int((yi==1).sum()):,})")
        clients_data.append((Xi, yi))

    return clients_data, X_te.astype("float32"), y_te.astype("int64"), feat_cols
