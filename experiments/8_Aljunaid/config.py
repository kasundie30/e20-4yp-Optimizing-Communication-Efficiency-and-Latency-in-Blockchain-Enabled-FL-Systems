"""
config.py — Hyperparameters for Aljunaid et al. (2025) XFL Framework
Dataset : European Credit Card Fraud (creditcard.csv)
Models  : GBM (primary), SVM, LR — best-model FL aggregation
XAI     : SHAP + LIME
"""
import os

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE_DIR, "..", "..", "data", "raw", "creditcard.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

# ── Dataset ────────────────────────────────────────────────────────────────────
TRAIN_RATIO  = 0.70      # Paper §5.3: "70% for training, 30% for testing"
RANDOM_SEED  = 42
IQR_CLIP     = True      # Paper §5.1: IQR outlier removal
USE_SMOTE    = True      # Handle class imbalance

# ── Federated Learning ─────────────────────────────────────────────────────────
NUM_CLIENTS  = 3         # Paper: multiple banks as local clients
FL_ROUNDS    = 10        # 10 FL rounds (consistent with project)

# ── Models ─────────────────────────────────────────────────────────────────────
# GBM (best per paper Table 6), SVM, LR trained at each client
GBM_N_ESTIMATORS  = 100
GBM_MAX_DEPTH      = 3
GBM_LEARNING_RATE  = 0.1
SVM_C              = 1.0
SVM_MAX_ITER       = 1000
LR_C               = 1.0
LR_MAX_ITER        = 1000

# ── Aggregation ────────────────────────────────────────────────────────────────
# Paper Eq. 2: W* = argmax A(Wi, Vi)  – select best local model as global model
AGGREGATION  = "best_model"      # NOT FedAvg
PERF_THRESH  = 0.90              # τ threshold (retrain if below)

# ── Evaluation ─────────────────────────────────────────────────────────────────
THRESHOLD    = 0.5               # Decision boundary for P/R/F1

# ── XAI ────────────────────────────────────────────────────────────────────────
SHAP_BACKGROUND_SAMPLES = 100    # Background samples for SHAP KernelExplainer
LIME_NUM_FEATURES        = 10    # Top features for LIME explanation
LIME_NUM_SAMPLES         = 500   # LIME perturbation samples
XAI_SAMPLE_SIZE          = 200   # Rows to explain (SHAP summary plot)
