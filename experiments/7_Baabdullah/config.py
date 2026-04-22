"""
config.py — Hyperparameters matching Baabdullah et al. (2024) Table 4
"""
import os

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE_DIR, "..", "..", "data", "raw", "creditcard.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

# ── Dataset ────────────────────────────────────────────────────────────────────
SAMPLE_FRAC    = 0.40    # Paper: "randomly select 40% of the data"
TRAIN_RATIO    = 0.70    # Paper: "70% for training, 30% for testing"
INPUT_DIM      = 30      # Paper Table 4: input_dim = 30 (V1-V28 + Time + Amount)
RANDOM_SEED    = 42

# ── Federated Learning ─────────────────────────────────────────────────────────
NUM_CLIENTS    = 3       # Paper: "three banks (fog nodes)"
FL_ROUNDS      = 10      # User-confirmed: 10 rounds
LOCAL_EPOCHS   = 10      # Paper Table 4: num_epochs = 10
BATCH_SIZE     = 64      # Paper Table 4: batch_size = 64

# ── Model (LSTM) ───────────────────────────────────────────────────────────────
HIDDEN_DIM     = 30      # Paper Table 4: hidden_dim = 30
NUM_LAYERS     = 1       # Paper Table 4: num_layers = 1

# ── Optimizer (ADAM) ───────────────────────────────────────────────────────────
LEARNING_RATE  = 0.001   # Paper Table 4: learning_rate = 0.001

# ── Evaluation ─────────────────────────────────────────────────────────────────
THRESHOLD      = 0.5     # Decision threshold for P / R / F1
LOG_INTERVAL   = 10      # Paper Table 4: log_interval = 10 batches
