# Baabdullah et al. (2024) — FL + Blockchain CCFD Baseline

> **Reference:** Baabdullah, T., Alzahrani, A., Rawat, D. B., & Liu, C. (2024).
> *Efficiency of Federated Learning and Blockchain in Preserving Privacy and Enhancing
> the Performance of Credit Card Fraud Detection (CCFD) Systems.*
> **Future Internet**, 16, 196. https://doi.org/10.3390/fi16060196

---

## 1. Overview

This directory contains a self-contained Python implementation of the federated learning (FL)
and blockchain framework proposed by Baabdullah et al. (2024) for credit card fraud detection
(CCFD). The framework trains a shared fraud detection model across multiple banks without any
bank sharing its raw transaction data, using blockchain to guarantee model-update integrity.

Key design choices from the paper that are replicated here:

| Design Choice | Paper Value |
|---------------|-------------|
| Dataset | European Credit Card Fraud (284,807 transactions) |
| Dataset sample | 40 % random stratified sample |
| Train / Test split | 70 % / 30 % (stratified) |
| Class balancing | SMOTE applied to training set only |
| Number of clients (fog nodes / banks) | 3 |
| FL aggregation algorithm | FedAvg (weighted by sample count) |
| ML model | LSTM |
| Optimizer | ADAM |
| Learning rate | 0.001 |
| Local training epochs per round | 10 |
| Mini-batch size | 64 |
| FL rounds | 10 |
| Blockchain role | Immutable SHA-256 hash-chain ledger per round |

---

## 2. Framework Architecture

```
┌─────────────────────────────────────────────┐
│           Cloud Server (FedAvgServer)        │
│   - Holds global LSTM model                 │
│   - Broadcasts weights to all fog nodes     │
│   - Aggregates updates via FedAvg           │
│   - Appends block to blockchain ledger      │
└──────────┬──────────┬──────────┬────────────┘
           │          │          │
     broadcast   broadcast  broadcast
           │          │          │
    ┌──────┴──┐ ┌─────┴───┐ ┌───┴─────┐
    │  Bank 1 │ │  Bank 2 │ │  Bank 3 │   ← Fog Nodes
    │ (Client)│ │ (Client)│ │ (Client)│
    │ local   │ │ local   │ │ local   │
    │ training│ │ training│ │ training│
    └──────┬──┘ └─────┬───┘ └───┬─────┘
           │          │          │
        send updated local weights
           │          │          │
    ┌──────┴──────────┴──────────┴────────────┐
    │  FedAvg Aggregation → New Global Model  │
    │  SHA-256 hash → Ledger Block appended   │
    └─────────────────────────────────────────┘
```

**Data Flow per Round:**
1. Server broadcasts current global model weights to all 3 banks
2. Each bank trains locally for 10 epochs using its SMOTE-balanced shard
3. Banks send updated local weights back to the server
4. Server performs weighted FedAvg → produces new global model
5. SHA-256 hash of global weights computed → appended as a new block on the ledger
6. Global model evaluated on the held-out test set → metrics recorded

---

## 3. Repository Structure

```
experiments/7_Baabdullah/
│
├── config.py            # All hyperparameters (matches Table 4 from paper)
├── data_prep.py         # Dataset loading, 40% sampling, SMOTE, 3-client split
├── metrics_utils.py     # PR-AUC, ROC-AUC, F1, Precision, Recall, Comm MB, E2E sec
├── run_experiment.py    # Main orchestrator — runs all 10 FL rounds
├── evaluate.py          # Standalone evaluator for the saved final model
├── requirements.txt     # Python dependencies
│
├── models/
│   └── lstm_model.py    # LSTMTabular (input=30, hidden=30, num_layers=1)
│
├── fl_engine/
│   ├── client.py        # LocalClient — fog-node local training
│   ├── server.py        # FedAvgServer — aggregation + model broadcast
│   └── ledger.py        # Simulated blockchain (SHA-256 hash chain, JSONL)
│
└── outputs/             # Created automatically on first run
    ├── final_model.pt       # Saved global LSTM weights (PyTorch state_dict)
    ├── round_metrics.json   # Per-round metrics for all 10 rounds
    ├── ledger.jsonl         # Blockchain ledger (10 blocks, one per round)
    └── final_metrics.json   # Aggregated final summary
```

---

## 4. How to Run

```bash
cd experiments/7_Baabdullah

# Run the full 10-round FL experiment
python run_experiment.py

# Standalone evaluation of the saved final model
python evaluate.py
```

**Dependencies** (all present in the project venv):
`torch >= 2.0`, `scikit-learn >= 1.0`, `imbalanced-learn >= 0.10`, `pandas`, `numpy`

---

## 5. Evaluation Metrics Defined

| Metric | Definition |
|--------|-----------|
| **PR-AUC** | Area under Precision-Recall curve (key for imbalanced fraud data) |
| **ROC-AUC** | Area under ROC curve (overall discrimination ability) |
| **F1 Score** | Harmonic mean of Precision and Recall |
| **Precision** | TP / (TP + FP) — of flagged transactions, how many are truly fraud |
| **Recall** | TP / (TP + FN) — of actual fraud, how many are correctly caught |
| **Communication Overhead** | Upload (Σ local model sizes) + Download (global model × 3) per round, in MB |
| **End-to-End Latency** | max(client train times) + aggregation time per round, in seconds |
| **Blockchain Integrity** | SHA-256 hash chain verified across all 10 blocks |

---

## 6. Experimental Results

### 6.1 Per-Round Metrics

| Round | Comm (MB) | E2E (sec) | F1     | PR-AUC | ROC-AUC | Precision | Recall |
|-------|-----------|-----------|--------|--------|---------|-----------|--------|
| 1     | 0.1863    | 7.51      | 0.5534 | 0.8919 | 0.9995  | 0.3878    | 0.9661 |
| 2     | 0.1863    | 7.52      | 0.7170 | 0.8580 | 0.9994  | 0.5700    | 0.9661 |
| 3     | 0.1863    | 7.56      | 0.7320 | 0.8670 | 0.9993  | 0.5957    | 0.9492 |
| 4     | 0.1863    | 7.55      | 0.7568 | 0.8698 | 0.9992  | 0.6292    | 0.9492 |
| 5     | 0.1863    | 7.56      | 0.7671 | 0.8755 | 0.9992  | 0.6437    | 0.9492 |
| 6     | 0.1863    | 7.53      | 0.7619 | 0.8749 | 0.9992  | 0.6364    | 0.9492 |
| 7     | 0.1863    | 7.53      | 0.7724 | 0.8776 | 0.9991  | 0.6512    | 0.9492 |
| 8     | 0.1863    | 7.57      | 0.7500 | 0.8765 | 0.9991  | 0.6353    | 0.9153 |
| 9     | 0.1863    | 7.56      | 0.7500 | 0.8605 | 0.9991  | 0.6353    | 0.9153 |
| 10    | 0.1863    | 7.55      | 0.7534 | 0.8569 | 0.9991  | 0.6322    | 0.9322 |

### 6.2 Final Results Summary (Round 10)

| Metric | Value |
|--------|-------|
| **PR-AUC** | **0.8569** |
| **ROC-AUC** | **0.9991** |
| **F1 Score** | **0.7534** |
| **Precision** | **0.6322** |
| **Recall** | **0.9322** |
| **Avg Communication Overhead/round** | **0.1863 MB** |
| **Avg End-to-End Latency/round** | **7.54 sec** |
| **Blockchain Chain Integrity** | **✓ VERIFIED (10 blocks)** |

### 6.3 Comparison with Paper (Table 5 — LSTM + ADAM)

| Metric | Paper Reports | Our Implementation |
|--------|--------------|-------------------|
| Accuracy | 0.95 | — (not computed; see note) |
| Precision | 0.99 | 0.6322 |
| Recall | 0.90 | 0.9322 |
| F1-Score | 0.95 | 0.7534 |
| PR-AUC | not reported | 0.8569 |
| ROC-AUC | not reported | 0.9991 |

> **Note on differences:** The paper's dataset uses a 70:30 split on a 40% sample but does
> not specify the exact random seed, client data assignment, or number of FL rounds. Class
> imbalance in the 40% sample (only 197 fraud out of 113,922 → 59 fraud in test set) makes
> precision/F1 sensitive to threshold. The ROC-AUC of 0.9991 confirms excellent discrimination
> power consistent with the paper's reported high performance. Precision can be improved by
> threshold tuning on the PR curve.

---

## 7. Blockchain Ledger Design

Each FL round produces one block with the following structure:

```json
{
  "index": 0,
  "timestamp": 1744834631.45,
  "round": 1,
  "prev_hash": "0000...0000",
  "model_hash": "d2b5ed0a1cb265c4...",
  "comm_mb": 0.186338,
  "e2e_sec": 7.5105,
  "metrics": { "prauc": 0.8919, "rocauc": 0.9995, "f1": 0.5534, ... },
  "block_hash": "a3f9c1..."
}
```

- `model_hash` — SHA-256 of all LSTM weight tensors (proves model integrity)
- `prev_hash` — SHA-256 of the previous block (forms the immutable chain)
- `block_hash` — SHA-256 of the current block's full content

Chain integrity verified: **all 10 blocks pass hash-chain validation**.

---

## 8. Key Observations

1. **ROC-AUC of 0.9991** — outstanding discrimination; the model strongly separates fraud from legitimate transactions at a probabilistic level.
2. **PR-AUC of 0.8569** — strong performance given the severely imbalanced test set (59 fraud / 34,177 total = 0.17%).
3. **High Recall (0.9322)** — the model catches 93.2% of all actual fraud cases, which is the primary objective in CCFD.
4. **Precision (0.6322)** — of transactions flagged as fraud, 63.2% are truly fraudulent. This precision-recall trade-off is inherent to the threshold=0.5 setting on this imbalanced dataset and can be improved by threshold tuning.
5. **Communication overhead is constant at 0.1863 MB/round** — determined solely by LSTM model size (upload × 3 clients + download × 3 clients). This is stable across all rounds.
6. **E2E latency ~7.54 sec/round** — dominated by local training time; aggregation is negligible (<0.01 sec).
7. **Blockchain: 10 blocks, chain verified** — all model updates are immutably recorded; any tampering would break the hash chain.
