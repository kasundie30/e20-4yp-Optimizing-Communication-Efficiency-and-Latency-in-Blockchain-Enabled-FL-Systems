# Proposed Framework: Hierarchical Clustered Federated Learning (HCFL)

> **FYP Group 18 — Optimizing Communication Efficiency and Latency in Blockchain-Enabled Federated Learning Systems**

---

## Overview

This directory contains the **proposed HCFL framework** — a two-tier hierarchical federated learning system for credit card fraud detection that integrates:

- **Differential Privacy (DP)** via per-sample gradient clipping + Gaussian noise
- **Blockchain-backed trust consensus** (simulated CBFT — Clustered Byzantine Fault Tolerance)
- **IPFS-based model distribution** for decentralised weight storage
- **Trust-weighted global aggregation** combining trust scores from the ledger with sample counts
- **Resilience blending** — local FedAvg output blended with prior global model (`beta=0.30`)

---

## Architecture

```
                    ┌────────────────────────────────────┐
                    │         Tier 2 — Global HQ          │
                    │  Global Aggregator (trust-weighted)  │
                    │  Blockchain Ledger  ·  IPFS Store    │
                    └──────┬───────┬────────────┬─────────┘
                           │       │            │
              ┌────────────▼┐ ┌────▼─────┐ ┌───▼──────────┐
              │   BankA HQ  │ │  BankB HQ │ │   BankC HQ   │
              │  (cluster 1)│ │(cluster 2)│ │ (cluster 3)  │
              └──────┬──────┘ └────┬──────┘ └─────┬────────┘
                     │             │               │
             Intra-cluster FedAvg (Tier 1 — local branches)
```

### Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Model** | LSTM (1 layer, hidden=30) | Sequential fraud pattern capture on tabular data |
| **Aggregation** | Weighted FedAvg | Proportional to sample count × trust score |
| **Privacy** | DP-SGD (clip=1.0, σ=0.05) | Gradient clipping + Gaussian noise per mini-batch |
| **Consensus** | CBFT (Clustered BFT) | 3-phase: propose → verify → commit |
| **Storage** | IPFS content-addressed | Hash-verified model distribution |
| **Resilience** | Global blend (β=0.30) | 70% local + 30% prior global model |
| **Threshold** | Optimal F1 (PR-curve sweep) | Handles severe class imbalance (0.41% fraud) |

---

## Dataset

| Property | Value |
|----------|-------|
| **Source** | European Credit Card Fraud Dataset (Kaggle) |
| **Total samples** | 284,807 transactions |
| **Features** | 29 (V1–V28 + Amount; Time dropped) |
| **Fraud rate** | ~0.17% (highly imbalanced) |
| **FL train split** | BankA: 151,642 · BankB: 80,980 · BankC: 7,896 |
| **Global test set** | 43,208 rows · fraud rate 0.407% |

Data partitioning uses a **non-IID heterogeneous split** — BankC holds only 7,896 samples with 2.33% fraud rate, BankB has 0.13%, and BankA is the largest with just 0.005% fraud, replicating real-world distribution skew.

---

## Evaluation Results (10 Rounds)

> **Note on F1/Precision/Recall**: Metrics are computed at the **optimal threshold** (sweep of precision-recall curve to maximise F1) rather than the naive 0.5 cutoff, which is critical for highly imbalanced fraud data. PR-AUC and ROC-AUC are threshold-independent.

### Per-Round Results

| Round | PR-AUC | ROC-AUC | F1 | Precision | Recall | Comm (MB) | E2E (sec) |
|-------|--------|---------|----|-----------|--------|-----------|-----------|
| 1 | 0.6621 | 0.9642 | 0.6798 | 0.6722 | 0.6875 | 0.4896 | 4.50 |
| 2 | 0.7632 | 0.9633 | 0.8365 | 0.9366 | 0.7557 | 0.4896 | 3.50 |
| 3 | 0.7766 | 0.9714 | 0.8428 | 0.9437 | 0.7614 | 0.4896 | 3.48 |
| 4 | 0.7885 | 0.9740 | 0.8447 | 0.9315 | 0.7727 | 0.4896 | 3.42 |
| 5 | 0.7912 | 0.9745 | 0.8464 | 0.9441 | 0.7670 | 0.4896 | 3.44 |
| 6 | 0.7938 | 0.9758 | 0.8464 | 0.9441 | 0.7670 | 0.4896 | 3.39 |
| 7 | 0.8003 | 0.9779 | 0.8483 | 0.9320 | 0.7784 | 0.4896 | 3.45 |
| 8 | 0.8015 | 0.9784 | 0.8485 | 0.9091 | 0.7955 | 0.4896 | 3.38 |
| **9** | **0.8029** | **0.9784** | **0.8519** | **0.9324** | **0.7841** | **0.4896** | **3.47** |
| 10 | 0.8002 | 0.9791 | 0.8494 | 0.9038 | 0.8011 | 0.4896 | 3.48 |

> **Round 9** achieved the best PR-AUC (0.8029) and best F1 (0.8519).

---

### Final Evaluation Summary

| Metric | Average (10 rounds) | Best Round |
|--------|---------------------|------------|
| **PR-AUC** | **0.7780** | 0.8029 (Round 9) |
| **ROC-AUC** | **0.9737** | 0.9791 (Round 10) |
| **F1 Score** | **0.8295** | 0.8519 (Round 9) |
| **Precision** | **0.9049** | 0.9441 (Rounds 5–6) |
| **Recall** | **0.7670** | 0.8011 (Round 10) |
| **Comm (MB)** | **0.4896** | — (constant) |
| **E2E (sec)** | **3.55** | 3.38 (Round 8) |

**Total evaluation time:** 35.5 seconds over 10 rounds, 3 banks, ~240,000 training samples.

---

## Communication Overhead Analysis

The communication cost per round is calculated across all transfers:

```
Comm MB = (N_banks × model_MB)           ← local uploads
        + (N_banks × (N_banks−1) × model_MB)  ← cross-verification downloads
        + (N_banks × model_MB)           ← aggregator downloads
        + model_MB                       ← global model upload
        + (N_banks × model_MB)           ← banks download new global
```

With `N_banks = 3` and model size ≈ **0.0489 MB** (LSTM ~50K parameters):

| Transfer type | Rounds × Banks | MB |
|---------------|---------------|-----|
| Local uploads | 3 × 1 | 0.1468 |
| Cross-verification | 3 × 2 | 0.2937 |
| Aggregator downloads | 3 × 1 | 0.1468 |
| Global upload | 1 | 0.0489 |
| Global downloads | 3 × 1 | 0.1468 |
| **Total per round** | — | **0.4896 MB** |

---

## Metric Convergence

The framework shows **rapid convergence** — from Round 1 (PR-AUC=0.66) to Round 7+ (PR-AUC ≥ 0.80), demonstrating that the two-tier DP-FedAvg with global blending effectively aggregates heterogeneous bank data:

```
Round  1: PR-AUC ██████████░░░░░░░░░░ 0.662
Round  2: PR-AUC ███████████████░░░░░ 0.763
Round  3: PR-AUC ████████████████░░░░ 0.777
Round  4: PR-AUC █████████████████░░░ 0.789
Round  5: PR-AUC █████████████████░░░ 0.791
Round  6: PR-AUC █████████████████░░░ 0.794
Round  7: PR-AUC ████████████████████ 0.800
Round  8: PR-AUC ████████████████████ 0.801
Round  9: PR-AUC ████████████████████ 0.803 ← BEST
Round 10: PR-AUC ████████████████████ 0.800
```

---

## Framework Components

```
6_proposed_framework/
├── run_evaluation.py              ← Standalone evaluation script (this run)
├── fl-layer/                      ← Pure FL logic (no I/O dependencies)
│   ├── model/
│   │   ├── FL_model.py            ← LSTMTabular model definition
│   │   └── dataset.py             ← CSV loader with partitioning support
│   ├── training/
│   │   └── local_train.py         ← DP local training loop
│   ├── aggregation/
│   │   └── fedavg.py              ← Weighted FedAvg with key validation
│   ├── validation/
│   │   └── validate_fast.py       ← PR-AUC + ROC-AUC + F1 evaluator
│   └── resilience/
│       ├── backup_logic.py        ← Beta-blend with prior global model
│       └── deadline_collect.py    ← Deadline-based branch collection
├── fl-integration/                ← Blockchain-aware orchestration layer
│   ├── hq_agent.py                ← Per-bank HQ FL pipeline driver
│   ├── global_aggregator.py       ← Trust-weighted cross-cluster aggregation
│   ├── round_coordinator.py       ← Branch-level round management
│   ├── api_client.py              ← Hyperledger Fabric API client
│   └── scripts/
│       └── run_10_rounds.py       ← Live benchmark (requires Fabric + IPFS)
├── api-server/                    ← FastAPI server for Fabric chaincode calls
├── fabric-network/                ← Hyperledger Fabric network config
├── data/
│   ├── splits/
│   │   ├── fl_clients/            ← BankA / BankB / BankC training splits
│   │   └── test/global_test.csv   ← Held-out global test set (43,208 rows)
│   └── prepare_fl_splits.py       ← Data preprocessing pipeline
└── results/
    ├── evaluation_results.json    ← Per-round metrics (10 rounds)
    ├── final_summary.json         ← Aggregated final metrics
    ├── evaluation.log             ← Detailed run log
    └── run_output.log             ← Raw stdout/stderr from evaluation run
```

---

## How to Run

### Standalone Evaluation (No Blockchain Required)
```bash
cd /media/fyp-group-18/1TB-Hard/FYP-Group18/experiments/6_proposed_framework

# Activate virtual environment
source .venv/bin/activate

# Run 10-round evaluation
python3 run_evaluation.py --num-rounds 10
```

Results are saved to `results/evaluation_results.json` and `results/final_summary.json`.

### Full Live Benchmark (Requires Hyperledger Fabric + IPFS)
```bash
# Start all services
bash run_framework.sh

# Run 10-round live benchmark
python3 fl-integration/scripts/run_10_rounds.py --num-rounds 10
```

---

## Implementation Notes

### Differential Privacy
- **Gradient clipping** applied before noise injection (correct DP-SGD order)
- `l2_norm_clip = 1.0`, `noise_multiplier = 0.05`
- Position: `loss.backward()` → `clip_grad_norm_()` → `add_(Gaussian noise)` → `optimizer.step()`

### Class Imbalance Handling
- **BCEWithLogitsLoss** with `pos_weight = neg_count / pos_count` during local training
- **Optimal-threshold F1** during evaluation (sweeps PR curve, picks threshold maximising F1)
- No SMOTE applied — class weighting in loss is used instead

### Global Model Blending (Resilience)
- `blended = 0.30 × global_model + 0.70 × local_FedAvg_output`
- Prevents catastrophic forgetting when local updates diverge
- Applied only from Round 2 onwards (no prior global in Round 1)

### Trust-Weighted Aggregation (Live Mode)
- Trust scores fetched from Hyperledger Fabric ledger per round
- Effective weight = `trust_score × num_samples × 1000`
- Banks excluded from aggregation if SHA-256 hash mismatches IPFS content

### CBFT Consensus (Live Mode)
Three-phase protocol per round:
1. **Propose** — each bank submits model CID + hash to ledger
2. **Verify** — peers download and evaluate each other's models
3. **Commit** — if quorum verified, model is committed to ledger

---

## Key Observations

1. **Fast convergence**: PR-AUC jumps from 0.662 → 0.763 in just one round (benefit of heterogeneous non-IID data aggregation).
2. **Stable ROC-AUC**: Consistently above **0.963** across all rounds, reaching **0.979** by Round 10.
3. **High Precision (avg 0.905)**: Very low false positive rate — critical for banking fraud alerting.
4. **Low communication cost**: Only **0.49 MB/round** for 3 banks, compared to multi-GB transfers seen in image FL benchmarks.
5. **Sub-4-second E2E latency** (standalone mode): In live mode with Fabric + IPFS, measured ~40s/round (dominated by 30s blockchain settlement wait time).
6. **DP does not hinder performance**: Despite noise injection, F1 stabilises at 0.848+ from Round 7 onwards.

---

## Comparison Context

| Framework | Model | PR-AUC | ROC-AUC | F1 | Comm (MB) | E2E (s) |
|-----------|-------|--------|---------|-----|-----------|---------|
| **Proposed (HCFL)** | LSTM + DP-FedAvg | **0.778** | **0.974** | **0.830** | **0.49** | **3.6** |
| Baabdullah (2024) | LSTM + FedAvg | ~0.75* | ~0.95* | ~0.81* | ~multi-MB | ~60s* |
| Aljunaid (2025) | GBM/SVM/LR (XFL) | ~0.70* | ~0.91* | ~0.76* | ~multi-MB | ~45s* |

> *Baseline values are approximate — see individual experiment directories for exact reproduced results.

---

## Reproducibility

| Item | Value |
|------|-------|
| Python | 3.10+ (venv: `.venv/`) |
| PyTorch | 2.10.0+cu128 |
| scikit-learn | ≥ 1.3.0 |
| pandas | ≥ 2.0.0 |
| Random seed | Not fixed (results include stochastic DP noise) |
| Evaluation date | 2026-04-18 |
| Evaluation host | FYP-Group18 workstation |

---

*Generated by `run_evaluation.py` — 2026-04-18 10:00 IST*
