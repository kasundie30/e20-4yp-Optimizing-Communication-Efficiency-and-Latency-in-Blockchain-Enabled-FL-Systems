# Experiment 8 — Aljunaid et al. (2025) XFL Baseline

> **Paper:** Aljunaid, S. K., Almheiri, S. J., Dawood, H., & Khan, M. A. (2025).
> *Secure and Transparent Banking: Explainable AI-Driven Federated Learning Model for Financial Fraud Detection.*
> Journal of Risk and Financial Management, 18(4), 179. https://doi.org/10.3390/jrfm18040179

---

## 1. Overview

This experiment replicates the **Explainable Federated Learning (XFL)** framework proposed by Aljunaid et al. (2025) for financial fraud detection. The paper addresses two core limitations of existing federated learning systems: **lack of privacy preservation** and **black-box opacity**. The proposed XFL model integrates FL with SHAP and LIME-based XAI to simultaneously ensure data privacy and decision-level transparency.

The implementation uses the **European Credit Card Fraud dataset** (adapted from the paper's original PaySim dataset) to allow direct comparison with other baselines in this project.

---

## 2. Framework Architecture

### 2.1 System Design

The XFL framework operates across three logical layers:

```
Banking Data Layer
        │  (transaction features)
        ▼
Preprocessing Layer
        │  IQR outlier removal → StandardScaler → SMOTE
        ▼
  ┌─────┴──────┐
  │  3 Banks   │  (local clients — train independently, no raw data shared)
  │  GBM │ GBM │ GBM
  └─────┬──────┘
        │  model parameters only (no raw data)
        ▼
  Global Server
  Best-Model Selection  ←── Paper Eq. 2: W* = argmax A(Wᵢ, Vᵢ)
        │
        ▼
  XAI Layer
  SHAP (Eq. 3) + LIME (Eq. 4)
        │
        ▼
  Blockchain Ledger  (SHA-256 hash-chained audit trail)
```

### 2.2 Key Design Decisions

| Component | Choice | Paper Reference |
|---|---|---|
| Primary model | Gradient Boosting Machine (GBM) | Table 6 — best accuracy (99.95%) |
| Comparison models | SVM, Logistic Regression | Table 6 |
| FL aggregation | **Best-model selection** (not FedAvg) | Eq. 2 |
| XAI — global | SHAP TreeExplainer (feature importance) | §5.4, Eq. 3 |
| XAI — local | LIME LimeTabularExplainer (per-sample) | §5.4, Eq. 4 |
| Privacy | No raw data shared; only model weights | §5.3 |
| Auditability | SHA-256 hash-chained ledger per round | §4.1 |

---

## 3. Dataset

| Property | Value |
|---|---|
| Dataset | European Credit Card Fraud (creditcard.csv) |
| Raw rows | 284,807 |
| After IQR outlier removal | 252,903 |
| Fraud (positive) class | 401 samples (after IQR) |
| Features | 30 (V1–V28 + Time + Amount) |
| Preprocessing | IQR clip → StandardScaler → SMOTE |
| Train / Test split | 70% / 30% (stratified) |
| Training rows (after SMOTE) | 353,502 (176,751 fraud + 176,751 normal) |
| Test rows | 75,871 (120 fraud) |
| Clients (IID shards) | 3 banks × ~117,834 samples each |

> **Note:** The paper uses the PaySim (Financial Fraud Detection) dataset with raw transaction columns
> (step, type, amount, nameOrig, etc.). This implementation maps the same algorithmic framework
> onto the European Credit Card dataset to enable direct cross-paper comparison within this project.

---

## 4. Federated Learning Protocol

### 4.1 Local Training (Bank Clients)

Each bank trains a GBM model on its local shard using warm-start extension across rounds:

```
Wᵢᵗ = Wᵢᵗ⁻¹ − η · ∇L(Wᵢᵗ⁻¹; Dᵢ)   (Paper Eq. 1)
```

- Clients train **independently** — no raw data leaves the bank
- From round 2 onwards, each client extends the previous global model via GBM `warm_start`
- Training time collapses from ~69 sec (round 1, cold) to ~14 sec (rounds 2–10, warm)

### 4.2 Global Aggregation — Best-Model Selection

Unlike FedAvg, the server selects the **single best-performing local model** rather than averaging weights:

```
W* = argmax_{Wᵢ} A(Wᵢ, Vᵢ)           (Paper Eq. 2)
```

where `A(Wᵢ, Vᵢ)` is accuracy of local model `Wᵢ` evaluated on the server's validation set `Vᵢ`.

### 4.3 XAI — SHAP (Eq. 3)

Global model feature importance via Shapley Additive Explanations:

```
phi_j = sum_{S ⊆ F\{j}} [|S|!(|F|-|S|-1)!/|F|!] * [f(S ∪ {j}) - f(S)]
```

Top features identified: **V14, V18, V17, V4, V11**

### 4.4 XAI — LIME (Eq. 4)

Per-sample explanations via local surrogate models:

```
f_hat(x) = argmin_{g ∈ G} L(f, g, π_x) + Ω(g)
```

Three individual transaction explanations generated (fraud + non-fraud samples).

---

## 5. Hyperparameters

| Parameter | Value |
|---|---|
| FL rounds | 10 |
| Clients (banks) | 3 |
| Aggregation | Best-model selection |
| GBM estimators (round 1) | 100 |
| GBM estimators increment | +20 per round (warm-start) |
| GBM max depth | 3 |
| GBM learning rate | 0.1 |
| Decision threshold (F1/P/R) | 0.5 |
| Random seed | 42 |
| SMOTE | Yes (training only) |
| IQR outlier removal | Yes (Amount & Time) |

---

## 6. Results

### 6.1 Per-Round Metrics

| Round | PR-AUC | ROC-AUC | F1 | Precision | Recall | Comm MB | E2E sec | Best Client |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 0.7806 | 0.9925 | 0.2572 | 0.1491 | 0.9333 | 0.8298 | 69.28 | bank_2 |
| 2 | 0.8097 | 0.9928 | 0.2917 | 0.1728 | 0.9333 | 0.9799 | 14.16 | bank_3 |
| 3 | 0.8277 | 0.9925 | 0.3348 | 0.2040 | 0.9333 | 1.1301 | 14.20 | bank_2 |
| 4 | 0.8507 | 0.9911 | 0.3584 | 0.2218 | 0.9333 | 1.2795 | 14.31 | bank_3 |
| 5 | 0.8780 | 0.9905 | 0.3916 | 0.2478 | 0.9333 | 1.4271 | 14.39 | bank_2 |
| 6 | 0.8757 | 0.9902 | 0.4250 | 0.2752 | 0.9333 | 1.5753 | 14.41 | bank_2 |
| 7 | 0.8815 | 0.9896 | 0.4590 | 0.3043 | 0.9333 | 1.7210 | 14.47 | bank_3 |
| 8 | 0.8761 | 0.9808 | 0.4774 | 0.3217 | 0.9250 | 1.8659 | 14.49 | bank_3 |
| 9 | 0.8738 | 0.9801 | 0.4966 | 0.3394 | 0.9250 | 2.0103 | 14.76 | bank_2 |
| **10** | **0.8791** | **0.9806** | **0.5187** | **0.3604** | **0.9250** | **2.1559** | **14.79** | **bank_3** |

### 6.2 Final Summary (Round 10)

| Metric | Value |
|---|---|
| **PR-AUC** | **0.8791** |
| **ROC-AUC** | **0.9806** |
| **F1 Score** | **0.5187** |
| **Precision** | **0.3604** |
| **Recall** | **0.9250** |
| **Avg Comm/round** | **1.497 MB** |
| **Avg E2E/round** | **19.93 sec** |
| **Blockchain** | **VERIFIED (10 blocks)** |

### 6.3 Communication Overhead

| Statistic | Value |
|---|---|
| Min (round 1, cold-start) | 0.830 MB |
| Max (round 10, warm) | 2.156 MB |
| Average across 10 rounds | 1.497 MB |

> Communication grows monotonically because warm-start GBM adds 20 new trees per round,
> increasing the serialised model size with each iteration.

### 6.4 End-to-End Latency

| Statistic | Value |
|---|---|
| Round 1 (cold-start, 100 trees) | 69.28 sec |
| Rounds 2–10 (warm-start, +20 trees) | ~14–15 sec |
| Average across 10 rounds | 19.93 sec |

---

## 7. XAI Outputs

SHAP and LIME explanations are saved to `outputs/xai/`:

| File | Description |
|---|---|
| `shap_bar_GradientBoostingClassifier.png` | Mean \|SHAP\| feature importance bar chart |
| `shap_beeswarm_GradientBoostingClassifier.png` | SHAP beeswarm summary plot |
| `lime_sample_0.png` | LIME explanation — fraud sample |
| `lime_sample_17.png` | LIME explanation — non-fraud sample |
| `lime_sample_100.png` | LIME explanation — additional sample |

**Top-5 features by SHAP importance:**

| Rank | Feature | Mean \|SHAP\| |
|:---:|---|:---:|
| 1 | V14 | 3.3586 |
| 2 | V18 | 1.7514 |
| 3 | V17 | 1.6613 |
| 4 | V4 | 1.4925 |
| 5 | V11 | 1.4473 |

---

## 8. Output Files

| File | Description |
|---|---|
| `outputs/final_metrics.json` | Final round metrics + averages + blockchain status |
| `outputs/round_metrics.json` | Per-round metrics for all 10 rounds |
| `outputs/ledger.jsonl` | Blockchain ledger (10 blocks, SHA-256 chained) |
| `outputs/final_model.joblib` | Serialised global GBM model (round 10) |
| `outputs/xai/` | SHAP bar, SHAP beeswarm, 3× LIME plots |

---

## 9. How to Run

```bash
# From the project root
/media/fyp-group-18/1TB-Hard/FYP-Group18/venv/bin/python3 \
    experiments/8_Aljunaid/run_experiment.py
```

**Dependencies** (installed in project venv):
```
scikit-learn >= 1.4     imbalanced-learn >= 0.14
shap >= 0.51            lime >= 0.2
numpy >= 1.26           pandas >= 2.1
joblib >= 1.2           matplotlib
```

---

## 10. Comparison with Baabdullah et al. (Experiment 7)

| Aspect | Aljunaid (Exp 8) | Baabdullah (Exp 7) |
|---|---|---|
| Model type | GBM (sklearn) | LSTM (PyTorch) |
| FL aggregation | Best-model selection | FedAvg (weighted avg) |
| XAI | SHAP + LIME | None |
| Comm/round (avg) | 1.497 MB | — |
| E2E/round (avg) | 19.93 sec | — |
| ROC-AUC | 0.9806 | — |
| PR-AUC | 0.8791 | — |
| Recall | 0.9250 | — |
| Blockchain | ✅ SHA-256 chained | ✅ SHA-256 chained |

---

*Generated: 2026-04-16 | Dataset: European Credit Card Fraud | Rounds: 10 | Clients: 3*
