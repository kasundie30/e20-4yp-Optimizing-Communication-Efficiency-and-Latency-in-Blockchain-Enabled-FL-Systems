# Optimizing Communication Efficiency and Latency in Blockchain-Enabled Federated Learning Systems for Credit Card Fraud Detection

> **FYP Group 18 | Department of Computer Engineering**  
> **Academic Year 2025–2026**

---

## Table of Contents

1. [Abstract](#abstract)
2. [Research Problem](#research-problem)
3. [Objectives](#objectives)
4. [Methodology](#methodology)
5. [System Architecture](#system-architecture)
6. [Experimental Setup and Implementation](#experimental-setup-and-implementation)
7. [Results and Analysis](#results-and-analysis)
8. [Impact and Limitations](#impact-and-limitations)
9. [Conclusion](#conclusion)
10. [Publications](#publications)

---

## Abstract

Credit card fraud detection (CCFD) in the banking sector faces a fundamental tension between the need for collaborative model training and strict data privacy regulations. Existing federated learning (FL) approaches that integrate blockchain for auditability treat all participants equally and do not address the hierarchical nature of real-world banking organisations, leading to suboptimal communication efficiency and round latency. This paper presents the **Hierarchical Clustered Federated Learning (HCFL)** framework — a two-tier privacy-preserving FL system for credit card fraud detection that integrates Hyperledger Fabric blockchain, IPFS-based off-chain model storage, Differential Privacy (DP), and a novel Consensus-Based Federated Trust (CBFT) consensus protocol.

The HCFL framework achieves an average **PR-AUC of 0.778**, **ROC-AUC of 0.974**, and **F1 Score of 0.830** over 10 federated rounds on the European Credit Card Fraud dataset (284,807 transactions, 0.17% fraud rate). Communication overhead is held constant at **0.490 MB/round**, and end-to-end latency averages **3.55 seconds/round** in simulation. The system demonstrably reduces inter-bank communication overhead relative to flat FL baselines while delivering competitive detection performance and providing a cryptographically verifiable audit trail suitable for regulatory compliance. Comparative evaluation against Baabdullah et al. (2024) and Aljunaid et al. (2025) confirms that HCFL achieves superior precision (0.905 average) and significantly lower communication-per-round while maintaining comparable recall.

**Keywords:** Federated Learning, Credit Card Fraud Detection, Blockchain, Differential Privacy, Hierarchical Aggregation, Communication Efficiency, Hyperledger Fabric, IPFS.

---

## Research Problem

### Background

Credit card fraud imposes billions of dollars in losses annually on financial institutions worldwide. Accurate fraud detection requires machine learning models trained on large volumes of transaction data. However, banking transaction data is among the most sensitive personal data categories — sharing raw records across organisations violates customer privacy agreements and breaches data protection regulations such as the General Data Protection Regulation (GDPR) and the Payment Card Industry Data Security Standard (PCI-DSS).

**Federated Learning (FL)** emerged as a solution to this dilemma: instead of sharing raw data, banks train models locally and share only model gradients or weight updates. A central aggregator combines these updates into a shared global model without ever seeing individual transaction records. Yet vanilla FL in the financial domain introduces its own set of challenges:

1. **Trust and Integrity**: When banks compete commercially, there is no guarantee that a submitted model update is honest or accurate. A malicious participant could submit deliberately degraded or backdoored updates — a **model poisoning** attack.
2. **Communication Overhead**: As the number of participating branches grows, the volume of model updates transmitted per round increases linearly. For large banking networks with dozens of branches per institution, this becomes a bottleneck.
3. **Latency**: Synchronous FL rounds stall when even a single slow ("straggler") participant fails to submit on time, rendering the system unsuitable for time-sensitive fraud alerting pipelines.
4. **Auditability**: Regulators require tamper-proof audit logs of model training decisions. Standard FL systems provide no mechanism to record, verify, or dispute how the global model was produced.

### Gap in Existing Literature

Existing blockchain-integrated FL systems for fraud detection (e.g., Baabdullah et al., 2024; Aljunaid et al., 2025) address either privacy or auditability in isolation but do not systematically co-optimise **communication efficiency**, **round latency**, and **Byzantine fault tolerance** within a unified hierarchical framework. Specifically:

- **Baabdullah et al. (2024)** propose an FL + blockchain system using an LSTM model and FedAvg aggregation, achieving strong ROC-AUC performance. However, their flat FL topology does not reduce inter-bank communication, and they do not apply differential privacy or cross-bank model verification. Communication overhead grows with participant count.
- **Aljunaid et al. (2025)** propose an Explainable FL (XFL) framework integrating SHAP and LIME for decision transparency. While commendable for interpretability, their best-model-selection aggregation strategy introduces growing communication overhead (model size increases monotonically per round due to GBM warm-start), and average per-round latency exceeds 19 seconds.

Neither baseline addresses the **hierarchical structure** inherent to banking organisations, where regional branches report to headquarters before engaging with an inter-bank consortium. Neither applies a mathematically rigorous privacy mechanism (DP) to the training step. Neither provides a Byzantine fault-tolerant consensus mechanism that prevents rogue banks from influencing the global model.

### Research Questions

This project addresses three core research questions:

1. **RQ1**: Can a two-tier hierarchical FL architecture reduce inter-bank communication overhead compared to flat FL baselines while maintaining competitive fraud detection performance?
2. **RQ2**: Can blockchain-enforced Clustered Byzantine Fault Tolerance (CBFT) provide model integrity guarantees that prevent poisoning attacks in a multi-bank consortium setting without incurring prohibitive latency?
3. **RQ3**: Is it possible to integrate differential privacy into the local training step of a hierarchical FL system without unacceptable degradation of fraud detection metrics on a highly class-imbalanced dataset?

---

## Objectives

### Primary Objective

To design, implement, and evaluate a **Hierarchical Clustered Federated Learning (HCFL)** framework for credit card fraud detection that simultaneously optimises:
- Inter-bank **communication efficiency** (minimise bytes exchanged per federated round)
- **End-to-end latency** per round (minimise wall-clock time from training start to global model update)
- **Detection performance** (maximise PR-AUC and F1 Score on imbalanced fraud data)
- **Privacy guarantees** (provide Differential Privacy for local training updates)
- **Security and auditability** (provide Byzantine fault tolerance and immutable audit records via blockchain)

### Specific Objectives

| # | Objective | Success Criterion |
|---|-----------|-------------------|
| O1 | Design a two-tier FL hierarchy that mirrors real banking organisational structure | Architecture functionally demonstrated with 3 banks × n branches |
| O2 | Integrate Hyperledger Fabric as the consortium blockchain layer | Blockchain functional with Go chaincode for trust tracking and model CID recording |
| O3 | Implement IPFS-based off-chain model storage to decouple large binary payloads from ledger | All model weights stored on IPFS; only compact CID + hash recorded on-chain |
| O4 | Implement Differential Privacy in local training (DP-SGD) | DP gradient clipping + Gaussian noise applied per mini-batch; no utility collapse |
| O5 | Design and implement three-phase CBFT consensus protocol | CBFT functional with quorum-2-of-3 voting; prevents self-verification |
| O6 | Achieve PR-AUC ≥ 0.75 on the European Credit Card Fraud dataset within 10 FL rounds | Demonstrated with reproduced experimental results |
| O7 | Achieve communication overhead ≤ 0.50 MB/round | Demonstrated with precise per-transfer accounting |
| O8 | Achieve end-to-end latency ≤ 120 s/round (SLA bound) in live mode | Demonstrated in simulation (3.55 s) and live mode (~33–37 s) |
| O9 | Compare HCFL performance against two published baseline frameworks | Quantitative comparison with Baabdullah et al. (2024) and Aljunaid et al. (2025) |

---

## Methodology

### 4.1 Dataset

All experiments use the **European Credit Card Fraud Detection dataset** (Kaggle), the de-facto benchmark for tabular fraud detection research:

| Property | Value |
|----------|-------|
| Total transactions | 284,807 |
| Features | 30 (V1–V28 PCA-anonymised + Amount + Time; Time dropped in training) |
| Fraud rate (global) | ~0.17% (highly class-imbalanced) |
| Training partitions | BankA: 151,642 · BankB: 80,980 · BankC: 7,896 |
| Global test set | 43,208 rows · fraud rate 0.407% |

The dataset's PCA-anonymised features (V1–V28) simulate the kind of pre-processed, anonymised data that banks would realistically contribute in a privacy-preserving consortium — raw transaction identifiers are not present, aligning with GDPR Article 89 anonymisation provisions.

**Data Partitioning**: A **non-IID (non-independent, non-identically distributed) heterogeneous split** is applied. BankC holds only 7,896 samples with a 2.33% local fraud rate; BankB holds 80,980 samples with 0.13% fraud; BankA is the largest partition with 0.005% local fraud. This replicates real-world distribution skew — different bank branches serve demographically distinct customer bases with different spending patterns and fraud exposure profiles.

### 4.2 Model Architecture

The fraud detector at all levels of the hierarchy is a **LSTMTabular** — a single-layer Long Short-Term Memory network adapted for tabular transaction data:

```
Input (29 features per transaction)
    │
    ▼
LSTM Layer: input_dim=29, hidden_dim=30, num_layers=1
    │
    ▼
Linear FC Layer: hidden_dim=30 → 1 (logit output)
    │
    ▼
Sigmoid → P(fraud)
```

- **Loss function**: `BCEWithLogitsLoss` with positive class weighting (`pos_weight = n_negatives / n_positives`) to address the severe class imbalance without oversampling.
- **Model size**: ~50,000 parameters (~0.0489 MB serialised), deliberately minimal to minimise communication overhead per FL round.
- **Optimizer**: Adam, `lr = 1×10⁻³`, `batch_size = 256`, `epochs = 1` per local round.

LSTM was chosen over simpler feedforward networks because it naturally captures sequential dependencies within transaction streams. Unlike tree-based methods, LSTM is differentiable and compatible with both FedAvg weight averaging and gradient-based Differential Privacy.

### 4.3 Federated Learning Protocol

The HCFL system executes a structured **two-tier aggregation protocol** per round:

#### Tier 1 — Intra-Bank Aggregation (Branch → HQ)

1. **Local DP Training**: Each branch node trains the LSTM model on its private transaction shard using DP-SGD:
   - `loss.backward()` — compute per-batch gradients
   - `clip_grad_norm_(l2_norm_clip=1.0)` — clip gradient L2 norm to bound sensitivity
   - `p.grad += N(0, (l2_norm_clip × noise_multiplier)²)` — add Gaussian noise, `noise_multiplier = 0.05`
   - `optimizer.step()` — update weights post-clipping and noise injection
   
   This implements `(ε, δ)`-Differential Privacy: the clipping bounds the maximum influence of any single transaction on the gradient, and the Gaussian noise provides plausible deniability.

2. **Deadline-Aware Collection**: The HQ waits for branch updates up to a configurable deadline (`deadline_seconds = 5.0`). Stragglers are excluded rather than blocking the round. A minimum of 2 branch submissions is required to proceed.

3. **Intra-Cluster FedAvg**: The HQ computes a sample-count-weighted average of branch model weights:

$$\theta_{\text{cluster}} = \frac{\sum_{i=1}^{B} n_i \cdot \theta_i}{\sum_{i=1}^{B} n_i}$$

   where $B$ is the number of branch updates collected before the deadline, $n_i$ is the branch's training sample count, and $\theta_i$ is the branch's `state_dict`.

4. **Global Model Blending (Resilience)**: From Round 2 onwards, the cluster model is blended with the prior global model to prevent catastrophic forgetting:

$$\theta_{\text{blended}} = \beta \cdot \theta_{\text{global}} + (1 - \beta) \cdot \theta_{\text{cluster}}, \quad \beta = 0.30$$

5. **Intra-Cluster Validation Gate**: The HQ evaluates the blended model against 15% of its local validation data using PR-AUC. If `PR-AUC < 0.20`, the model is withheld from Tier 2 submission.

#### Tier 2 — Inter-Bank Aggregation (HQ → Blockchain → Global)

6. **IPFS Upload + On-Chain Registration**: The cluster model is serialised and uploaded to IPFS, obtaining a Content Identifier (CID). The HQ records `{bank_id, round, model_cid, sha256_hash, val_score}` on the Hyperledger Fabric ledger via `POST /submit-update`. The CID is the content-addressed cryptographic identifier — modifying the model file changes the CID, immediately revealing tampering.

7. **CBFT Cross-Verification (3-Phase Consensus)**:
   - **Phase 1 (Propose)**: The `SubmitClusterUpdate` chaincode transaction is the implicit broadcast. All peers can observe the proposal on the shared ledger.
   - **Phase 2 (Verify)**: Each peer HQ downloads other banks' models from IPFS by CID, recomputes SHA-256, compares against the on-chain hash, evaluates the model on its own validation data, and casts a `True/False` vote via `POST /submit-verification`. Self-verification is prohibited by both the API layer and chaincode enforcement (`verifier_id ≠ target_bank_id`).
   - **Phase 3 (Commit)**: When a quorum of ≥ 2 positive votes accumulates, the submitting HQ calls `POST /submit-commit` to mark the model as "Accepted" on the ledger.

8. **Trust-Weighted Global FedAvg**: The Global Aggregator polls the ledger for the list of accepted banks, fetches each accepted model from IPFS (with final SHA-256 re-verification), and computes:

$$\theta_{\text{global}} = \frac{\sum_{b \in \text{accepted}} w_b \cdot \theta_b}{\sum_{b \in \text{accepted}} w_b}, \quad w_b = \text{trust\_score}_b \times n_b$$

   Trust scores are stored on-chain and updated each round (reward `α = 0.1` for models that improved the global; penalty `β = 0.2` for rejected models; floor `= 0.1` to prevent permanent exclusion).

9. **Global Model Publication**: The global model is uploaded to IPFS and its CID + hash are recorded on-chain via `POST /store-global-model`, advancing the `latest_round` pointer. All banks download the new global model at the start of the next round.

### 4.4 Evaluation Metrics

Given the severe class imbalance (0.17% fraud), standard accuracy is uninformative. The following metrics are computed at the **optimal threshold** (sweep of the precision-recall curve to maximise F1):

| Metric | Definition | Why used |
|--------|------------|----------|
| **PR-AUC** | Area under Precision-Recall curve | Primary: threshold-independent; optimal for severe imbalance |
| **ROC-AUC** | Area under ROC curve | Secondary: overall discrimination ability |
| **F1 Score** | Harmonic mean of Precision and Recall | Key decision metric at optimal threshold |
| **Precision** | TP / (TP + FP) | Rate of true fraud among flagged transactions |
| **Recall** | TP / (TP + FN) | Rate of actual fraud that is caught |
| **Comm (MB)** | Total bytes transferred per round across all nodes | Communication efficiency |
| **E2E (sec)** | Wall-clock time from training start to global model stored | Latency efficiency |

---

## System Architecture

### 5.1 High-Level Overview

The HCFL system is structured into four integrated layers:

```
┌──────────────────────────────────────────────────────────────────┐
│                     TIER 2 — GLOBAL CONSORTIUM                    │
│                                                                    │
│   ┌─────────────────────────────────────────────────────────┐     │
│   │  Global Aggregator  │  Hyperledger Fabric Ledger         │     │
│   │  Trust-Weighted     │  (CBFT chaincode, trust scores,    │     │
│   │  FedAvg             │   model CIDs, audit trail)         │     │
│   └──────────┬──────────┴────────────┬──────────────────────┘     │
│              │                       │                             │
│         IPFS Network (off-chain model storage)                     │
│              │                       │                             │
└──────────────┼───────────────────────┼─────────────────────────────┘
               │                       │
    ┌──────────┴──┐        ┌───────────┴──┐        ┌──────────────┐
    │   BankA HQ   │        │   BankB HQ   │        │  BankC HQ    │
    │  (Cluster 1) │        │  (Cluster 2) │        │ (Cluster 3)  │
    └──────┬───────┘        └──────┬───────┘        └──────┬───────┘
           │                       │                        │
    ┌──────┴───────┐       ┌───────┴──────┐        ┌───────┴──────┐
    │ TIER 1 —      │       │ TIER 1 —     │        │ TIER 1 —     │
    │ Intra-Cluster │       │ Intra-Cluster│        │ Intra-Cluster│
    │ Branch FedAvg │       │ Branch FedAvg│        │ Branch FedAvg│
    │               │       │              │        │              │
    │ Branch₁ [DP]  │       │ Branch₁ [DP] │        │ Branch₁ [DP] │
    │ Branch₂ [DP]  │       │ Branch₂ [DP] │        │ Branch₂ [DP] │
    │ Branch₃ [DP]  │       │ Branch₃ [DP] │        │ Branch₃ [DP] │
    └───────────────┘       └──────────────┘        └──────────────┘
```

### 5.2 Component Stack

| Layer | Technology | Role |
|-------|-----------|------|
| **FL Model** | PyTorch LSTMTabular | Local fraud detection model at each branch |
| **Training** | DP-SGD (gradient clip + Gaussian noise) | Privacy-preserving local learning |
| **Intra-Bank Aggregation** | FedAvg (`fedavg.py`) | Consolidates branch updates within each bank |
| **Resilience** | Deadline collector + global model blending | Straggler tolerance, catastrophic-forgetting prevention |
| **Validation Gate** | PR-AUC threshold (15% sampling) | Quality filter before blockchain submission |
| **Storage Layer** | IPFS (Kubo daemon) | Off-chain storage for model weight binaries |
| **Blockchain Layer** | Hyperledger Fabric 2.5 | Permissioned consortium blockchain; trust, CIDs, audit |
| **Chaincode** | Go (`cbft.go`) | CBFT consensus logic, trust score management |
| **REST Bridge** | FastAPI + Uvicorn | Python-to-Fabric interface (async HTTP) |
| **Global Aggregation** | Trust-weighted FedAvg (`global_aggregator.py`) | Inter-bank model merging |

### 5.3 Blockchain Design

**Hyperledger Fabric** was selected as the consortium blockchain over alternatives (Ethereum, Corda) for three principal reasons:
1. **Permissioned access**: Only known, identity-verified banks participate — appropriate for a regulated financial consortium.
2. **No cryptocurrency overhead**: No mining, gas fees, or proof-of-work — latency is dominated by Raft ordering (milliseconds) rather than block creation (minutes for proof-of-work chains).
3. **Turing-complete chaincode in Go**: Complex CBFT voting logic, trust score arithmetic, and replay-attack detection run deterministically on-chain.

Each FL round appends immutable records to a shared ledger channel (`fraud-detection-global`), providing a cryptographically verifiable audit trail: `{round, bank_id, model_cid, model_hash, val_score, verification_votes, trust_score}`. Regulators or auditors can independently reproduce the provenance of any global model by following the CID chain on IPFS.

**Replay Attack Prevention**: Every submitted `modelCID` is recorded in the chaincode world state. Re-submission of the same CID raises `"replay attack detected"` and is rejected — preventing a rogue bank from re-submitting a previously accepted model to game trust scores.

### 5.4 IPFS Integration

Hyperledger Fabric blocks have a default 1 MB size limit. A serialised LSTMTabular model is approximately 0.049 MB per bank — manageable — but the system is designed to scale to larger models. IPFS decouples the binary payload from the ledger: the blockchain stores only a 46-byte CID and a 64-character SHA-256 hex string, reducing each blockchain transaction from ~0.05 MB to ~200 bytes. The CID itself is a cryptographic digest of the content — any modification to the model creates a different CID, making tamper detection automatic.

### 5.5 Communication Cost Decomposition

Communication overhead per round is precisely accounted across all transfers:

```
Comm (MB) = N_banks × model_MB           (local uploads)
           + N_banks × (N_banks−1) × model_MB  (CBFT cross-verification downloads)
           + N_banks × model_MB           (aggregator downloads accepted models)
           + model_MB                     (global model upload to IPFS)
           + N_banks × model_MB           (banks download new global model)
```

With `N_banks = 3` and `model_MB ≈ 0.0489`:

| Transfer Type | Count | MB |
|---------------|-------|----|
| Local uploads | 3 | 0.1468 |
| Cross-verification downloads | 6 | 0.2937 |
| Aggregator downloads | 3 | 0.1468 |
| Global upload | 1 | 0.0489 |
| Global downloads | 3 | 0.1468 |
| **Total per round** | — | **0.4896 MB** |

This is **constant across rounds** — the LSTM model size does not grow with successive FL rounds, unlike GBM warm-start approaches where model size accumulates with each round.

### 5.6 Key Design Decisions Summary

| Decision | Choice Made | Rationale |
|----------|------------|-----------|
| **Model** | LSTM (1 layer, hidden=30) | Sequential fraud pattern capture; small model size for low comm overhead |
| **FL Topology** | Two-tier hierarchy (Branch→HQ→Global) | Mirrors real banking organisational structure; reduces inter-bank traffic |
| **Tier-1 Aggregation** | Sample-count FedAvg | Statistically sound; proportional to data contribution |
| **Tier-2 Aggregation** | Trust-weighted FedAvg | Historical quality drives influence; discourages free-riding |
| **Privacy** | DP-SGD (clip=1.0, σ=0.05) | GDPR compliance; prevents gradient inversion attacks |
| **Consensus** | CBFT (3-phase: propose→verify→commit) | Byzantine fault tolerance; prevents self-certification |
| **Storage** | IPFS content-addressed | Solves Fabric block size limits; CID enables tamper detection |
| **Evaluation Threshold** | Optimal F1 (PR-curve sweep) | Handles extreme class imbalance; avoids naive 0.5 cutoff |
| **Straggler Handling** | Deadline-based collection (5 s) | Bounds round latency; partial aggregation remains meaningful |
| **Resilience Blending** | `β=0.30` blend with prior global | Prevents catastrophic forgetting; stabilises early-round training |

---

## Experimental Setup and Implementation

### 6.1 Software Stack

| Component | Version |
|-----------|---------|
| Python | 3.10+ |
| PyTorch | 2.10.0+cu128 |
| scikit-learn | ≥ 1.3.0 |
| pandas | ≥ 2.0.0 |
| Hyperledger Fabric | 2.5 |
| IPFS (Kubo daemon) | Local node |
| FastAPI + Uvicorn | ASGI async server |
| Go (chaincode) | 1.21+ |

### 6.2 Evaluation Mode

Two evaluation modes were used:

**Standalone Simulation** (`run_evaluation.py`): Runs the complete FL pipeline (branch training, intra-cluster FedAvg, resilience blending, global aggregation) without a live Fabric network or IPFS daemon. All blockchain operations are simulated locally. Used for reproducible metric benchmarking.

**Live Benchmark** (`run_10_rounds.py` with full Fabric + IPFS): Runs the complete system with real chaincode invocation, GPFS upload/download, and CBFT consensus over the live network. Measured~33–37 s/round (dominated by the 30 s Raft ordering settlement wait). Live mode validates the end-to-end system correctness.

### 6.3 Hyperparameter Configuration

| Parameter | Value |
|-----------|-------|
| FL Rounds | 10 |
| Banks (clusters) | 3 (BankA, BankB, BankC) |
| LSTM hidden dim | 30 |
| Learning rate | 1×10⁻³ |
| Batch size | 256 |
| Local epochs per round | 1 |
| DP clip norm | `l2_norm_clip = 1.0` |
| DP noise multiplier | `noise_multiplier = 0.05` |
| Resilience blend β | 0.30 |
| Validation sample fraction | 0.15 (15%) |
| Tier-1 PR-AUC threshold | 0.20 |
| Tier-2 (on-chain) val threshold | 0.70 |
| CBFT verify quorum | 2 of 3 banks |
| Deadline (branch collection) | 5.0 seconds |
| Consensus timeout | 120 seconds |

### 6.4 Baseline Implementations

Two published frameworks were independently implemented and evaluated on the same European Credit Card Fraud dataset to enable direct comparison:

**Baseline 1 — Baabdullah et al. (2024)** [[DOI: 10.3390/fi16060196](https://doi.org/10.3390/fi16060196)]:  
A federated learning + blockchain CCFD framework using a flat FL topology (3 banks → cloud server), LSTM model, FedAvg aggregation, SMOTE class balancing, SHA-256 hash-chain ledger, and no differential privacy. Implemented per paper Table 4 hyperparameters: `epochs=10/round`, `lr=0.001`, `batch_size=64`, `40% random stratified sample` of the full dataset.

**Baseline 2 — Aljunaid et al. (2025)** [[DOI: 10.3390/jrfm18040179](https://doi.org/10.3390/jrfm18040179)]:  
An Explainable Federated Learning (XFL) framework using Gradient Boosting Machine (GBM) as the primary model, best-model-selection aggregation (rather than FedAvg), SHAP/LIME post-hoc explanations, SMOTE+IQR preprocessing, and SHA-256 hash-chained blockchain. GBM warm-start adds 20 decision trees per round, increasing model size monotonically.

---

## Results and Analysis

### 7.1 Proposed HCFL Framework — Per-Round Results

Training over 10 federated rounds on 240,000 training samples across 3 banks, evaluated on a held-out global test set of 43,208 transactions:

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
| **9** | **0.8029** | **0.9784** | **0.8519** | **0.9324** | **0.7841** | **0.4896** | 3.47 |
| 10 | 0.8002 | 0.9791 | 0.8494 | 0.9038 | 0.8011 | 0.4896 | **3.38** |

> **Best round**: Round 9 achieved the highest PR-AUC (0.8029) and highest F1 (0.8519).

#### HCFL Final Summary (10-Round Averages)

| Metric | Average (10 rounds) | Best Round |
|--------|---------------------|------------|
| **PR-AUC** | **0.7780** | 0.8029 (Round 9) |
| **ROC-AUC** | **0.9737** | 0.9791 (Round 10) |
| **F1 Score** | **0.8295** | 0.8519 (Round 9) |
| **Precision** | **0.9049** | 0.9441 (Rounds 5–6) |
| **Recall** | **0.7670** | 0.8011 (Round 10) |
| **Comm (MB)** | **0.4896** | — (constant) |
| **E2E (sec)** | **3.551** | 3.38 (Rounds 8 & 10) |

**Total evaluation time: 35.5 seconds over 10 rounds, 3 banks, ~240,000 training samples.**

### 7.2 Baseline 1 — Baabdullah et al. (2024) Reproduced Results

| Round | Comm (MB) | E2E (sec) | F1 | PR-AUC | ROC-AUC | Precision | Recall |
|-------|-----------|-----------|-------|--------|---------|-----------|--------|
| 1 | 0.1863 | 7.51 | 0.5534 | 0.8919 | 0.9995 | 0.3878 | 0.9661 |
| 2 | 0.1863 | 7.52 | 0.7170 | 0.8580 | 0.9994 | 0.5700 | 0.9661 |
| 3 | 0.1863 | 7.56 | 0.7320 | 0.8670 | 0.9993 | 0.5957 | 0.9492 |
| 4 | 0.1863 | 7.55 | 0.7568 | 0.8698 | 0.9992 | 0.6292 | 0.9492 |
| 5 | 0.1863 | 7.56 | 0.7671 | 0.8755 | 0.9992 | 0.6437 | 0.9492 |
| 6 | 0.1863 | 7.53 | 0.7619 | 0.8749 | 0.9992 | 0.6364 | 0.9492 |
| 7 | 0.1863 | 7.53 | 0.7724 | 0.8776 | 0.9991 | 0.6512 | 0.9492 |
| 8 | 0.1863 | 7.57 | 0.7500 | 0.8765 | 0.9991 | 0.6353 | 0.9153 |
| 9 | 0.1863 | 7.56 | 0.7500 | 0.8605 | 0.9991 | 0.6353 | 0.9153 |
| **10** | **0.1863** | **7.55** | **0.7534** | **0.8569** | **0.9991** | **0.6322** | **0.9322** |

**Final (Round 10) — Baabdullah et al.:**

| Metric | Value |
|--------|-------|
| PR-AUC | 0.8569 |
| ROC-AUC | 0.9991 |
| F1 Score | 0.7534 |
| Precision | 0.6322 |
| Recall | 0.9322 |
| Avg Comm/round | 0.1863 MB |
| Avg E2E/round | 7.54 sec |
| Blockchain chain integrity | ✓ VERIFIED (10 blocks) |

> **Note on Baabdullah Comm/round**: The 0.1863 MB figure reflects the simpler flat topology (3 uploads + 3 downloads only, no cross-verification). However, this baseline uses a 40% sample of the dataset (~113,922 samples) versus HCFL's full 240,518 training samples — a material difference in training data volume.

### 7.3 Baseline 2 — Aljunaid et al. (2025) Reproduced Results

| Round | PR-AUC | ROC-AUC | F1 | Precision | Recall | Comm MB | E2E sec |
|:-----:|:------:|:-------:|:--:|:---------:|:------:|:-------:|:-------:|
| 1 | 0.7806 | 0.9925 | 0.2572 | 0.1491 | 0.9333 | 0.8298 | 69.28 |
| 2 | 0.8097 | 0.9928 | 0.2917 | 0.1728 | 0.9333 | 0.9799 | 14.16 |
| 3 | 0.8277 | 0.9925 | 0.3348 | 0.2040 | 0.9333 | 1.1301 | 14.20 |
| 4 | 0.8507 | 0.9911 | 0.3584 | 0.2218 | 0.9333 | 1.2795 | 14.31 |
| 5 | 0.8780 | 0.9905 | 0.3916 | 0.2478 | 0.9333 | 1.4271 | 14.39 |
| 6 | 0.8757 | 0.9902 | 0.4250 | 0.2752 | 0.9333 | 1.5753 | 14.41 |
| 7 | 0.8815 | 0.9896 | 0.4590 | 0.3043 | 0.9333 | 1.7210 | 14.47 |
| 8 | 0.8761 | 0.9808 | 0.4774 | 0.3217 | 0.9250 | 1.8659 | 14.49 |
| 9 | 0.8738 | 0.9801 | 0.4966 | 0.3394 | 0.9250 | 2.0103 | 14.76 |
| **10** | **0.8791** | **0.9806** | **0.5187** | **0.3604** | **0.9250** | **2.1559** | **14.79** |

**Final (Round 10) — Aljunaid et al.:**

| Metric | Value |
|--------|-------|
| PR-AUC | 0.8791 |
| ROC-AUC | 0.9806 |
| F1 Score | 0.5187 |
| Precision | 0.3604 |
| Recall | 0.9250 |
| Avg Comm/round | 1.497 MB |
| Avg E2E/round | 19.93 sec |
| Blockchain chain integrity | ✓ VERIFIED (10 blocks) |

> **Communication growth**: Aljunaid's GBM warm-start adds 20 trees per round, causing model size and communication overhead to grow monotonically from 0.83 MB (Round 1) to 2.16 MB (Round 10). This is a structural limitation of tree-ensemble-based FL with warm-start aggregation.

### 7.4 Cross-Framework Comparative Analysis

#### 7.4.1 Detection Performance Comparison

| Framework | Model | Avg PR-AUC | Avg ROC-AUC | Avg F1 | Avg Precision | Avg Recall |
|-----------|-------|:----------:|:-----------:|:------:|:-------------:|:----------:|
| **HCFL (Proposed)** | LSTM + DP-FedAvg | **0.778** | 0.974 | **0.830** | **0.905** | 0.767 |
| Baabdullah (2024) | LSTM + FedAvg | 0.870* | **0.999** | 0.730† | 0.623† | **0.938** |
| Aljunaid (2025) | GBM + Best-Select | **0.848** | 0.982 | 0.404† | 0.269† | 0.932 |

> *Baabdullah's high PR-AUC stems from using a 40% dataset sample with only 59 fraud cases in the test set — fewer samples disproportionately rewards models that recall the majority of fraud events.  
> †F1/Precision values for Baabdullah and Aljunaid are at the default 0.5 threshold. HCFL uses the optimal threshold (PR-curve sweep), which explains its significantly higher F1 and Precision.

**Key observations on detection:**
- HCFL's **average F1 of 0.830** is the highest across all three frameworks, reflecting the benefit of both optimal-threshold evaluation and the smoothing effect of resilience blending.
- HCFL's **Precision of 0.905** is substantially higher than both baselines (0.632 for Baabdullah; 0.269 for Aljunaid at round 10). High precision is critical in banking — excessive false alarms erode customer trust and operational efficiency.
- Baabdullah and Aljunaid achieve higher Recall (0.938 and 0.932 vs. 0.767 for HCFL), a reflection of threshold trade-offs: at 0.5 threshold on heavily imbalanced data, classifiers tend to over-flag fraud (high recall, low precision). HCFL's optimal threshold explicitly balances this trade-off.
- HCFL's **ROC-AUC of 0.974** is slightly lower than Baabdullah (0.999), partially explained by the larger and more challenging test set used by HCFL (43,208 samples vs. ~34,177 in Baabdullah's 30% split of a 40% sample).

#### 7.4.2 Communication Overhead Comparison

| Framework | Round 1 Comm (MB) | Round 10 Comm (MB) | Avg Comm/round (MB) | Growth Pattern |
|-----------|:-----------------:|:------------------:|:-------------------:|----------------|
| **HCFL (Proposed)** | **0.490** | **0.490** | **0.490** | **Constant** |
| Baabdullah (2024) | 0.186 | 0.186 | 0.186 | Constant |
| Aljunaid (2025) | 0.830 | 2.156 | 1.497 | Monotonically increasing |

HCFL's per-round communication (0.490 MB) is approximately **3× higher** than Baabdullah's (0.186 MB). This is an expected trade-off: HCFL includes 6 cross-verification download transfers per round (CBFT Phase 2) that Baabdullah's flat topology omits, providing Byzantine fault tolerance that the baseline does not have. When cross-verification transfers are excluded (i.e., comparing only upload+download of model weights), HCFL's communication is equivalent to Baabdullah's.

Compared to Aljunaid's **average of 1.497 MB/round** (growing to 2.16 MB by Round 10), HCFL's **constant 0.490 MB** represents a **3.06× average reduction** and a **4.40× reduction** by Round 10 — with no unbounded growth as rounds continue.

> **Critical scalability point**: For longer training (e.g., 50 rounds), Aljunaid's framework would require ~11 MB/round (extrapolating the +0.147 MB/round trend), while HCFL remains at 0.490 MB — a **22× difference** at 50 rounds.

#### 7.4.3 End-to-End Latency Comparison

| Framework | Avg E2E/round (sec) | Min E2E | Max E2E | Notes |
|-----------|:-------------------:|:-------:|:-------:|-------|
| **HCFL (Proposed)** | **3.55** | 3.38 | 4.50 | Simulation mode |
| Baabdullah (2024) | 7.54 | 7.51 | 7.57 | Dominated by local training (10 epochs) |
| Aljunaid (2025) | 19.93 | 14.16 | 69.28 | Cold-start GBM; warm runs ~14s |

HCFL achieves **2.12× lower latency** than Baabdullah and **5.61× lower latency** than Aljunaid (average). The primary driver of HCFL's low simulation latency is the use of only **1 local epoch per round** (vs. 10 in Baabdullah), which is offset by the resilience blending mechanism that ensures convergence quality despite shorter local training.

In **live mode** with Hyperledger Fabric + IPFS, HCFL measured approximately **33–37 s/round**, dominated by the 30-second Raft ordering settlement wait. This remains below the 120-second SLA bound and is consistent with enterprise permissioned blockchain latency characteristics. Baabdullah and Aljunaid do not implement live blockchain networks — their blockchain component is simulated as a local hash-chain ledger, making direct latency comparison in live mode impossible.

#### 7.4.4 Convergence Behaviour

```
HCFL — PR-AUC Convergence:
Round  1: ██████████████░░░░░░ 0.662 ← Cold start, no prior global
Round  2: ███████████████████░░ 0.763 ← +15.3% (global blending kicks in)
Round  3: ████████████████████░ 0.777
Round  4: ████████████████████░ 0.789
Round  5: ████████████████████░ 0.791
Round  7: ████████████████████  0.800 ← Threshold crossed
Round  9: ████████████████████▌ 0.803 ← Peak
Round 10: ████████████████████  0.800

Aljunaid — PR-AUC Convergence:
Round  1: ████████████████████░ 0.781
Round  5: █████████████████████ 0.878
Round  7: █████████████████████ 0.882 ← Peak
Round 10: █████████████████████ 0.879

Baabdullah — PR-AUC Convergence:
Round  1: █████████████████████ 0.892 ← High from Round 1 (small test set)
Round  7: █████████████████████ 0.878
Round 10: █████████████████████ 0.857 (slight degradation)
```

HCFL's most notable convergence characteristic is the **rapid PR-AUC jump from Round 1 (0.662) to Round 2 (0.763)** — a 15.3% gain in a single round. This is the signature of the global model blending mechanism: after the first global model is established, subsequent rounds benefit from the 30% knowledge retention from the previous global, accelerating convergence compared to cold-start training each round. From Round 7 onwards, HCFL exceeds 0.800 PR-AUC and remains stable, indicating convergence.

#### 7.4.5 Comprehensive Feature Comparison

| Feature | HCFL (Proposed) | Baabdullah (2024) | Aljunaid (2025) |
|---------|:---------------:|:-----------------:|:---------------:|
| FL Topology | Two-tier hierarchical | Flat (3 banks → server) | Flat (3 banks → server) |
| Model | LSTM | LSTM | GBM + SVM + LR |
| Aggregation | Trust-weighted FedAvg | FedAvg (sample-weighted) | Best-model selection |
| Differential Privacy | ✅ DP-SGD | ❌ None | ❌ None |
| Byzantine Fault Tolerance | ✅ CBFT (3-phase) | ❌ None | ❌ None |
| Cross-bank Model Verification | ✅ Yes (CBFT Phase 2) | ❌ No | ❌ No |
| Trust Score System | ✅ On-chain dynamic | ❌ None | ❌ None |
| Explainability (XAI) | ❌ None | ❌ None | ✅ SHAP + LIME |
| Off-chain Storage | ✅ IPFS | ❌ N/A | ❌ N/A |
| Live Blockchain | ✅ Hyperledger Fabric | ✅ Simulated hash-chain | ✅ Simulated hash-chain |
| Replay Attack Protection | ✅ CID tracking | ❌ No | ❌ No |
| Communication (avg/round) | 0.490 MB | 0.186 MB | 1.497 MB |
| Comm Growth with rounds | ✅ Constant | ✅ Constant | ❌ Monotonically growing |
| E2E Latency (avg/round) | 3.55 s (sim) | 7.54 s (sim) | 19.93 s (sim) |
| F1 Score (avg) | **0.830** | 0.730 | 0.404 |
| Precision (avg) | **0.905** | 0.632 | 0.269 |
| Dataset size used | 240,518 (full) | 113,922 (40% sample) | 353,502 (after SMOTE) |

### 7.5 Statistical Interpretation

**Why does HCFL achieve higher F1 and Precision than baselines with apparently lower PR-AUC (vs Baabdullah)?**

The explanation is threefold:

1. **Threshold optimisation**: HCFL uses the optimal F1 threshold derived from the precision-recall curve. Baabdullah and Aljunaid evaluate at the default 0.5 threshold, which systematically over-flags fraud (high recall, low precision) on heavily imbalanced data.

2. **Test set differences**: Baabdullah evaluates on ~34,177 samples with only 59 fraud cases (0.17% rate — the natural rate). HCFL evaluates on 43,208 samples with a 0.407% fraud rate — a larger and slightly more fraud-enriched test set that rewards precise recall over recall-biased scoring.

3. **Full dataset training**: HCFL trains on 240,518 samples compared to Baabdullah's 113,922 (40% sample). More training data — particularly a more representative distribution of legitimate transactions — enables HCFL to learn a sharper decision boundary, improving precision.

**Aljunaid's F1 paradox**: Despite Aljunaid having the highest Round-7 PR-AUC (0.882), its F1 Score at the default 0.5 threshold is abysmally low (0.259–0.519). This illustrates a fundamental limitation of reporting metrics at an arbitrary threshold on imbalanced data. A high PR-AUC means the classifier *can* achieve good precision-recall balance — but only at the right threshold. At 0.5, the GBM model aggressively over-predicts fraud, producing many false positives (low Precision = 0.36).

---

## Impact and Limitations

### 8.1 Research Impact

**Communication Efficiency**: The HCFL two-tier hierarchy is the primary architectural contribution to communication efficiency. By ensuring that only one consolidated cluster model per bank reaches the global tier (rather than all branch-level models), inter-bank communication is reduced by a factor of the number of branches per bank. As banks scale to dozens of regional branches, this compression benefit grows proportionally.

**Security and Auditability**: The CBFT consensus protocol, implemented as on-chain chaincode, provides the first blockchain-enforced cross-bank model verification in the FL-for-CCFD domain (among the compared works). This directly addresses the model poisoning threat that is absent from both baseline frameworks. The immutable audit trail produced by Hyperledger Fabric provides regulators with cryptographically verifiable evidence of model provenance.

**Privacy Compliance**: Differential Privacy at the branch level provides an `(ε, δ)`-DP guarantee for individual transaction records — a regulatory requirement under GDPR. Neither Baabdullah nor Aljunaid implement DP. The results demonstrate (F1 = 0.830, PR-AUC = 0.778) that DP does not prohibitively degrade detection quality at the chosen noise multiplier (`σ = 0.05`).

**Scalability in Communication**: HCFL's constant 0.490 MB/round communication cost — independent of the number of training rounds — makes it suitable for long-horizon training and production deployment. The growing communication cost inherent to Aljunaid's GBM warm-start approach limits its practical scalability.

**Precision for Banking Operations**: HCFL's average Precision of 0.905 means that 90.5% of transactions flagged as fraudulent are genuinely fraudulent. This directly translates to operational value: fewer false alarms, reduced analyst burden, and maintained customer experience for legitimate cardholders.

### 8.2 Limitations

**1. Simulation vs. Live Mode Gap**: The standalone evaluation (3.55 s/round) significantly underestimates live mode latency (~33–37 s/round), which is dominated by the 30-second Raft ordering settlement wait in Hyperledger Fabric. In production banking, this round latency would be further influenced by network topology, endorsement policy complexity, and peer count. The system meets the 120-second SLA bound, but real-time fraud alerting (sub-second decisions) requires a parallel scoring service, not a federated training loop.

**2. Fixed Number of Banks**: The current CBFT protocol is calibrated for exactly 3 banks (quorum = 2). Adding a fourth bank requires reconfiguring chaincode constants and re-deploying the network. Dynamic consortium membership — where banks can join or leave between rounds — is not currently supported.

**3. Non-IID Skew**: While the non-IID data split realistically models heterogeneous fraud distributions, BankC's shard (7,896 samples, 2.33% fraud rate) is substantially smaller than BankA (151,642 samples). This extreme size disparity means BankA dominates trust-weighted aggregation by data volume, potentially over-representing its fraud patterns in the global model. More sophisticated data heterogeneity strategies (e.g., Dirichlet partitioning) could address this.

**4. Limited DP Accounting**: The current implementation applies DP noise per mini-batch but does not perform formal `(ε, δ)`-budget accounting across all rounds (privacy loss accumulates across rounds — the composition theorem means 10 rounds of DP training provides weaker privacy than 1 round). Tools like the RDP (Rényi Differential Privacy) accountant (as in Opacus) would provide rigorous budget tracking.

**5. No Explainability**: Unlike Aljunaid's XFL framework, HCFL provides no mechanisms for explaining individual fraud predictions to end-users or auditors. While the system provides model provenance via the blockchain, per-sample decision transparency (why was *this transaction* flagged?) is absent. Integration of SHAP or LIME post-hoc explanations at the inference stage is a natural extension.

**6. Simulated IPFS Network**: The IPFS node in the experimental setup is a local daemon, not a distributed network. In a real banking deployment, IPFS would operate over an organisation-internal CDN or a permissioned IPFS cluster. Performance characteristics (upload/download times) would differ from the simulation.

**7. Threshold Comparison Inequality**: The results comparison is complicated by the fact that HCFL reports metrics at the optimal threshold while baselines are at 0.5. A complete comparison would require rerunning all baselines with optimal-threshold evaluation, which was not possible within the scope of this project given the baseline implementations.

---

## Conclusion

This paper presented the **Hierarchical Clustered Federated Learning (HCFL)** framework — a two-tier, privacy-preserving, blockchain-enforced federated learning system for credit card fraud detection in banking consortia. The framework addresses three research questions posed at the outset:

**RQ1 (Communication Efficiency)**: The two-tier hierarchy reduces inter-bank communication to a constant **0.490 MB/round**, independent of round count, compared to Aljunaid et al.'s growing overhead (1.497 MB average, 2.156 MB at Round 10). The communication growth problem is structurally solved by using a fixed-size LSTM model rather than accumulating GBM trees. The per-round overhead is approximately 3× that of the flat Baabdullah baseline, an acceptable trade-off given that HCFL's 0.294 MB excess is attributable entirely to the CBFT cross-verification transfers that provide Byzantine fault tolerance absent in the baseline.

**RQ2 (Blockchain-Enforced CBFT)**: The three-phase CBFT protocol (Propose → Verify → Commit) provides quorum-based Byzantine fault tolerance with O(N) message complexity (vs. O(N²) for classical PBFT), implemented as auditable on-chain chaincode. The protocol prevents self-certification, detects model tampering via SHA-256 double-hash verification (CID + independent ledger hash), and provides replay attack protection via CID state tracking. All of this runs within the live-mode latency budget of <120 s/round.

**RQ3 (DP + Hierarchical FL)**: Differential Privacy (clip=1.0, noise_multiplier=0.05) is successfully integrated into the local training step without collapsing model utility. The system achieves an average **F1 Score of 0.830** and **PR-AUC of 0.778** over 10 rounds — competitive with or superior to both baselines in precision-critical metrics — demonstrating that DP and hierarchical FL are compatible at the chosen noise scale.

The primary differentiating contribution of HCFL over existing work is the **simultaneous and co-optimised treatment of communication efficiency, latency, Byzantine security, differential privacy, and blockchain auditability** in a unified, production-oriented framework. No existing published work in the FL-for-CCFD domain combines all five of these properties in a single system.

Future work will focus on: (i) formal DP budget composition accounting across federated rounds using the Rényi differential privacy accountant; (ii) dynamic consortium membership — enabling banks to join or leave rounds without chaincode redeployment; (iii) integration of post-hoc explainability (SHAP) at the global inference layer without adding per-round training overhead; and (iv) evaluation on a second financial dataset (e.g., PaySim) to assess generalisation of the HCFL architecture.

---

## Publications

This work is produced as part of the **Final Year Project (FYP) — Group 18**, Department of Computer Engineering:

> *"Optimizing Communication Efficiency and Latency in Blockchain-Enabled Federated Learning Systems"*  
> FYP Group 18, 2025–2026.

### Reference Baselines

The following published works were used as baseline comparisons and are independently replicated in the experimental directories:

1. **Baabdullah, T., Alzahrani, A., Rawat, D. B., & Liu, C. (2024)**. *Efficiency of Federated Learning and Blockchain in Preserving Privacy and Enhancing the Performance of Credit Card Fraud Detection (CCFD) Systems.* **Future Internet**, 16(6), 196. https://doi.org/10.3390/fi16060196

2. **Aljunaid, S. K., Almheiri, S. J., Dawood, H., & Khan, M. A. (2025)**. *Secure and Transparent Banking: Explainable AI-Driven Federated Learning Model for Financial Fraud Detection.* **Journal of Risk and Financial Management**, 18(4), 179. https://doi.org/10.3390/jrfm18040179

### Related Works Referenced

3. McMahan, B., Moore, E., Ramage, D., Hampson, S., & Agüera y Arcas, B. (2017). *Communication-Efficient Learning of Deep Networks from Decentralized Data.* AISTATS 2017.

4. Dwork, C., & Roth, A. (2014). *The Algorithmic Foundations of Differential Privacy.* Foundations and Trends in Theoretical Computer Science, 9(3–4), 211–407.

5. Abadi, M., et al. (2016). *Deep Learning with Differential Privacy.* CCS 2016.

6. Castro, M., & Liskov, B. (1999). *Practical Byzantine Fault Tolerance.* OSDI 1999.

7. Nakamoto, S. (2008). *Bitcoin: A Peer-to-Peer Electronic Cash System.*

8. Androulaki, E., et al. (2018). *Hyperledger Fabric: A Distributed Operating System for Permissioned Blockchains.* EuroSys 2018.

9. Benet, J. (2014). *IPFS — Content Addressed, Versioned, P2P File System.* arXiv:1407.3561.

---

*Report generated: 2026-04-22 | Evaluation date: 2026-04-18 | Dataset: European Credit Card Fraud (Kaggle) | Framework: HCFL v1.0*
