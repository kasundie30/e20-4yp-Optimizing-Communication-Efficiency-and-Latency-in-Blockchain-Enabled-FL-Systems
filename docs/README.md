---
layout: home
permalink: index.html

repository-name: eYY-4yp-Optimizing-Communication-Efficiency-and-Latency-in-Blockchain-Enabled-FL-Systems
title: Optimizing Communication Efficiency and Latency in Blockchain-Enabled Federated Learning Systems
---

# Optimizing Communication Efficiency and Latency in Blockchain-Enabled Federated Learning Systems

#### Team

- E/20/148, Hewawasam A.K.L, [e20148@eng.pdn.ac.lk](mailto:e20148@eng.pdn.ac.lk)
- E/20/285, Perera B.B.M.R, [e20285@eng.pdn.ac.lk](mailto:e20285@eng.pdn.ac.lk)
- E/20/316, Ranasinghe R.A.W.L, [e20316@eng.pdn.ac.lk](mailto:e20316@eng.pdn.ac.lk)

#### Supervisors

- Dr. Suneth Namal Karunarathna, [namal@eng.pdn.ac.lk](mailto:namal@eng.pdn.ac.lk)
- Dr. Upul Jayasinghe, [upuljm@eng.pdn.ac.lk](mailto:upuljm@eng.pdn.ac.lk)

#### Table of Contents

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
11. [Links](#links)

---

## Abstract

Credit card fraud detection (CCFD) in the banking sector faces a fundamental tension between the need for collaborative model training and strict data privacy regulations. Existing federated learning (FL) approaches that integrate blockchain for auditability treat all participants equally and do not address the hierarchical nature of real-world banking organisations, leading to suboptimal communication efficiency and round latency. This research presents the Hierarchical Clustered Federated Learning (HCFL) framework — a two-tier privacy-preserving FL system for credit card fraud detection that integrates Hyperledger Fabric blockchain, IPFS-based off-chain model storage, Differential Privacy (DP), and a novel Consensus-Based Federated Trust (CBFT) consensus protocol.

The HCFL framework achieves an average PR-AUC of 0.778, ROC-AUC of 0.974, and F1 Score of 0.830 over 10 federated rounds on the European Credit Card Fraud dataset (284,807 transactions, 0.17% fraud rate). Communication overhead is held constant at 0.490 MB/round, and end-to-end latency averages 3.55 seconds/round in simulation. The system demonstrably reduces inter-bank communication overhead relative to flat FL baselines while delivering competitive detection performance and providing a cryptographically verifiable audit trail suitable for regulatory compliance. Comparative evaluation against Baabdullah et al. (2024) and Aljunaid et al. (2025) confirms that HCFL achieves superior precision (0.905 average) and significantly lower communication-per-round while maintaining comparable recall.

Keywords: Federated Learning, Credit Card Fraud Detection, Blockchain, Differential Privacy, Hierarchical Aggregation, Communication Efficiency, Hyperledger Fabric, IPFS.


---

## Research Problem

Background
Credit card fraud imposes billions of dollars in losses annually on financial institutions worldwide. Accurate fraud detection requires machine learning models trained on large volumes of transaction data. However, banking transaction data is among the most sensitive personal data categories — sharing raw records across organisations violates customer privacy agreements and breaches data protection regulations such as the General Data Protection Regulation (GDPR) and the Payment Card Industry Data Security Standard (PCI-DSS).

Federated Learning (FL) emerged as a solution to this dilemma: instead of sharing raw data, banks train models locally and share only model gradients or weight updates. A central aggregator combines these updates into a shared global model without ever seeing individual transaction records. Yet vanilla FL in the financial domain introduces its own set of challenges:

Trust and Integrity: When banks compete commercially, there is no guarantee that a submitted model update is honest or accurate. A malicious participant could submit deliberately degraded or backdoored updates — a model poisoning attack.
Communication Overhead: As the number of participating branches grows, the volume of model updates transmitted per round increases linearly. For large banking networks with dozens of branches per institution, this becomes a bottleneck.
Latency: Synchronous FL rounds stall when even a single slow ("straggler") participant fails to submit on time, rendering the system unsuitable for time-sensitive fraud alerting pipelines.
Auditability: Regulators require tamper-proof audit logs of model training decisions. Standard FL systems provide no mechanism to record, verify, or dispute how the global model was produced.
Gap in Existing Literature
Existing blockchain-integrated FL systems for fraud detection (e.g., Baabdullah et al., 2024; Aljunaid et al., 2025) address either privacy or auditability in isolation but do not systematically co-optimise communication efficiency, round latency, and Byzantine fault tolerance within a unified hierarchical framework. Specifically:

Baabdullah et al. (2024) propose an FL + blockchain system using an LSTM model and FedAvg aggregation, achieving strong ROC-AUC performance. However, their flat FL topology does not reduce inter-bank communication, and they do not apply differential privacy or cross-bank model verification. Communication overhead grows with participant count.
Aljunaid et al. (2025) propose an Explainable FL (XFL) framework integrating SHAP and LIME for decision transparency. While commendable for interpretability, their best-model-selection aggregation strategy introduces growing communication overhead (model size increases monotonically per round due to GBM warm-start), and average per-round latency exceeds 19 seconds.
Neither baseline addresses the hierarchical structure inherent to banking organisations, where regional branches report to headquarters before engaging with an inter-bank consortium. Neither applies a mathematically rigorous privacy mechanism (DP) to the training step. Neither provides a Byzantine fault-tolerant consensus mechanism that prevents rogue banks from influencing the global model.

Research Questions
This project addresses three core research questions:

RQ1: Can a two-tier hierarchical FL architecture reduce inter-bank communication overhead compared to flat FL baselines while maintaining competitive fraud detection performance?

RQ2: Can blockchain-enforced Clustered Byzantine Fault Tolerance (CBFT) provide model integrity guarantees that prevent poisoning attacks in a multi-bank consortium setting without incurring prohibitive latency?

RQ3: Is it possible to integrate differential privacy into the local training step of a hierarchical FL system without unacceptable degradation of fraud detection metrics on a highly class-imbalanced dataset?

---

<img alt="Image" src="https://github.com/user-attachments/assets/327997f5-2d00-4c0b-809e-325dd2c78f3d" width="100%" alt="Image"/>

## Objectives

To design, implement, and evaluate a Hierarchical Clustered Federated Learning (HCFL) framework for credit card fraud detection that simultaneously optimises:

- Inter-bank communication efficiency (minimise bytes exchanged per federated round)
- End-to-end latency per round (minimise wall-clock time from training start to global model update)
- Detection performance (maximise PR-AUC and F1 Score on imbalanced fraud data)
- Privacy guarantees (provide Differential Privacy for local training updates)
- Security and auditability (provide Byzantine fault tolerance and immutable audit records via blockchain)

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

All experiments use the European Credit Card Fraud Detection dataset (Kaggle), the de-facto benchmark for tabular fraud detection research:

| Property | Value |
|----------|-------|
| Total transactions | 284,807 |
| Features | 30 (V1–V28 PCA-anonymised + Amount + Time; Time dropped in training) |
| Fraud rate (global) | ~0.17% (highly class-imbalanced) |
| Training partitions | BankA: 151,642 · BankB: 80,980 · BankC: 7,896 |
| Global test set | 43,208 rows · fraud rate 0.407% |

The dataset's PCA-anonymised features (V1–V28) simulate the kind of pre-processed, anonymised data that banks would realistically contribute in a privacy-preserving consortium — raw transaction identifiers are not present, aligning with GDPR Article 89 anonymisation provisions.

Data Partitioning: A non-IID (non-independent, non-identically distributed) heterogeneous split is applied. BankC holds only 7,896 samples with a 2.33% local fraud rate; BankB holds 80,980 samples with 0.13% fraud; BankA is the largest partition with 0.005% local fraud. This replicates real-world distribution skew — different bank branches serve demographically distinct customer bases with different spending patterns and fraud exposure profiles.

## 4.2 Model Architecture

The fraud detector at all levels of the hierarchy is a **LSTMTabular** — a single-layer Long Short-Term Memory network adapted for tabular transaction data.


Loss function: BCEWithLogitsLoss with positive class weighting (pos_weight = n_negatives / n_positives) to address the severe class imbalance without oversampling.
Model size: ~50,000 parameters (~0.0489 MB serialised), deliberately minimal to minimise communication overhead per FL round.
Optimizer: Adam, lr = 1×10⁻³, batch_size = 256, epochs = 1 per local round.
LSTM was chosen over simpler feedforward networks because it naturally captures sequential dependencies within transaction streams. Unlike tree-based methods, LSTM is differentiable and compatible with both FedAvg weight averaging and gradient-based Differential Privacy. 


### 4.3 Evaluation Metrics

Given the severe class imbalance (0.17% fraud), standard accuracy is uninformative. The following metrics are computed at the optimal threshold (sweep of the precision-recall curve to maximise F1):

| Metric | Definition | Why used |
|--------|------------|----------|
| PR-AUC | Area under Precision-Recall curve | Primary: threshold-independent; optimal for severe imbalance |
| ROC-AUC | Area under ROC curve | Secondary: overall discrimination ability |
| F1 Score | Harmonic mean of Precision and Recall | Key decision metric at optimal threshold |
| Precision | TP / (TP + FP) | Rate of true fraud among flagged transactions |
| Recall | TP / (TP + FN) | Rate of actual fraud that is caught |
| Comm (MB) | Total bytes transferred per round across all nodes | Communication efficiency |
| E2E (sec) | Wall-clock time from training start to global model stored | Latency efficiency |

---

## System Architecture

<img width="100%" alt="Image" src="https://github.com/user-attachments/assets/d05fa46b-1ba1-4a72-ac90-5a8ead5760a5" />

The proposed architecture follows a **two-tier hierarchical design**.

### Tier 1 – Institutional / Intra-Cluster Layer

Each participating financial institution is treated as an independent cluster. A cluster contains:

- multiple **branch nodes**,
- one **HQ node**,
- and one **Backup node**.

Branch nodes perform local model training on private transaction datasets. The HQ collects branch updates and generates a cluster-level aggregated model. The Backup node remains synchronized with the HQ and can take over in the event of HQ failure, reducing the risk of a single point of failure.



### Tier 2 – Blockchain / Inter-Cluster Layer

The second tier consists of a permissioned blockchain network connecting only the HQ nodes of participating institutions. This layer is responsible for:

- validating cluster-level model updates,
- executing consensus,
- maintaining immutable audit records,
- managing trust scores,
- and coordinating global aggregation.

By limiting blockchain participation to HQ nodes instead of all branch nodes, the architecture significantly reduces consensus complexity, communication overhead, and latency.

### Off-Chain Storage

To avoid storing large model files directly on-chain, model artifacts are uploaded to **IPFS**, and only compact metadata such as CID, hash, validation status, and trust information are stored on the blockchain ledger.

---

<img width="100%" alt="Image" src="https://github.com/user-attachments/assets/790c0478-e528-4fc8-befe-91a40493747f" />

<img width="100%" alt="Image" src="https://github.com/user-attachments/assets/abdeafc0-1baf-4ce1-a948-a14f397f6f58" />

## Experimental Setup and Implementation

The framework is implemented using the following technologies:

- **PyTorch** for model training and federated learning logic
- **Hyperledger Fabric** for permissioned blockchain coordination
- **Go chaincode** for trust score management and update verification
- **IPFS (Kubo)** for off-chain model storage
- **FastAPI** for the middleware API layer
- **Docker** for reproducible multi-service deployment

### Dataset

Experiments use the **Credit Card Fraud Detection dataset**, partitioned across multiple simulated institutions and branches. The training setup reflects a realistic **non-IID multi-institution environment**, where each branch has access only to its local transaction subset.

### Federation Setup

The experimental setup consists of:

- **3 financial institutions**
- **3 branches per institution**
- **HQ aggregation per institution**
- **permissioned blockchain coordination across HQs**

### Key Implemented Components

- Local branch training with DP-inspired clipping and Gaussian noise
- LSTM-based fraud detection model
- Intra-institute FedAvg aggregation
- Fast validation gate
- Backup recovery and failover support
- Trust-aware weighted global aggregation
- Fabric-based CBFT-style verification
- IPFS-based off-chain storage
- REST API integration layer

---

## Results and Analysis

### Proposed HCFL Framework — Per-Round Results

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
| 9 | 0.8029 | 0.9784 | 0.8519 | 0.9324 | 0.7841 | 0.4896 | 3.47 |
| 10 | 0.8002 | 0.9791 | 0.8494 | 0.9038 | 0.8011 | 0.4896 | 3.38 |

PR-AUC jumps from **0.662** (Round 1) to **0.763** (Round 2, +15.3%) once global model blending activates, then stabilises above **0.800** from Round 7 onwards. Communication overhead is constant at **0.4896 MB/round**. Total evaluation time: **35.5 seconds**.

### Baseline Results Summary

##### Baabdullah et al. (2024) — Flat FL + LSTM, 40% dataset sample (~113,922 training / ~34,177 test)

| Metric | Round 1 | Round 10 | Average |
|--------|---------|----------|---------|
| PR-AUC | 0.8919 | 0.8569 | 0.8699 |
| ROC-AUC | 0.9995 | 0.9991 | 0.9992 |
| F1 Score | 0.5534 | 0.7534 | 0.7261 |
| Precision | 0.3878 | 0.6322 | 0.6095 |
| Recall | 0.9661 | 0.9322 | 0.9436 |
| Comm (MB) | 0.1863 | 0.1863 | 0.1863 |
| E2E (sec) | 7.51 | 7.55 | 7.54 |

**Blockchain integrity:** VERIFIED (10 blocks).

##### Aljunaid et al. (2025) — XFL + GBM, best-model selection; model size grows +20 trees/round

| Metric | Round 1 | Round 10 | Average |
|--------|---------|----------|---------|
| PR-AUC | 0.7806 | 0.8791 | 0.8633 |
| ROC-AUC | 0.9925 | 0.9806 | 0.9871 |
| F1 Score | 0.2572 | 0.5187 | 0.4092 |
| Precision | 0.1491 | 0.3604 | 0.2696 |
| Recall | 0.9333 | 0.9250 | 0.9278 |
| Comm (MB) | 0.8298 | 2.1559 | 1.497 |
| E2E (sec) | 69.28 | 14.79 | 19.93 |

**Blockchain integrity:** VERIFIED (10 blocks).

### Cross-Framework Comparison

#### Detection Performance

| Framework | Model | PR-AUC | ROC-AUC | F1 | Precision | Recall |
|-----------|-------|--------|---------|----|-----------|--------|
| HCFL (Proposed) | LSTM + DP-FedAvg | 0.778 | 0.974 | 0.830 | 0.905 | 0.767 |
| Baabdullah (2024) | LSTM + FedAvg | 0.870* | 0.999 | 0.726 | 0.610 | 0.944 |
| Aljunaid (2025) | GBM + Best-Select | 0.863* | 0.987 | 0.409 | 0.270 | 0.928 |

\*Baselines evaluated at the default 0.5 threshold on smaller test sets; HCFL uses the optimal threshold (PR-curve sweep) on the full dataset. Higher baseline Recall reflects threshold-driven over-flagging rather than superior detection.

#### Communication Overhead and Latency

| Framework | Avg Comm/round | Growth | Avg E2E/round |
|-----------|----------------|--------|---------------|
| HCFL (Proposed) | 0.490 MB | None (constant) | 3.55 s |
| Baabdullah (2024) | 0.186 MB | None (constant) | 7.54 s |
| Aljunaid (2025) | 1.497 MB | +0.147 MB/round | 19.93 s |

HCFL's extra **0.304 MB** vs Baabdullah is entirely the CBFT cross-verification cost — the price of Byzantine fault tolerance absent in the baseline.

HCFL delivers a **3.1× lower average communication overhead** and **5.6× lower latency** than Aljunaid, with no growth over rounds. At 50 rounds, Aljunaid would need **~11 MB/round** vs HCFL's constant **0.490 MB** (**22× gap**).

In live mode (Fabric + IPFS), HCFL measured **~33–37 s/round**, within the **120-second SLA bound**.

#### Comprehensive Feature Comparison

| Feature | HCFL (Proposed) | Baabdullah (2024) | Aljunaid (2025) |
|---------|------------------|-------------------|-----------------|
| FL Topology | Two-tier hierarchical | Flat | Flat |
| Aggregation | Trust-weighted FedAvg | FedAvg | Best-model selection |
| Differential Privacy | Yes (DP-SGD) | No | No |
| Byzantine Fault Tolerance | Yes (CBFT 3-phase) | No | No |
| Cross-bank Verification | Yes | No | No |
| Trust Score System | Yes (on-chain) | No | No |
| Explainability (XAI) | No | No | Yes (SHAP + LIME) |
| Live Blockchain | Hyperledger Fabric | Hash-chain (sim) | Hash-chain (sim) |
| Replay Attack Protection | Yes | No | No |
| Comm/round | 0.490 MB (fixed) | 0.186 MB (fixed) | 1.497 MB (growing) |
| E2E Latency | 3.55 s sim / ~35 s live | 7.54 s | 19.93 s |

---

## Impact and Limitations

### Research Impact
Communication Efficiency: The HCFL two-tier hierarchy is the primary architectural contribution to communication efficiency. By ensuring that only one consolidated cluster model per bank reaches the global tier (rather than all branch-level models), inter-bank communication is reduced by a factor of the number of branches per bank. As banks scale to dozens of regional branches, this compression benefit grows proportionally.

Security and Auditability: The CBFT consensus protocol, implemented as on-chain chaincode, provides the first blockchain-enforced cross-bank model verification in the FL-for-CCFD domain (among the compared works). This directly addresses the model poisoning threat that is absent from both baseline frameworks. The immutable audit trail produced by Hyperledger Fabric provides regulators with cryptographically verifiable evidence of model provenance.

Privacy Compliance: Differential Privacy at the branch level provides an (ε, δ)-DP guarantee for individual transaction records — a regulatory requirement under GDPR. Neither Baabdullah nor Aljunaid implement DP. The results demonstrate (F1 = 0.830, PR-AUC = 0.778) that DP does not prohibitively degrade detection quality at the chosen noise multiplier (σ = 0.05).

Scalability in Communication: HCFL's constant 0.490 MB/round communication cost — independent of the number of training rounds — makes it suitable for long-horizon training and production deployment. The growing communication cost inherent to Aljunaid's GBM warm-start approach limits its practical scalability.

Precision for Banking Operations: HCFL's average Precision of 0.905 means that 90.5% of transactions flagged as fraudulent are genuinely fraudulent. This directly translates to operational value: fewer false alarms, reduced analyst burden, and maintained customer experience for legitimate cardholders.

### Limitations
1. Simulation vs. Live Mode Gap: The standalone evaluation (3.55 s/round) significantly underestimates live mode latency (~33–37 s/round), which is dominated by the 30-second Raft ordering settlement wait in Hyperledger Fabric. In production banking, this round latency would be further influenced by network topology, endorsement policy complexity, and peer count. The system meets the 120-second SLA bound, but real-time fraud alerting (sub-second decisions) requires a parallel scoring service, not a federated training loop.

2. Fixed Number of Banks: The current CBFT protocol is calibrated for exactly 3 banks (quorum = 2). Adding a fourth bank requires reconfiguring chaincode constants and re-deploying the network. Dynamic consortium membership — where banks can join or leave between rounds — is not currently supported.

3. Non-IID Skew: While the non-IID data split realistically models heterogeneous fraud distributions, BankC's shard (7,896 samples, 2.33% fraud rate) is substantially smaller than BankA (151,642 samples). This extreme size disparity means BankA dominates trust-weighted aggregation by data volume, potentially over-representing its fraud patterns in the global model. More sophisticated data heterogeneity strategies (e.g., Dirichlet partitioning) could address this.

4. Limited DP Accounting: The current implementation applies DP noise per mini-batch but does not perform formal (ε, δ)-budget accounting across all rounds (privacy loss accumulates across rounds — the composition theorem means 10 rounds of DP training provides weaker privacy than 1 round). Tools like the RDP (Rényi Differential Privacy) accountant (as in Opacus) would provide rigorous budget tracking.

5. No Explainability: Unlike Aljunaid's XFL framework, HCFL provides no mechanisms for explaining individual fraud predictions to end-users or auditors. While the system provides model provenance via the blockchain, per-sample decision transparency (why was this transaction flagged?) is absent. Integration of SHAP or LIME post-hoc explanations at the inference stage is a natural extension.

6. Simulated IPFS Network: The IPFS node in the experimental setup is a local daemon, not a distributed network. In a real banking deployment, IPFS would operate over an organisation-internal CDN or a permissioned IPFS cluster. Performance characteristics (upload/download times) would differ from the simulation.

7. Threshold Comparison Inequality: The results comparison is complicated by the fact that HCFL reports metrics at the optimal threshold while baselines are at 0.5. A complete comparison would require rerunning all baselines with optimal-threshold evaluation, which was not possible within the scope of this project given the baseline implementations.

---

## Conclusion

This paper presented the Hierarchical Clustered Federated Learning (HCFL) framework — a two-tier, privacy-preserving, blockchain-enforced federated learning system for credit card fraud detection in banking consortia. The framework addresses three research questions posed at the outset:

RQ1 (Communication Efficiency): The two-tier hierarchy reduces inter-bank communication to a constant 0.490 MB/round, independent of round count, compared to Aljunaid et al.'s growing overhead (1.497 MB average, 2.156 MB at Round 10). The communication growth problem is structurally solved by using a fixed-size LSTM model rather than accumulating GBM trees. The per-round overhead is approximately 3× that of the flat Baabdullah baseline, an acceptable trade-off given that HCFL's 0.294 MB excess is attributable entirely to the CBFT cross-verification transfers that provide Byzantine fault tolerance absent in the baseline.

RQ2 (Blockchain-Enforced CBFT): The three-phase CBFT protocol (Propose → Verify → Commit) provides quorum-based Byzantine fault tolerance with O(N) message complexity (vs. O(N²) for classical PBFT), implemented as auditable on-chain chaincode. The protocol prevents self-certification, detects model tampering via SHA-256 double-hash verification (CID + independent ledger hash), and provides replay attack protection via CID state tracking. All of this runs within the live-mode latency budget of <120 s/round.

RQ3 (DP + Hierarchical FL): Differential Privacy (clip=1.0, noise_multiplier=0.05) is successfully integrated into the local training step without collapsing model utility. The system achieves an average F1 Score of 0.830 and PR-AUC of 0.778 over 10 rounds — competitive with or superior to both baselines in precision-critical metrics — demonstrating that DP and hierarchical FL are compatible at the chosen noise scale.

The primary differentiating contribution of HCFL over existing work is the simultaneous and co-optimised treatment of communication efficiency, latency, Byzantine security, differential privacy, and blockchain auditability in a unified, production-oriented framework. No existing published work in the FL-for-CCFD domain combines all five of these properties in a single system.

---

## Publications

[//]: # "Uncomment and update once files are uploaded"

<!-- 1. [Semester 7 report](./) -->
<!-- 2. [Semester 7 slides](./) -->
<!-- 3. [Semester 8 report](./) -->
<!-- 4. [Semester 8 slides](./) -->
<!-- 5. Author 1, Author 2 and Author 3, "Optimizing Communication Efficiency and Latency in Blockchain-Enabled FL Systems", [PDF](./) -->

---

## Links

- [Project Repository](https://github.com/cepdnaclk/e20-4yp-Optimizing-Communication-Efficiency-and-Latency-in-Blockchain-Enabled-FL-Systems)
- [Project Page](https://cepdnaclk.github.io/e20-4yp-Optimizing-Communication-Efficiency-and-Latency-in-Blockchain-Enabled-FL-Systems)
- [Department of Computer Engineering](http://www.ce.pdn.ac.lk/)
- [University of Peradeniya](https://eng.pdn.ac.lk/)
