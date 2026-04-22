# Experimental Setup
*(Draft — Research Paper: "Optimizing Communication Efficiency and Latency in Blockchain-Enabled Federated Learning Systems")*

---

## III. Experimental Setup

### A. Hardware and Software Environment

All experiments were conducted on a single commodity workstation running **Ubuntu 22.04 LTS** on a 64-bit x86 processor with no dedicated GPU. All model training and inference ran exclusively on CPU, demonstrating that the proposed HCFL framework is deployable on standard banking server hardware without specialised accelerators.

The software stack consisted of the following components:

| Component | Version | Role |
|---|---|---|
| Python | 3.12 | FL pipeline, API client, training |
| PyTorch | 2.x (CPU build) | LSTMTabular model, training, serialisation |
| Hyperledger Fabric | 2.5 | Permissioned blockchain substrate |
| Go | 1.21 | CBFT chaincode (`cbft-fl`) |
| IPFS (Kubo) | 0.27 | Off-chain model weight storage |
| FastAPI + Uvicorn | 0.110 / latest | Async REST gateway (Python ↔ Fabric) |
| Docker + Compose | 24.x / 2.x | Container orchestration for bank nodes |
| CouchDB | 3.x | Hyperledger Fabric world-state database (per peer) |
| scikit-learn | 1.x | PR-AUC, ROC-AUC, F1 metric computation |
| pytest | 8.x | Unit and integration test runner |

The Hyperledger Fabric network ran entirely within Docker containers on a single-host Docker bridge network (`fabric_network`), with 16 containers total: 3 orderers, 6 peers (2 per bank), 6 CouchDB instances, 1 CLI container, 1 IPFS daemon, and 1 FastAPI server.

---

### B. Dataset

We evaluate the proposed framework on the **Kaggle Credit Card Fraud Detection dataset** [CITE], a widely-used benchmark for tabular fraud detection systems. The dataset contains **284,807 credit card transactions** collected over two days in September 2013 by European cardholders, of which **492 (0.173%) are fraudulent**. The features consist of:

- **V1–V28**: 28 numerical features produced by PCA transformation to anonymise sensitive cardholder information.
- **Amount**: the transaction amount, not PCA-transformed.
- **Time**: the elapsed seconds from the first transaction in the dataset. **Dropped during preprocessing** to prevent temporal leakage between train/test splits.
- **Class**: the binary label (0 = legitimate, 1 = fraud).

This results in **29 input features** ($d = 29$) used during training, matching `MODEL_CFG = {input_dim: 29, hidden_dim: 30}` in the benchmarking script.

#### Dataset Preprocessing and Non-IID Partitioning

The preprocessing pipeline produces the following directory structure before FL training begins:

```
data/
├── raw/creditcard.csv                     ← original Kaggle dataset
├── splits/
│   ├── fl_clients/
│   │   ├── BankA/train_ready.csv          ← BankA's non-IID training silo
│   │   ├── BankB/train_ready.csv
│   │   └── BankC/train_ready.csv
│   └── test/global_test.csv              ← held-out global test set (not seen during training)
```

The partitioning follows three steps:
1. **Stratified bank split**: the dataset is divided among $K = 3$ banks using a stratified split that preserves the overall fraud ratio across banks but introduces heterogeneity in transaction volume and feature distributions, simulating real-world differences in customer demographics and spending patterns.
2. **Per-branch local balancing**: within each bank, the minority class (fraud) is oversampled locally to ensure no branch silo is entirely devoid of fraud labels, which would cause degenerate local training.
3. **Per-branch local scaling**: each branch applies its own `StandardScaler` (zero-mean, unit-variance) fitted only on its own training data, preserving privacy by ensuring no branch can infer global feature statistics from shared scaler parameters.

The global held-out test set (`global_test.csv`) is reserved exclusively for post-aggregation model evaluation and is not used during training or within the CBFT validation gate.

---

### C. Federated Learning Configuration

The FL hyperparameters, loaded from `fl-integration/config/fl_config.yaml`, are listed below:

| Parameter | Value | Description |
|---|---|---|
| `local_epochs` | 2 | Training epochs per branch per round |
| `learning_rate` | 0.001 | Adam optimiser learning rate |
| `batch_size` | 256 | Mini-batch size for local training |
| `l2_norm_clip` | 1.0 | DP gradient clipping norm ($\Delta$) |
| `noise_multiplier` | 0.05 | DP Gaussian noise scale ($\sigma$) |
| `deadline_seconds` | 5.0 | Tier-1 branch collection timeout |
| `min_branches_required` | 2 | Minimum branch submissions before Tier-1 FedAvg proceeds |
| `validation_threshold` | 0.20 | Tier-1 PR-AUC gate threshold ($\tau_{\text{val}}$) |
| `backup_beta` | 0.30 | Model blending coefficient ($\beta$); proportion from prior global |
| `val_score_threshold` | 0.70 | On-chain Tier-2 CBFT PR-AUC gate (chaincode constant) |
| `verify_quorum` | 2 | Minimum positive CBFT votes for consensus |
| `commit_quorum` | 2 | Minimum commits for global acceptance |
| `consensus_timeout` | 120.0 s | Maximum wait for global consensus before partial aggregation |
| `poll_interval` | 2.0 s | Global aggregator consensus polling interval |
| Number of FL rounds | 10 | Total training rounds in the full benchmark |
| Number of banks ($K$) | 3 | BankA, BankB, BankC |
| Branches per bank ($M_k$) | 3 | Branch 0 (HQ), Branch 1 (Backup), Branch 2 (training-only) |

#### Model Architecture

| Parameter | Value |
|---|---|
| Model class | `LSTMTabular` |
| Input dimension | 29 |
| Hidden dimension | 30 |
| LSTM layers | 1 |
| Output | Single logit → sigmoid fraud probability |
| Loss function | `BCEWithLogitsLoss` with dynamic `pos_weight = n_neg / n_pos` |
| Model size (serialised) | ~0.49 MB |
| Total parameters | ~10,000 |

---

### D. Blockchain Network Configuration

The Hyperledger Fabric network was configured with three organisations (BankA, BankB, BankC), each with two peers (HQ and Backup), and three Raft orderer nodes providing crash fault-tolerant block ordering.

| Configuration Item | Value |
|---|---|
| Fabric version | 2.5 |
| Channel name | `fraud-detection-global` |
| Organisations | BankA (BankAMSP), BankB (BankBMSP), BankC (BankCMSP) |
| Peers per org | 2 (peer0 = HQ/Anchor, peer1 = Backup) |
| Orderer nodes | 3 (Raft consensus) |
| Endorsement policy | 2-of-3 MSPs must endorse |
| State database | CouchDB (one instance per peer) |
| Chaincode | `cbft-fl` (Go, Fabric Contract API) |
| Chaincode channel | `fraud-detection-global` |
| Block size limit | 1 MB (default) |
| Trust score initialisation | 1.0 per bank |
| Trust reward ($\alpha$) | 0.1 per accepted round |
| Trust penalty ($\beta$) | 0.2 per rejected round |
| Trust score floor | 0.1 |
| Trust score ceiling | 3.0 |

---

### E. Evaluation Metrics

The framework is evaluated on both **model quality** and **system efficiency** metrics, measured at the end of each FL round after global aggregation.

**Model quality metrics** are computed by loading the freshly-committed global model from IPFS and evaluating it on the **held-out global test set** (`global_test.csv`) using 100% of the test data (`sample_fraction=1.0`):

| Metric | Formula | Rationale |
|---|---|---|
| **PR-AUC** | Area under Precision-Recall curve | Primary metric; robust to class imbalance; threshold-free |
| **ROC-AUC** | Area under Receiver Operating Characteristic curve | Standard secondary metric; reported for comparison |
| **F1 Score** | $2 \cdot \frac{P \cdot R}{P + R}$ at threshold 0.5 | Combined precision-recall measure |
| **Precision** | $\text{TP} / (\text{TP} + \text{FP})$ | Fraction of fraud alerts that are genuine |
| **Recall** | $\text{TP} / (\text{TP} + \text{FN})$ | Fraction of actual frauds detected |

**System efficiency metrics** are computed from the benchmarking script's timing and payload measurements:

| Metric | Formula | Rationale |
|---|---|---|
| **Communication cost (MB/round)** | $\sum_k \text{upload}_k + \sum_{k} \text{download}_k + \text{global upload}$ | Total model bytes transferred across all banks per round |
| **End-to-end (E2E) latency (s)** | Wall-clock time from round start to global model publication | Measures operational feasibility; SLA bound = 120 s |

The per-round communication cost is computed as:

$$C_{\text{round}} = K \cdot s_m + K(K-1) \cdot s_m + K \cdot s_m + s_m$$

where $s_m \approx 0.49$ MB is the serialised model size and the four terms correspond to: (1) bank uploads to IPFS, (2) cross-bank CBFT verification downloads (each bank downloads $K-1$ peers' models), (3) global aggregator downloads, and (4) global model upload. With $K = 3$, this yields $C_{\text{round}} = (3 + 6 + 3 + 1) \times 0.49 \approx \mathbf{6.37}$ MB total bytes moved, of which only 46-byte CID strings traverse the Fabric ledger.

---

### F. Validation and Testing Protocol

The framework was validated through eleven structured phases before running the final benchmark:

| Phase | Component Tested | Outcome |
|---|---|---|
| Phase 1 | Fabric network bootstrap (peers, orderers, channel creation) | 6 peers joined, anchor peers configured ✓ |
| Phase 2 | CBFT chaincode deployment & unit tests (`go test`) | All chaincode functions pass (0.004 s) ✓ |
| Phase 3 | IPFS storage layer (`test_ipfs_client.py`) | 3 tests passed (2.15 s), including 10 MB model upload ✓ |
| Phase 4 | FastAPI REST gateway (`test_main.py`) | 16 tests passed (4.88 s), including cross-bank spoofing rejection ✓ |
| Phase 5 | FL layer functions in isolation (`fl-layer` pytest) | 10 tests passed (12.55 s): LSTM forward pass, FedAvg, DP, blending ✓ |
| Phase 6 & 9 | HQ Agent integration (`test_hq_agent.py`) | 3 tests passed (1.22 s): training, upload, cross-verification logic ✓ |
| Phase 10.1 | Dynamic org addition (`addOrg.sh BankD`) | BankD peer joined and approved chaincode on live network ✓ |
| Phase 10.2 | HQ failover (`ActivateBackup` chaincode) | Backup activation confirmed on-chain with status 200 ✓ |
| Phase 10.3 | Raft orderer fault tolerance (leader crash) | `orderer1` became leader at term 5; channel transitioned seamlessly ✓ |
| Phase 10.4 | 10-round performance benchmark (`run_10_rounds.py`) | Avg latency 33–37 s/round; comm cost 0.49 MB; SLA met ✓ |
| Phase 10.5 | FastAPI load test (Locust, 50 users) | Zero failures under sustained concurrent load ✓ |
| **Phase 11** | **End-to-end live CCFD run (real data, 10 rounds)** | **PR-AUC 0.9694, F1 0.8667, E2E latency 33.00 s ✓** |

The Phase 11 end-to-end run constitutes the primary empirical result of this paper. All reported metrics in Section IV are taken from this phase.

---

> **Note to authors**: The `[CITE]` for the Kaggle CCFD dataset should point to: *Dal Pozzolo, A. et al. (2015). Calibrating Probability with Undersampling for Unbalanced Classification. In 2015 IEEE Symposium Series on Computational Intelligence (SSCI)*. Alternatively, cite the dataset directly: *ULB Machine Learning Group, "Credit Card Fraud Detection," Kaggle, 2016.*
