___
# HBFL-CCFD: Hierarchical Blockchain enabled Federated Learning framework for Credit Card Fraud Detection
___


##  Overview
This project introduces **HBFL-CCFD: Hierarchical Blockchain enabled Federated Learning framework for Credit Card Fraud Detection** — a two-tier federated learning framework designed for **secure and efficient credit card fraud detection** across multiple banks.

It combines:
- Federated Learning (FL)
- Blockchain (Hyperledger Fabric)
- Differential Privacy (DP)
- IPFS off-chain storage
- Byzantine Fault Tolerance (CBFT)

to address **privacy, trust, communication efficiency, and auditability** in collaborative banking systems.

---

##  Key Features
-  **Privacy Preservation** – Differential Privacy (DP-SGD)
-  **Hierarchical FL Architecture** – Branch → HQ → Global
-  **Blockchain Integration** – Immutable audit trail & trust tracking
-  **CBFT Consensus** – Prevents malicious model updates
-  **IPFS Storage** – Efficient off-chain model handling
-  **Optimized Performance** – Low latency & constant communication cost

---

##  System Architecture
<img width="1348" alt="solution architecture" src="https://github.com/cepdnaclk/e20-4yp-Optimizing-Communication-Efficiency-and-Latency-in-Blockchain-Enabled-FL-Systems/blob/main/docs/images/System_overview.png "/>

### Tier 1: Intra-Bank Layer
- Branch nodes train local models
- HQ aggregates updates (FedAvg)
- Backup node ensures fault tolerance

### Tier 2: Inter-Bank Layer
- HQ nodes connected via blockchain
- Perform validation, consensus, and global aggregation

<img width="1348" alt="high level architecture" src="https://github.com/cepdnaclk/e20-4yp-Optimizing-Communication-Efficiency-and-Latency-in-Blockchain-Enabled-FL-Systems/blob/main/docs/images/Two_tier_architecture.png" />

<img width="1348" alt="work flow" src="https://github.com/cepdnaclk/e20-4yp-Optimizing-Communication-Efficiency-and-Latency-in-Blockchain-Enabled-FL-Systems/blob/main/docs/images/Workflow_1.png" />

---

##  Tech Stack
- **PyTorch** – Model training
- **Hyperledger Fabric** – Blockchain network
- **Go (Chaincode)** – Trust & validation logic
- **IPFS (Kubo)** – Off-chain storage
- **FastAPI** – API layer
- **Docker** – Deployment

---

##  Dataset
- European Credit Card Fraud Detection Dataset
- 284,807 transactions
- Highly imbalanced (~0.17% fraud)
- Non-IID distributed across banks

---

##  Model Details
- LSTM-based tabular model (~50K parameters)
- Loss: BCEWithLogits (class-weighted)
- Optimizer: Adam
- 1 epoch per federated round

---

##  Results (10 Federated Rounds)
| Metric        | Value |
|--------------|------|
| PR-AUC       | 0.778 |
| ROC-AUC      | 0.974 |
| F1 Score     | 0.830 |
| Precision    | 0.905 |
| Comm/round   | 0.490 MB |
| Latency      | ~3.55s (sim) / ~35s (live) |

---

##  Comparison Highlights
- ✔️ Lower latency than existing blockchain-FL frameworks
- ✔️ Constant communication overhead (no growth per round)
- ✔️ Higher precision (reduces false fraud alerts)
- ✔️ Added security via CBFT (absent in baselines)

---

##  Contributions
- Novel **two-tier hierarchical FL architecture**
- Blockchain-based **model verification & auditability**
- Integration of **Differential Privacy in FL**
- Efficient **communication and latency optimization**
- Secure **trust-based aggregation using CBFT**

---

##  Limitations
- Fixed number of participating banks
- No explainability (XAI) support
- DP budget not formally tracked
- Higher latency in live blockchain setup

---

##  Conclusion
HCFL delivers a **secure, scalable, and privacy-compliant** framework for fraud detection in banking systems by unifying federated learning, blockchain, and differential privacy.


