# Introduction
*(Draft — Research Paper: "Optimizing Communication Efficiency and Latency in Blockchain-Enabled Federated Learning Systems")*

---

## I. Introduction

The rapid proliferation of digital financial services has produced an unprecedented volume of transaction data, creating both opportunity and obligation for financial institutions to develop robust fraud detection mechanisms. Credit card fraud alone inflicts hundreds of billions of dollars in global losses annually, motivating a sustained research effort into machine learning-based detection systems [CITE]. Historically, the most accurate models have been trained on large, centralised repositories of labelled transaction data. However, the sensitive and regulated nature of customer financial records makes data centralisation legally and commercially untenable under frameworks such as the General Data Protection Regulation (GDPR) and the Basel III banking accords. Federated Learning (FL) has emerged as a promising paradigm to resolve this tension: participating institutions collaboratively train a shared model by exchanging only model parameter updates rather than raw data, thereby preserving local data privacy while still benefiting from knowledge distributed across the federation [CITE McMahan et al., 2017].

Despite its conceptual appeal, the deployment of federated learning in real-world multi-institutional settings—particularly in the banking sector—faces three interlocking challenges that remain inadequately addressed in the literature.

**Communication overhead and scalability.** In the canonical flat FL topology, every participating client transmits a full copy of its model to a central aggregation server at each training round. When the federation spans dozens of branches across multiple banks, this results in a communication bottleneck proportional to the number of participants and the model size. Existing work has explored gradient compression [CITE], sparse update [CITE], and quantisation [CITE] to mitigate this overhead, but these approaches often compromise model accuracy or require specialised hardware. A hierarchical aggregation strategy—wherein a first aggregation step occurs locally within each institution before an inter-institutional aggregation—can reduce cross-organisational traffic by an order of magnitude without sacrificing model fidelity. However, hierarchical FL approaches have rarely been studied in the context of real financial datasets or integrated with trust mechanisms.

**Trust, integrity, and Byzantine resilience.** A central aggregator constitutes a single point of trust; if it is compromised, or if a malicious participant submits poisoned gradient updates, the integrity of the global model is destroyed. Byzantine-robust aggregation rules such as Krum [CITE] and Bulyan [CITE] offer statistical defences but rely on assumptions about the fraction of malicious clients that may not hold in an open consortium. Blockchain technology offers an alternative trust model: by recording every model submission, validation vote, and aggregation outcome on an immutable, permissioned ledger, it enables any consortium member to independently audit the entire history of training. Prior work combining blockchain and FL [CITE Nguyen et al., 2021; Li et al., 2020] has demonstrated the feasibility of on-chain gradient verification, but has typically overlooked the latency costs of storing large model weight files directly on the ledger, or replaced statistical validation with computationally expensive cryptographic proofs.

**Latency and real-time operational constraints.** Fraud detection is a latency-sensitive application: a model that takes minutes per training round is of limited operational value in an environment where transaction patterns shift in near real-time. The overhead of blockchain consensus, cross-institution communication, and model evaluation must collectively fit within a time budget acceptable to banking operations. Existing blockchain-FL proposals often report convergence results but neglect end-to-end round-trip latency as a primary design criterion [CITE].

To address these three challenges simultaneously, this paper presents a **Hierarchical Clustered Federated Learning (HCFL)** framework specifically designed for blockchain-enabled multi-bank fraud detection. Our framework integrates:

1. **A two-tier aggregation hierarchy** that separates intra-bank branch aggregation from inter-bank global aggregation, reducing the number of model transmissions to the blockchain by a factor equal to the number of branches per bank.

2. **Hyperledger Fabric** as a permissioned blockchain substrate, providing identity-authenticated participation (via Membership Service Providers), immutable audit logs, and deterministic smart contract execution without proof-of-work overhead.

3. **IPFS-based off-chain model storage**, where the blockchain ledger records only a 46-byte Content Identifier (CID) and a SHA-256 integrity hash per model, while full weight tensors are held in a distributed peer-to-peer store. This circumvents the block-size limitations of Hyperledger Fabric (default 1 MB per block) for the ~0.5 MB model files used in our experiments.

4. **A Consensus-Based Federated Trust (CBFT) protocol**, a lightweight three-phase (propose–verify–commit) on-chain consensus mechanism in which each bank independently downloads and evaluates peer models from IPFS before casting a verifiable vote. CBFT achieves Byzantine fault tolerance with O(N) vote messages, compared to the O(N²) complexity of classical PBFT, while providing stronger fraud detection guarantees than self-reported validation scores.

5. **A dynamic trust score system** that accumulates each bank's historical contribution quality on-chain and uses it as a multiplier in the inter-bank Federated Averaging step. Banks that consistently contribute high-quality updates gain proportionally greater influence; poor-quality banks are progressively down-weighted without full exclusion, preserving a floor of participation.

6. **A suite of latency-reduction optimisations**, including deadline-based straggler handling, 15% fractional validation sampling, backup HQ failover with model blending, and an asynchronous FastAPI gateway that decouples the Python training pipeline from the Go-based Fabric CLI.

We evaluate the framework on a non-IID partition of the Kaggle Credit Card Fraud Detection dataset [CITE], a standard benchmark with 284,807 transactions and a severe 0.17% fraud rate. The model is a single-layer LSTM (LSTMTabular) trained with Gaussian differential privacy noise and evaluated using Precision-Recall AUC (PR-AUC) as the primary metric—a choice motivated by the well-established inadequacy of ROC-AUC for heavily imbalanced classification tasks [CITE]. Our experiments demonstrate that the HCFL framework achieves a PR-AUC of **0.9694** and a global F1 score of **0.8667** on the held-out validation data by Round 10, with an average end-to-end round latency of **33–37 seconds** and a per-round communication overhead of only **0.49 MB**—well within the 120-second SLA boundary imposed by our consortium design.

The principal contributions of this paper are as follows:

- **A hierarchical FL topology for banking consortia** that reduces inter-bank communication overhead relative to flat FL without degrading model performance, validated on a real credit card fraud dataset partitioned into heterogeneous (non-IID) bank silos.

- **CBFT: a lightweight, on-chain Byzantine-fault-tolerant consensus mechanism for FL** that replaces the expensive multi-round messaging of classical BFT protocols with a single-vote-per-bank quorum check, while preventing self-verification and replay attacks via chaincode-enforced constraints.

- **An IPFS-backed hybrid storage architecture** for blockchain-integrated FL that resolves the ledger bloat problem by separating model integrity guarantees (on-chain CID + hash) from model storage (IPFS), enabling Hyperledger Fabric to function as a verifiable coordination layer rather than a file store.

- **A dynamic, on-chain trust scoring system** integrated into the FedAvg aggregation weighting, providing a continuous, tamper-evident incentive mechanism aligned with long-term consortium participation quality.

- **An open-source, end-to-end experimental framework** combining Hyperledger Fabric 2.5, IPFS (Kubo), PyTorch, FastAPI, and Docker Compose, fully reproducible on CPU-only commodity hardware, enabling future researchers to benchmark blockchain-integrated FL under realistic multi-bank conditions.

The remainder of this paper is organised as follows. Section II reviews related work on federated learning for fraud detection, blockchain-integrated FL, and hierarchical FL architectures. Section III describes the system model and threat assumptions. Section IV presents the proposed HCFL framework in detail, covering the two-tier topology, the CBFT protocol, the IPFS storage layer, and the trust scoring mechanism. Section V describes the experimental setup, dataset partitioning strategy, and evaluation metrics. Section VI reports and analyses the experimental results. Section VII discusses limitations and directions for future work. Section VIII concludes the paper.

---

> **Note to authors**: Placeholder citations marked `[CITE]` should be resolved before submission. Key references to confirm:
> - McMahan et al., *Communication-Efficient Learning of Deep Networks from Decentralized Data* (FedAvg, AISTATS 2017)
> - Nguyen et al., *Federated Learning Meets Blockchain in Edge Computing* (IoT, 2021)
> - Li et al., *A Survey on Federated Learning Systems* (TIST, 2021)
> - Davis & Goadrich, *The Relationship Between Precision-Recall and ROC Curves* (PR-AUC, ICML 2006)
> - Blanchard et al., *Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent* (Krum, NeurIPS 2017)
> - El Mhamdi et al., *The Hidden Vulnerability of Distributed Learning in Byzantium* (Bulyan, ICML 2018)
