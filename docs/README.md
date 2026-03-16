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

Credit card fraud detection increasingly requires collaborative learning across multiple financial institutions in order to capture fraud patterns that are not visible within isolated organizational datasets. However, direct sharing of raw financial transaction data is restricted by privacy regulations, institutional policies, and security concerns. Federated Learning (FL) provides a privacy-preserving alternative by enabling institutions to train models collaboratively without exchanging raw data. Nevertheless, conventional blockchain-enabled federated learning systems often suffer from high communication overhead, increased end-to-end latency, and limited scalability when deployed in realistic multi-institution settings.

This project proposes a **Hierarchical Collaborative Federated Learning (HCFL)** framework for credit card fraud detection, designed to improve communication efficiency, reduce latency, and enhance trust in collaborative training. The framework organizes participants into institutional clusters, where branch-level model updates are first aggregated at headquarters (HQ) nodes before being coordinated globally. To provide decentralized trust and verifiability, the framework integrates **Hyperledger Fabric** as a permissioned blockchain layer. To reduce blockchain storage overhead, model artifacts are stored off-chain using **IPFS**, while only hashes, content identifiers (CIDs), validation outcomes, and trust metadata are recorded on-chain.

The system also incorporates resilience and optimization mechanisms, including deadline-aware collection, validation-based filtering, trust-weighted aggregation, backup failover handling, and lightweight blockchain verification. Experimental results show that the proposed architecture achieves strong fraud detection performance while maintaining low communication cost and acceptable end-to-end latency in a simulated multi-institution environment.

---

## Research Problem

Fraud detection models trained independently within a single financial institution often fail to capture cross-institution fraud patterns. Although federated learning allows collaborative model training without sharing raw data, existing approaches still face several limitations:

- reliance on centralized aggregation servers,
- limited mechanisms for verifying model updates,
- increased communication overhead and latency in blockchain-enabled settings,
- weak resilience against faulty or malicious participants,
- and limited scalability in hierarchical financial networks.

This project addresses the challenge of designing a **trustworthy, efficient, and scalable collaborative fraud detection framework** that preserves privacy while reducing communication overhead and latency.

---

## Objectives

The main objective of this research is to design and evaluate a blockchain-enabled hierarchical federated learning framework for collaborative credit card fraud detection.

The specific objectives are:

- To design a **hierarchical federated learning architecture** that reduces global communication by aggregating updates at institutional HQ level.
- To integrate **Hyperledger Fabric** for decentralized verification, trust management, and tamper-evident logging of model-update events.
- To use **IPFS** for off-chain storage of model artifacts in order to reduce blockchain payload size.
- To incorporate **resilience mechanisms** such as validation filtering, trust scoring, deadline-aware aggregation, and backup failover.
- To evaluate the framework in terms of **fraud detection performance, communication overhead, end-to-end latency, and robustness**.

---

## Methodology

The project follows a staged methodology to separately validate the learning framework and the blockchain-integrated system.

### Stage 1: Hierarchical Federated Learning Core

In the first stage, the core fraud detection pipeline is developed and validated independently of the blockchain layer. This includes:

- local branch-level model training,
- hierarchical aggregation at institutional HQ nodes,
- validation-based filtering of cluster-level models,
- backup recovery via model blending,
- and trust-aware global aggregation.

This stage ensures that the learning framework is numerically correct, robust under non-IID data, and effective for fraud detection before introducing distributed blockchain overhead.

### Stage 2: Blockchain-Enabled HCFL

In the second stage, the HCFL pipeline is integrated with:

- **Hyperledger Fabric** for permissioned verification and trust management,
- **IPFS** for off-chain storage of model artifacts,
- and a **FastAPI-based integration layer** for communication between the machine learning and blockchain subsystems.

This stage evaluates the full distributed system under realistic coordination, verification, and infrastructure constraints.

---

## System Architecture

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

The proposed HCFL framework was evaluated using both machine learning and system-level performance metrics.

### Model Performance

The framework achieved strong fraud detection capability on the held-out test data, with results including:

- **F1 Score:** 0.8667
- **PR-AUC:** 0.9694
- **ROC-AUC:** 0.9758
- **Precision:** 0.8667
- **Recall:** 0.8667

These results indicate that the hierarchical and trust-aware aggregation strategy is effective even in a distributed multi-institution setting.

### System Efficiency

System-level evaluation demonstrated that the proposed design can support collaborative model training with manageable overhead:

- **Communication Cost:** approximately 0.49 MB per round
- **End-to-End Latency:** approximately 33 seconds per round

The hierarchical structure reduced the number of global submissions, while IPFS minimized blockchain payload size by storing only references and integrity metadata on-chain.

### Robustness Validation

The framework was also tested under adverse system conditions, including:

- Byzantine/poisoned model submissions
- replay attack scenarios
- HQ failover conditions
- orderer crash recovery
- dynamic organization addition
- API load testing

These experiments confirmed that the system remains operational and verifiable under fault and attack scenarios.

---

## Impact and Limitations

### Impact

This research contributes to the growing field of secure and scalable collaborative AI for finance by demonstrating that:

- privacy-preserving fraud detection across institutions is feasible,
- hierarchical aggregation can reduce overhead in blockchain-enabled FL,
- and permissioned blockchain infrastructure can provide trust and auditability without storing raw data or full models on-chain.

The proposed framework is relevant not only to credit card fraud detection, but also to other collaborative learning settings in regulated domains such as healthcare, insurance, and cybersecurity.

### Limitations

Despite promising results, several limitations remain:

- the experiments were conducted in a **simulated multi-institution environment** rather than a live banking deployment,
- the number of participating institutions is limited compared to real-world financial ecosystems,
- the model currently uses **single-step tabular LSTM input** rather than longer temporal sequences,
- and the privacy mechanism is **DP-inspired** but does not include formal privacy accounting.

These limitations provide directions for future work.

---

## Conclusion

This project presented a **Hierarchical Collaborative Federated Learning (HCFL)** framework for optimizing communication efficiency and latency in blockchain-enabled federated learning systems for credit card fraud detection. The framework combines hierarchical aggregation, trust-aware global coordination, Hyperledger Fabric–based verification, and IPFS-backed off-chain storage to support privacy-preserving and verifiable multi-institution learning.

The experimental results demonstrate that the proposed architecture can achieve strong fraud detection performance while maintaining low communication overhead and reasonable latency. The framework also improves resilience through validation filtering, trust scoring, and failover support. Overall, this work shows that blockchain-enabled federated learning can be made more practical for financial fraud detection through careful hierarchical design and communication-aware optimization.

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
