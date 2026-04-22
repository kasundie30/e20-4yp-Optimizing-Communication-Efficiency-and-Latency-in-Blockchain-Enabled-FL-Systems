# HCFL Framework: Structural and Operational Flow

This document provides a comprehensive analysis of the Hierarchical Clustered Federated Learning (HCFL) framework designed for Credit Card Fraud Detection. It details how the system is initialized and traces the exact flow of data and execution across its two distinct aggregation tiers.

## 1. System Initialization

The initialization of the HCFL framework is divided into two main stages: **Infrastructure Bootstrapping** and **Machine Learning Orchestration**.

### 1.1. Infrastructure Bootstrapping (Backend Layer)
The backend decentralized architecture is strictly initialized via the root `./start_system.sh` script. This script orchestrates three critical components sequentially:
1.  **Hyperledger Fabric Network**: Spun up using `network.sh` and `createChannel.sh`. It establishes a permissioned blockchain consortium consisting of 3 organizations (Bank A, Bank B, Bank C), each with 2 peers (HQ and Backup), backed by a Raft consensus orderer service. The custom `cbft-fl` Go chaincode is packaged, installed on all peers, approved by the consortium, and committed to the `fraud-detection-global` channel.
2.  **IPFS Off-Chain Storage**: The IPFS daemon (`kubo`) is started locally. This serves as the decentralized, distributed file system for the heavy ML model weights, ensuring the blockchain only handles lightweight cryptographic hashes and metadata, thereby avoiding block size limits and reducing latency.
3.  **FastAPI REST Server**: An asynchronous Python `uvicorn` server is launched (`api-server/main.py`). This microservice acts as the crucial I/O bridge between the CPU-heavy Python FL agents and the Go-based Fabric CLI, handling ledger reads and writes asynchronously to prevent training bottlenecks.

### 1.2. Machine Learning Orchestration (FL Layer)
Once the backend is healthy and reporting status, the Machine Learning pipeline is initialized. This is done either via distributed Docker containers (`docker compose up` inside the `fl-integration/` directory) or through an automated evaluation script like `run_10_rounds.py`. 
Upon startup, each Bank initializes an `HQAgent` process and prepares its simulated local `Branch` nodes with preprocessed, non-IID partitioned datasets representing highly imbalanced real-world credit card transactions.

---

## 2. Two-Tiered Aggregation Architecture

The framework is explicitly engineered with a two-tiered hierarchy to drastically minimize global communication overhead, mitigate the impact of slow nodes (stragglers), and maintain robust security against Byzantine (malicious) actors.

### Tier 1: Intra-Cluster Aggregation (Branch to HQ)
This tier operates entirely within the secure, private boundaries of a single organization (e.g., the isolated intranet of Bank A). Data privacy is absolute, as raw transaction logs never leave the local branch.

1.  **Local Branch Training**: 
    *   **Entity**: Local Branch Nodes (e.g., regional branches).
    *   **Process**: Each branch retrieves the latest Global Model. If it is Round 1, it initializes a standardized random weight state. The branch trains a PyTorch `LSTMTabular` model on its localized dataset for a specified number of local epochs using a differential privacy-aware training loop (`local_train.py`).
    *   **Network Flow**: Branch Node $\rightarrow$ HQ Agent. Branches transmit their updated model weights (`state_dict`) inward to their respective Headquarter (HQ) server.

2.  **HQ Aggregation (Cluster Model Generation)**:
    *   **Entity**: Organization HQ Agent.
    *   **Process**: The HQ waits for updates from its branches, utilizing a strict timeout mechanism (`deadline_collect.py`) to bypass unresponsive branches and prevent stalling. Once the deadline passes or all updates arrive, the HQ performs a synchronous Federated Averaging (`fedavg.py`) across the branch weights to synthesize a unified **Cluster Model**.
    *   **Validation**: The HQ evaluates this synthesized Cluster Model against an internal validation set to ensure its PR-AUC score meets the baseline threshold $\tau$.

### Tier 2: Inter-Cluster Aggregation (HQ to Blockchain to Global)
This tier operates across the decentralized, trustless consortium of different Banks. It utilizes the blockchain ledger for verifiable consensus and IPFS for peer-to-peer transport. 

3.  **Cluster Model Registration (Off-chain Storage & On-chain Hash)**:
    *   **Entity**: HQ Agent $\rightarrow$ IPFS $\rightarrow$ Fabric Chaincode.
    *   **Process**: The HQ uploads its evaluated Cluster Model directly to IPFS, receiving a unique Content Identifier (CID). It then computes a rigid SHA-256 cryptographic hash of the model byte stream.
    *   **Network Flow**: The HQ requests the API (`POST /submit-update`), passing its CID, Hash, and local validation score. The REST API submits this transaction to the chaincode, recording the Bank's update proposal immutably on the ledger.

4.  **Consensus Based Federated Trust (CBFT) Cross-Verification**:
    *   **Entity**: All peer HQ Agents across the consortium.
    *   **Process**: 
        *   **Fetch**: Each HQ polls the ledger to discover cluster updates submitted by the *other* participating banks.
        *   **Download & Verify**: The HQ downloads the peer's model from IPFS using the provided CID. Crucially, it recalculates the SHA-256 hash locally and compares it to the hash on the blockchain to ensure data integrity (preventing man-in-the-middle manipulation).
        *   **Fractional Validation**: The HQ evaluates the peer's model on its own validation data utilizing a rapid fractional sample (`validate_fast.py`). If the performance is acceptable, it casts a `True` vote; if the performance is poor, or if the hash mismatches indicating tampering, it casts a `False` vote.
    *   **Network Flow**: HQ $\rightarrow$ API $\rightarrow$ Chaincode (`POST /submit-verification`). The chaincode transparently tallies these boolean votes.

5.  **Commitment & Global Aggregation**:
    *   **Entity**: Global Aggregator (an elected routine executed when consensus is finalized).
    *   **Process**: 
        *   When a proposed cluster model receives a quorum of positive validations (e.g., $vCount \ge 2$), it is marked as "Accepted". The proposing HQ agent invokes `POST /submit-commit` to lock the model on the ledger as officially verified.
        *   The **Global Aggregator** module polls the ledger (`GET /check-consensus`) to retrieve the list of all Accepted cluster CIDs for the current round. It downloads them from IPFS.
        *   **Trust-Weighted FedAvg**: The aggregator executes the final inter-cluster merge. Instead of treating all banks equally, it retrieves the historical **Trust Scores** of each bank from the Fabric ledger. Models originating from highly trusted, historically accurate banks are weighted more heavily in the final average than models from unverified or penalized banks.
    *   **Network Flow**: Global Aggregator $\rightarrow$ IPFS $\rightarrow$ Blockchain. The newly minted Global Model is uploaded to IPFS.

6.  **Round Finalization & Dynamic Progression**:
    *   **Entity**: Chaincode.
    *   **Process**: The aggregator triggers `StoreGlobalModel` on the chaincode. This saves the exact Global CID to the ledger and unequivocally updates the `latest_round` pointer in the world state. 
    *   **Flow**: The system dynamically increments. The next training round initializes, automatically querying `GET /latest-round`, and the pipeline recurses back to Tier 1, with local branches fetching the fresh Global CID to begin their next epoch.
