# Blockchain Layer Implementation Overview

This document provides a comprehensive overview of the Hyperledger Fabric blockchain layer implementation located in the `fabric-network` directory. This layer is designed to support a Hierarchical Clustered Federated Learning (HCFL) system for fraud detection among three participating banks.

## 1. Network Topology and Configuration

The network is configured to support a permissioned blockchain representing a consortium of three banking organizations and one ordering service. 

- **Organizations:** 
  - `BankA`, `BankB`, `BankC` (Peer Organizations)
  - `OrdererOrg` (Ordering Service Organization)
- **Channel:** `fraud-detection-global`
- **Consensus Type:** Raft (Crash Fault Tolerant ordering service)
- **State Database:** CouchDB (used to support rich JSON state queries)
- **Endorsement Policy:** Requires signatures from at least 2 out of 3 banks (`OutOf(2, 'BankAMSP.peer', 'BankBMSP.peer', 'BankCMSP.peer')`).

### Nodes in the Network:
- **Orderer Nodes:** 3-node Raft cluster (`orderer0`, `orderer1`, `orderer2`) to provide a fault-tolerant ordering service.
- **Peer Nodes:** 6 peers in total (2 per bank). 
  - `peer0` serves as the Headquarter (HQ) and acts as the anchor peer and primary committee member for each bank.
  - `peer1` serves as the Backup HQ, ensuring high availability if the primary peer fails.
- **State Databases:** 6 CouchDB instances, one associated with each peer.

## 2. Infrastructure Management Scripts

The `scripts/` directory provides essential lifecycle management scripts:

- **`network.sh`**: The main entry point for managing the network lifecycle (`up`, `down`, `teardown`, `status`). It uses `cryptogen` to generate cryptographic certificates from `crypto-config.yaml` and `configtxgen` to create the genesis block and channel transaction artifacts based on `configtx.yaml`. It boots up the containers using Docker Compose.
- **`deployChaincode.sh`**: Handles the Fabric 2.x chaincode lifecycle. It packages the `cbft-fl` chaincode, installs it across all 6 peers, approves the chaincode definition on behalf of the three banks, commits the definition to the channel, and finally invokes the `InitLedger` function to bootstrap initial state data (trust scores).
- **`createChannel.sh` / `addOrg.sh`**: Scripts to join the peers to the `fraud-detection-global` channel and optionally manage organizational scaling.

## 3. Smart Contract (Chaincode) Implementation

The core business logic is implemented in Go within `chaincode/cbft/cbft.go`. The chaincode manages the global state of the federated learning process, enforces consensus during model updates, and tracks the reputation (trust score) of each bank.

### High-level Logic and Data Models:
1. **Trust Scores (`TrustScore`)**: 
   - Dynamically tracks the reputation of each bank. 
   - Initialized at `1.0` and bounded by a minimum of `0.1` (`ScoreMin`).
   - Trust scores are adjusted (`UpdateTrustScore`) by granting a reward ($\alpha = 0.1$) for improving the global model's validation performance and applying a penalty ($\beta = 0.2$) for degrading it.
   - The method `GetTrustScores` allows external applications to retrieve these scores to compute the trust-weighted aggregated global model in Level-2 (inter-cluster) aggregation.

2. **Cluster Updates (`ClusterUpdate`)**: 
   - Banks submit their locally aggregated intra-cluster models via `SubmitClusterUpdate`. 
   - Records include the training round, an off-chain IPFS content identifier (`ModelCID`), a SHA-256 hash of the model for integrity verification, and a validation score (`ValScore`). 
   - The chaincode enforces a strict cluster-level validation threshold cutoff (e.g., F1-score $\ge 0.7$); updates below this threshold are outright rejected.

3. **CBFT Consensus Mechanism**:
   The chaincode enacts a Cluster-Based Byzantine Fault Tolerance (CBFT)-styled verification protocol consisting of two voting phases:
   - **Phase 2 - Verification (`SubmitVerification`)**: Verifier banks evaluate the submitted cluster update. A minimum quorum of positive verification votes (`VerifyQuorum = 2`) is required.
   - **Phase 3 - Commit (`SubmitCommit`)**: Committer banks finalize the acceptance of the target bank's update. This requires checking that enough verifications were received. Commit quorum is also set to 2 (`CommitQuorum = 2`).
   - The combined consensus outcome is finalized through the `CheckConsensus` transaction which marks the updates as globally accepted.

4. **Global Model Management**:
   - `StoreGlobalModel` and `GetGlobalModel` transactions are used to persist and retrieve the immutable record (CID and hash) of the final global model produced in each FL training round.

5. **Fault Tolerance and Failover**:
   - The `ActivateBackup` transaction allows failover. If a bank's HQ (`peer0`) goes offline, the Backup HQ (`peer1`) steps in to submit updates, and the chaincode explicitly records `BackupActive: true` within the ledger state.

## Summary

The blockchain layer uses Hyperledger Fabric to construct an enterprise-grade trust component for the Hierarchical Clustered Federated Learning system. It reliably captures intermediate cluster updates, robustly enforces cross-verification consensus before merging, maintains an immutable ledger of all round parameters (CIDs and hashes for models stored off-chain), and autonomously manages a trust economy that penalizes underperforming or malicious nodes.
