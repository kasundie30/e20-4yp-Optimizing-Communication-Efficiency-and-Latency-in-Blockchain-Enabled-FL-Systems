# HCFL Framework Documentation

This repository contains the Hierarchical Clustered Federated Learning (HCFL) framework integrated with a Hyperledger Fabric blockchain.

## Directory Structure & Purpose

### 1. `fabric-network/`
**Purpose:** The foundation. Contains the Hyperledger Fabric blockchain network.
- **`crypto-config/` & `channel-artifacts/`**: Certificates and genesis blocks for BankA, BankB, BankC, and the Orderer consensus cluster.
- **`docker-compose.yaml`**: Defines the 16 containers (Peers, CouchDBs, Orderers, CLIs) that run the permissioned network.
- **`chaincode/cbft/`**: The Go-based Smart Contract. Enforces the 3-phase Comprehensive Byzantine Fault Tolerance (CBFT) protocol, stores IPFS CIDs, and maintains mathematical Trust Scores.
- **`start_system.sh` / `stop_system.sh`**: Root level Bash wrappers that automate the cold-start and teardown process for the Fabric blockchain, IPFS daemon, and FastAPI gateway layer simultaneously.

### 2. `api-server/`
**Purpose:** The Bridge. A FastAPI REST interface.
- Python ML code cannot natively talk to Hyperledger Fabric Go code easily. This server runs alongside each HQ node, providing standard REST endpoints (`POST /submit-update`, `GET /trust-scores`). It uses standard HTTP and executes Docker CLI commands via `fabric_client.py` under the hood.

### 3. `ipfs/`
**Purpose:** Distributed Model Storage.
- Machine learning models form heavy `.pt` files. The blockchain cannot store megabytes of unstructured weights. The IPFS daemon stores the actual weights and returns a CID (Content Identifier) hash. Only the tiny CID strings are persisted on the blockchain.

### 4. `fl-layer/`
**Purpose:** Pure Machine Learning Physics Engine.
- Contains only PyTorch, Scikit-Learn, and numpy math. It has **zero knowledge** of the blockchain or Docker. 
- **`model/`**: LSTM architecture and dataset loaders.
- **`training/`**: Local training with strict Differential Privacy (DP) clipping and noising rules.
- **`aggregation/`**: Weighted Federated Averaging math.
- **`validation/`**: Calculates PR-AUC scores.

### 5. `fl-integration/`
**Purpose:** The HQ Orchestrator.
- Glues the `fl-layer/` math to the `api-server/` APIs.
- **`hq_agent.py`**: Runs a round for a single bank (evaluates local branches, uploads to IPFS, submits to blockchain).
- **`global_aggregator.py`**: Polls the blockchain waiting for 2/3 consensus, downloads models, checks SHA-256 hashes, applies Trust Score scaling mathematically, and creates the single unified Global Model.
- **`round_coordinator.py`**: Enforces strict timeout deadlines. If a branch is slow, it is left behind.
- **`docker-compose.yml`**: Docker wrapper that isolates and spins up 3 decentralized container images `Bank A`, `Bank B`, and `Bank C` sequentially to handle active ML simulations.

### 6. `docs/`
**Purpose:** Output reporting and analysis documentation.
- Stores historical walkthrough logs, integration guides, execution checklists, and architectural evaluations detailing the structural transitions from a legacy monolithic state to the current HCFL framework.

---

## How to Run It Manually (The E2E Pipeline)

You can verify that all 5 layers work interactively by running the automated testing suites in your `.venv`. 

### Prerequisites
Activate the virtual environment where all dependencies (torch, pytest, requests, fastapi) reside:
```bash
source "/media/fyp-group-18/1TB-Hard/FYP-Group18/experiments/6_proposed_framework/.venv/bin/activate"
```

### 1. Verify the Blockchain Layer (Go Chaincode)
Runs the Go mock-stub tests for the Smart Contract (consensus enforcement).
```bash
cd "/media/fyp-group-18/1TB-Hard/FYP-Group18/experiments/6_proposed_framework/fabric-network/chaincode/cbft"
go test -v
```

### 2. Verify the API Server Layer (FastAPI)
Starts an embedded mock network and tests all HTTP endpoints, identity enforcement, and schema validation.
```bash
cd "/media/fyp-group-18/1TB-Hard/FYP-Group18/experiments/6_proposed_framework/api-server"
python -m pytest tests/unit/ -v
```

### 3. Verify the Pure ML Layer
Generates synthetic transaction models and tests differential privacy offsets, model serialization identicality, and FedAvg math in pure isolation.
```bash
cd "/media/fyp-group-18/1TB-Hard/FYP-Group18/experiments/6_proposed_framework/fl-layer"
python -m pytest tests/ -v
```

### 4. Verify the Orchestration Integration Layer
Simulates 3 banks, mock IPFS servers, and mock blockchain polling, ensuring the HQ agents talk to the Global Aggregator correctly on a strict timeline.
```bash
cd "/media/fyp-group-18/1TB-Hard/FYP-Group18/experiments/6_proposed_framework/fl-integration"
python -m pytest tests/ -v
```

### 5. Execute 10 Rounds of Live Decentralized Evaluation
Once all tests succeed, initiate the true environment mapped across distributed containers yielding exact PR-AUC metrics:
```bash
cd "/media/fyp-group-18/1TB-Hard/FYP-Group18/experiments/6_proposed_framework"
./start_system.sh
cd fl-integration
docker compose up --build
```
