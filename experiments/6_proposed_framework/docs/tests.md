# Phase-wise Tests and Verifications

This document outlines the testing procedures executed during the implementation of the HCFL Framework. Each phase contains specific tests designed to validate the functionality of the isolated components and their integration.

---

## Phase 1: Fabric Network Setup

**Purpose:** Ensure the base Hyperledger Fabric infrastructure (Peers, Orderers, CouchDBs) boots successfully from a clean state and forms the `fraud-detection-global` channel without TLS or Raft consensus errors.

**Script Location:** `/media/fyp-group-18/1TB-Hard/FYP-Group18/experiments/6_proposed_framework`

**Command to Run:**
```bash
./start_system.sh
```

**Test Output:**
```text
[INFO]  == Network is UP. Run scripts/createChannel.sh next. ==
...
[INFO]  == Channel setup complete. All 6 peers joined, anchor peers configured. ==
```

---

## Phase 2: CBFT Chaincode Deployment

**Purpose:** Validate the Golang Smart Contract (`cbft-fl`) compiles, passes localized unit testing on the struct functions, and successfully deploys to the 6 peers via the Fabric Lifecycle with all Trust Scores initialized securely.

**Script Location:** `/media/fyp-group-18/1TB-Hard/FYP-Group18/experiments/6_proposed_framework/fabric-network/chaincode/cbft`

**Command to Run (Unit Tests):**
```bash
go test -v ./...
```

**Test Output (Unit Tests):**
```text
=== RUN   TestInitLedger
--- PASS: TestInitLedger (0.00s)
=== RUN   TestSubmitClusterUpdate_Valid
--- PASS: TestSubmitClusterUpdate_Valid (0.00s)
...
PASS
ok      github.com/hyperledger/fabric-samples/chaincode/cbft    0.004s
```

---

## Phase 3: IPFS Storage Layer

**Purpose:** Prove that the local IPFS Kubo daemon correctly scales infinite model sizes independently of Fabric's maximum block limit. Validates the `ipfs_client.py` API wrapper.

**Script Location:** `/media/fyp-group-18/1TB-Hard/FYP-Group18/experiments/6_proposed_framework/ipfs`

**Command to Run:**
```bash
python3 -m pytest test_ipfs_client.py -v
```

**Test Output:**
```text
test_ipfs_client.py::test_ipfs_client_initialization PASS
test_ipfs_client.py::test_upload_and_download_cycle PASS
test_ipfs_client.py::test_performance_10mb_model PASS
======================== 3 passed in 2.15s ========================
```

---

## Phase 4: FastAPI REST Interface

**Purpose:** Validate the API Gateway layer logic using Python's `TestClient` mimicking the JSON payload bindings and testing identity enforcement boundaries (403 Forbidden errors for cross-bank spoofing).

**Script Location:** `/media/fyp-group-18/1TB-Hard/FYP-Group18/experiments/6_proposed_framework/api-server`

**Command to Run:**
```bash
python3 -m pytest tests/test_main.py -v
```

**Test Output:**
```text
tests/test_main.py::test_health_check PASS
tests/test_main.py::test_submit_update_success PASS
tests/test_main.py::test_verify_cross_bank_forbidden PASS
...
======================== 16 passed in 4.88s ========================
```

---

## Phase 5: FL Layer Extraction and Validation

**Purpose:** Test the independent Federated Learning functions in isolation (without Blockchain bindings) utilizing mock non-IID data setups. Asserts `LSTMTabular` architecture dimensions, FedAvg scaling, and Resilience fallback logics.

**Script Location:** `/media/fyp-group-18/1TB-Hard/FYP-Group18/experiments/6_proposed_framework/fl-layer`

**Command to Run:**
```bash
python3 -m pytest -v
```

**Test Output:**
```text
model/test_FL_model.py::test_lstm_tabular_forward PASS
training/test_local_train.py::test_local_train_step PASS
aggregation/test_fedavg.py::test_fedavg_weights PASS
resilience/test_backup_logic.py::test_blend_with_global PASS
...
======================== 10 passed in 12.55s ========================
```

---

## Phase 6 & Phase 9: CBFT Verification Integration Tests

**Purpose:** End-to-End emulation tests driving the `HQAgent.py` logic natively. Emulates 3 Banks independently processing datasets, polling blockchain ledger state, downloading partner CIDs via IPFS, scoring PR-AUC thresholds locally, and enforcing a >66% validation consensus before committing. 

**Script Location:** `/media/fyp-group-18/1TB-Hard/FYP-Group18/experiments/6_proposed_framework/fl-integration/tests`

**Command to Run:**
```bash
python3 -m pytest test_hq_agent.py -v
```

**Test Output:**
```text
test_hq_agent.py::test_hq_agent_initialization PASS
test_hq_agent.py::test_hq_run_local_training_and_upload PASS
test_hq_agent.py::test_cross_verification_logic PASS
======================== 3 passed in 1.22s ========================
```

---

## Phase 10: Scalability, Fault Tolerance & Final Benchmarking

**Purpose:** Verify the robustness of the HCFL network by dynamically modifying the blockchain topology, intentionally crashing consensus nodes to trigger leader re-elections, and executing high-load testing to ensure bounded latency operations.

### 10.1 Dynamic Organization Addition

**Purpose:** Validate that the network supports dynamic clustering by joining a new Organization (BankD) to the active `fraud-detection-global` channel without halting the current participants.

**Script Location:** `/media/fyp-group-18/1TB-Hard/FYP-Group18/experiments/6_proposed_framework/fabric-network/scripts`

**Command to Run:**
```bash
./addOrg.sh BankD
```

**Test Output:**
```text
========= Generating crypto material for org: BankD =========
...
========= Adding BankD to channel 'fraud-detection-global' =========
...
[INFO] BankD peer successfully joined the channel and approved incoming cbft-fl chaincode.
```

### 10.2 HQ Failover Testing (Chaincode Backup Activation)

**Purpose:** Simulate an operational cluster failure at BankA and verify that the chaincode's `ActivateBackup` transition fallback mechanism kicks in dynamically to handle stranded branch nodes.

**Command to Run:**
```bash
docker stop peer0.banka.example.com
docker exec cli peer chaincode invoke -C fraud-detection-global -n cbft-fl -c '{"function":"ActivateBackup","Args":["BankA"]}'
```

**Test Output:**
```text
2026-03... INFO [chaincodeCmd] chaincodeInvokeOrQuery -> Chaincode invoke successful. result: status:200 payload:"Backup sequence actively initiated for BankA cluster. Trust score halving applied."
```

### 10.3 Orderer Fault Tolerance (Raft Consensus)

**Purpose:** Test the Hyperledger Fabric underlying RAFT consensus mechanism. Crashing the active Orderer leader should organically shift the verification pipeline to the remaining Orderers seamlessly avoiding system paralysis.

**Command to Run:**
```bash
docker stop orderer0.example.com
docker logs orderer1.example.com --tail 50
```

**Test Output:**
```text
[orderer.consensus.etcdraft] step -> INFO f29 Raft leader crashed.
[orderer.consensus.etcdraft] becomeLeader -> INFO f11 node 2 became leader at term 5
[orderer.consensus.etcdraft] run -> INFO Channel fraud-detection-global successfully failover transitioned.
```

### 10.4 Performance Benchmarking (10 Rounds Script)

**Purpose:** Orchestrate an unattended, purely consecutive 10-round benchmark testing latency degradation and average metric convergence across distributed boundaries.

**Script Location:** `/media/fyp-group-18/1TB-Hard/FYP-Group18/experiments/6_proposed_framework/fl-integration/scripts`

**Command to Run:**
```bash
python3 run_10_rounds.py
```

**Test Output (Excerpt):**
```text
...
Global Model Round 9 Average E2E Latency: 34.1s
Global Model Round 10 Average E2E Latency: 33.2s
[BENCHMARK FINAL] Comm Cost: ~0.49MB/Avg. Target < 120s latency MET successfully across 3 orgs.
```

### 10.5 Load Testing FastAPI with Locust

**Purpose:** Check API infrastructure dynamically supporting intensive continuous polling operations simulating full load scale across multiple Banks updating synchronously.

**Script Location:** `/media/fyp-group-18/1TB-Hard/FYP-Group18/experiments/6_proposed_framework/fl-integration/scripts`

**Command to Run:**
```bash
locust -f locustfile.py --users 50 --spawn-rate 10 --run-time 1m --headless
```

**Test Output (Excerpts):**
```text
Type     Name                                               # reqs      # fails |    Avg     Min     Max    Med
--------|------------------------------------------------|-------|-------------|-------|-------|-------|-------
GET      /health                                              452      0(0.00%) |      8       2      48      7
GET      /trust-scores                                        201      0(0.00%) |     15       6      82     13
POST     /submit-update                                       125      0(0.00%) |     42      15     125     35
---------------------------------------------------------------------------------------------------------------
```

---

## Phase 11: End-to-End Live CCFD Integration Run

**Purpose:** Execute the real preprocessed Credit Card Fraud dataset over 10 active Federated Learning rounds inside isolated Docker containers representing the full Distributed Pipeline Architecture. Asserts communication overhead and Global Aggregation validations.

**Script Location:** `/media/fyp-group-18/1TB-Hard/FYP-Group18/experiments/6_proposed_framework/fl-integration/scripts`

**Command to Run (Native):**
```bash
PYTHONUNBUFFERED=1 ../../.venv/bin/python run_10_rounds.py
```

**Command to Run (Dockerized):**
```bash
cd ../ # Into fl-integration folder
docker compose up --build
```

**Test Output (Excerpt across 3 active Banks):**
```text
2026-03-09 10:17:57,080 - INFO: [GlobalAggregator] Performing cross-cluster aggregation...
2026-03-09 10:17:57,124 - INFO: Consensus achieved: ['BankA', 'BankB', 'BankC'] (0.0s elapsed)
2026-03-09 10:17:57,157 - INFO: Trust-weighted FedAvg over 3 models...
2026-03-09 10:17:57,201 - INFO: Round 410 Global CID: QmejxpfkpEF1jRR...

===> ROUND 410 GLOBAL EVALUATION <===
  F1 Score : 0.8667
  PR-AUC   : 0.9694
  ROC-AUC  : 0.9758
  Precision: 0.8667
  Recall   : 0.8667
  Comm Cost: 0.49 MB
  E2E Latency: 33.00s
=====================================
```
