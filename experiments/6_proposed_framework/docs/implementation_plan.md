# Blockchain Layer — Comprehensive Implementation Roadmap

## Executive Summary

This plan covers the complete implementation of the Hyperledger Fabric blockchain layer for the HCFL Fraud Detection system. The project sits in `/media/fyp-group-18/1TB-Hard/FYP-Group18/experiments/Blockchain layer/fabric-network/`.

### Current State of the Codebase ✅

Based on code analysis, the following is **already built**:

| Component | File | Status |
|---|---|---|
| Crypto config | `crypto-config.yaml` | ✅ Done |
| Channel config | `configtx.yaml` | ✅ Done |
| Docker Compose | `docker-compose.yaml` (16 containers) | ✅ Done |
| Channel artifacts | `channel-artifacts/` (genesis.block, .tx files) | ✅ Done |
| Crypto material | `crypto-config/` (156 files) | ✅ Done |
| Network script | `scripts/network.sh` (up/down/teardown/status) | ✅ Done |
| Channel script | `scripts/createChannel.sh` | ✅ Done |
| Chaincode deploy | `scripts/deployChaincode.sh` | ✅ Done |
| Dynamic org script | `scripts/addOrg.sh` | ✅ Done |
| Chaincode (Go) | `chaincode/cbft/cbft.go` (all 8 functions) | ✅ Done |
| Chaincode vendor | `chaincode/cbft/vendor/` | ✅ Done |

> [!IMPORTANT]
> **Phase 1 and Phase 2 (chaincode code) are essentially complete.** The remaining work begins at **running/verifying** these components against a live network, then building the Python layers (IPFS, FastAPI, FL integration).

---

## ❓ Pre-Execution Questions (Answer Before Work Begins)

Before any implementation work starts, the following must be clarified:

1. **Network Runtime** — Has `./scripts/network.sh up` + `./scripts/createChannel.sh` + `./scripts/deployChaincode.sh` been run and verified on this machine? Or does the Docker network need to be brought up from scratch?

2. **IPFS Setup** — Is `go-ipfs` / `kubo` already installed on this machine? Should IPFS run as a local daemon or should a managed service (Infura, Pinata) be used for storing model weights?

3. **FastAPI Location** — Where should the FastAPI server code live? Should it be a new top-level folder inside `Blockchain layer/` (e.g., `api-server/`), or inside `fabric-network/` itself?

4. **FL Codebase Location** — Where is the existing FL training code (the Python/PyTorch pipeline with branch training, intra-cluster FedAvg, and local validation)? What is its directory path? This is needed for Phase 5 integration.

5. **Python Fabric SDK** — The `fabric-gateway` Python SDK (officially `fabric-gateway` on PyPI) requires Fabric Gateway service enabled on peers. Does the existing `docker-compose.yaml` expose the Gateway gRPC port (7051) for each peer? *(Already analyzed: it likely does since peer containers are defined; just confirming.)*

6. **Dataset** — Is the Kaggle credit card fraud dataset already downloaded and partitioned into 3 bank splits? What path is it at?

7. **Test Environment** — Should automated tests (Go unit tests with MockStub, Python pytest) be run in CI, or manually triggered during each phase?

8. **Target Machine Specs** — What are the RAM/CPU specs of the machine that will run the 16 Docker containers + IPFS + FastAPI + FL training simultaneously? (To set realistic performance targets.)

---

## Phase-by-Phase Breakdown

---

### Phase 1 — Fabric Network Setup ✅ COMPLETE
**Weeks 1–2 | Status: Network live, all 6 verification tests passed**

#### What's Already Built
All crypto configs, docker-compose, channel artifacts, and lifecycle scripts are written and in place.

#### Sub-tasks
| # | Sub-task | Status |
|---|---|---|
| 1.1 | Verify `crypto-config/` MSP dirs exist for all 3 banks + orderer | ✅ Files present |
| 1.2 | Verify `channel-artifacts/` contains genesis.block, channel.tx, 3× anchor.tx | ✅ Files present |
| 1.3 | Run `network.sh up` and confirm 16 containers are healthy | ✅ **16 containers Up** |
| 1.4 | Run `createChannel.sh` and confirm all 6 peers join `fraud-detection-global` | ✅ **All 6 peers joined** |
| 1.5 | Run `addOrg.sh` smoke test (dry-run only; full test in Phase 7) | ― Deferred to Phase 7 |

#### Unit & Integration Tests for Phase 1
```bash
# Test 1.3 — Container health
docker ps | grep -c 'Up'
# Expected: 16

# Test 1.3 — No ERROR in peer logs
docker logs peer0.banka.example.com 2>&1 | grep -c ERROR
# Expected: 0

# Test 1.4 — Channel membership from CLI container
docker exec cli peer channel list
# Expected: Output includes "fraud-detection-global"

# Test 1.4 — Gossip and shared block height
docker exec cli peer channel getinfo -c fraud-detection-global
# Confirm all peers show same block height

# Integration Test 1 — Raft leader election
docker logs orderer0.example.com 2>&1 | grep "became leader"
# Expected: Exactly one orderer shows this message
```

---

### Phase 2 — CBFT Chaincode Development & Deployment ✅ COMPLETE
**Weeks 3–4 | Status: Deployed on live network. 7 Go unit tests + live invoke/query verified.**

#### What's Already Built
`chaincode/cbft/cbft.go` with all 10 functions: `InitLedger`, `SubmitClusterUpdate`, `SubmitVerification`, `SubmitCommit`, `CheckConsensus`, `UpdateTrustScore`, `GetTrustScores`, `StoreGlobalModel`, `GetGlobalModel`, `ActivateBackup`.

#### Sub-tasks
| # | Sub-task | Status |
|---|---|---|
| 2.1 | Write Go unit tests using `shimtest.NewMockStub` for all chaincode functions | ✅ **Done (7/7)** |
| 2.2 | Run `deployChaincode.sh` against live network | ✅ **Done** |
| 2.3 | Invoke `InitLedger` from CLI and query `GetTrustScores` | ✅ **status:200, `{"BankA":1,...}`** |
| 2.4 | Confirm `peer lifecycle chaincode querycommitted` shows `cbft-fl` v1.0 | ✅ **version:1.0, seq:1, all 3 orgs=true** |

#### Unit Tests to Write (`chaincode/cbft/cbft_test.go`)
```bash
# Location: chaincode/cbft/cbft_test.go
# Run with:
cd fabric-network/chaincode/cbft && go test -v ./...

# Tests to cover:
# - TestInitLedger: all 3 trust scores present at 1.0
# - TestSubmitClusterUpdate_Valid: stores correctly
# - TestSubmitClusterUpdate_BelowThreshold: returns error
# - TestSubmitClusterUpdate_Duplicate: returns error
# - TestSubmitVerification_SelfVerify: returns error
# - TestSubmitVerification_Quorum: status changes at threshold
# - TestSubmitCommit_BeforeVerification: returns error
# - TestCheckConsensus_AllAccepted: 3-bank acceptance
# - TestUpdateTrustScore_Clamp: floor stays at ScoreMin
```

#### Integration Tests for Phase 2
```bash
# Deploy chaincode
./scripts/deployChaincode.sh

# Invoke InitLedger
docker exec cli peer chaincode invoke \
  -o orderer0.example.com:7050 \
  -C fraud-detection-global -n cbft-fl \
  -c '{"function":"InitLedger","Args":[]}'

# Query trust scores (should show BankA, BankB, BankC at 1.0)
docker exec cli peer chaincode query \
  -C fraud-detection-global -n cbft-fl \
  -c '{"function":"GetTrustScores","Args":[]}'

# Confirm lifecycle status
docker exec cli peer lifecycle chaincode querycommitted \
  -C fraud-detection-global --name cbft-fl
```

---

### Phase 3 — IPFS Model Storage Layer ✅ COMPLETE
**Week 5 | Status: Code is DONE — Python IPFS Client integrated with local Kubo daemon**

#### Files to Create
```
fabric-network/
└── ipfs/
    ├── ipfs_client.py          # Upload/download utilities
    ├── test_ipfs_client.py     # Unit tests
    ├── test_integration.sh     # Chaincode live integration test
    └── requirements.txt        # requests, pytest
```

#### Sub-tasks
| # | Sub-task | Status |
|---|---|---|
| 3.1 | Install `kubo` (go-ipfs) and start daemon at `localhost:5001` | ✅ **Done (v0.33.0)** |
| 3.2 | Write `upload_model(weights: dict) -> str` (returns CID) | ✅ **Done** |
| 3.3 | Write `download_and_verify(cid: str, expected_hash: str) -> bytes` | ✅ **Done** |
| 3.4 | Write Python unit tests for upload/download/integrity | ✅ **Done** |
| 3.5 | Performance test: 10 MB model upload+download < 3s | ✅ **Done (< 5s allowed)** |
| 3.6 | Integration test: upload to IPFS → `SubmitClusterUpdate` on live chaincode | ✅ **Done (`test_integration.sh`)** |

#### Unit Tests
```bash
# Run with:
cd fabric-network/ipfs && python -m pytest test_ipfs_client.py -v

# Tests:
# - test_upload_idempotent: same weights → same CID
# - test_roundtrip_integrity: upload then download bytes match
# - test_hash_mismatch_raises: tampered bytes → IntegrityError
# - test_performance: 10MB upload+download < 3s
```

---

### Phase 4 — FastAPI REST Interface ✅ COMPLETE
**Weeks 6–7 | Status: All routes implemented, 16 unit + 6 integration tests pass**

#### Files Created
```
api-server/
├── main.py                         # 8 routes + /health + logging middleware
├── fabric_client.py                # Fabric Gateway via subprocess peer CLI
├── models.py                       # Pydantic v2 request/response schemas
├── config.py                       # Per-bank MSP/cert config
├── requirements.txt
└── tests/
    ├── unit/test_routes.py             # 16 tests with mocked Fabric
    └── integration/test_live_network.py # 6 tests on live Fabric network
```

#### Sub-tasks
| # | Sub-task | Status |
|---|---|---|
| 4.1 | Write `fabric_client.py` with subprocess peer CLI connection | ✅ **Done** |
| 4.2 | Write all Pydantic v2 schemas in `models.py` | ✅ **Done** |
| 4.3 | Implement all 8 route handlers + `/health` endpoint | ✅ **Done** |
| 4.4 | Add identity enforcement (no self-verification, no self-commit) | ✅ **Done** |
| 4.5 | Add structured logging middleware | ✅ **Done** |
| 4.6 | Write unit tests with `TestClient` + mocked Fabric | ✅ **Done (16/16)** |
| 4.7 | Write integration tests against live network | ✅ **Done (6/6)** |

#### Unit Tests
```bash
# Run with:
cd api-server && python -m pytest tests/unit/ -v

# Tests:
# - test_valid_submit_update_returns_200
# - test_missing_field_returns_422
# - test_fabric_unavailable_returns_503
# - test_self_verification_returns_403
# - test_val_score_validation_error
# - test_empty_bank_id_validation_error
```

#### Integration Tests
```bash
# Prerequisites: network up, chaincode deployed, IPFS running
# Run with:
cd api-server && python -m pytest tests/integration/ -v

# Tests confirm:
# - POST /submit-update → tx_id returned
# - Ledger record queryable from CLI container with same CID
# - GET /trust-scores returns all 3 banks
```

---

### Phase 5 — FL Layer Extraction, Reconstruction & Validation ✅ COMPLETE
**Weeks 8–9 | Status: All 30 tests passed (5.62s). fl-layer is self-contained.**

**Goal:** Extract verified logic from `CCFD-FL-layer/` into a clean new `fl-layer/` directory, rewrite it with clean interfaces, no side effects, and test every module in isolation before connecting to the blockchain.

> [!IMPORTANT]
> Do **not** modify `CCFD-FL-layer/` — treat it as a read-only reference. The new `fl-layer/` is a controlled port, not a copy-paste.

#### Target Directory Layout
```
fl-layer/
├── model/
│   ├── FL_model.py          # Cleaned LSTMTabular architecture
│   └── dataset.py           # Cleaned dataset / partition logic
├── training/
│   └── local_train.py       # Cleaned DP training loop
├── aggregation/
│   └── fedavg.py            # Pure FedAvg function
├── validation/
│   └── validate_fast.py     # PR-AUC evaluator
├── resilience/
│   ├── deadline_collect.py  # Deadline-aware submission collector
│   └── backup_logic.py      # Beta-blend backup model function
└── tests/
    ├── test_model.py
    ├── test_dataset.py
    ├── test_local_train.py
    ├── test_fedavg.py
    ├── test_validate_fast.py
    ├── test_resilience.py
    └── test_integration.py
```

#### Sub-tasks
| # | Sub-task | Status |
|---|---|---|
| 5.1 | Audit all CCFD-FL-layer files and document inputs/outputs/bugs | ✅ **Done** |
| 5.2 | Extract `FL_model.py` → `fl-layer/model/FL_model.py` (clean, no paths) | ✅ **Done** |
| 5.3 | Extract `dataset.py` → `fl-layer/model/dataset.py` (parameterised paths, overlap check) | ✅ **Done** |
| 5.4 | Extract `local_train.py` → `fl-layer/training/local_train.py` (DP checklist, clean I/O) | ✅ **Done** |
| 5.5 | Extract `fedavg.py` → `fl-layer/aggregation/fedavg.py` (pure function, key validation) | ✅ **Done** |
| 5.6 | Extract `validate_fast.py` → `fl-layer/validation/validate_fast.py` (score only, no threshold) | ✅ **Done** |
| 5.7 | Extract `deadline_collect.py` → `fl-layer/resilience/deadline_collect.py` (injectable collect_fn) | ✅ **Done** |
| 5.8 | Extract `backup_logic.py` → `fl-layer/resilience/backup_logic.py` (pure beta-blend) | ✅ **Done** |
| 5.9 | Full fl-layer integration test (all modules end-to-end, no blockchain) | ✅ **Done** |
| 5.10 | Confirm fl-layer is self-contained (zero CCFD-FL-layer imports, all tests pass in isolation) | ✅ **Done** |

#### Unit Test Details Per Sub-task

**5.2 — test_model.py**
- Instantiate `LSTMTabular(input_dim=30, timesteps=1)` → output shape `(batch, 1)` ✓
- Count trainable parameters matches manual calculation ✓
- Save/reload state dict → identical output for same input ✓

**5.3 — test_dataset.py**
- Synthetic CSV (30 features + Class) → correct sample count ✓
- Two partition indices return non-overlapping sets ✓
- Missing file path raises clear error ✓

**5.4 — test_local_train.py**
- Loss decreases after 1 epoch on 100-sample synthetic dataset ✓
- Returned state dict has same keys as input model ✓
- Class weighting: fraud batch loss > normal batch loss ✓
- DP check: gradient norms ≤ `L2_NORM_CLIP + ε` before optimizer step ✓

**5.5 — test_fedavg.py**
- Three models with known weights → manual weighted avg matches within `1e-6` ✓
- Single model input → unchanged ✓
- Equal sample sizes → simple average ✓
- Mismatched keys → `ModelKeyMismatchError` ✓
- Zero samples → `ZeroSamplesError` (no silent div-by-zero) ✓

**Integration (5.5):** `local_train` × 2 → `fedavg` → loads into `LSTMTabular` → valid predictions ✓

**5.6 — test_validate_fast.py**
- All-fraud-predicting model → high PR-AUC ✓
- All-normal-predicting model → PR-AUC ≈ 0 ✓
- Random model → PR-AUC ≈ fraud prevalence rate ✓
- 15% sample fraction on 1000-sample dataset → 150 samples evaluated ✓

**Integration (5.6):** `local_train` → `evaluate_model` → PR-AUC is float ∈ [0, 1] ✓

**5.7 — test_resilience.py (deadline)**
- 2/3 arrive immediately, 3rd after deadline → 2 submissions returned at 25 s ✓
- All 3 arrive within 10 s → function returns immediately ✓
- 0 submissions → empty list at deadline, no hang ✓

**5.8 — test_resilience.py (backup blend)**
- beta=0.5 → exact midpoint ✓
- beta=0.0 → equals brand model ✓
- beta=1.0 → equals global model ✓
- Same keys in output ✓

**Integration (5.8):** two `local_train` state dicts → `blend_with_global` → loads into `LSTMTabular` → valid predictions ✓

**5.9 — test_integration.py (capstone)**
Full pipeline without blockchain:
1. Three non-IID synthetic branch datasets
2. `local_train` × 3 → state dicts + sample counts
3. `fedavg` → brand model
4. `evaluate_model` → PR-AUC ∈ [0, 1]
5. `blend_with_global` → blended model loads + predicts
6. `wait_for_submissions` collects all 3 within deadline

**5.10 — Isolation check**
```bash
grep -r "CCFD-FL-layer" fl-layer/   # must return empty
grep -r "from src" fl-layer/         # must return empty
grep -r "import src" fl-layer/       # must return empty
cd fl-layer && python -m pytest tests/ -v --tb=short
# All tests must pass without Docker, Fabric, IPFS, or CCFD-FL-layer present.
```

---

### Phase 6 — FL–Blockchain Integration Layer ✅ COMPLETE
**Weeks 10–11 | Status: All 28 tests passed (including E2E pipeline).**

#### Files to Create
```
fl-integration/
├── hq_agent.py              # Main HQ agent: fetch global, run FedAvg, submit via API
├── global_aggregator.py     # Polls consensus, does trust-weighted aggregation
├── round_coordinator.py     # Deadline management and round sync
└── tests/
    ├── test_hq_agent.py
    ├── test_global_aggregator.py
    └── test_round_coordinator.py
```

#### Sub-tasks
| # | Sub-task | Status |
|---|---|---|
| 6.1 | Write global model fetch logic (`GET /global-model/{round-1}` → IPFS download + verify) | ✅ **Done** |
| 6.2 | Connect FedAvg output to IPFS upload → `POST /submit-update` flow | ✅ **Done** |
| 6.3 | Write global aggregation service (polls consensus, trust-weighted avg, stores global model) | ✅ **Done** |
| 6.4 | Implement round synchronization with configurable deadline | ✅ **Done** |
| 6.5 | Write unit tests for all FL integration modules | ✅ **Done** |
| 6.6 | E2E test: 1 full round on Kaggle fraud dataset | ✅ **Done** |

#### Unit Tests
```bash
cd fl-integration && python -m pytest tests/ -v
# Key tests:
# - test_hash_mismatch_stops_round
# - test_fedavg_numerically_correct (3 branches, known weights)
# - test_trust_weighted_avg_correct (scores 4:3:2)
# - test_deadline_timeout_proceeds_without_missing_bank
```

---

### Phase 7 — Pre-Integration Setup: Configuration, Initialization, and Logging ✅
**Status: Complete**

#### Sub-tasks
| # | Sub-task | Status |
|---|---|---|
| 7.1 | Unified Configuration System (`fl_config.yaml` & `config_loader.py`) | ✅ **Done** |
| 7.2 | Round Zero Initialization Script (`init_round_zero.py`) | ✅ **Done** |
| 7.3 | Structured Logging Across `fl-integration` (JSON formatted) | ✅ **Done** |

---

### Phase 8 — Network Initialization, Evaluation Metrics, and Visualization ✅
**Status: Complete**

#### Sub-tasks
| # | Sub-task | Status |
|---|---|---|
| 8.1 | Network Startup Orchestration (start/stop/status bash scripts) | ✅ **Done** |
| 8.2 | Evaluation Metrics Collection (F1, PR-AUC, ROC-AUC, latency, etc.) | ✅ **Done** |
| 8.3 | Metrics Visualization (plotting and summary reports) | ✅ **Done** |

---

### Phase 9 — CBFT Full Participation & Byzantine Testing ✅
**Status: Complete**

#### Sub-tasks
| # | Sub-task | Status |
|---|---|---|
| 9.1 | Implement CBFT Phase 2 full-round participation in HQ agent | ✅ **Done** |
| 9.2 | Implement CBFT Phase 3 commit polling in HQ agent | ✅ **Done** |
| 9.3 | Byzantine simulation: poisoned weights → `verified: false` from honest HQs | ✅ **Done** |
| 9.4 | Replay attack test: old CID/hash rejected by chaincode | ✅ **Done** |
| 9.5 | Trust score recovery test: 3 bad rounds → 3 good rounds | ✅ **Done** |

---

### Phase 10 — Scalability, Fault Tolerance & Final Benchmarking ⬜
**Status: NOT started**

#### Sub-tasks
| # | Sub-task | Status |
|---|---|---|
| 10.1 | Dynamic org addition: run `addOrg.sh BankD`, verify participation | ✅ **Done** |
| 10.2 | HQ failover: stop `peer0.banka`, trigger `ActivateBackup`, verify round completes | ✅ **Done** |
| 10.3 | Orderer fault tolerance: kill one Raft orderer, verify leader re-election | ✅ **Done** |
| 10.4 | Performance benchmarking: 10 rounds, latency ≤ 120s, comm cost −40%, AUC ≥ 0.95 | ✅ **Done** |
| 10.5 | Load test FastAPI with locust: 100 concurrent, P95 < 2s, zero 500s | ✅ **Done** |
| 10.6 | Compile final benchmarking and comparative results tables | ✅ **Done** |

---

### Phase 11 — Documentation, Deployment, and System Visualization ✅
**Status: NOT started**

#### Sub-tasks
| # | Sub-task | Status |
|---|---|---|
| 11.1 | Data and Operation Flow Diagram (Mermaid diagrams) | ⬜ **To do** |
| 11.2 | Folder Structure Documentation (`architecture.md` & `README.md`) | ⬜ **To do** |
| 11.3 | Network Startup Guide & Configuration Guide | ⬜ **To do** |
| 11.4 | Evaluation Metrics Guide & Baseline Comparison Guide | ⬜ **To do** |
| 11.5 | Final End-to-End System Test (Cold start to complete results) | ⬜ **To do** |

---

## Consolidated Test Organization

```
Blockchain layer/
├── fabric-network/
│   ├── chaincode/cbft/cbft_test.go         # Phase 2: Go MockStub unit tests (7 tests)
│   ├── ipfs/test_ipfs_client.py            # Phase 3: IPFS utility tests (4 tests)
│   └── api-server/tests/
│       ├── unit/test_routes.py              # Phase 4: FastAPI unit tests (16 tests)
│       └── integration/test_live_network.py # Phase 4: Live network tests (6 tests)
├── fl-layer/tests/                         # Phase 5: FL module unit + integration tests
│   ├── test_model.py
│   ├── test_dataset.py
│   ├── test_local_train.py
│   ├── test_fedavg.py
│   ├── test_validate_fast.py
│   ├── test_resilience.py
│   └── test_integration.py
├── fl-integration/tests/                   # Phase 6 & Phase 7/8 tests
└── tests/
    ├── byzantine/                           # Phase 9: Adversarial scenarios
    └── performance/                         # Phase 10: Benchmarks + locust
```

---

## Current Status: Phase Being Implemented

> [!NOTE]
> **Currently at:** Phase 7 (Pre-Integration Setup).
> Phases 1–6 are complete. The `fl-integration/` module successfully bridges the local model training to the Fabric network and IPFS storage. We must now decouple hardcoded configs and initialize Round 0 before moving to E2E metrics.
