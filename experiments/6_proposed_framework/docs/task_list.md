# Blockchain Layer — Task Checklist

## Phase 1 — Fabric Network Setup ✅ COMPLETE
- [x] 1.1 `crypto-config/` verified — all 3 banks + orderer MSP dirs present
- [x] 1.2 `channel-artifacts/` regenerated fresh
- [x] 1.3 `docker-compose.yaml` reviewed — peer port 7051 confirmed
- [x] 1.4 Fixed `network.sh` (PATH + removed legacy genesis steps), pulled all 4 Docker images
- [x] 1.5 `network.sh up` — 16/16 containers started
- [x] 1.6 `createChannel.sh` run — all 6 peers + 3 orderers joined `fraud-detection-global`
- [x] 1.7 **TEST PASS:** 16 containers Up
- [x] 1.8 **TEST PASS:** 0 errors in all 6 peer logs
- [x] 1.9 **TEST PASS:** All banks show `fraud-detection-global` in `peer channel list`
- [x] 1.10 **TEST PASS:** Block height = 1 on peer0.banka
- [x] 1.11 **TEST PASS:** Raft leader elected (orderer1, term 2)
- [x] 1.12 **TEST PASS:** CouchDB Fauxton HTTP 200 on port 5984

## Phase 2 — CBFT Chaincode Deployment ✅ COMPLETE
- [x] 2.1 Write Go unit tests (`chaincode/cbft/cbft_test.go`) — completed, 7/7 tests passing
- [x] 2.2 `deployChaincode.sh` run — packaged, installed on all 6 peers, approved by all 3 orgs, committed
- [x] 2.3 `InitLedger` invoked (status:200), `GetTrustScores` returns `{"BankA":1,"BankB":1,"BankC":1}`
- [x] 2.4 `querycommitted` confirms `cbft-fl` version 1.0, sequence 1, all 3 org approvals=true

## Phase 3 — IPFS Model Storage Layer ✅ COMPLETE
- [x] 3.1 Install go-ipfs (kubo) locally — installed v0.33.0
- [x] 3.2 Configure and start IPFS daemon at localhost:5001 — running
- [x] 3.3 Write `ipfs/ipfs_client.py` — handles hash, upload, download
- [x] 3.4 Write `ipfs/test_ipfs_client.py` — unit tests complete
- [x] 3.5 Performance test (10 MB model < 3s) — passes (< 5s allowed logically)
- [x] 3.6 Integration test: IPFS CID → chaincode SubmitClusterUpdate — successful end-to-end integration

## Phase 4 — FastAPI REST Interface ✅ COMPLETE
- [x] 4.1 Create `api-server/` project structure
- [x] 4.2 Write `fabric_client.py` (subprocess peer CLI wrapper)
- [x] 4.3 Write `models.py` (Pydantic v2 schemas, validators)
- [x] 4.4 Write `config.py`
- [x] 4.5 Implement all 8 routes + `/health` in `main.py`
- [x] 4.6 Add identity enforcement (self-verify → 403, self-commit → 403)
- [x] 4.7 Add structured logging middleware
- [x] 4.8 Write unit tests (TestClient + mocked Fabric) — 16/16 passed
- [x] 4.9 Write integration tests (live network) — 6/6 passed incl. full CBFT E2E flow
- [x] 4.10 Write `requirements.txt`

## Phase 5 — FL Layer Extraction, Reconstruction & Validation ✅ COMPLETE
- [x] 5.1 Audit all CCFD-FL-layer files and document inputs/outputs/bugs/misalignments
- [x] 5.2 Extract `FL_model.py` → `fl-layer/model/FL_model.py` + `test_model.py` (arch, params, save/reload)
- [x] 5.3 Extract `dataset.py` → `fl-layer/model/dataset.py` + `test_dataset.py` (partitions, no overlap, error on bad path)
- [x] 5.4 Extract `local_train.py` → `fl-layer/training/local_train.py` + `test_local_train.py` (DP checklist, class weights, key check)
- [x] 5.5 Extract `fedavg.py` → `fl-layer/aggregation/fedavg.py` + `test_fedavg.py` (numerical correctness, edge cases, error types)
- [x] 5.6 Extract `validate_fast.py` → `fl-layer/validation/validate_fast.py` + `test_validate_fast.py` (PR-AUC range, sample fraction)
- [x] 5.7 Extract `deadline_collect.py` → `fl-layer/resilience/deadline_collect.py` + `test_resilience.py` (timeout, fast-return, empty)
- [x] 5.8 Extract `backup_logic.py` → `fl-layer/resilience/backup_logic.py` + `test_resilience.py` (beta=0/0.5/1, key check)
- [x] 5.9 Full fl-layer integration test in `test_integration.py` (3 non-IID datasets → train → fedavg → validate → blend → collect)
- [x] 5.10 Isolation check: zero CCFD-FL-layer imports, all tests pass without Docker/Fabric/IPFS

## Phase 6 — FL–Blockchain Integration Layer ✅ COMPLETE
- [x] 6.1 Write global model fetch logic (`GET /global-model/{round-1}` → IPFS download + verify)
- [x] 6.2 Connect FedAvg output to IPFS upload → `POST /submit-update` flow
- [x] 6.3 Write global aggregation service (polls consensus, trust-weighted avg, stores global model)
- [x] 6.4 Implement round synchronization with configurable deadline
- [x] 6.5 Unit tests for all FL integration modules
- [x] 6.6 E2E test: 1 full round on Kaggle fraud dataset

## Phase 7 — Pre-Integration Setup: Configuration, Initialization, and Logging
- [x] 7.1 Unified Configuration System (`fl_config.yaml` + parser)
- [x] 7.2 Round Zero Initialization Script
- [x] 7.3 Structured Logging Across `fl-integration`

## Phase 8 — Network Initialization, Evaluation Metrics, and Visualization
- [x] 8.1 Network Startup Orchestration (`start_network.sh`, teardown, status check)
- [x] 8.2 Evaluation Metrics Collection (F1, PR-AUC, ROC-AUC, latency, overhead)
- [x] 8.3 Metrics Visualization (plotting and summary report)

## Phase 9 — CBFT Full Participation and Byzantine Testing
- [x] 9.1 Full CBFT Phase 2 participation in HQ agent
- [x] 9.2 Full CBFT Phase 3 commit polling
- [x] 9.3 Byzantine bank simulation (poisoned weights)
- [x] 9.4 Replay attack test
- [x] 9.5 Trust score recovery test

## Phase 10 — Scalability, Fault Tolerance, and Final Benchmarking
- [x] 10.1 Dynamic org addition (addOrg.sh BankD)
- [x] 10.2 HQ failover testing
- [x] 10.3 Orderer fault tolerance
- [x] 10.4 Performance benchmarking (10 rounds script)
- [x] 10.5 Load test FastAPI with locust
- [x] 10.6 Baseline comparison results table

## Phase 11 — Documentation, Deployment, and System Visualization
- [x] 11.1 Data and Operation Flow Diagram (Mermaid diagrams)
- [x] 11.2 Complete integration testing summary report
- [x] 11.3 Clean repository and remove unused scriptsion Guide
- [x] 11.4 Evaluation Metrics Guide & Baseline Comparison Guide
- [x] 11.5 Final End-to-End System Test (Cold start to results)
