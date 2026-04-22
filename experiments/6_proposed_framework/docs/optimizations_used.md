# Latency & Communication Optimization Techniques

This document catalogues every optimization technique applied in the HCFL project across:
- `experiments/4_CCFD_Model` — the FL-layer baseline
- `experiments/6_proposed_framework` — the full blockchain-integrated system

Each entry explains **what the technique is**, **how it was applied here**, and **the exact code location**.

---

## 1. Compact Model Architecture (Minimal LSTM)

**What it is:** Choosing the smallest neural network that can still solve the task. A simpler model means smaller weight tensors, which directly translates to smaller payloads whenever a model is uploaded, downloaded, or transmitted between nodes.

**How it is used here:** Both experiments use a single-layer LSTM with a hidden dimension of only 30 units and a single linear output head. The model has on the order of **~10,000 parameters** — far smaller than standard deep learning models, keeping each serialized `.pt` file below 0.5 MB.

| Experiment | File | Key Code |
|---|---|---|
| 4_CCFD_Model | [`FL_model.py`](file:///media/fyp-group-18/1TB-Hard/FYP-Group18/experiments/4_CCFD_Model/FL_model.py) | `nn.LSTM(input_size=input_dim, hidden_size=30, num_layers=1)` |
| 6_proposed_framework | [`fl-layer/model/FL_model.py`](file:///media/fyp-group-18/1TB-Hard/FYP-Group18/experiments/6_proposed_framework/fl-layer/model/FL_model.py) | `LSTMTabular(input_dim=29, hidden_dim=30, num_layers=1)` |

**Impact:** ↓ Communication Overhead — smaller models per upload/download.

---

## 2. Differential Privacy — Gradient Clipping + Gaussian Noise

**What it is:** Differential Privacy (DP) limits the sensitivity of each training update by clipping per-sample gradients to a maximum L2 norm, then adding calibrated Gaussian noise. This is primarily a **privacy** mechanism, but it also acts as a form of gradient compression — clipped gradients are uniformly bounded, preventing exploding updates that would otherwise require extra coordination rounds.

**How it is used here:** Applied during local branch training in both experiments. The `L2_NORM_CLIP=1.0` and `NOISE_MULTIPLIER=0.05` (framework) / `0.05` (baseline) are applied per-parameter after each backward pass.

| Experiment | File | Key Code |
|---|---|---|
| 4_CCFD_Model | [`local_train.py`](file:///media/fyp-group-18/1TB-Hard/FYP-Group18/experiments/4_CCFD_Model/local_train.py) lines 51–55 | `clip_grad_norm_(…, 1.0) → p.grad += randn_like(p.grad) * (1.0 * 0.05)` |
| 6_proposed_framework | [`fl-layer/training/local_train.py`](file:///media/fyp-group-18/1TB-Hard/FYP-Group18/experiments/6_proposed_framework/fl-layer/training/local_train.py) | Same DP pattern, configurable via `l2_norm_clip`, `noise_multiplier` |

**Impact:** ↓ Latency — bounded gradients prevent divergent runs that require extra rounds to converge.

---

## 3. Hierarchical Two-Level Aggregation (Intra-Cluster FedAvg Before Global)

**What it is:** Instead of every branch node submitting its model directly to the global aggregator (a flat topology), branches first submit to their local HQ. The HQ performs an intra-cluster FedAvg and submits only one model per cluster to the blockchain. This compresses N-branch submissions into 1 per cluster.

**How it is used here:** In `4_CCFD_Model`, each brand's `hq_aggregate.py` collects branch models and runs FedAvg before writing one `brand_model.pt`. In `6_proposed_framework`, the `HQAgent.run_round()` method performs the same intra-bank FedAvg before uploading to IPFS.

| Experiment | File | Key Code |
|---|---|---|
| 4_CCFD_Model | [`src/aggregation/hq_aggregate.py`](file:///media/fyp-group-18/1TB-Hard/FYP-Group18/experiments/4_CCFD_Model/src/aggregation/hq_aggregate.py) line 83 | `brand_sd = fedavg_state_dicts(state_dicts)` |
| 6_proposed_framework | [`fl-integration/hq_agent.py`](file:///media/fyp-group-18/1TB-Hard/FYP-Group18/experiments/6_proposed_framework/fl-integration/hq_agent.py) | `HQAgent.run_round(round_num, branch_updates)` |

**Impact:** ↓ Communication Overhead — only 3 models uploaded to blockchain per round (one per bank) regardless of how many branches each bank has. ↓ Latency — fewer CBFT verification events needed.

---

## 4. IPFS Off-Chain Storage (CID Pointer on Blockchain)

**What it is:** Blockchain ledgers are extremely expensive for large binary payloads — every transaction is replicated across all peers, and large blocks congested gossip propagation. IPFS separates the concern: the large model binary is stored off-chain in a distributed content-addressed store (IPFS), and only the compact 46-byte Content Identifier (CID) plus a SHA-256 hash string are stored on the Hyperledger Fabric ledger.

**How it is used here:** Every model submission — cluster models from banks and the final global model — is uploaded to the local IPFS daemon before the API call to the fabric chaincode. The chaincode stores only `modelCID` and `modelHash`.

| Experiment | File | Key Code |
|---|---|---|
| 6_proposed_framework | [`fl-integration/hq_agent.py`](file:///media/fyp-group-18/1TB-Hard/FYP-Group18/experiments/6_proposed_framework/fl-integration/hq_agent.py) | `cid = self.ipfs_upload(buf.getvalue())` → `submit_update(…, model_cid=cid, …)` |
| 6_proposed_framework | [`fl-integration/global_aggregator.py`](file:///media/fyp-group-18/1TB-Hard/FYP-Group18/experiments/6_proposed_framework/fl-integration/global_aggregator.py) | `global_cid = self.ipfs_upload(buf.getvalue())` |
| 6_proposed_framework | [`fabric-network/chaincode/cbft/cbft.go`](file:///media/fyp-group-18/1TB-Hard/FYP-Group18/experiments/6_proposed_framework/fabric-network/chaincode/cbft/cbft.go) | Ledger stores `ModelCID string`, `ModelHash string` only |

**Impact:** ↓ Communication Overhead — blockchain transactions reduced from ~0.5 MB to ~200 bytes per submission. ↓ Latency — faster Fabric block propagation and consensus due to tiny transaction payload.

---

## 5. Straggler Mitigation via Deadline-Based Collection

**What it is:** In synchronous FL, the entire round stalls waiting for the slowest node (the "straggler"). Deadline collection sets a hard time budget. When the deadline expires, whatever models have arrived are aggregated and the round proceeds — stragglers are simply excluded from that round.

**How it is used here:** Both experiments use a deadline polling loop. `4_CCFD_Model` has an explicit `deadline_sec` config parameter (default 25s). `6_proposed_framework` also uses `deadline_seconds: 5.0` in the config and enforces it in the `RoundCoordinator`.

| Experiment | File | Key Code |
|---|---|---|
| 4_CCFD_Model | [`src/resilience/deadline_collect.py`](file:///media/fyp-group-18/1TB-Hard/FYP-Group18/experiments/4_CCFD_Model/src/resilience/deadline_collect.py) lines 25–51 | `while True: … if time.time() - start >= deadline_sec: break` |
| 4_CCFD_Model | [`src/aggregation/hq_aggregate.py`](file:///media/fyp-group-18/1TB-Hard/FYP-Group18/experiments/4_CCFD_Model/src/aggregation/hq_aggregate.py) line 34 | `deadline_sec = int(p2.get("deadline_sec", 25))` |
| 6_proposed_framework | [`fl-integration/round_coordinator.py`](file:///media/fyp-group-18/1TB-Hard/FYP-Group18/experiments/6_proposed_framework/fl-integration/round_coordinator.py) | Enforces `deadline_seconds` from `fl_config.yaml` |
| 6_proposed_framework | [`fl-integration/config/fl_config.yaml`](file:///media/fyp-group-18/1TB-Hard/FYP-Group18/experiments/6_proposed_framework/fl-integration/config/fl_config.yaml) | `deadline_seconds: 5.0` |

**Impact:** ↓ Latency — prevents rounds from blocking indefinitely on slow or failed nodes.

---

## 6. Backup / Global Model Blending for Failed Validators

**What it is:** When a cluster model fails validation (PR-AUC below threshold), instead of completely discarding it and wasting that round's local training, the system blends the failed cluster model with the previous global model using a `beta` mixing coefficient: `w_alt = beta * w_prev_global + (1 - beta) * w_cluster`. The blended model is then re-validated before deciding on exclusion.

**How it is used here:** Implemented in `4_CCFD_Model` as `blend_with_prev_global()`. Default `beta = 0.7` means 70% of the previous stable global and 30% of the new cluster update. This allows partial learning signal to survive even when a cluster underperforms, reducing the number of rounds needed to recover.

| Experiment | File | Key Code |
|---|---|---|
| 4_CCFD_Model | [`src/resilience/backup_logic.py`](file:///media/fyp-group-18/1TB-Hard/FYP-Group18/experiments/4_CCFD_Model/src/resilience/backup_logic.py) lines 11–25 | `out[k] = beta * prev_sd[k] + (1.0 - beta) * brand_sd[k]` |
| 4_CCFD_Model | [`src/aggregation/hq_aggregate.py`](file:///media/fyp-group-18/1TB-Hard/FYP-Group18/experiments/4_CCFD_Model/src/aggregation/hq_aggregate.py) lines 96–110 | Calls `blend_with_prev_global(…, beta=0.7)` when validation fails |

**Impact:** ↓ Latency — reduces round count wasted on failed clusters; partial signal is recovered immediately.

---

## 7. Fractional Validation Sampling (`validate_fast`)

**What it is:** Computing full dataset metrics (PR-AUC, F1) during the cross-validation phase of each round is expensive. Fractional sampling selects a statistically representative random subset of the validation dataset (15% by default) for fast metric estimation, drastically cutting the per-model evaluation time during CBFT.

**How it is used here:** Both experiments use `fast_val_fraction = 0.15`. The fast validator loads only a fraction of the dataset into a DataLoader and runs a single inference pass.

| Experiment | File | Key Code |
|---|---|---|
| 4_CCFD_Model | [`src/aggregation/hq_aggregate.py`](file:///media/fyp-group-18/1TB-Hard/FYP-Group18/experiments/4_CCFD_Model/src/aggregation/hq_aggregate.py) line 37 | `frac = float(p2.get("fast_val_fraction", 0.15))` |
| 6_proposed_framework | [`fl-layer/validation/validate_fast.py`](file:///media/fyp-group-18/1TB-Hard/FYP-Group18/experiments/6_proposed_framework/fl-layer/validation/validate_fast.py) | `sample_fraction=0.15` limits the DataLoader size |

**Impact:** ↓ Latency — each CBFT cross-verification step runs in seconds instead of minutes.

---

## 8. Trust Score — Weighted FedAvg Global Aggregation

**What it is:** Standard FedAvg weights model contributions purely by dataset size. Trust-weighted FedAvg additionally factors in a historical trust score per bank, accumulated over multiple rounds. Banks that consistently submit high-quality models gain higher trust scores and have proportionally more influence on the global model. Poor-quality banks are down-weighted without full exclusion.

**How it is used here:** In `4_CCFD_Model`, scores are stored in `shared/scores.json` and updated with `reward=0.05` / `penalty=0.10` per round. In `6_proposed_framework`, trust scores are stored on the Hyperledger Fabric ledger and fetched via the API before the global FedAvg.

| Experiment | File | Key Code |
|---|---|---|
| 4_CCFD_Model | [`src/aggregation/hq_aggregate.py`](file:///media/fyp-group-18/1TB-Hard/FYP-Group18/experiments/4_CCFD_Model/src/aggregation/hq_aggregate.py) lines 128, 142 | `update_score(scores, brand_id, True/False, …)` → `save_scores(score_file, scores)` |
| 6_proposed_framework | [`fl-integration/global_aggregator.py`](file:///media/fyp-group-18/1TB-Hard/FYP-Group18/experiments/6_proposed_framework/fl-integration/global_aggregator.py) | Fetches `trust_scores` via `GET /trust-score/{bank_id}` → uses as FedAvg weights |

**Impact:** ↓ Communication Overhead (indirectly) — unreliable banks are down-weighted rather than triggering retries; fewer rounds needed to reach stable global quality.

---

## 9. CBFT Lightweight Boolean Cascade Consensus

**What it is:** Traditional consensus (PBFT, PoW) requires intensive multi-round message exchanges among all nodes. The CBFT scheme used here is a simple positional boolean-vote cascade: each bank downloads a peer's model from IPFS, runs the fast validator, and casts a single `True/False` vote on-chain. When a quorum threshold is reached (e.g., 2-of-3 banks vote True), consensus is immediately finalized without further messaging.

**How it is used here:** Banks poll `GET /check-consensus` which queries `GetConsensusState` on the chaincode. The chaincode counts `SubmitVerification` votes and marks a bank's update as `accepted=true` once quorum is met.

| Experiment | File | Key Code |
|---|---|---|
| 6_proposed_framework | [`fl-integration/hq_agent.py`](file:///media/fyp-group-18/1TB-Hard/FYP-Group18/experiments/6_proposed_framework/fl-integration/hq_agent.py) | `verify_peer_updates(round_num, banks)` → `POST /submit-verification` |
| 6_proposed_framework | [`fabric-network/chaincode/cbft/cbft.go`](file:///media/fyp-group-18/1TB-Hard/FYP-Group18/experiments/6_proposed_framework/fabric-network/chaincode/cbft/cbft.go) | `SubmitVerification` + `CheckConsensus` functions |

**Impact:** ↓ Latency — consensus in O(N) vote messages vs O(N²) in PBFT; no mining overhead.

---

## 10. Asynchronous REST API Gateway (FastAPI + Uvicorn)

**What it is:** Synchronous shell-out calls to Fabric peer CLI are blocking — the Python training process would stall waiting for each blockchain transaction to return. The FastAPI microservice decouples the ML pipeline from the peer CLI, exposing async HTTP endpoints. The ML layer fires non-blocking requests and can proceed with other work.

**How it is used here:** The `api-server` exposes endpoints like `POST /submit-update`, `GET /check-consensus`, `GET /trust-score/{bank_id}`. It handles all Fabric interaction internally using the Fabric Python SDK. The FL layer uses a lightweight `APIClient` that wraps `requests` calls.

| Experiment | File | Key Code |
|---|---|---|
| 6_proposed_framework | [`api-server/main.py`](file:///media/fyp-group-18/1TB-Hard/FYP-Group18/experiments/6_proposed_framework/api-server/main.py) | `@app.post("/submit-update")` async FastAPI route |
| 6_proposed_framework | [`fl-integration/api_client.py`](file:///media/fyp-group-18/1TB-Hard/FYP-Group18/experiments/6_proposed_framework/fl-integration/api_client.py) | `APIClient.submit_update(…)` → `requests.post(…)` |

**Impact:** ↓ Latency — training and blockchain I/O are decoupled; training continues while transactions propagate.

---

## 11. Consensus Timeout — Bounded Global Aggregation Wait

**What it is:** The global aggregator polls the blockchain for consensus state in a loop. Without a bound, a stalled network or a bank that never submits would hang the entire system indefinitely. A `consensus_timeout` cap ensures the aggregator proceeds with whichever banks have been accepted at expiration.

**How it is used here:** `GlobalAggregator` uses `poll_interval=2.0s` and `consensus_timeout=120.0s`. If consensus is not reached in 120 seconds, it aggregates whatever accepted banks are available.

| Experiment | File | Key Code |
|---|---|---|
| 6_proposed_framework | [`fl-integration/global_aggregator.py`](file:///media/fyp-group-18/1TB-Hard/FYP-Group18/experiments/6_proposed_framework/fl-integration/global_aggregator.py) | `while time.time() - start < self.consensus_timeout: poll …` |
| 6_proposed_framework | [`fl-integration/scripts/run_10_rounds.py`](file:///media/fyp-group-18/1TB-Hard/FYP-Group18/experiments/6_proposed_framework/fl-integration/scripts/run_10_rounds.py) | `GlobalAggregator(…, consensus_timeout=120.0)` |

**Impact:** ↓ Latency — global aggregation always completes within a bounded wall-clock time.

---

## Summary Table

| # | Technique | Primary Target | Experiments |
|---|---|---|---|
| 1 | Compact LSTM model (30 hidden units) | ↓ Comm Overhead | Both |
| 2 | DP gradient clipping + noise | ↓ Latency (convergence stability) | Both |
| 3 | Hierarchical 2-level FedAvg (Branch → HQ → Global) | ↓ Comm Overhead, ↓ Latency | Both |
| 4 | IPFS off-chain storage (CID on ledger only) | ↓ Comm Overhead, ↓ Latency | 6_proposed_framework |
| 5 | Deadline-based straggler exclusion | ↓ Latency | Both |
| 6 | Backup blending with previous global model | ↓ Latency (fewer wasted rounds) | 4_CCFD_Model |
| 7 | Fractional validation sampling (15%) | ↓ Latency | Both |
| 8 | Trust-weighted FedAvg global aggregation | ↓ Comm Overhead (indirectly) | Both |
| 9 | CBFT lightweight boolean vote consensus | ↓ Latency | 6_proposed_framework |
| 10 | Async FastAPI/Uvicorn gateway | ↓ Latency | 6_proposed_framework |
| 11 | Consensus timeout cap (120s) | ↓ Latency | 6_proposed_framework |
