# Major Design Decisions — HCFL Credit Card Fraud Detection

> **Project**: Optimizing Communication Efficiency and Latency in Blockchain-Enabled Federated Learning Systems  
> **Directories Covered**: `4_CCFD_Model/` (FL prototype) · `6_proposed_framework/` (full system)  
> **Date**: March 2026

This document records every significant technical and architectural decision made across the project, along with the alternatives that were considered and the rationale for each choice.

---

## Table of Contents

1. [Problem Framing & Dataset](#1-problem-framing--dataset)
2. [Model Architecture — LSTMTabular](#2-model-architecture--lstmtabular)
3. [Federated Learning Topology — Two-Tier Hierarchy](#3-federated-learning-topology--two-tier-hierarchy)
4. [Aggregation Algorithm — FedAvg with Trust Weighting](#4-aggregation-algorithm--fedavg-with-trust-weighting)
5. [Differential Privacy](#5-differential-privacy)
6. [Blockchain Platform — Hyperledger Fabric](#6-blockchain-platform--hyperledger-fabric)
7. [Off-Chain Storage — IPFS for Model Weights](#7-off-chain-storage--ipfs-for-model-weights)
8. [Consensus Protocol — CBFT (Custom BFT for FL)](#8-consensus-protocol--cbft-custom-bft-for-fl)
9. [REST API Bridge — FastAPI](#9-rest-api-bridge--fastapi)
10. [Validation Gate & Primary Metric — PR-AUC](#10-validation-gate--primary-metric--pr-auc)
11. [Resilience: Deadline-Based Straggler Handling](#11-resilience-deadline-based-straggler-handling)
12. [Resilience: Backup HQ & Model Blending](#12-resilience-backup-hq--model-blending)
13. [Trust Score System](#13-trust-score-system)
14. [Data Partitioning Strategy — Non-IID Silos](#14-data-partitioning-strategy--non-iid-silos)
15. [Containerization — Docker Compose per Branch](#15-containerization--docker-compose-per-branch)
16. [Code Refactoring: Pure Functions in `fl-layer`](#16-code-refactoring-pure-functions-in-fl-layer)
17. [Security: Replay Attack Protection in Chaincode](#17-security-replay-attack-protection-in-chaincode)
18. [Evaluation Metric Sampling — Fractional Validation](#18-evaluation-metric-sampling--fractional-validation)

---

## 1. Problem Framing & Dataset

### Decision
Use the **Kaggle Credit Card Fraud Detection** dataset (284,807 transactions, 30 PCA-anonymised features, 0.17% fraud rate) as the shared benchmark, partitioned into bank silos.

### Rationale
- **Real-world relevance**: The dataset is the de-facto benchmark for tabular fraud detection and is directly comparable to published baselines.
- **Imbalance challenge**: The severe class imbalance (fraud ≈ 0.17%) makes it a harder and more realistic FL scenario than balanced datasets.
- **Anonymisation fits the privacy narrative**: PCA features simulate the kind of anonymised data banks would realistically share without exposing raw customer records.
- **30 features**: Small enough to keep LSTM model dimensions tractable (input_dim=30, hidden_dim=30) without GPU requirements, enabling reproducible experiments on CPU-only machines.

### Alternatives Considered
| Alternative | Why rejected |
|-------------|--------------|
| Synthetic fraud dataset | Not comparable to published literature |
| PaySim | Less directly applicable to credit card use case |
| Full transaction logs | Privacy concerns; infeasible without real bank partnership |

---

## 2. Model Architecture — LSTMTabular

### Decision
Use a **single-layer LSTM** (`LSTMTabular`) with one fully-connected output head:

```python
class LSTMTabular(nn.Module):
    def __init__(self, input_dim=30, hidden_dim=30, num_layers=1):
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc   = nn.Linear(hidden_dim, 1)   # logit output
```

**Loss**: `BCEWithLogitsLoss` with class-imbalance positive weighting  
**Key config**: `epochs=1`, `lr=1e-3`, `batch_size=256`, `hidden_dim=30`

### Rationale
- **Temporal patterns**: LSTM naturally captures sequential dependencies within transaction streams, unlike a simple MLP that treats each transaction independently.
- **Compatibility between sites**: A fixed `input_dim=30` (matching the 30 PCA features) and `hidden_dim=30` ensures every branch and HQ works with identical architecture, making FedAvg weight averaging numerically well-defined.
- **Lightweight for FL**: A single LSTM layer keeps the model file small (~10 MB), minimising communication overhead per round.
- **Weight file stability**: The architecture was intentionally kept unchanged between `4_CCFD_Model` and `6_proposed_framework` so that `.pt` files produced by the prototype remain fully compatible with the production framework.
- **BCEWithLogitsLoss with pos_weight**: Addresses the extreme class imbalance by upweighting the loss on the minority (fraud) class: `pos_weight = n_negatives / n_positives`.

### Alternatives Considered
| Alternative | Why rejected |
|-------------|--------------|
| MLP (feedforward) | No temporal context; lower PR-AUC in experiments |
| Transformer encoder | Much heavier; overkill for 30-feature tabular data; slow to train per round |
| Multi-layer LSTM | Higher latency per training round; marginal accuracy gain on this dataset |
| XGBoost / tree ensemble | Not differentiable; incompatible with gradient-based FedAvg and DP |

---

## 3. Federated Learning Topology — Two-Tier Hierarchy

### Decision
Organise participants into a **two-tier hierarchy**:

```
Tier 1 (Intra-Bank):  Branch_i → HQ (FedAvg within one bank)
Tier 2 (Inter-Bank):  HQ_A + HQ_B + HQ_C → Global Aggregator
```

**Topology config** (from `4_CCFD_Model`):
```yaml
brands:
  brand_1:
    hq: brand_1_branch_0
    backup: brand_1_branch_1
    branches: [brand_1_branch_0, brand_1_branch_1, brand_1_branch_2]
```

In `6_proposed_framework` this maps to `BankA`, `BankB`, `BankC`, each with 2 Fabric peers (HQ = peer0, Backup = peer1).

### Rationale
- **Minimises global communication**: Only the HQ-level aggregated model ever leaves the bank's private network. Raw branch weights (×N per bank) never cross organisational boundaries, reducing inter-bank bandwidth by a factor of N (number of branches).
- **Mirrors real organisational structure**: Banks naturally operate with regional branches reporting to a headquarters; the hierarchy reflects this without forcing an artificial flat topology.
- **Isolates privacy boundaries**: Tier 1 operates entirely within a bank's intranet. Raw local data never leaves a branch. Only aggregated weight updates propagate upward, limiting the adversarial surface.
- **Straggler containment**: A slow branch only delays its own bank's Tier-1 round, not the global round, because the HQ can proceed after its deadline.

### Alternatives Considered
| Alternative | Why rejected |
|-------------|--------------|
| Flat FL (all branches → global aggregator) | Massive communication overhead; exposes per-branch updates globally |
| Three-tier (branch → regional → national → global) | Over-engineered for 3 banks × 3 branches; would need 4 rounds of synchronisation |
| Fully decentralised gossip | No deterministic consensus; hard to integrate with permissioned blockchain |

---

## 4. Aggregation Algorithm — FedAvg with Trust Weighting

### Decision
Use **Federated Averaging (FedAvg)** at both tiers, but with different weighting:

- **Tier 1 (intra-bank)**: sample-count-proportional FedAvg (`weight = n_samples_i / Σ n_samples`)
- **Tier 2 (inter-bank)**: **trust-weighted** FedAvg: `effective_weight = trust_score × num_samples`

```python
# global_aggregator.py
trust = trust_scores.get(bank_id, 1.0)
effective_weight = trust * num_samples
model_updates.append((sd, int(max(1, effective_weight * 1000))))
```

### Rationale
- **FedAvg is the standard baseline**: Well-understood, communication-efficient, and straightforward to implement with any model whose parameters are real-valued tensors.
- **Sample proportionality at Tier 1**: Ensures that a branch with more data exerts proportionally more influence in the bank's cluster model, which is statistically sound.
- **Trust weighting at Tier 2**: Banks have historically different data quality and participation rates. Weighting by trust score means that a bank with a long track record of high-quality updates is given more influence than a new or repeatedly penalised participant. This is a key contribution of the project — vanilla FedAvg treats all banks equally regardless of history.
- **Non-float tensor handling**: Integer tensors (e.g., LSTM's `num_batches_tracked`) are copied from the first model to avoid nonsensical averaging, a subtle correctness concern addressed explicitly in `fedavg.py`.

### Alternatives Considered
| Alternative | Why rejected |
|-------------|--------------|
| Uniform averaging | Ignores data-size differences; biased toward small contributors |
| FedProx | Adds proximal term overhead; minimal benefit over FedAvg on this dataset scale |
| Personalised FL (per-client models) | Defeats the purpose of global model convergence |
| Byzantine-robust aggregators (Krum, Bulyan) | Computationally expensive; CBFT consensus serves the same Byzantine-detection role |

---

## 5. Differential Privacy

### Decision
Apply **Gaussian DP noise** to local gradients during branch training:

```yaml
l2_norm_clip: 1.0       # gradient clipping bound
noise_multiplier: 0.05  # Gaussian noise σ = l2_norm_clip × noise_multiplier
```

Implementation: per-sample gradient clipping followed by noise addition (compatible with Opacus-style DP-SGD).

### Rationale
- **Regulatory alignment**: Financial institutions operate under GDPR and equivalent regulations. DP provides a mathematically rigorous privacy guarantee bounding information leakage from model updates.
- **Gradient inversion protection**: Without DP, published research shows that model gradients can be inverted to partially reconstruct training samples. The (ε, δ)-DP guarantee quantifies this risk.
- **Small noise multiplier (0.05)**: Chosen deliberately to provide a light privacy guarantee without seriously degrading model utility. At this scale the noise is small relative to signal; a larger multiplier (e.g., 1.0) would require more rounds to converge.
- **Clip norm = 1.0**: Standard choice that prevents a single large-gradient sample from dominating the update and amplifies the per-sample noise effect.

### Alternatives Considered
| Alternative | Why rejected |
|-------------|--------------|
| No DP | Unacceptable in a banking scenario; violates regulatory requirements |
| Local DP (noise at data level) | Much stronger noise required; would destroy model utility |
| Secure aggregation (cryptographic MPC) | Far more implementation complexity; DP+blockchain serves the same trust goal |
| Larger noise_multiplier (0.5–1.0) | Empirically degraded PR-AUC significantly in initial experiments |

---

## 6. Blockchain Platform — Hyperledger Fabric

### Decision
Use **Hyperledger Fabric** as the permissioned blockchain with:
- **3 organisations** (BankA, BankB, BankC), each with 2 peers (peer0=HQ, peer1=Backup)
- **Raft consensus** orderer service
- **1 channel**: `fraud-detection-global`
- **Custom chaincode**: `cbft-fl` (Go, using `fabric-contract-api-go`)

### Rationale
- **Permissioned access**: In a real banking consortium, participants are known and regulated entities. Fabric's MSP (Membership Service Provider) enforces identity at the TLS level, which is far more appropriate than a public, proof-of-work chain.
- **No cryptocurrency overhead**: Fabric has no native token or mining concept, removing the financial and computational overhead of public blockchains.
- **Smart contract (chaincode) flexibility in Go**: Go chaincode allows complex validation logic (CBFT quorum counting, trust score updates, replay attack detection) to run deterministically on-chain.
- **Raft orderer**: Simpler and faster than Kafka for a 3-org consortium; provides crash-fault tolerance without the complexity of Byzantine-fault-tolerant orderer consensus (which is handled at the FL level by CBFT).
- **Immutability for audit**: Every model CID, hash, trust score update, and verification vote is permanently recorded on the ledger, providing a cryptographically verifiable audit trail for regulators.

### Alternatives Considered
| Alternative | Why rejected |
|-------------|--------------|
| Ethereum (public) | Mining overhead; permissionless; gas fees; not suitable for production banking |
| Ethereum (permissioned / private) | Less tooling support; EVM gas model complicates on-chain FL logic |
| Hyperledger Besu | More Ethereum-compatible but less suited to multi-org channel model |
| Corda (R3) | Designed for bilateral contracts, not multi-party global model aggregation |
| Simple ledger file (no chain) | No tamper-evidence; no decentralised consensus |

---

## 7. Off-Chain Storage — IPFS for Model Weights

### Decision
Store **model weight files on IPFS** (Kubo daemon, local node). The blockchain only records:
- `model_cid` — IPFS Content Identifier (CID, SHA-256 content hash)
- `model_hash` — independent SHA-256 of the raw bytes (double verification)

### Rationale
- **Block size limits**: A Hyperledger Fabric block has a default size limit of 1 MB (configurable). A single `LSTMTabular` model is ~10 MB. Storing weights on-chain directly would be infeasible.
- **Content addressing is self-verifying**: IPFS CIDs are derived from the file content itself. If a malicious actor modifies the weight file, the CID changes, immediately making the mismatch detectable.
- **Separation of concerns**: The blockchain is optimised for small, high-integrity records. IPFS is optimised for large file distribution. Using each for its strength avoids trying to force a blockchain into a file system role.
- **Double hash verification**: Before loading any peer's model, the system independently recalculates `SHA-256` of the downloaded bytes and compares it to both the CID and the on-chain `model_hash`. This closes the gap between IPFS addressing and ledger integrity.
- **Decentralisation**: IPFS is inherently peer-to-peer. Any bank can pin content and serve it to others without a single point of failure.

### Alternatives Considered
| Alternative | Why rejected |
|-------------|--------------|
| Store weights directly on Fabric | Block size limit; high ledger bloat; slow endorsement for large payloads |
| Central file server / S3 | Single point of failure / trust; no content addressing |
| Filecoin | More complex incentive layer; unnecessary for a consortium setting with trusted peers |
| Shared network filesystem (NFS) | Centralised; not resilient; no cryptographic integrity |

---

## 8. Consensus Protocol — CBFT (Custom BFT for FL)

### Decision
Implement a **3-phase Consensus-Based Federated Trust (CBFT)** protocol executed on-chain:

| Phase | Smart Contract Function | Quorum |
|-------|------------------------|--------|
| 1 — Propose | `SubmitClusterUpdate` | `val_score ≥ 0.7` enforced on-chain |
| 2 — Verify  | `SubmitVerification` | `VerifyQuorum = 2` positive votes |
| 3 — Commit  | `SubmitCommit`       | `CommitQuorum = 2` commits |

A bank's update is considered **Accepted** only after passing all three phases (`CheckConsensus`).

### Rationale
- **Byzantine fault tolerance**: With 3 banks, a quorum of 2 means a single malicious bank cannot unilaterally accept its own poisoned model; two other banks must independently verify it.
- **Cross-bank model validation**: Each verifying HQ downloads the peer's model from IPFS, evaluates it on their own local validation data, and votes only if `PR-AUC ≥ threshold`. This catches gradient poisoning attacks that pass a bank's self-reported `val_score`.
- **On-chain vote recording**: All votes are immutable ledger records; there is no off-chain signalling channel a bank could manipulate to fake consensus.
- **Self-verification prevention**: Both the API server and chaincode enforce `verifier_id ≠ target_bank_id`, preventing a bank from voting for itself.
- **Replay attack protection**: Each `modelCID` is recorded in the chaincode state (`cid~{cid}` key) on first submission; re-submission of the same CID fails immediately.

### Alternatives Considered
| Alternative | Why rejected |
|-------------|--------------|
| No cross-bank verification (trust self-reported score) | Trivially defeated by a bank inflating its `val_score` |
| PBFT (classical) | Requires synchronous messaging rounds at the application level; complex to implement correctly |
| Nakamoto-style probabilistic consensus | Introduces latency and non-finality unacceptable for FL rounds with tight deadlines |
| Threshold signature schemes | Cryptographic complexity; overkill when the verifier computation is already done in Python |

---

## 9. REST API Bridge — FastAPI

### Decision
Use a **FastAPI (Python, async) microservice** (`api-server/main.py`) running under `uvicorn` as the interface between the Python FL pipeline and the Go-based Hyperledger Fabric CLI.

Endpoints include: `POST /submit-update`, `POST /submit-verification`, `POST /submit-commit`, `POST /store-global-model`, `GET /trust-scores`, `GET /check-consensus/{round}`, `GET /global-model/{round}`.

### Rationale
- **Language boundary**: The FL agents are Python (PyTorch). The Fabric chaincode is Go. Fabric's primary client SDK is also Go, or Node.js. A REST bridge allows the Python FL code to call Fabric without adopting a new SDK language in the critical training path.
- **Async I/O**: FastAPI with `uvicorn`'s ASGI server handles multiple concurrent HQ agent requests without blocking on Fabric CLI round-trips, preventing training-side bottlenecks.
- **OpenAPI documentation**: FastAPI auto-generates an `/docs` Swagger UI, making the API self-documenting and testable without additional tooling.
- **Testability**: The `APIClient` Python wrapper (`fl-integration/api_client.py`) is injectable — unit tests can stub it without a live Fabric network, enabling fast CI.

### Alternatives Considered
| Alternative | Why rejected |
|-------------|--------------|
| Fabric Python SDK (fabric-sdk-py) | Unmaintained; lacked full support for endorsement policy control |
| Go FL agents | Too deep a rewrite of the existing PyTorch pipeline |
| gRPC bridge | More complex contract definition; REST is sufficient for the latency requirements |
| Fabric Gateway API (direct) | Requires the Python process to hold Fabric credentials; harder to isolate in tests |

---

## 10. Validation Gate & Primary Metric — PR-AUC

### Decision
Gate model acceptance at both tiers on **PR-AUC (Precision-Recall Area Under Curve)**:

- **Tier 1 (intra-bank, local)**: `PR-AUC ≥ 0.20` (prototype threshold, `4_CCFD_Model`)
- **Tier 1 (production)**: configurable via `fl_config.yaml` (`validation_threshold: 0.20`)
- **Tier 2 (on-chain)**: `val_score ≥ 0.7` enforced by chaincode (`ClusterValThreshold = 0.7`)
- **Fast evaluation**: sample only 15% of validation data (`sample_fraction=0.15`)

### Rationale
- **PR-AUC over ROC-AUC for imbalanced data**: ROC-AUC is known to be over-optimistic on highly imbalanced datasets because it weights true-negative performance equally with true-positive performance. Fraud detection cares overwhelmingly about correctly identifying the minority positive class. PR-AUC is both a stricter and a more budget-conscious metric for this use case.
- **Two different thresholds**: The Tier 1 threshold (0.20) is deliberately lenient to avoid excluding early-round models that haven't yet converged. The on-chain Tier 2 threshold (0.7) is tighter because it guards the global model, which affects all banks.
- **15% sampling**: Running full validation on every model at every round would add significant latency. 15% of a typical local silo (~10,000 samples) gives ~1,500 samples — statistically sufficient for a reliable PR-AUC estimate while keeping evaluation time under 1 second on CPU.
- **Additional metrics logged**: ROC-AUC and F1 are also computed and logged for analysis even though PR-AUC is the gate metric.

### Alternatives Considered
| Alternative | Why rejected |
|-------------|--------------|
| Accuracy | Useless for 0.17% fraud rate; 99.83% accuracy trivially from predicting "no fraud" |
| ROC-AUC only | Over-optimistic on imbalanced data; masks recall failures |
| F1 score | Threshold-dependent; requires choosing a decision threshold separately |
| Full validation (100% data) | Adds significant latency per round; unacceptable for tight deadlines |

---

## 11. Resilience: Deadline-Based Straggler Handling

### Decision
At both tiers, use a **configurable deadline** after which aggregation proceeds with whatever updates have arrived:

```yaml
# fl_config.yaml
fl:
  deadline_seconds: 5.0
  min_branches_required: 2

# 4_CCFD_Model config
phase2:
  deadline_sec: 25
  min_models_required: 2
```

Implemented in `fl-layer/resilience/deadline_collect.py` (`wait_for_submissions`) and `fl-integration/round_coordinator.py` (`collect_branch_updates`).

### Rationale
- **Avoids head-of-line blocking**: In a synchronous FL setup, one slow branch can stall an entire round. The deadline ensures the round completes within a predictable time window regardless of straggler behaviour.
- **Partial aggregation is still useful**: FedAvg on N-1 out of N branch models still produces a meaningful aggregate. Waiting for the Nth model past the deadline provides diminishing marginal benefit.
- **Configurable per environment**: The 25-second deadline in `4_CCFD_Model` is appropriate for Docker containers on a single machine. The 5-second deadline in `6_proposed_framework` assumes network delays are minimal in the lab setup. Both are tunable.
- **Minimum model guard**: `min_branches_required=2` prevents aggregation from proceeding if only 1 update arrived, which would be identical to no aggregation at all.

### Alternatives Considered
| Alternative | Why rejected |
|-------------|--------------|
| Synchronous wait (no deadline) | Single slow branch locks the entire round indefinitely |
| Fixed-fraction collection (e.g., wait for 2/3) | Harder to tune; doesn't naturally handle dropout or node failure |
| Asynchronous FL (updates applied immediately as they arrive) | No clear round structure; incompatible with blockchain-round semantics |

---

## 12. Resilience: Backup HQ & Model Blending

### Decision
Each bank designates a **Backup HQ** (peer1, configured in topology). When the primary HQ fails:
1. Backup activates via `ActivateBackup` chaincode call (records `backupActive=true` on ledger)
2. Backup takes over `run_round()` for that bank

When a cluster model **fails the validation gate**, instead of discarding it, blend it with the previous global model:

```python
# beta=0.3 in production (fl_config.yaml: backup_beta: 0.3)
# beta=0.7 in prototype (4_CCFD_Model: blend_beta: 0.70)
w_recovered = beta * w_global + (1 - beta) * w_branch
```

Then re-validate the blended model; only then exclude if it still fails.

### Rationale for Backup HQ
- **Single point of failure elimination**: Without a backup, a crashed HQ leaves its bank unable to participate until manually restarted, potentially excluding it from multiple rounds and triggering trust-score penalties unfairly.
- **Ledger-recorded failover**: Other banks can query the ledger to see that `backupActive=true` and understand why the model submission came from a secondary peer, maintaining auditability.

### Rationale for Model Blending (varying beta)
- **Conservative baseline preserves convergence**: Blending with the previous global pulls the unstable cluster update toward a known-good baseline. This is particularly useful in early rounds where local models haven't yet converged.
- **Beta evolution between versions**: The prototype (`4_CCFD_Model`) used `beta=0.7` (70% global, 30% new), which was very conservative. Production (`6_proposed_framework`) uses `beta=0.3` (30% global, 70% new). This shift reflects learning from prototype experiments — the new local model tends to carry more useful information once the global model has already converged to a decent baseline.
- **Second-chance before exclusion**: Excluding a bank entirely has a trust-score cost. Attempting recovery first avoids unfair penalty for a model that was simply noisy rather than malicious.

### Alternatives Considered
| Alternative | Why rejected |
|-------------|--------------|
| No backup; skip round on HQ failure | Trust-score penalty accumulates unfairly for infrastructure failures |
| Full replacement by global model | Loses all local training signal; equivalent to that bank sitting out the round |
| Re-training from scratch on failure | Too slow; would hold up the global round |

---

## 13. Trust Score System

### Decision
Maintain an on-chain **dynamic trust score** per bank, updated after each round's global aggregation:

**Chaincode constants** (from `cbft.go`):
```go
InitialTrustScore = 1.0
ScoreMin          = 0.1   // floor — prevents permanent exclusion
Alpha             = 0.1   // reward (model improved global)
Beta              = 0.2   // penalty (model degraded global)
```

**Config-level** (from `4_CCFD_Model`):
```yaml
scoring:
  init_score: 1.0
  floor: 0.2
  max: 3.0
  reward: 0.05
  penalty: 0.10
```

The trust score directly multiplies `num_samples` in the Tier-2 trust-weighted FedAvg:
`effective_weight = trust_score × num_samples`

### Rationale
- **Long-term incentive alignment**: A bank that consistently submits high-quality updates earns a growing weight in the global model. A bank that repeatedly fails validation has diminishing influence. This naturally discourages free-riding or passive participation.
- **Score floor prevents permanent exclusion**: Setting `ScoreMin = 0.1` (chaincode) or `floor = 0.2` (prototype) ensures even a heavily penalised bank retains a minimal stake in the global model. This matters because complete exclusion might incentivise a desperate bank to attempt more aggressive attacks.
- **Asymmetric reward/penalty**: The penalty magnitudes (Beta=0.2, prototype penalty=0.10) are larger than reward magnitudes (Alpha=0.1, prototype reward=0.05). This is intentional — it is harder to rebuild trust than to lose it, which is consistent with real-world reputation dynamics and discourages intermittent misbehaviour.
- **Stored on-chain for verifiability**: Trust scores are part of the shared ledger state. Any bank can independently verify another's score by querying `GetTrustScores`. The immutable history of updates provides an auditable reputation record.

### Alternatives Considered
| Alternative | Why rejected |
|-------------|--------------|
| Uniform weight for all banks | No incentive for consistent participation |
| Off-chain reputation (central database) | Not tamper-proof; single point of trust |
| Binary accept/exclude (no score) | Binary decisions are too harsh; no gradation of trust |
| Data-volume-only weighting | Ignores historical quality; large-data but low-quality banks would dominate |

---

## 14. Data Partitioning Strategy — Non-IID Silos

### Decision
Split the Kaggle credit card dataset into **non-IID bank silos** using the pipeline:

```
raw/creditcard.csv
  → 1_feature_scaled/    (global StandardScaler)
  → 2_bank_silos/        (split by bank)
  → 3_local_silo_balancing/  (per-branch local class balancing)
```

Each branch receives its own `train_ready.csv` with `StandardScaler` applied **locally** (per branch).

### Rationale
- **Non-IID simulates reality**: Real bank branches serve different customer demographics, merchant categories, and geographic regions. Their fraud distributions will differ. Using non-IID splits more faithfully reflects the challenge that motivates federated learning in the first place.
- **Local scaling prevents information leakage**: Applying `StandardScaler` globally (with global mean/variance) would allow each branch to infer global statistics from the scaler parameters, which is a privacy leak. Per-branch scaling ensures each node only sees its own distribution.
- **Local class balancing**: Each branch's silo is balanced locally to ensure the minority class (fraud) is not entirely absent from small branches, which would make local training degenerate.
- **Fixed processed path (`3_local_silo_balancing`)**: Storing processed splits on disk means training runs are reproducible and don't re-process data each round.

### Alternatives Considered
| Alternative | Why rejected |
|-------------|--------------|
| IID random split | Unrealistic; removes the core FL motivation |
| Dirichlet-partitioned split | Could be used in future work for more sophisticated heterogeneity control |
| Global scaling | Privacy-leaking; branches observe global stats |
| Raw features without any preprocessing | LSTM performance degrades significantly on unscaled tabular data |

---

## 15. Containerization — Docker Compose per Branch

### Decision
In `4_CCFD_Model`, run each branch's `local_train.py` in an **isolated Docker container** via `docker-compose.yml`:

```yaml
services:
  brand_1_branch_0:
    image: fyp-bank:latest
    volumes:
      - ../data/processed/3_local_silo_balancing:/data  # read-only
      - ${LOGS_DIR}:/logs                                # round-specific write
    environment:
      - BANK_ID=brand_1_branch_0
```

In `6_proposed_framework`, the FL-integration layer uses Docker Compose for bank-level isolation (`fl-integration/docker-compose.yml`, `Dockerfile.bank`).

### Rationale
- **Isolation enforces data privacy**: The container mount scheme means branch data directories are strictly read-only, and output directories are per-round. A branch container cannot access another branch's data even if the code is compromised.
- **Parallel local training**: `docker compose up` launches all 9 branch containers simultaneously and in parallel, matching the FL assumption that local training is concurrent.
- **Reproducibility**: The same Docker image (`fyp-bank:latest`) runs identically on any machine with the same environment, eliminating "works on my machine" issues.
- **Round isolation via `LOGS_DIR`**: Each training round mounts a different directory into `/logs`, preventing models from a previous round from accidentally being read as outputs of the current round.

### Alternatives Considered
| Alternative | Why rejected |
|-------------|--------------|
| Sequential Python subprocesses | No parallel training; slower; no isolation |
| Kubernetes | Over-engineered for a 9-branch lab setup |
| Virtual machines per branch | Enormous overhead; slow startup; excessive resource use |
| Single process with multiprocessing | Shared memory space violates the isolation assumption; harder to reproduce |

---

## 16. Code Refactoring: Pure Functions in `fl-layer`

### Decision
When evolving from `4_CCFD_Model` to `6_proposed_framework`, all core FL functions were extracted into a **pure-function `fl-layer` package** with no filesystem or network side effects:

- `fl-layer/aggregation/fedavg.py` — no file I/O, no config reads
- `fl-layer/resilience/backup_logic.py` — accepts `state_dict` arguments directly instead of reading from `round_dir`
- `fl-layer/model/FL_model.py` — no weight saving/loading on import
- `fl-layer/validation/validate_fast.py` — injectable dataset

**The key change** from the prototype (which read from disk inside aggregation functions):
```python
# BEFORE (4_CCFD_Model): filesystem-coupled
brand_model_alt = blend_with_prev_global(round_dir, brand_model, beta=0.70)

# AFTER (6_proposed_framework): pure function, injectable
avg_sd = blend_with_global(avg_sd, global_sd, beta=beta)
```

### Rationale
- **Testability**: Pure functions can be unit-tested without a real filesystem, Docker network, or Fabric network. All external I/O in `HQAgent` and `GlobalAggregator` is injected as callable arguments, so tests can pass stubs.
- **Separation of concerns**: Business logic (averaging weights, blending, validation) should not be tangled with infrastructure concerns (file paths, network calls). This made it possible to unit-test the FL math independently of the blockchain integration.
- **Model file compatibility**: Keeping `LSTMTabular` architecturally identical between prototype and production ensured that weights trained in `4_CCFD_Model` could be loaded directly in `6_proposed_framework` without conversion.

---

## 17. Security: Replay Attack Protection in Chaincode

### Decision
In `cbft.go`, every `SubmitClusterUpdate` call records the submitted `modelCID` in the world state under the key `cid~{modelCID}`. If the same CID is submitted again, the chaincode rejects it:

```go
if cidBytes != nil {
    return fmt.Errorf("replay attack detected, modelCID %s already exists", modelCID)
}
```

### Rationale
- **CID reuse is a valid attack vector**: Without this check, a malicious (or buggy) bank could re-submit a model from a previous round to try to influence the current round's aggregation. Since IPFS CIDs are content-addressed (deterministic from file content), the same weights always produce the same CID.
- **Immutable detection**: Once a CID is recorded on the ledger, it cannot be deleted (Fabric does not support deletes that remove history). The detection is therefore permanent and auditable.
- **Cheap to enforce**: The check is a single `GetState` ledger read in the chaincode — effectively zero overhead.

---

## 18. Evaluation Metric Sampling — Fractional Validation

### Decision
All validation during training (`hq_agent.py`, `validate_fast.py`) uses **15% random sampling** of the validation set (`sample_fraction=0.15`).

### Rationale
- **Latency**: Full validation on every branch's full test set at every round would add 5–10 seconds per HQ per round on CPU hardware. With 3 banks × 10 rounds, this accumulates to minutes of pure evaluation overhead with no training benefit.
- **Statistical sufficiency**: A 15% sample of even a 5,000-sample validation set gives 750 samples. For PR-AUC estimation on a dataset with ~0.17% fraud, this includes roughly 1–2 fraudulent samples per sample. While small, this is consistent across rounds and sufficient to distinguish a good model from a bad one.
- **Consistent across peers**: All HQs use the same fraction, so verification votes in CBFT Phase 2 are based on comparable measurement effort.

### Alternatives Considered
| Alternative | Why rejected |
|-------------|--------------|
| Full dataset evaluation | ~7× slower per validation call; unacceptable for tight round deadlines |
| Fixed N samples | Less robust to different dataset sizes across branches |
| Evaluating only on fraud samples | Biased estimate; does not penalise excessive false positives |

---

## Summary Table

| # | Decision | Key Technology | Primary Goal |
|---|----------|---------------|--------------|
| 1 | Kaggle CCFD dataset | CSV, silo split | Realistic benchmark |
| 2 | LSTMTabular model | PyTorch, BCEWithLogitsLoss | Temporal fraud detection |
| 3 | Two-tier FL hierarchy | Branches → HQ → Global | Reduce cross-bank communication |
| 4 | FedAvg + trust weighting | `fedavg.py`, `global_aggregator.py` | Fair & historical-quality-aware average |
| 5 | Differential privacy | Gradient clip + Gaussian noise | Regulatory compliance, prevent gradient inversion |
| 6 | Hyperledger Fabric | Go chaincode, Raft, MSP | Permissioned, auditable consensus |
| 7 | IPFS for model storage | CID, SHA-256 double-hash | Off-chain large files, content integrity |
| 8 | CBFT 3-phase consensus | `cbft.go` | Byzantine fault tolerance, cross-bank verification |
| 9 | FastAPI REST bridge | Python AsyncIO + Uvicorn | Language-agnostic Fabric integration |
| 10 | PR-AUC validation gate | `validate_fast.py` | Imbalance-robust quality gate |
| 11 | Deadline-based straggler handling | `deadline_collect.py` | Bound round latency |
| 12 | Backup HQ + model blending | `backup_logic.py`, `ActivateBackup` | High availability, recovery without penalty |
| 13 | Dynamic trust scores | On-chain `TrustScore` struct | Long-term participation incentive |
| 14 | Non-IID silo partitioning | `3_local_silo_balancing` | Realistic heterogeneous FL scenario |
| 15 | Docker Compose per branch | `docker-compose.yml` | Isolation, parallelism, reproducibility |
| 16 | Pure functions in `fl-layer` | Injected I/O callables | Testability, separation of concerns |
| 17 | Replay attack protection | CID state tracking in chaincode | Security hardening |
| 18 | 15% fractional validation | `sample_fraction=0.15` | Balance accuracy vs. round latency |

---

*Document maintained in `6_proposed_framework/docs/decisions_made.md`.*
