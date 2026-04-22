# Two-Tier Hierarchy in the HCFL Framework

> **System**: Hierarchical Clustered Federated Learning (HCFL) for Credit Card Fraud Detection  
> **File**: `experiments/6_proposed_framework/`  
> **Date**: 2026-03-17

---

## Overview

The HCFL framework is built around a **two-tier aggregation hierarchy** — a deliberate architectural choice that splits federated learning into two distinct communication scopes: one that operates *inside* a single bank, and one that operates *across* multiple banks through a blockchain consortium. This separation solves three fundamental problems simultaneously:

1. **Communication overhead**: Banks do not broadcast raw local branch gradients to the world — only one consolidated cluster update per bank reaches the global tier.
2. **Straggler resilience**: Slow or failing nodes at either tier are handled by deadline-based timeouts without stalling the entire system.
3. **Byzantine fault tolerance**: The cross-bank tier uses cryptographic consensus (CBFT) before any model is accepted into the global aggregate, preventing a rogue bank from poisoning the global model.

---

## The Actors

Before diving into the tiers, it is essential to understand who does what:

| Actor | Scope | Source File | Role |
|---|---|---|---|
| **Branch Node** | Intra-bank (Tier 1) | `fl-layer/training/local_train.py` | Trains model locally on its own private data slice |
| **HQ Agent** | Intra-bank + cross-bank boundary | `fl-integration/hq_agent.py` | Aggregates branch models, submits to blockchain, verifies peers |
| **Round Coordinator** | System-wide orchestrator | `fl-integration/round_coordinator.py` | Times and sequences collection → HQ rounds → global aggregation |
| **Global Aggregator** | Inter-bank (Tier 2) | `fl-integration/global_aggregator.py` | Trust-weighted FedAvg across all banks, stores result on-chain |
| **Hyperledger Fabric Chaincode** | Blockchain ledger | `fabric-network/chaincode/` | Immutable record of updates, verifications, trust scores, global model |
| **IPFS Node** | Off-chain storage | `fabric-network/ipfs/` | Stores large model weight files; only the hash goes on-chain |

---

## Tier 1 — Intra-Cluster Aggregation (Branch → HQ)

### Definition

**Tier 1** is the aggregation that happens *within a single bank's private network*. Each bank is called a **cluster**, and the local branches of that bank are the cluster's participants. The bank's Headquarter (HQ) server is the **cluster aggregator**.

### Step 1.1 — Local Branch Training with Differential Privacy

**Entity**: Individual branch nodes (e.g., regional offices of Bank A)  
**Source**: `fl-layer/training/local_train.py :: train_local()`

Each branch holds a **non-IID partitioned** slice of the credit card transaction dataset. Non-IID (non-independent, non-identically distributed) means each branch's data reflects its own geographic spending patterns, making the fraud signals locally unique and privacy-sensitive.

The training uses a **`LSTMTabular`** model — a Long Short-Term Memory network adapted for tabular transaction data. The branch trains for a configured number of **local epochs** using the Adam optimizer.

What makes this training privacy-preserving is the **Differential Privacy (DP)** loop. DP is a mathematical guarantee that the model update cannot be reverse-engineered to reveal any individual transaction. The exact DP mechanism applied is:

```
1. loss.backward()                  → compute per-batch gradients
2. clip_grad_norm_(l2_norm_clip=1.0) → clip gradient norm to bound sensitivity
3. grad += randn_like(grad) * (l2_norm_clip * noise_multiplier=0.05)
                                     → add calibrated Gaussian noise
4. optimizer.step()                  → update weights AFTER clip+noise
```

The **clipping** bounds the maximum influence any single sample can have (limiting sensitivity $\Delta f$). The **Gaussian noise** adds controlled randomness that prevents the gradient from uniquely fingerprinting individual records. The result satisfies $(\\varepsilon, \\delta)$-differential privacy.

**Output**: A `state_dict` — a Python dictionary of `{layer_name: weight_tensor}` pairs — representing the branch's updated model weights. **No raw transaction data ever leaves the branch.**

### Step 1.2 — Deadline-Aware Branch Collection

**Entity**: Round Coordinator / HQ  
**Source**: `fl-layer/resilience/deadline_collect.py :: wait_for_submissions()`

The HQ cannot wait forever for branches. `wait_for_submissions()` polls a `collect_fn` every `poll_interval` seconds (default 0.5 s). It returns as soon as either:
- All `expected_count` branch updates have arrived, **or**
- The `deadline_sec` wall-clock limit is reached.

This is the **straggler mitigation** mechanism at Tier 1. Branches that are slow, offline, or training on unusually large partitions are simply excluded from the current round. The round proceeds with whatever has arrived by the deadline, preventing one slow branch from blocking the bank's entire participation in the global round.

### Step 1.3 — Intra-Cluster FedAvg (Cluster Model Generation)

**Entity**: HQ Agent  
**Source**: `fl-layer/aggregation/fedavg.py :: fedavg()` called from `fl-integration/hq_agent.py :: run_round()`

Once branch updates are collected, the HQ performs **Federated Averaging (FedAvg)** — the canonical aggregation algorithm in federated learning. FedAvg computes a sample-count-weighted average of all branch model weights:

$$\theta_{\text{cluster}} = \frac{\sum_{i=1}^{B} n_i \cdot \theta_i}{\sum_{i=1}^{B} n_i}$$

Where:
- $B$ = number of branch updates collected before the deadline
- $n_i$ = number of training samples used by branch $i$
- $\theta_i$ = `state_dict` (weight vector) of branch $i$'s model

Branches that trained on more data get proportionally larger influence on the **Cluster Model** $\theta_{\text{cluster}}$. The `fedavg()` function operates purely on in-memory `state_dict` dictionaries — no file system I/O, no network calls. Non-floating-point tensors (e.g., LSTM batch trackers) are copied from the first model unchanged.

**Output**: A single `state_dict` — the **Cluster Model** — which is a weighted synthesis of all branches' locally learned fraud detection knowledge.

### Step 1.4 — Global Model Blending (Resilience)

**Entity**: HQ Agent  
**Source**: `fl-layer/resilience/backup_logic.py :: blend_with_global()`

If a previous global model exists (i.e., it is not round 1), the HQ fetches it from IPFS (verified by SHA-256 hash against the on-chain record). Instead of discarding it, the cluster model is **blended** with the prior global model:

$$\theta_{\text{blended}} = \beta \cdot \theta_{\text{global}} + (1 - \beta) \cdot \theta_{\text{cluster}}$$

With the default $\beta = 0.30$, the blended model is 70% the new cluster knowledge and 30% the previously validated global knowledge. This prevents catastrophic forgetting — the scenario where a new round of purely local training completely overwrites globally learned patterns. It also acts as a **backup mechanism**: if a bank has very few branches respond before the deadline, blending with the global model keeps the submission meaningful rather than garbage.

### Step 1.5 — Intra-Cluster Validation (Quality Gate)

**Entity**: HQ Agent  
**Source**: `fl-layer/validation/validate_fast.py :: evaluate_model()`

Before the cluster model is allowed to cross into Tier 2 (the blockchain), the HQ evaluates it against an internal **validation dataset** using only a 15% random sample for speed (`sample_fraction=0.15`). The metric computed is **PR-AUC** (Precision-Recall Area Under the Curve):

- PR-AUC is the chosen metric because credit card fraud is **heavily class-imbalanced** (typically <0.5% fraud rate). Accuracy is useless for imbalanced data; PR-AUC directly measures how well the model ranks fraudulent transactions above legitimate ones.
- A threshold $\tau$ (validation threshold, loaded from config) gates submission: if `pr_auc < τ`, the cluster model is **not submitted** to the blockchain. The bank effectively abstains from this round rather than polluting the global aggregate with a degraded update.

Additional metrics computed (but secondary): `roc_auc`, `f1`, `precision`, `recall`.

---

## Tier 2 — Inter-Cluster Aggregation (HQ → Blockchain → Global)

### Definition

**Tier 2** is the aggregation that happens *across the bank consortium* — a trustless, multi-organizational setting where no bank can be implicitly trusted by any other. This is where the blockchain (Hyperledger Fabric) and the CBFT consensus protocol become essential.

### Step 2.1 — IPFS Upload and On-Chain Registration

**Entity**: HQ Agent  
**Source**: `fl-integration/hq_agent.py :: run_round()` — steps 5–6

Because Hyperledger Fabric blocks cannot hold large binary objects like PyTorch model weights (typical model size: megabytes), the system uses a **hybrid storage pattern**:

1. The cluster model's `state_dict` is serialized to bytes using `torch.save()`.
2. The bytes are uploaded to **IPFS** (InterPlanetary File System), the decentralized content-addressable storage layer. IPFS returns a **CID** (Content Identifier) — a cryptographic hash-derived address that uniquely and immutably identifies the file content.
3. The HQ independently computes a **SHA-256 hash** of the same byte stream.
4. The HQ calls `POST /submit-update` on the REST API, which invokes the Fabric chaincode to record `{bank_id, round_num, model_cid, model_hash, val_score}` on the ledger.

The blockchain now has a **tamper-evident receipt**: the CID tells peers where to find the model, and the hash lets them verify the file has not been altered after upload.

### Step 2.2 — CBFT Cross-Verification (Consensus Phase)

**Entity**: All peer HQ Agents across the consortium  
**Source**: `fl-integration/hq_agent.py :: verify_peer_updates()`

**CBFT** (Consensus-Based Federated Trust) is the custom Byzantine-fault-tolerant consensus protocol layered on top of Fabric's Raft ordering service. It has three explicit phases modeled after PBFT (Practical Byzantine Fault Tolerance):

#### Phase 1 — Pre-Prepare (Proposal)
The submitting HQ's `submit_update` transaction is the implicit pre-prepare broadcast. All peers can see the proposal on the shared ledger.

#### Phase 2 — Prepare (Cross-Verification)
Each HQ fetches peer updates from the chaincode (`GET /cluster-update/{bank_id}/{round_num}`), then:

1. Downloads the peer's model from IPFS using the `model_cid`.
2. Recomputes SHA-256 of the downloaded bytes.
3. Compares against the `model_hash` stored on-chain. A **mismatch** → the model was tampered with after upload → automatic `False` vote.
4. If the hash matches, loads the model into a `LSTMTabular` instance and evaluates it using `evaluate_model()` on the verifying HQ's **own** validation set (15% sample).
5. If `pr_auc >= τ`: casts a `True` (verified) vote. If below threshold: casts a `False` (rejected) vote.
6. Submits the boolean vote via `POST /submit-verification` to the chaincode.

This is the critical security mechanism: **no bank can self-certify its own model**. A Byzantine (malicious) bank that submits a backdoored or garbage model will receive `False` votes from honest peers who independently evaluate it. The chaincode tallies all votes transparently on the ledger.

#### Phase 3 — Commit (Quorum Finalization)
Each HQ checks whether a **quorum** of positive verifications has accumulated (`GET /check-verify-quorum`). With 3 banks in the consortium, a quorum of ≥ 2 `True` votes is required (majority). If quorum is met, the HQ calls `POST /submit-commit` to lock the bank's model as **"Accepted"** on the ledger. Only "Accepted" models are eligible for global aggregation.

**Source**: `fl-integration/hq_agent.py :: commit_peer_updates()`

### Step 2.3 — Consensus Polling by the Global Aggregator

**Entity**: Global Aggregator (runs on the designated aggregator bank, BankA by default)  
**Source**: `fl-integration/global_aggregator.py :: wait_for_consensus()`

The Global Aggregator continuously polls `GET /check-consensus/{round_num}` at a configurable `poll_interval` (default 5 s). The chaincode returns the list of `accepted_banks` — the banks whose cluster models have passed CBFT commit. The aggregator waits until at least one bank appears in this list, or until `consensus_timeout` expires.

This polling is the gateway: the global aggregation **cannot start** until blockchain consensus proves that at least some submitted models are honest and performant.

### Step 2.4 — Trust-Weighted Inter-Cluster FedAvg (Global Aggregation)

**Entity**: Global Aggregator  
**Source**: `fl-integration/global_aggregator.py :: aggregate_round()`

This is the definitive aggregation step. Unlike Tier 1 FedAvg (which is purely sample-count weighted), the Tier 2 aggregation applies **trust scores** retrieved from the Fabric ledger:

$$\theta_{\text{global}} = \frac{\sum_{b \in \text{accepted}} w_b \cdot \theta_b}{\sum_{b \in \text{accepted}} w_b}$$

Where the **effective weight** for each bank $b$ is:

$$w_b = \text{trust\_score}_b \times n_b$$

- $\text{trust\_score}_b$: a float retrieved from the chaincode, reflecting the bank's historical reliability. A bank that consistently submits high-quality, verified models accumulates a higher trust score. A bank that submitted a model that was rejected by peers in previous rounds has a penalized score.
- $n_b$: number of samples (branch updates) the bank's cluster model was trained on in this round.

The multiplication ensures that both **data volume** (how much the bank contributed) and **historical trustworthiness** (how reliable the bank has been) jointly determine its influence on the global model. A first-time participant with no history defaults to `trust_score = 1.0`.

Before including any bank's model in this FedAvg, the Global Aggregator performs a **final hash verification**: it re-downloads the model from IPFS and re-checks the SHA-256 against the on-chain record. A hash mismatch at this stage results in **exclusion** — even a model that passed CBFT voting is re-validated to prevent replay or substitution attacks.

The resulting `global_sd` (global state dict) is the mathematically merged, trust-adjusted synthesis of all accepted banks' fraud detection knowledge.

### Step 2.5 — Global Model Publication

**Entity**: Global Aggregator  
**Source**: `fl-integration/global_aggregator.py :: aggregate_round()` continued

1. The global `state_dict` is serialized to bytes via `torch.save()`.
2. Uploaded to IPFS → obtaining `global_cid`.
3. SHA-256 of bytes computed → `global_hash`.
4. The API call `POST /store-global-model` invokes the chaincode function `StoreGlobalModel`, which:
   - Writes `{round_num, global_cid, global_hash}` to the ledger's world state.
   - Advances the `latest_round` pointer.

All participants can now query `GET /latest-round` and `GET /global-model/{round_num}` to retrieve the verified, immutable CID of the new global model for the next training round.

---

## Round Coordination: The Glue Between Tiers

**Source**: `fl-integration/round_coordinator.py :: RoundCoordinator.run()`

The `RoundCoordinator` is the orchestrator that sequences all of the above steps in a single round execution:

```
RoundCoordinator.run()
│
├── 1. collect_branch_updates()          ← Tier 1: deadline-gated branch collection
│        └── wait_for_submissions()
│
├── 2. for each bank's HQ agent:
│        agent.run_round()               ← Tier 1: intra-cluster FedAvg + validation
│            ├── fetch_global_model()    ← fetch prior global from IPFS
│            ├── fedavg(branch_updates) ← weighted average of branches
│            ├── blend_with_global()    ← resilience blending (if round > 1)
│            ├── evaluate_model()       ← quality gate (PR-AUC >= τ ?)
│            └── submit_update()        ← upload to IPFS, post to blockchain
│
└── 3. if is_aggregator:
         global_agg.run_full_aggregation()  ← Tier 2: CBFT + trust-weighted FedAvg
             ├── wait_for_consensus()        ← poll blockchain for accepted banks
             ├── aggregate_round()           ← trust-weighted FedAvg
             │     ├── get_trust_scores()    ← fetch historical trust from ledger
             │     ├── ipfs_download()       ← download+verify each bank's model
             │     └── fedavg(weighted)      ← global merge
             └── store_global_model()        ← upload to IPFS, record on-chain
```

The `RoundConfig.is_aggregator` flag determines which node runs the global aggregation step. In the current configuration, BankA's node is designated as the global aggregator.

---

## Data Flow Summary

```
                    ┌─────────────────────────────────────────────────────┐
                    │                   TIER  1                           │
                    │            (Inside each Bank's network)             │
  Branch 1 ──[DP]──┐                                                     │
  Branch 2 ──[DP]──┼──► wait_for_submissions() ──► fedavg() ──► blend() ─┼──► Cluster Model
  Branch 3 ──[DP]──┘    (deadline_collect.py)    (fedavg.py)  (backup)   │
                    │                                                     │
                    └──────────────────────┬──────────────────────────────┘
                                           │
                                evaluate_model()
                                (PR-AUC >= τ ?)
                                           │ YES
                    ┌──────────────────────▼──────────────────────────────┐
                    │                   TIER  2                           │
                    │         (Across the blockchain consortium)          │
                    │                                                     │
                    │  ① torch.save() → IPFS upload → CID               │
                    │  ② SHA-256(bytes) → POST /submit-update            │
                    │      └─► Chaincode records {CID, hash, val_score}  │
                    │                                                     │
                    │  ③ CBFT Cross-Verification (all peer HQ agents)    │
                    │      ├── Download peer IPFS model by CID           │
                    │      ├── Recompute SHA-256 → verify vs on-chain    │
                    │      ├── evaluate_model() on own validation data   │
                    │      └── POST /submit-verification (True/False)   │
                    │                                                     │
                    │  ④ Quorum check (≥ 2/3 banks verified ?)          │
                    │      └── POST /submit-commit → "Accepted"          │
                    │                                                     │
                    │  ⑤ GlobalAggregator polls GET /check-consensus    │
                    │      └── trust_score × n_samples weighted FedAvg  │
                    │          → Global Model → IPFS → POST /store-global│
                    └─────────────────────────────────────────────────────┘
                                           │
                              All banks query GET /latest-round
                              Download Global Model CID from IPFS
                                           │
                              ─────────── NEXT ROUND ───────────►
```

---

## Key Terms Glossary

| Term | Definition |
|---|---|
| **FedAvg** | Federated Averaging — the sample-count-weighted average of model `state_dict`s from multiple participants |
| **state_dict** | PyTorch's internal dictionary `{layer_name → weight_tensor}` representing all learned parameters of a model |
| **Non-IID** | Non-independent, non-identically distributed — each client's data has unique patterns not representative of the global distribution |
| **LSTMTabular** | Long Short-Term Memory network adapted for tabular (row/column) financial transaction data |
| **Differential Privacy (DP)** | A mathematical privacy guarantee: gradient clipping + Gaussian noise injection that prevents model updates from revealing individual training records |
| **PR-AUC** | Precision-Recall Area Under Curve — model quality metric ideal for class-imbalanced datasets like fraud detection |
| **CID** | Content Identifier — IPFS's cryptographic address derived from file content; same bytes always produce same CID |
| **SHA-256** | Cryptographic hash function producing a 256-bit digest; used to verify model file integrity |
| **CBFT** | Consensus-Based Federated Trust — the framework's custom 3-phase Byzantine consensus protocol |
| **PBFT** | Practical Byzantine Fault Tolerance — classical distributed consensus protocol that CBFT is inspired by |
| **Byzantine fault** | A failure mode where a participant sends incorrect, conflicting, or adversarial messages |
| **Trust Score** | A per-bank reliability score stored on the Fabric ledger, increased for accurate submissions and penalized for rejected ones |
| **World State** | Hyperledger Fabric's key-value database (maintained by each peer) representing the current state of all ledger variables |
| **Cluster Model** | The intermediate model produced by Tier 1 FedAvg — represents one bank's aggregated knowledge |
| **Global Model** | The final model produced by Tier 2 trust-weighted FedAvg — represents the consortium's merged, verified knowledge |
| **Straggler** | A slow or unresponsive participant node; mitigated by the deadline collector |
| **Backup Blending** | The `β·global + (1−β)·cluster` interpolation that stabilizes training when branch participation is low |
| **Intra-cluster** | Synonymous with Tier 1 — within a single bank's private infrastructure |
| **Inter-cluster** | Synonymous with Tier 2 — across different banks via the public blockchain channel |
| **Quorum** | The minimum number of agreeing nodes required to finalize a consensus decision (typically ≥ f+1 for f Byzantine nodes) |
| **Aggregator Bank** | The designated bank node (BankA) responsible for running GlobalAggregator — the only node that executes inter-cluster FedAvg |
| **HQ Peer** | Each bank's Hyperledger Fabric peer that represents the "headquarters" and participates in CBFT voting |
