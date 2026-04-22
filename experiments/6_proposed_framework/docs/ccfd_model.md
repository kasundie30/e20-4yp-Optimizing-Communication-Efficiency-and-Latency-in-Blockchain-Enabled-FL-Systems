# CCFD Model — Architecture, Integration, and Design Decisions

> **Source directory**: `experiments/4_CCFD_Model/`  
> **System**: Credit Card Fraud Detection via Hierarchical Federated Learning (pre-blockchain prototype)  
> **Date**: 2026-03-17

---

## 1. Overview

The `4_CCFD_Model` is the **pure FL prototype** of the fraud detection system. It establishes the full hierarchical federated learning pipeline — model architecture, local training with differential privacy, two-tier aggregation, straggler resilience, reputation scoring, and evaluation — all **without any blockchain**. The mock ledger (a local JSONL file) simulates the trust and audit trail that Hyperledger Fabric provides in the later `6_proposed_framework`.

This document explains every component, how they connect, the execution flow across multiple FL rounds, and the key design decisions made at each step.

---

## 2. System Topology

**Source**: `config/topology.yaml`, `src/clustering/topology_loader.py`

The system is organized into **brands** (clusters/banks) and **branches** (local data-holding nodes within each cluster).

```
Brand 1                Brand 2                Brand 3
├── branch_0 (HQ)      ├── branch_0 (HQ)      ├── branch_0 (HQ)
├── branch_1 (Backup)  ├── branch_1 (Backup)  ├── branch_1 (Backup)
└── branch_2           └── branch_2           └── branch_2
```

Each brand has:
- **3 branches** — the local training nodes with isolated data partitions
- **1 HQ** — the intra-cluster aggregator (maps to `brand_X_branch_0`)
- **1 Backup** — the fallback aggregator (maps to `brand_X_branch_1`)

The `Topology` dataclass (`topology_loader.py`) builds four lookup maps from the YAML:

| Map | Purpose |
|-----|---------|
| `brand_to_branches` | Lists all branches for each brand |
| `brand_to_hq` | Identifies the HQ node per brand |
| `brand_to_backup` | Identifies the backup node per brand |
| `branch_to_brand` | Reverse map: which brand does a branch belong to? |

**Design decision**: The HQ and Backup are defined in static configuration (YAML), not dynamically elected at runtime. This simplifies orchestration for the prototype phase while leaving the role structure explicit and auditable.

---

## 3. Machine Learning Model

**Source**: `FL_model.py`

```python
class LSTMTabular(nn.Module):
    def __init__(self, input_dim=30, hidden_dim=30, num_layers=1):
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim,
                            num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, (h, c) = self.lstm(x)
        return self.fc(h[-1])      # logit (scalar per sample)
```

### Design Decision: Why LSTM for Tabular Data?

Credit card transaction data is inherently sequential — a user's spending pattern over time encodes behavioral context that is essential for distinguishing fraud from legitimate anomalies. An LSTM (Long Short-Term Memory) network captures this temporal dependency through its cell state $c_t$ and hidden state $h_t$, updated via the forget, input, and output gates:

$$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$
$$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$$
$$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$$

The final hidden state $h[-1]$ is passed through a single linear layer to produce a **logit** (unbounded scalar), which is converted to a fraud probability via sigmoid during inference.

The architecture is deliberately **minimal** (`hidden_dim=30`, `num_layers=1`, `input_dim=30`) to match the 30-feature structure of the standard credit card fraud dataset. This makes the model fast to train locally and efficient to transmit — the entire `state_dict` serializes to approximately 30–50 KB.

---

## 4. Data Loading and Preprocessing

**Source**: `dataset.py`

```python
def load_bank_dataset(bank_id, data_path, filename_candidates):
    folder = os.path.join(data_path, bank_id)
    # Looks for: train_ready.csv | local_data.csv | train.csv | data.csv
    df = pd.read_csv(path)
    X = df.iloc[:, :-1].values      # all columns except last
    y = df.iloc[:, -1].values       # last column is label (0=normal, 1=fraud)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)     # per-node normalization
    return TensorDataset(X_tensor, y_tensor)
```

### Design Decisions

1. **Per-node StandardScaler**: Each branch fits a `StandardScaler` **locally** on its own data, not globally. This preserves data privacy (no statistics are shared) and ensures the LSTM receives zero-mean, unit-variance inputs — critical for stable gradient flow in RNNs.

2. **Non-IID data silos**: Each branch loads from its own folder (`data_path/{bank_id}/`). The data partitioning is done beforehand (in the `3_local_silo_balancing` processing step), resulting in non-identically distributed partitions that reflect realistic geographic/demographic differences in spending behavior.

3. **Filename fallback list**: The loader tries four candidate filenames in order. This makes the loader forward-compatible across different preprocessing stages without requiring code changes.

4. **Last-column label convention**: The label is always the rightmost column, which matches the standard Kaggle credit card fraud dataset schema.

---

## 5. Local Branch Training

**Source**: `local_train.py`

```python
BANK_ID = os.environ["BANK_ID"]       # injected by Docker
LOCAL_EPOCHS = 2
LR = 1e-3
BATCH_SIZE = 256
L2_NORM_CLIP = 1.0
NOISE_MULTIPLIER = 0.05
```

Each branch runs inside a Docker container and performs the following:

### Step 5.1 — Load Global Model
```python
global_model.load_state_dict(torch.load("/logs/global_model.pt"))
```
The branch starts from the **current global model weights** rather than random initialization. This is the distinguishing property of federated learning — all clients begin each round from the same global starting point, ensuring that local updates are additive improvements rather than independent restarts.

### Step 5.2 — Class-Weighted Loss
```python
pos = (all_labels == 1).sum().item()
neg = (all_labels == 0).sum().item()
weight = torch.tensor([neg / max(pos, 1)])
criterion = nn.BCEWithLogitsLoss(pos_weight=weight)
```
**Design decision**: Credit card fraud is severely imbalanced — typically 0.1–0.5% fraud rate. Standard cross-entropy loss would result in a model that predicts "not fraud" for everything and still achieves >99.5% accuracy. The `pos_weight = neg/pos` multiplier penalizes false negatives (missed fraud) by a factor proportional to the class ratio, pushing the model to learn fraud patterns rather than defaulting to the majority class.

### Step 5.3 — Differential Privacy Training Loop

```
for each batch:
  1. loss.backward()                           ← compute gradients
  2. clip_grad_norm_(max_norm=L2_NORM_CLIP)    ← bound sensitivity
  3. grad += randn * (L2_NORM_CLIP * NOISE_MULTIPLIER)  ← inject noise
  4. optimizer.step()                          ← update AFTER clip+noise
```

**Design decision — DP ordering is strict**: Clipping must happen before noise injection, and the optimizer must step after both. This is the standard $(ε, δ)$-DP mechanism. The `L2_NORM_CLIP=1.0` bounds the maximum gradient contribution any single sample can make (bounding sensitivity $\Delta f$). The `NOISE_MULTIPLIER=0.05` is deliberately small — it injects just enough noise to provide formal privacy guarantees without significantly degrading model convergence.

### Step 5.4 — Save Local Model
```python
torch.save(global_model.state_dict(), f"/logs/{BANK_ID}_local_model.pt")
```
The trained weights are written to the shared Docker volume at `/logs/`, which maps to the current round's folder on the host (`shared/round_XXXX/`). Only the weights (`state_dict`) are saved — no raw training data ever leaves the branch.

---

## 6. Run Orchestration

**Source**: `src/run_rounds.py`

The `run_rounds.py` script orchestrates the complete multi-round FL process using subprocess calls. For each round from 1 to `--rounds`:

```
Round t:
  1. Copy prev_global.pt → shared/round_XXXX/global_model.pt
  2. docker compose up --build        ← all 9 branches train in parallel
  3. python -m src.aggregation.hq_aggregate   ← Tier 1 (intra-cluster)
  4. python -m src.aggregation.global_aggregate ← Tier 2 (inter-cluster)
  5. new global_model.pt becomes prev_global for round t+1
  6. docker compose down
```

**Design decision**: The orchestrator uses the **filesystem as the communication bus**. Model weights are exchanged by reading and writing `.pt` files from a shared directory. This is simple and avoids network complexity in the prototype, but it becomes the key difference vs. the `6_proposed_framework` which replaces filesystem I/O with IPFS + blockchain.

---

## 7. Component Integration Map

```
run_rounds.py
│
├── [Docker launch] ──► local_train.py (× 9 branches in parallel)
│     ├── dataset.py::load_bank_dataset()       ← loads branch CSV
│     ├── FL_model.py::LSTMTabular              ← initializes from global .pt
│     └── writes: /logs/{BANK_ID}_local_model.pt
│
├── [HQ Aggregation] ──► src/aggregation/hq_aggregate.py::main()
│     ├── src/clustering/topology_loader.py     ← reads topology.yaml
│     ├── src/resilience/deadline_collect.py    ← waits for branch .pt files
│     │     └── src/aggregation/fedavg.py::load_state_dict()
│     ├── src/aggregation/fedavg.py::fedavg_state_dicts()  ← Tier 1 FedAvg
│     ├── src/validation/validate_fast.py        ← PR-AUC quality gate
│     │     └── src/validation/metrics.py::fraud_metrics()
│     ├── src/resilience/backup_logic.py         ← blend with prev global if failed
│     ├── src/utils/score_store.py              ← update brand reputation
│     ├── src/utils/ledger.py                   ← append audit record
│     └── writes: shared/round_XXXX/brand_models/{brand_id}_brand_model.pt
│
└── [Global Aggregation] ──► src/aggregation/global_aggregate.py::main()
      ├── src/clustering/topology_loader.py     ← discovers which brands submitted
      ├── src/aggregation/fedavg.py::fedavg_state_dicts()  ← Tier 2 FedAvg
      └── writes: shared/round_XXXX/global_model.pt
```

---

## 8. Tier 1 — Intra-Brand Aggregation (HQ)

**Source**: `src/aggregation/hq_aggregate.py`

### Step 8.1 — Deadline Collection

```python
arrived_ids, state_dicts = collect_until_deadline(round_dir, branch_ids, deadline_sec=25)
```

`deadline_collect.py` polls the round directory every 1 second for branch `.pt` files. It returns whatever has arrived within 25 seconds. Stragglers are excluded, not waited for.

**Design decision — filesystem polling vs. event-driven**: The prototype uses naive polling (`time.sleep(1.0)`) because simplicity is prioritized over efficiency. The `6_proposed_framework` replaces this with an injectable `collect_fn` that can be swapped without changing the timeout logic.

### Step 8.2 — Intra-Brand FedAvg

```python
brand_sd = fedavg_state_dicts(state_dicts)   # uniform average (no weights in this version)
```

**Design decision — uniform vs. sample-weighted**: At this stage, `fedavg_state_dicts` uses a **uniform average** by default (each branch contributes equally regardless of its local dataset size). This is simpler than sample-weighted FedAvg and is sufficient when branches are configured with roughly equal data volumes. The `6_proposed_framework` switches to explicit sample-count weighting.

### Step 8.3 — Fast Validation Gate

```python
metrics = fast_validate_state_dict(brand_sd, node_id_for_data=hq_id, fraction=0.15)
passed = metrics["prauc"] >= 0.20
```

The HQ validates the merged brand model against a **15% random sample** of its own data. Threshold: PR-AUC ≥ 0.20.

**Design decision — PR-AUC threshold is intentionally permissive (0.20)**: A random classifier on 0.1% fraud rate achieves ~0.001 PR-AUC; 0.20 is 200× better than chance. Setting it low ensures the system doesn't exclude brands in early rounds where the global model is still initializing and PR-AUC naturally starts low before converging.

### Step 8.4 — Backup Recovery (Blending)

If validation fails OR fewer than `min_required=2` branches responded:

```python
brand_sd_alt = blend_with_prev_global(round_dir, brand_sd, beta=0.70)
```

$$\theta_{\text{alt}} = 0.70 \cdot \theta_{\text{prev\_global}} + 0.30 \cdot \theta_{\text{brand}}$$

**Design decision — beta=0.70 (favor global over local)**: In a failure scenario (bad validation, few branches), trusting the *previously verified* global model at 70% weight is safer than trusting the current underpowered brand model at full weight. This prevents a degraded round from corrupting the brand's contribution to the global model.

The blended model is then re-validated. If it also fails: the brand is **excluded** entirely from this round.

### Step 8.5 — Reputation Scoring and Ledger

```python
update_score(scores, brand_id, ok=True,  reward=0.05, penalty=0.10, floor=0.2, max=3.0)
update_score(scores, brand_id, ok=False, ...)
append_record(ledger_file, {"brand": brand_id, "status": "accepted", "metrics": ..., "brand_hash": ...})
```

**Scoring mechanism**:
- Each accepted round: `score += 0.05`
- Each rejected round: `score -= 0.10`
- Score never falls below `floor=0.2` (prevents permanent exclusion)
- Score never exceeds `max=3.0` (prevents runaway dominance)

**Design decision — asymmetric reward/penalty**: Penalizing failures at 2× the reward rate (0.10 vs 0.05) reflects the fact that contributing a bad model is more harmful than withholding a good one. The floor at 0.2 ensures even a consistently poor performer can eventually recover.

**Ledger hashing**: `hash_state_dict()` computes SHA-256 over the sorted keys and tensor bytes. This is the primitive prototype of the cryptographic binding used in IPFS+blockchain in `6_proposed_framework`.

---

## 9. Tier 2 — Global Aggregation

**Source**: `src/aggregation/global_aggregate.py`

```python
for brand_id in topo.brand_to_branches.keys():
    path = os.path.join(brand_models_dir, brand_model_filename(brand_id))
    if os.path.exists(path):          # only include brands that passed Tier 1
        state_dicts.append(load_state_dict(path))
        used_brands.append(brand_id)

global_sd = fedavg_state_dicts(state_dicts)   # uniform average over accepted brands
save_state_dict(global_sd, out_path)           # writes global_model.pt
```

**Design decision — implicit trust through participation**: In this prototype, all brands that survive the HQ quality gate contribute **equally** to the global model (uniform FedAvg). There is no cross-brand verification of whether a brand's submitted model is honest. This is the key limitation addressed in `6_proposed_framework` via CBFT consensus and trust-score-weighted FedAvg.

**Design decision — file existence as acceptance signal**: A brand model file existing in `brand_models/` is the only signal that a brand was accepted. There is no explicit commit protocol. This simplicity works in a single-machine prototype but is not Byzantine-tolerant — a malicious actor could write a fake model file.

---

## 10. FedAvg Implementation

**Source**: `src/aggregation/fedavg.py`

```python
def fedavg_state_dicts(state_dicts, weights=None):
    # Default: uniform weights (1/n each)
    # Custom: normalized weighted average
    for k in keys:
        if is_float_tensor(v):
            avg[k] = sum(w * sd[k] for sd, w in zip(state_dicts, weights))
        else:
            avg[k] = state_dicts[0][k].clone()    # copy non-float tensors from first
```

Two additional helpers:
- `load_state_dict(path)` — always loads to CPU to avoid GPU dependency during aggregation
- `save_state_dict(state_dict, path)` — simple `torch.save` wrapper

**Design decision — non-float tensor handling**: LSTM state dicts may contain integer counters (e.g., `num_batches_tracked` from BatchNorm). Averaging integers produces meaningless results. The code copies these from the first model — a pragmatic choice that avoids errors without affecting the meaningful floating-point parameters.

---

## 11. Validation Stack

**Source**: `src/validation/validate_fast.py`, `src/validation/metrics.py`

```
fast_validate_state_dict(state_dict, node_id, data_root, fraction=0.15)
  ├── load_bank_dataset(node_id, data_root)
  ├── sample 15% randomly
  ├── LSTMTabular.load_state_dict(state_dict)
  ├── model.eval()
  └── fraud_metrics(y_true, y_prob)
       ├── PR-AUC  (average_precision_score)
       ├── Precision
       ├── Recall
       └── F1
```

**Design decision — fractional validation (15%) over full validation**: Running full inference on 100% of each HQ's dataset for every brand model in every round is expensive, especially when scaled. 15% provides sufficient statistical confidence for the quality gate while keeping validation fast enough to fit within the round deadline.

**Design decision — no ROC-AUC in the gate**: The quality gate uses **PR-AUC** exclusively (not ROC-AUC or F1). PR-AUC is the only metric that correctly penalizes false negatives in highly imbalanced datasets — ROC-AUC is inflated by the abundance of true negatives (legitimate transactions), giving an overly optimistic view of model quality. F1 depends on the decision threshold while PR-AUC is threshold-free.

---

## 12. Global Evaluation

**Source**: `evaluate_global.py`

The evaluation module assesses the final global model across **7 metrics**:

### Classification Metrics (on all 9 branches combined)
| Metric | How Measured |
|--------|-------------|
| **F1 Score** | Harmonic mean of precision and recall at threshold=0.5 |
| **PR-AUC** | Area under the Precision-Recall curve (threshold-free) |
| **ROC-AUC** | Area under the ROC curve |
| **Precision** | TP / (TP + FP) |
| **Recall** | TP / (TP + FN) |

### FL System Metrics (from round artefacts)
| Metric | Formula |
|--------|---------|
| **Comm / round (MB)** | Upload: Σ branch_model sizes + Download: global_model × num_branches |
| **E2E latency / round (sec)** | mtime(global_model.pt) − min(mtime(branch_local_model.pt)) |

**Design decision — communication overhead measurement**: Communication overhead is measured using actual **file sizes on disk** rather than network packet captures. This is accurate for prototype measurement and avoids the need for network monitoring infrastructure. The formula separates upload (branches → aggregator) from download (global model → branches), matching the theoretical analysis commonly used in FL literature.

**Design decision — E2E latency proxy via mtime**: True end-to-end latency would require synchronized clocks across containers. The `mtime` of the first branch output file approximates round start, and `mtime` of the output `global_model.pt` approximates round end. This is a pragmatic approximation that works within a single-machine Docker environment.

**Output**: Per-branch breakdown table + per-round FL system metrics + final aggregate summary including all 7 comparison metrics.

---

## 13. Docker Infrastructure

**Source**: `docker/docker-compose.yml`, `docker/Dockerfile.bank`

```yaml
x-branch-template: &branch
  image: fyp-bank:latest
  volumes:
    - ../data/processed/3_local_silo_balancing:/data     # read-only data
    - ${LOGS_DIR:-../shared/round_0001}:/logs            # read-write round artefacts

services:
  brand_1_branch_0: { <<: *branch, environment: [BANK_ID=brand_1_branch_0] }
  brand_1_branch_1: { <<: *branch, environment: [BANK_ID=brand_1_branch_1] }
  ... (9 services total)
```

**Design decisions in the Docker setup**:

1. **Single shared image** (`fyp-bank:latest`): All 9 branches use the same Docker image. The only difference is the `BANK_ID` environment variable, which `local_train.py` reads to determine which data folder to load. This dramatically simplifies CI/build pipelines.

2. **`LOGS_DIR` environment variable for round isolation**: Each round's folder is passed as `${LOGS_DIR}` and mounted to `/logs`. This allows the same compose file to be reused across all rounds without modifying it — the orchestrator (`run_rounds.py`) simply changes `LOGS_DIR` per round invocation.

3. **YAML anchor (`&branch`)**: The `x-branch-template` anchor with `<<: *branch` merge key eliminates duplication across all 9 service definitions. Any change to the common template (e.g., resource limits, volume mounts) propagates to all services automatically.

4. **`--remove-orphans`**: Ensures containers from aborted previous runs don't interfere with the current round.

---

## 14. End-to-End Round Flow

```
Initialization
└── init_init_model.py → creates random global_model.pt  (round 0 seed)

Round 1 ... N:

  [Host Orchestrator] run_rounds.py
  │
  ├── 1. cp global_model.pt → shared/round_XXXX/global_model.pt
  │
  ├── 2. docker compose up (9 branches in parallel)
  │     └── Each container:
  │           ├── loads /data/{BANK_ID}/train_ready.csv
  │           ├── applies StandardScaler (local)
  │           ├── loads /logs/global_model.pt into LSTMTabular
  │           ├── trains 2 epochs with DP (clip → noise → step)
  │           └── saves /logs/{BANK_ID}_local_model.pt
  │
  ├── 3. HQ Aggregation (hq_aggregate.py — sequential per brand)
  │     For each brand (brand_1, brand_2, brand_3):
  │       ├── wait up to 25s for branch .pt files (deadline_collect)
  │       ├── FedAvg over arrived branch state_dicts (uniform weights)
  │       ├── fast_validate (PR-AUC on 15% of HQ data)
  │       ├── if fail → blend(0.70 global + 0.30 brand) → re-validate
  │       ├── if still fail → EXCLUDE (score -= 0.10, log to ledger)
  │       └── if pass → save brand_model.pt, score += 0.05, log to ledger
  │
  ├── 4. Global Aggregation (global_aggregate.py)
  │       ├── discover which brand_model.pt files exist
  │       ├── load all accepted brand state_dicts
  │       ├── FedAvg (uniform average)
  │       └── save global_model.pt
  │
  └── 5. new global_model.pt → prev_global for round+1

Post-training:
  └── evaluate_global.py → 7 metrics (F1, PR-AUC, ROC-AUC, P, R, Comm/round, E2E/round)
```

---

## 15. Major Design Decisions Summary

| Decision | Choice Made | Rationale |
|---|---|---|
| **Model architecture** | LSTM (not MLP or CNN) | Captures temporal transaction sequences; handles sequential fraud patterns |
| **Input dimension** | 30 features | Matches standard credit card fraud dataset (PCA-reduced from raw Visa/Mastercard features) |
| **Loss function** | BCEWithLogitsLoss + pos_weight | Addresses class imbalance (~0.1% fraud rate) without oversampling |
| **DP mechanism** | Gaussian noise + L2 clip | Standard $(ε, δ)$-DP; clip before noise is mandatory for correctness |
| **Noise multiplier** | 0.05 (small) | Provides DP guarantee with minimal model quality degradation |
| **Aggregation** | FedAvg (uniform, then trust-weighted in v2) | FedAvg is statistically unbiased when data is IID; trust-weighting adds Byzantine resistance |
| **Validation metric** | PR-AUC (not accuracy, not ROC-AUC) | Only metric robust to extreme class imbalance for fraud detection |
| **Quality threshold** | PR-AUC ≥ 0.20 | Permissive in early rounds; 200× above random baseline for heavily imbalanced data |
| **Backup blending** | β = 0.70 (favor global) | In a degraded local round, the validated global model is safer than an underpowered brand model |
| **Scoring system** | reward=0.05, penalty=0.10, floor=0.2, max=3.0 | Asymmetric (2:1 penalty:reward) discourages bad submissions; floor prevents lockout |
| **Communication bus** | Shared filesystem / Docker volumes | Simplest approach for single-machine prototype; replaced by IPFS in v2 |
| **Deadline (25s)** | Hardcoded in config | Balances waiting for stragglers vs. round latency; configurable for production tuning |
| **Docker architecture** | Single image, BANK_ID env var | Reduces build complexity; `LOGS_DIR` mount provides per-round isolation |
| **No cross-brand verification** | Implicit trust in this prototype | Byzantine resistance (CBFT) added in `6_proposed_framework` |
| **Evaluation sampling** | 15% random fraction | Fast enough for per-round quality gating; full evaluation done only at the end |

---

## 16. Limitations and What Changed in 6_proposed_framework

| CCFD Model Limitation | Resolution in Proposed Framework |
|---|---|
| **Filesystem-based model sharing** | IPFS: content-addressed, distributed, tamper-evident |
| **No cross-bank verification** | CBFT: 3-phase Byzantine consensus across bank HQ peers |
| **No immutable audit trail** | Hyperledger Fabric ledger: append-only, cryptographically linked |
| **Uniform FedAvg across brands** | Trust-score-weighted FedAvg: reputation from blockchain history |
| **Scores stored in local JSON** | Trust scores stored and updated on-chain by chaincode |
| **Mock ledger (JSONL file)** | Real distributed ledger (Fabric world state + block history) |
| **Single-machine Docker** | Multi-organization Fabric network (3 orgs, 2 peers each) |
| **No hash verification of peer models** | SHA-256 hash checked against on-chain record before aggregation |
| **Sequential HQ aggregation** | Injectable `collect_fn` + decoupled `HQAgent` class for parallelism |
| **Script-based orchestration** | `RoundCoordinator` class with clean separation of timing and logic |
