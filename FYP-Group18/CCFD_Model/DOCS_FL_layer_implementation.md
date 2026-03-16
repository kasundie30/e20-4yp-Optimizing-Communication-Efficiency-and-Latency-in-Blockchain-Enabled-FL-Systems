# FL Layer Implementation Documentation

## Overview

The 4_CCFD_Model directory implements a **Hierarchical Federated Learning (FL) system** for Credit Card Fraud Detection (CCFD) across multiple financial institutions (brands) with multiple branches. The system is built on a two-tier aggregation architecture with resilience mechanisms including deadline-based straggler handling, validation gates, and backup recovery.

---

## System Architecture

### Hierarchical Clustering Structure

The FL system organizes participants in a **two-level hierarchy**:

```
                           Global Node
                                |
                    ┌───────────┼───────────┐
                  Brand_1       Brand_2     Brand_3
                    |             |           |
          ┌─────────┼─────────┐  (3 HQ)  ┌──┴──┐
      Branch_0   Branch_1   Branch_2      ...
```

**Key Topology Elements** (defined in `config/topology.yaml`):

- **Brands**: 3 clusters of branches (e.g., Brand_1, Brand_2, Brand_3)
- **Branches**: Individual financial institutions (e.g., brand_1_branch_0, brand_1_branch_1, brand_1_branch_2)
- **HQ Node**: Designated leader per brand (e.g., brand_1_branch_0 for Brand_1)
- **Backup Node**: Fallback for recovery (e.g., brand_1_branch_1 for Brand_1)

**Topology Configuration** example:
```yaml
brands:
  brand_1:
    hq: brand_1_branch_0      # Aggregation leader
    backup: brand_1_branch_1  # Recovery fallback
    branches:
      - brand_1_branch_0
      - brand_1_branch_1
      - brand_1_branch_2
```

---

## Multi-Round Dataflow

### High-Level Round Workflow

Each **training round** follows this sequence orchestrated by `src/run_rounds.py`:

```
Round Start
    ↓
[Phase 1] Local Training (Docker containers)
    ↓
[Phase 2a] Intra-Brand Aggregation (HQ nodes)
    ↓
[Phase 2b] Inter-Brand Aggregation (Global node)
    ↓
Round End
```

### Detailed Round Flow

#### **1. Round Initialization**

```python
# From: src/run_rounds.py::main()
for t in range(1, args.rounds + 1):
    round_dir = shared_root / f"round_{t:04d}"
    # Copy previous global model to current round
    copy_global_model(prev_global, round_dir)
```

**Outputs created:**
- `shared/round_0001/` through `shared/round_000N/`
- Each round folder contains: `global_model.pt`, local models, brand models, metrics

---

#### **2. Phase 1: Distributed Local Training**

**Trigger**: Docker Compose spins up 9 containers (one per branch)

**File**: `local_train.py` (executes inside each container)

**Data Flow**:

```
Global Model (round N-1)
    ↓ (copied to /logs/global_model.pt in container)
Local Dataset (/data/{BANK_ID}/)
    ↓
Local Training Loop (LSTM Model)
    ↓ (with DP noise injection)
Local Model Update
    ↓ (saved to /logs/{BANK_ID}_local_model.pt)
Shared Volume (Round Folder)
```

**Training Details**:

```python
# Model Architecture
class LSTMTabular(nn.Module):
    - LSTM layer (input_dim → hidden_dim)
    - Fully connected layer (hidden_dim → 1, for fraud logits)

# Training Parameters
LOCAL_EPOCHS = 2
LR = 1e-3
BATCH_SIZE = 256
HIDDEN_DIM = 30
NUM_LAYERS = 1

# Privacy Protection (Differential Privacy)
L2_NORM_CLIP = 1.0          # Gradient clipping
NOISE_MULTIPLIER = 0.05     # Noise scale
```

**Local Loss Function**:
- **Criterion**: BCEWithLogitsLoss with positive weight
- **Weighting**: Handles class imbalance (fraud vs. non-fraud)
  ```python
  weight = (# of negatives) / (# of positives)
  ```

**DP Mechanism**:
1. Compute gradients normally
2. Clip gradient norms to 1.0
3. Add Gaussian noise: `noise ~ N(0, (1.0 * 0.05)²)`
4. Apply noisy update

**Output**: `{BANK_ID}_local_model.pt` in round folder

---

#### **3. Phase 2a: Intra-Brand Aggregation (HQ Level)**

**File**: `src/aggregation/hq_aggregate.py`

**Trigger**: Runs after Docker containers finish

**Data Flow** (per brand):

```
Branch Local Models (up to 3)
    ↓ (with deadline collection: 25 seconds)
Load Models [Collection Phase]
    ↓
FedAvg Aggregation [Aggregation Phase]
    ↓
Fast Validation on HQ Data [Validation Phase]
    ↓
    ├─ PASS → Save Brand Model
    │         (shared/round_000N/brand_models/{brand_id}.pt)
    │
    └─ FAIL → Try Backup Recovery [Recovery Phase]
              (blend with prev global: 0.7*w_prev + 0.3*w_brand)
              ├─ RECOVERED → Save Brand Model
              └─ FAILED → Exclude Brand (score penalty)
```

**Key Components**:

**3a. Deadline Collection**
```python
# From: src/resilience/deadline_collect.py
collect_until_deadline(
    round_dir="shared/round_0001",
    branch_ids=["brand_1_branch_0", "brand_1_branch_1", "brand_1_branch_2"],
    deadline_sec=25  # Wait max 25 seconds
)
```

**Behavior**: 
- Waits up to 25 seconds for branch models to arrive
- Stragglers (late models) are excluded from this round
- Continues aggregation with available models

**3b. FedAvg Aggregation**
```python
# From: src/aggregation/fedavg.py
def fedavg_state_dicts(state_dicts, weights=None):
    # Uniform average: w_brand = (1/N) * Σ w_branch_i
    # Handles non-float tensors (BatchNorm) by copying from first
```

**3c. Fast Validation Gate**
```python
# From: src/validation/validate_fast.py
metrics = fast_validate_state_dict(
    state_dict=brand_model,
    node_id_for_data=hq_branch_id,  # Use HQ's local data
    fraction=0.15,                   # Validate on 15% of data
    metric="prauc"                   # Primary metric
)
# Threshold check: metrics['prauc'] >= 0.20
```

**Validation Strategy**:
- Samples 15% of HQ's local test data
- Computes fraud detection metrics
- Acceptance criteria: `PRAUC >= 0.20`

**Available Metrics** (from `src/validation/metrics.py`):
- `prauc`: Precision-Recall AUC
- `recall`: True Positive Rate
- `f1`: F1 Score
- Others: precision, rocauc

**3d. Backup Recovery Logic**
```python
# From: src/resilience/backup_logic.py
if validation_failed or weak_collection:
    brand_model_alt = blend_with_prev_global(
        round_dir, brand_model, beta=0.70
    )
    # w_alt = 0.70 * w_prev_global + 0.30 * w_brand
    re_validate(brand_model_alt)
```

**Recovery Strategy**:
- If model fails validation: blend with previous global
- Blend formula: `w = β*w_prev + (1-β)*w_new`
- Re-validate blended model
- Only accept if second validation passes

**3e. Scoring System**

```python
# From: src/utils/score_store.py (inferred from config)
scores = {
    "brand_1": 1.0,  # Initial score
    "brand_2": 1.0,
    "brand_3": 1.0,
}

# Update scores based on performance
if accepted:
    score += reward (0.05)
else:
    score -= penalty (0.10)

# Bounds: floor (0.2) to max (3.0)
```

**Ledger Recording**:
```python
# From: src/utils/ledger.py
append_record("shared/ledger.jsonl", {
    "brand": "brand_1",
    "status": "accepted",  # or "excluded_*"
    "arrived_branches": ["brand_1_branch_0", "brand_1_branch_1"],
    "metrics": {"prauc": 0.25, ...},
    "score": 1.05,
    "brand_hash": "abc123...",  # SHA256 of model weights
    "ts": 1703057088.5
})
```

**Output**: 
- Brand models: `shared/round_000N/brand_models/{brand_id}.pt`
- Updated scores: `shared/scores.json`
- Audit log: `shared/ledger.jsonl`

---

#### **4. Phase 2b: Inter-Brand Aggregation (Global Level)**

**File**: `src/aggregation/global_aggregate.py`

**Data Flow**:

```
Brand Models (filtered: only accepted brands)
    ↓
Load Brand Models [Collection Phase]
    ↓
FedAvg Aggregation [Aggregation Phase]
    ↓
Save Global Model
    ↓
Next Round Input
```

**Implementation**:

```python
# Load all available brand models
state_dicts = []
for brand_id in topo.brand_to_branches.keys():
    path = f"shared/round_000N/brand_models/{brand_id}.pt"
    if exists(path):
        state_dicts.append(load_state_dict(path))

# Aggregate all brands uniformly
global_model = fedavg_state_dicts(state_dicts)

# Save for next round
save_state_dict(global_model, f"shared/round_000N/global_model.pt")
```

**Key Difference from Phase 2a**:
- NO validation (trust brand aggregation)
- NO backup recovery
- Simple uniform averaging

**Output**: `shared/round_000N/global_model.pt` (becomes input for Round N+1)

---

## Key Components Deep Dive

### Model Architecture

**File**: `FL_model.py`

```python
class LSTMTabular(nn.Module):
    def __init__(self, input_dim=30, hidden_dim=30, num_layers=1):
        self.lstm = LSTM(input_dim → hidden_dim, num_layers)
        self.fc = Linear(hidden_dim → 1)
    
    def forward(self, x):
        # x: (batch, seq_len=1, features)
        out, (h, c) = lstm(x)
        return fc(h[-1])  # Last hidden state → fraud logit
```

**Design Rationale**:
- LSTM captures temporal patterns in transaction sequences
- Single fully-connected layer for binary classification (fraud/non-fraud)
- Output is logit (pre-sigmoid), used with BCEWithLogitsLoss

### Data Loading & Preprocessing

**File**: `dataset.py`

```python
def load_bank_dataset(bank_id, data_path="data/processed/3_local_silo_balancing"):
    # Load: data/processed/3_local_silo_balancing/{bank_id}/train_ready.csv
    # X: (n_samples, 30) features (already feature-engineered)
    # y: (n_samples, 1) binary labels
    
    # Local feature scaling (StandardScaler per branch)
    X = StandardScaler().fit_transform(X)
    
    return TensorDataset(X_tensor, y_tensor)
```

**Data Characteristics**:
- **Input**: 30 features per transaction
- **Format**: Tabular financial transaction data
- **Location**: `data/processed/3_local_silo_balancing/{bank_id}/`
- **Variants**: train_ready.csv, local_data.csv, train.csv, data.csv

**Feature Engineering**: Pre-processed and stored in silo-balanced format (each bank's data locally scaled)

### Docker Containerization

**File**: `docker/docker-compose.yml`

```yaml
services:
  brand_1_branch_0:
    image: fyp-bank:latest
    volumes:
      - ../data/processed/3_local_silo_balancing:/data   # Read-only data
      - ${LOGS_DIR:-../shared/round_0001}:/logs          # Write outputs
    environment:
      - BANK_ID=brand_1_branch_0
```

**Execution Flow**:
1. `docker compose up --build --remove-orphans`
2. Each container runs `local_train.py`
3. Container accesses: global model from `/logs/global_model.pt`
4. Container saves: local model to `/logs/{BANK_ID}_local_model.pt`
5. `docker compose down` after training

**Volume Mounting Strategy**:
- **Data volume** (`/data`): Shared read-only access to all banks' data
- **Logs volume** (`/logs`): Round-specific directory, prevents model overwrites
  - Each round gets its own `${LOGS_DIR}` (e.g., `shared/round_0001`)

---

## Configuration & Deployment

### Main Configuration Files

#### `config/topology.yaml`

**Purpose**: Define organizational structure (brands, branches, hierarchy)

**Content**:
- Brand definitions with branch assignments
- HQ node per brand (aggregation leader)
- Backup node per brand (recovery fallback)

#### `config/config.yaml`

**Purpose**: Control FL system behavior and thresholds

**Sections**:

```yaml
phase2:
  deadline_sec: 25              # Max wait for stragglers
  min_models_required: 2        # Min branches for valid aggregation
  fast_val_fraction: 0.15       # Validation sample size
  metric: prauc                 # Primary validation metric
  threshold:
    prauc: 0.20                 # Acceptance threshold
  blend_beta: 0.70              # Recovery blend ratio
  ledger_commit_every_k: 3      # Batch ledger writes

scoring:
  init_score: 1.0               # Starting brand score
  floor: 0.2                    # Minimum score (prevents permanent exclusion)
  max: 3.0                      # Maximum score
  reward: 0.05                  # Score increase for accepted brands
  penalty: 0.10                 # Score decrease for excluded brands

paths:
  scores_file: "shared/scores.json"
  ledger_file: "shared/ledger.jsonl"
```

### Execution Commands

#### Initialize Models
```bash
python3 init_init_model.py --output init/global_model.pt
```

#### Run Full Training Pipeline
```bash
python3 -m src.run_rounds \
    --topology config/topology.yaml \
    --compose_file docker/docker-compose.yml \
    --project_root . \
    --shared_root shared \
    --rounds 5 \
    --init_global_model init/global_model.pt
```

**Key Arguments**:
- `--rounds`: Number of FL training rounds
- `--init_global_model`: Path to initial global model
- `--shared_root`: Directory for round outputs
- `--topology`: Cluster configuration file

#### Evaluate Global model
```bash
python3 evaluate_global.py \
    --model_path shared/round_0005/global_model.pt \
    --rounds 5 \
    --threshold 0.5
```

---

## Resilience & Safety Mechanisms

### 1. Straggler Handling (Deadline-Based)

**Problem**: Slow branches delay training rounds

**Solution** (from `src/resilience/deadline_collect.py`):
```python
collect_until_deadline(
    round_dir, branch_ids, 
    deadline_sec=25,  # 25-second cutoff
    poll_interval=1.0  # Check every 1 second
)
```

**Behavior**:
- Wait max 25 seconds for branch models
- Accept any models received before deadline
- Exclude slow branches from this round
- Continue aggregation with available models

**Impact**: Enables system to tolerate 30% stragglers (example)

### 2. Validation Gate (Quality Control)

**Problem**: Low-quality models harm global performance

**Solution** (from `src/validation/validate_fast.py`):
```python
fast_validate_state_dict(
    state_dict=brand_model,
    fraction=0.15,      # 15% fast validation
    metric="prauc",
    threshold=0.20      # Only accept if PRAUC >= 0.20
)
```

**Benefits**:
- Catches models with poor generalization
- Uses fast validation (15% data) for efficiency
- Prevents poisoning of global model

### 3. Backup Recovery (Model Blending)

**Problem**: Validation failure leaves brand with no contribution

**Solution** (from `src/resilience/backup_logic.py`):
```python
if validation_failed:
    brand_model_alt = blend_with_prev_global(
        round_dir, brand_model, beta=0.70
    )
    # Retry validation on blended model
```

**Formula**: `w_alt = 0.70*w_prev_global + 0.30*w_new_brand`

**Justification**:
- Stabilizes unstable updates
- Leverages previous global knowledge
- Second chance before exclusion

### 4. Scoring System (Long-Term Incentives)

**Purpose**: Incentivize reliable participation

**Mechanism**:
- Each brand maintains a score (starts at 1.0)
- Accepted models: +0.05 reward
- Excluded models: -0.10 penalty
- Score bounds: [0.2, 3.0] (prevents permanent exclusion)

**Use Case**: Could be extended for weighted averaging or resource allocation

### 5. Audit Trail (Ledger)

**File**: `shared/ledger.jsonl` (one JSON record per line)

**Example Record**:
```json
{
  "ts": 1703057088.5,
  "round": 1,
  "brand": "brand_1",
  "status": "accepted",
  "arrived_branches": ["brand_1_branch_0", "brand_1_branch_1"],
  "metrics": {"prauc": 0.25, "recall": 0.65, "f1": 0.18},
  "score": 1.05,
  "brand_hash": "abc123def456...",
  "brand_model_path": "shared/round_0001/brand_models/brand_1.pt"
}
```

**Use Cases**:
- Trace model evolution per brand
- Audit model acceptance/rejection decisions
- Identify suspicious patterns
- Reproducibility and compliance

---

## Directory Structure

```
4_CCFD_Model/
├── config/
│   ├── config.yaml          # FL system hyperparameters
│   └── topology.yaml        # Brand/branch organization
├── data/
│   ├── raw/
│   │   └── creditcard.csv   # Original dataset
│   ├── processed/
│   │   ├── 1_feature_scaled/
│   │   ├── 2_bank_silos/
│   │   ├── 3_local_silo_balancing/  # Final split (main input)
│   │   └── Previous_balanced/
│   └── splits/
│       ├── centralized/
│       ├── fl_clients/
│       └── test/
├── docker/
│   ├── docker-compose.yml   # Run all 9 branch containers
│   ├── Dockerfile.bank      # Docker image for branches
│   └── shared/              # Shared files in containers
├── init/
│   └── global_model.pt      # Initial global model
├── shared/                  # Round outputs
│   ├── scores.json          # Brand score history
│   ├── ledger.jsonl         # Audit trail
│   └── round_000N/
│       ├── global_model.pt  # Global model for round N
│       ├── {BANK_ID}_local_model.pt  # Local models (Phase 1)
│       └── brand_models/
│           └── {brand_id}.pt  # Brand models (Phase 2a output)
├── src/
│   ├── aggregation/
│   │   ├── fedavg.py        # FedAvg implementation
│   │   ├── hq_aggregate.py  # Phase 2a (intra-brand)
│   │   └── global_aggregate.py  # Phase 2b (inter-brand)
│   ├── clustering/
│   │   ├── ids.py           # Naming conventions
│   │   └── topology_loader.py  # Load topology.yaml
│   ├── resilience/
│   │   ├── deadline_collect.py  # Straggler handling
│   │   └── backup_logic.py  # Model blending
│   ├── validation/
│   │   ├── validate_fast.py  # Fast validation
│   │   └── metrics.py       # Fraud detection metrics
│   ├── utils/
│   │   ├── config_loader.py  # Load config.yaml
│   │   ├── ledger.py        # Audit trail
│   │   └── score_store.py   # Score management
│   └── run_rounds.py        # Round orchestrator
├── FL_model.py              # LSTM model definition
├── dataset.py               # Data loading
├── local_train.py           # Branch-side training (in Docker)
├── evaluate_global.py       # Final evaluation
├── init_init_model.py       # Initialize global model
├── requirements.txt         # Python dependencies
└── README.md
```

---

## Data Flow Summary

### Complete Multi-Round Workflow

```
Initial Setup
├─ Load topology (3 brands × 3 branches)
├─ Initialize global model
└─ Prepare data silos (brand_1_branch_0, ..., brand_3_branch_2)

Round 1..N Loop
├─ [PHASE 1: Local Training]
│  ├─ Docker compose up (9 containers)
│  ├─ Each container:
│  │  ├─ Load global model from /logs/global_model.pt
│  │  ├─ Load local data from /data/{BANK_ID}/
│  │  ├─ Train LSTM for 2 epochs with DP
│  │  └─ Save to /logs/{BANK_ID}_local_model.pt
│  └─ Docker compose down
│
├─ [PHASE 2a: Intra-Brand Aggregation]
│  ├─ For each brand (brand_1, brand_2, brand_3):
│  │  ├─ Collect local models (deadline: 25s)
│  │  ├─ FedAvg aggregation
│  │  ├─ Fast validation (15% data, PRAUC >= 0.20)
│  │  ├─ If failed: try backup recovery
│  │  ├─ Update brand score
│  │  └─ Save brand model or mark as excluded
│  └─ Log results to ledger.jsonl
│
├─ [PHASE 2b: Inter-Brand Aggregation]
│  ├─ Collect accepted brand models
│  ├─ FedAvg aggregation (uniform)
│  └─ Save global model (input for Round N+1)
│
└─ Evaluate (optional):
   ├─ Load final global model
   ├─ Test on all branches' test data
   └─ Report accuracy, PRAUC, latency

Outputs:
├─ shared/round_000N/global_model.pt
├─ shared/scores.json
├─ shared/ledger.jsonl
└─ Evaluation metrics (F1, PRAUC, ROC-AUC, Precision, Recall, Comm, Latency)
```

### Data at Each Stage

| Stage | Data | Source | Format |
|-------|------|--------|--------|
| Phase 1 Input | Global model | `shared/round_N-1/global_model.pt` | PyTorch state_dict |
| Phase 1 Input | Local data | `data/processed/3_local_silo_balancing/{BANK_ID}/` | CSV features |
| Phase 1 Output | Local models | `/logs/{BANK_ID}_local_model.pt` | PyTorch state_dict |
| Phase 2a Input | Local models | `shared/round_N/` | PyTorch state_dict (9 files) |
| Phase 2a Output | Brand models | `shared/round_N/brand_models/` | PyTorch state_dict (3 files) |
| Phase 2b Input | Brand models | `shared/round_N/brand_models/` | PyTorch state_dict (3 files) |
| Phase 2b Output | Global model | `shared/round_N/global_model.pt` | PyTorch state_dict |
| Metadata | Scores | `shared/scores.json` | JSON dict {brand_id: score} |
| Audit | Ledger | `shared/ledger.jsonl` | JSONL (one record/line) |

---

## Performance & Scaling

### Computational Complexity

**Per Round**:
- Phase 1: 9 parallel trainings (2 epochs each, ~2-3 min per branch)
- Phase 2a: 3 HQ aggregations (sequential, <1 min total)
- Phase 2b: 1 global aggregation (<1 sec)

**Total per round**: ~3-5 minutes (dominated by Phase 1 Docker execution)

### Communication Overhead

**Exchanged Models per Round**:
- Phase 1 → Phase 2a: 9 local models (~10 MB each = 90 MB)
- Phase 2a → Phase 2b: 3 brand models (~10 MB each = 30 MB)
- Total: ~120 MB per round

**Optimization Opportunities**:
- Model compression (quantization, pruning)
- Differential privacy overhead (currently minimal)
- Round skip for non-stragglers

### Scalability Considerations

**Current Setup**: 3 brands × 3 branches = 9 total clients

**To Scale to Larger Deployments**:
1. Add more brands (linear increase in Phase 2b)
2. Add more branches per brand (polynomial increase in Phase 2a)
3. Implement async aggregation (reduce deadline wait)
4. Use weighted FedAvg by data distribution size

---

## Conclusion

This FL layer implementation provides:

✅ **Hierarchical Aggregation**: Two-tier (HQ-level, global-level) prevents bottlenecking
✅ **Robustness**: Deadline-based, validation gates, backup recovery
✅ **Privacy**: Differential privacy via gradient clipping and noise injection
✅ **Auditability**: Complete ledger of all model decisions
✅ **Flexibility**: Configuration-driven (topology, thresholds, deadlines)
✅ **Containerization**: Docker-based deployment for easy scaling

The system balances **model quality** (validation, recovery) with **system resilience** (straggler handling, backup mechanisms) to enable reliable collaborative training across distributed financial institutions.
