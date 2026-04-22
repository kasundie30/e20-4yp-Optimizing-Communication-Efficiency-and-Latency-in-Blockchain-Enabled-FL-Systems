# Methods
*(Draft — Research Paper: "Optimizing Communication Efficiency and Latency in Blockchain-Enabled Federated Learning Systems")*

---

## II. Methods

### A. System Model and Architectural Design

#### 1. Network Topology and Participants

The proposed system models a banking consortium in which $K$ independent financial institutions (banks) collaborate to train a shared fraud detection model without disclosing raw customer transaction records. Each bank $b_k$ ($k = 1, \ldots, K$) operates a set of $M_k$ branch nodes $\{v_{k,1}, \ldots, v_{k,M_k}\}$, each holding a private, non-overlapping data silo. In our experiments $K = 3$ (BankA, BankB, BankC) with $M_k = 3$ branches per bank, yielding nine data-holding participants in total.

Within each bank, a designated **Headquarters (HQ) peer** ($v_{k,0}$, corresponding to `peer0.bank*` in the Hyperledger Fabric network) acts as the intra-bank aggregator and the bank's representative to the global consortium. A **Backup HQ** ($v_{k,1}$, `peer1.bank*`) mirrors the ledger and assumes the HQ role if the primary peer fails. The full network therefore comprises:

| Node class | Count | Role |
|---|---|---|
| Branch nodes | $K \times M_k$ | Local training with private data |
| HQ peers | $K$ | Intra-bank aggregation, cross-bank verification |
| Backup HQ peers | $K$ | High-availability failover |
| Raft orderers | 3 | Crash-fault-tolerant transaction ordering |
| IPFS daemon | 1 (shared) | Off-chain model weight storage |
| FastAPI server | 1 | Python-to-Fabric REST bridge |

All HQ and Backup peers are members of a single Hyperledger Fabric channel (`fraud-detection-global`), with an endorsement policy requiring signatures from at least 2-of-3 bank Membership Service Providers (MSPs). The three Raft orderer nodes provide crash-fault-tolerant block ordering without proof-of-work overhead.

#### 2. Threat Model

We consider a semi-honest to Byzantine threat model with the following assumptions:

- **Data privacy**: No branch node or HQ peer releases raw transaction records. Only serialised model weight tensors (state dictionaries) are transmitted between participants.
- **Byzantine participants**: Up to $f = 1$ bank out of $K = 3$ may behave arbitrarily—submitting poisoned model weights, inflating self-reported validation scores, or attempting to replay previously accepted model CIDs. The system tolerates this under the condition $K \geq 2f + 1$.
- **Infrastructure integrity**: The IPFS daemon and Raft orderers are assumed honest. Their outputs are independently verified by SHA-256 hash checks on every model download, providing integrity even if the storage layer is compromised at the byte level.
- **Self-certification prevention**: Both the API server and the Fabric chaincode enforce that a bank cannot vote to verify its own submitted model ($\text{verifier\_id} \neq \text{target\_bank\_id}$), eliminating self-endorsement as an attack vector.

#### 3. Two-Tier Hierarchical Architecture

The central architectural contribution is a **two-tier aggregation hierarchy** that partitions FL communication into two scopes with distinct trust requirements.

**Tier 1 — Intra-bank (Branch → HQ):** All branch-to-HQ communication remains within a bank's private network. The HQ performs a Federated Averaging (FedAvg) over branch model updates, producing a single *cluster model* $\theta_k^{(r)}$ for round $r$. Only this one aggregated model per bank ever exits the private network, reducing upstream traffic by a factor of $M_k$ relative to a flat topology where every branch submits directly to the global aggregator.

**Tier 2 — Inter-bank (HQ → Blockchain → Global Aggregator):** Each bank's cluster model is published to the consortium via the Fabric ledger (by CID pointer) and subjected to cross-bank CBFT consensus before eligibility for global aggregation. The global aggregator performs a trust-weighted FedAvg over all accepted cluster models to produce the round's global model $\theta_{\text{global}}^{(r)}$.

This separation provides three simultaneous benefits: (i) communication overhead scales with $K$ rather than $\sum_k M_k$; (ii) straggler failures are contained within Tier 1 and do not propagate to the global round; and (iii) Byzantine resistance is concentrated at Tier 2 where the cryptographic audit trail is most valuable.

#### 4. Blockchain Substrate — Hyperledger Fabric

Hyperledger Fabric 2.5 is selected as the permissioned blockchain because its MSP-based identity model, Raft-based crash-fault-tolerant ordering, and Go chaincode execution environment align with the requirements of a regulated banking consortium without introducing the token economics or probabilistic finality of public blockchains.

The custom Go chaincode (`cbft-fl`) implements the following on-chain state machine per round:

| On-chain function | Purpose |
|---|---|
| `SubmitClusterUpdate` | Records `{bank_id, round, model_cid, model_hash, val_score}`; enforces CID uniqueness to prevent replay attacks |
| `SubmitVerification` | Records a boolean vote from a peer HQ; enforces $\text{verifier} \neq \text{submitter}$ |
| `SubmitCommit` | Locks a bank's model as "Accepted" after quorum of positive votes |
| `StoreGlobalModel` | Publishes the global model CID and advances the `latest_round` pointer |
| `UpdateTrustScore` | Adjusts per-bank trust score based on whether the bank's model improved or degraded the global aggregate |
| `GetTrustScores` | Returns current trust score vector for all banks; consumed by global aggregator |

#### 5. Hybrid Off-Chain Storage — IPFS

A Hyperledger Fabric block has a default maximum size of 1 MB. The LSTMTabular model used in this work serialises to approximately 0.49 MB—close to this limit. To prevent ledger bloat and enable efficient peer-to-peer model distribution, we adopt a hybrid storage pattern: model weight files are stored on **IPFS** (Kubo daemon), and only the 46-byte Content Identifier (CID) and an independent SHA-256 hash of the raw bytes are recorded on the Fabric ledger.

This design provides two security properties. First, IPFS CIDs are derived deterministically from file content (SHA-256 Merkle DAG), so any modification to the downloaded bytes changes the CID and immediately flags tampering. Second, the independent SHA-256 on-chain record provides a second verification layer: the global aggregator recomputes the hash of every downloaded model and rejects any file whose hash does not match the ledger record, closing the gap between IPFS addressing and ledger integrity.

---

### B. Proposed Model

#### 1. Local Model Architecture — LSTMTabular

Each participating branch trains a **LSTMTabular** model—a single-layer Long Short-Term Memory network with one fully-connected output head:

$$\text{LSTM}: \mathbb{R}^{T \times d} \rightarrow \mathbb{R}^{T \times h}, \quad \text{FC}: \mathbb{R}^{h} \rightarrow \mathbb{R}$$

where $d = 30$ (input feature dimension), $h = 30$ (hidden dimension), and $T = 1$ (each transaction is modelled as a sequence of length one). The model is defined as:

```python
class LSTMTabular(nn.Module):
    def __init__(self, input_dim=30, hidden_dim=30, num_layers=1):
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc   = nn.Linear(hidden_dim, 1)   # scalar logit output
```

The output of `self.fc` is an unbounded logit; fraud probability is obtained by applying a sigmoid function during inference. The LSTM is chosen over a feedforward MLP because credit card transactions encode temporal behavioural context—sequential spending patterns that LSTM's gating mechanism can capture through its cell state $c_t$ and hidden state $h_t$:

$$f_t = \sigma(W_f [h_{t-1}, x_t] + b_f), \quad i_t = \sigma(W_i [h_{t-1}, x_t] + b_i)$$
$$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t, \quad h_t = o_t \odot \tanh(c_t)$$

The minimal architecture (~10,000 parameters, ~0.49 MB serialised) ensures the communication overhead per round remains low and the model is tractable to train on CPU-only hardware within a single local epoch.

#### 2. Dataset and Non-IID Partitioning

Experiments use the **Kaggle Credit Card Fraud Detection dataset** [CITE], consisting of 284,807 transactions with 30 PCA-anonymised features and a severe class imbalance of approximately 0.17% fraud. This dataset is partitioned into non-IID bank silos through the following pipeline:

1. **Global feature scaling**: A `StandardScaler` is fit on the full dataset to produce comparable feature ranges across silos.
2. **Bank silo split**: Records are distributed among $K = 3$ banks to create organizationally distinct subsets reflecting heterogeneous geographic or demographic transaction patterns.
3. **Per-branch local balancing**: Within each bank, branch data is locally balanced using oversampling of the minority class to ensure that no branch silo is entirely devoid of fraud labels, preventing degenerate local training.
4. **Per-branch local re-scaling**: Each branch applies its own `StandardScaler` fitted only on its own silo. This prevents privacy leakage through global statistics while ensuring stable gradient flow in the LSTM.

The resulting partitions are non-identically distributed (non-IID): each branch's fraud rate, transaction volume, and feature distributions differ, directly reflecting the federated learning challenge that motivates this work.

#### 3. Tier-1: Local Branch Training with Differential Privacy

At the start of each FL round $r$, every branch $v_{k,j}$ downloads the global model $\theta_{\text{global}}^{(r-1)}$ from IPFS (using the CID published on the blockchain) and uses it as the initialisation point for local training. Local training runs for one epoch using the Adam optimiser with learning rate $\eta = 10^{-3}$ and batch size 256.

To address the severe class imbalance, the binary cross-entropy loss is modified with a positive class weight computed from the branch's own label distribution:

$$\mathcal{L} = -\frac{1}{N} \sum_{i=1}^{N} \left[ w_+ \cdot y_i \log \hat{y}_i + (1 - y_i) \log (1 - \hat{y}_i) \right], \quad w_+ = \frac{n_{\text{neg}}}{n_{\text{pos}}}$$

This per-branch dynamic weighting ensures that each node's loss function is calibrated to its own fraud rate without requiring inter-branch communication of label statistics.

**Differential Privacy (DP)** is applied at every gradient step to bound information leakage from model updates. The implemented mechanism follows the DP-SGD order:

1. Compute per-batch gradients via backpropagation.
2. Clip the gradient L2 norm: $\nabla \leftarrow \nabla \cdot \min\!\left(1, \frac{\Delta}{\|\nabla\|_2}\right)$, with clipping bound $\Delta = 1.0$.
3. Add calibrated Gaussian noise: $\nabla \leftarrow \nabla + \mathcal{N}(0, (\Delta \cdot \sigma)^2 \mathbf{I})$, with noise multiplier $\sigma = 0.05$.
4. Update parameters via the optimizer after clipping and noise injection.

The clipping step bounds the sensitivity $\Delta f$ of each update, while the Gaussian mechanism provides $(ε, δ)$-differential privacy. The noise multiplier of 0.05 is chosen empirically to provide a formal privacy guarantee without significantly degrading convergence.

After training, the branch saves its updated `state_dict` (weight tensor dictionary) to the shared Docker volume. **Raw transaction data never leaves the branch container.**

#### 4. Tier-1: Intra-Bank Aggregation

The **HQ Agent** of each bank $b_k$ performs the following sequence after collecting branch updates:

**Step 4a — Deadline-based collection.** The HQ polls for branch model submissions with a configurable timeout ($\tau_{\text{deadline}} = 5$ s in production). Any branch that has not responded by the deadline is excluded from the current round without blocking global progress.

**Step 4b — Intra-cluster FedAvg.** The HQ computes a sample-count-weighted average of all received branch state dictionaries:

$$\theta_k^{(r)} = \frac{\sum_{j=1}^{B_k} n_{k,j} \, \theta_{k,j}^{(r)}}{\sum_{j=1}^{B_k} n_{k,j}}$$

where $B_k$ is the number of branches that responded before the deadline and $n_{k,j}$ is the number of training samples used by branch $j$ of bank $k$ in round $r$.

**Step 4c — Global model blending.** If a prior global model $\theta_{\text{global}}^{(r-1)}$ exists (i.e., $r > 1$), the cluster model is blended with it to stabilise training and prevent catastrophic forgetting:

$$\tilde{\theta}_k^{(r)} = \beta \cdot \theta_{\text{global}}^{(r-1)} + (1 - \beta) \cdot \theta_k^{(r)}, \quad \beta = 0.30$$

**Step 4d — Validation gate.** The blended cluster model is evaluated on a random 15% sample of the HQ's local validation set using **PR-AUC** (Precision-Recall Area Under the Curve) as the gating metric. PR-AUC is adopted over ROC-AUC and accuracy because both of the latter metrics are known to be misleading under severe class imbalance [CITE Davis & Goadrich, 2006]: ROC-AUC is inflated by the large number of true negatives, and accuracy trivially exceeds 99% by predicting the majority class exclusively. The gate threshold is $\tau_{\text{val}} = 0.20$ at Tier 1. If the gate is not met, the blended model is re-evaluated once; if it still fails, the bank abstains from submitting for this round.

#### 5. Tier-2: CBFT Consensus Protocol

Banks that pass the Tier-1 validation gate participate in the **Consensus-Based Federated Trust (CBFT)** protocol—a lightweight three-phase Byzantine-fault-tolerant consensus mechanism executed on the Fabric ledger:

**Phase 1 — Propose.** The submitting HQ Agent uploads the cluster model to IPFS (obtaining a CID) and computes its SHA-256 hash. It then calls `SubmitClusterUpdate` on the chaincode, which records `{bank_id, round, model_cid, model_hash, val_score}` on the ledger and enforces CID uniqueness to prevent replay attacks.

**Phase 2 — Verify.** Each peer HQ agent independently performs cross-bank evaluation:
1. Fetches the submitted bank's `{model_cid, model_hash}` from the chaincode.
2. Downloads the model from IPFS using the CID.
3. Recomputes SHA-256 of the downloaded bytes and verifies it against the on-chain `model_hash`. A mismatch produces an automatic `False` vote.
4. If the hash matches, loads the model into a `LSTMTabular` instance and evaluates it on the verifier's own local validation data (15% sample, PR-AUC ≥ $\tau_{\text{chain}}$ = 0.7).
5. Submits the boolean vote via `SubmitVerification` to the chaincode (self-verification is rejected by the chaincode).

**Phase 3 — Commit.** Each HQ polls `CheckConsensus` on the chaincode. When the number of positive verification votes reaches the quorum threshold $Q = \lceil K/2 \rceil + 1 = 2$, the bank calls `SubmitCommit`, marking its model as **"Accepted"** on the ledger. Only "Accepted" models are eligible for global aggregation.

With $K = 3$ banks and $f = 1$ maximum Byzantine participants, CBFT achieves Byzantine fault tolerance ($K \geq 2f + 1$) with $O(K)$ vote messages per phase—significantly lower complexity than classical PBFT which requires $O(K^2)$ message exchanges [CITE Castro & Liskov, 1999].

#### 6. Tier-2: Trust-Weighted Global Aggregation

Once CBFT consensus is reached, the **Global Aggregator** (running on BankA's node) computes the inter-bank federated average weighted jointly by sample count and historical trust score:

$$\theta_{\text{global}}^{(r)} = \frac{\sum_{k \in \mathcal{A}^{(r)}} w_k \cdot \tilde{\theta}_k^{(r)}}{\sum_{k \in \mathcal{A}^{(r)}} w_k}, \quad w_k = \tau_k^{(r-1)} \times n_k$$

where $\mathcal{A}^{(r)}$ is the set of banks accepted by CBFT in round $r$, $\tau_k^{(r-1)}$ is bank $k$'s trust score at the start of round $r$ (initialised to 1.0 for new participants), and $n_k = \sum_j n_{k,j}$ is the total number of training samples aggregated by bank $k$'s HQ in this round.

After global aggregation, the **trust scores** are updated on-chain by the chaincode based on whether each bank's contribution improved the global model:

$$\tau_k^{(r)} = \text{clip}\!\left(\tau_k^{(r-1)} + \begin{cases} \alpha & \text{if bank's model improved global PR-AUC} \\ -\beta & \text{otherwise} \end{cases}, \; \tau_{\min}, \; \tau_{\max}\right)$$

with $\alpha = 0.1$ (reward), $\beta = 0.2$ (penalty), $\tau_{\min} = 0.1$ (floor preventing permanent exclusion), and $\tau_{\max} = 3.0$ (ceiling preventing runaway dominance). The asymmetric reward-to-penalty ratio (1:2) is deliberate: contributing a degraded model is more harmful than withholding a good one, which discourages intermittent misbehaviour.

The global model is then uploaded to IPFS and its CID published on the ledger via `StoreGlobalModel`, completing the round and making the new model available for all banks to fetch at the start of round $r+1$.

#### 7. Communication and Latency Optimisations

Beyond the architectural hierarchy, the following implementation-level optimisations reduce per-round communication overhead and end-to-end latency:

| Optimisation | Mechanism | Impact |
|---|---|---|
| **Compact model** | Single-layer LSTM, $h = 30$, ~0.49 MB serialised | ↓ Upload/download size per round |
| **IPFS off-chain storage** | Only 46-byte CID + SHA-256 string on ledger | ↓ Fabric block size; faster endorsement propagation |
| **Hierarchical aggregation** | Branch → HQ → Global reduces blockchain submissions from $\sum M_k$ to $K$ | ↓ Cross-bank communication by factor $M_k$ |
| **Deadline-based straggler handling** | `wait_for_submissions(deadline_sec=5)` at Tier 1; `consensus_timeout=120 s` at Tier 2 | ↓ Round latency; prevents indefinite blocking |
| **Fractional validation (15%)** | 15% random sample of validation data for PR-AUC gating | ↓ Per-model evaluation time from ~7 s to <1 s |
| **Backup HQ + model blending** | Failed cluster models blended with prior global ($\beta = 0.30$) before exclusion | ↓ Wasted rounds; partial training signal recovered |
| **Asynchronous REST gateway** | FastAPI + Uvicorn ASGI decouples Python FL pipeline from Fabric CLI latency | ↓ Per-round blocking time during blockchain I/O |
| **CBFT lightweight voting** | Boolean vote cascade: $O(K)$ messages vs. $O(K^2)$ for PBFT | ↓ Consensus overhead |

---

> **Note to authors**: The mathematical notation uses standard FL conventions. Placeholders `[CITE]` to resolve: Davis & Goadrich (ICML 2006) for PR-AUC justification; Castro & Liskov (OSDI 1999) for PBFT complexity comparison; McMahan et al. (AISTATS 2017) for FedAvg; and the original Kaggle dataset citation.
