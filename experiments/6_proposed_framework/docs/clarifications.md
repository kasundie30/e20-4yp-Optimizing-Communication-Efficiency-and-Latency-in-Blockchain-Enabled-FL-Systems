# Framework Clarifications

---

## 1. Where is the Ledger Stored? What is the Purpose of CouchDB?

### Where the ledger lives

Hyperledger Fabric maintains **two separate data stores** per peer node:

| Store | What it holds | Default backend |
|-------|--------------|-----------------|
| **Block store** | The immutable, append-only chain of blocks (the actual blockchain) | LevelDB on disk (`/var/hyperledger/production/ledgersData/chains/`) |
| **World State** | The *current* value of every key — a snapshot derived from replaying the chain | **CouchDB** (our setup) or LevelDB |

In our framework, each Fabric peer (`peer0` / `peer1` of BankA, BankB, BankC) stores its own full copy of both. There is no single "central" ledger — every peer has an identical, independently verified replica.

### What CouchDB stores in our project

The `cbft.go` chaincode writes these key families to the world state (and therefore into CouchDB):

| Key pattern | Content |
|-------------|---------|
| `trust~BankA` | `TrustScore {bankID, score, updatedAt}` |
| `update~BankA~3` | `ClusterUpdate {bankID, round, modelCID, modelHash, valScore, accepted}` |
| `verify~BankA~3~BankB` | `VerificationVote {verifierID, targetBankID, round, verified}` |
| `commit~BankA~3~BankC` | `CommitRecord {committerID, targetBankID, round}` |
| `global~3` | `GlobalModel {round, globalCID, globalHash}` |
| `latest_round` | Integer — highest completed round |
| `cid~<ipfs-cid>` | Anti-replay marker for each submitted CID |

### Why CouchDB over LevelDB?

| Feature | LevelDB | CouchDB |
|---------|---------|---------|
| Data format | Key-value bytes | JSON documents |
| Rich queries | ❌ Key-range only | ✅ Mango (MongoDB-style) queries |
| Visibility | No web UI | Built-in Fauxton UI for inspection |
| Use case | Simple counters / small structs | Structured records with multiple fields |

Our chaincode stores JSON structs (`TrustScore`, `ClusterUpdate`, etc.). CouchDB lets the chaincode — and external tools — query these with filters such as *"give me all cluster updates for round 5 where `accepted = true`"* without scanning the full ledger. LevelDB would require iterating all keys manually.

> **TL;DR**: CouchDB is the live, queryable snapshot of the blockchain's current state. The blockchain itself (immutable block chain) lives in the file-system block store alongside it. Both together constitute "the ledger".

---

## 2. Diagram Corrections & Clarifications

### 2a. After PR-AUC check fails — shouldn't the Backup node act?

**Short answer: No — not at this point. The Backup HQ is triggered by *hardware/availability* failure of the primary HQ, not by a *model quality* failure.**

Here is the distinction:

| Failure type | What triggers it | Who handles it |
|---|---|---|
| **Primary HQ crashes / unreachable** | `peer0` goes offline | Backup HQ (`peer1`) takes over; chaincode `ActivateBackup` called |
| **Cluster model fails PR-AUC gate** | Low-quality aggregate (model is bad, not the server) | `backup_logic.py` — model blending, not a node switch |

When PR-AUC fails the validation threshold (`< 0.20` locally / `< 0.7` on-chain):

```
Model fails PR-AUC gate
        │
        ▼
blend_with_global(brand_model, prev_global, beta=0.3)
  → w_recovered = 0.3 × w_global + 0.7 × w_brand
        │
        ▼
Re-evaluate recovered model
        │
   ┌────┴────┐
PASS         FAIL
  │            │
Submit        Skip round
to chain    (no submission this round,
              trust score not penalised
              for infrastructure; future
              rounds continue normally)
```

The "Backup" that runs here is a **software backup** (model blending) inside `hq_agent.py` / `backup_logic.py`. The **hardware backup** (`peer1` / Backup HQ) only activates through `ActivateBackup` chaincode when `peer0` is down.

> **Diagram fix**: The PR-AUC failure branch in the diagram should say "Model Blending Recovery (`backup_logic.py`)" rather than implying a node failover.

---

### 2b. Before inter-cluster aggregation — does the HQ validate models from *other* banks?

**Yes — this is exactly what CBFT Phase 2 (Verification) does, and it happens *before* the GlobalAggregator runs.** The current diagram shows this correctly but labeled it vaguely. Here is the precise sequence:

```
Each HQ has submitted its own cluster model (CBFT Phase 1)
                    │
                    ▼
     CBFT Phase 2 — Cross-HQ Validation
  ┌─────────────────────────────────────────┐
  │  BankB-HQ downloads BankA's model CID   │
  │  from IPFS → recalculates SHA-256 →     │
  │  evaluates PR-AUC on BankB's own data → │
  │  POST /submit-verification (vote=True/  │
  │  False) → on-chain                      │
  └─────────────────────────────────────────┘
  (Same happens: BankC verifies BankA, BankA verifies BankB, etc.)
                    │
                    ▼
     CBFT Phase 3 — Commit
  Each HQ checks if peer has ≥ 2 positive votes → POST /submit-commit
                    │
                    ▼
     CheckConsensus → returns accepted banks list
                    │
                    ▼
     GlobalAggregator.aggregate_round()  ← runs ONLY with accepted banks
```

So the validation of *other banks' models* is done by **all HQs in Phase 2**, using their own local validation data as the reference. The GlobalAggregator does not re-validate; it trusts the on-chain consensus result.

---

## 3. The 3 Stages of CBFT — Exactly Who Calls What

CBFT stands for **Consensus-Based Federated Trust**. It is a 3-phase protocol modelled after the Prepare/Commit phases of classical BFT, adapted for federated learning.

---

### Phase 1 — PROPOSE (SubmitClusterUpdate)

**Goal**: A bank makes its trained model available and registers it on-chain.

```
hq_agent.py::run_round()
    │
    ├─ torch.save(avg_sd) → bytes
    ├─ SHA-256(bytes) → model_hash
    ├─ ipfs_upload(bytes) → model_cid          [IPFS]
    │
    └─ api_client.py::submit_update(
           bank_id, round, model_cid,
           model_hash, val_score
       )
           │
           ▼
       FastAPI POST /submit-update
           │
           ▼
       fabric_client.py::invoke("SubmitClusterUpdate")
           │
           ▼
       cbft.go::SubmitClusterUpdate()
           ├─ Enforce valScore ≥ 0.7 (on-chain)
           ├─ Check CID not already submitted (replay attack guard)
           └─ Write ClusterUpdate to ledger (accepted=false initially)
```

**Quorum needed**: None (unilateral — any bank can propose).

---

### Phase 2 — VERIFY (SubmitVerification)

**Goal**: Every other bank independently evaluates the proposer's model and casts a vote.

```
hq_agent.py::verify_peer_updates(round, peers)
    │
    For each peer bank (that is NOT self):
    │
    ├─ api_client.py::get_cluster_update(target, round)
    │       │
    │       ▼
    │   FastAPI GET /cluster-update/{bank_id}/{round}
    │       │
    │       ▼
    │   cbft.go::GetClusterUpdate()  → {modelCID, modelHash}
    │
    ├─ ipfs_download(model_cid) → bytes             [IPFS]
    ├─ compute_sha256(bytes) vs stored modelHash      [integrity check]
    ├─ torch.load(bytes) → peer_state_dict
    ├─ evaluate_model(peer_state_dict, own_val_data)
    │       → metrics["pr_auc"]
    │
    ├─ verified = (pr_auc >= val_threshold)
    │
    └─ api_client.py::submit_verification(
           verifier_id=self,
           target_bank_id=peer,
           round, verified
       )
           │
           ▼
       FastAPI POST /submit-verification
           │
           ▼
       cbft.go::SubmitVerification()
           └─ Write VerificationVote to ledger
```

**Quorum needed**: `VerifyQuorum = 2`  
(i.e., at least 2 banks must vote `verified=True` before Phase 3 can proceed)

---

### Phase 3 — COMMIT (SubmitCommit)

**Goal**: Lock in the acceptance of a bank's model once enough verification votes exist.

```
hq_agent.py::commit_peer_updates(round, peers)
    │
    For each peer bank:
    │
    ├─ api_client.py::check_verify_quorum(target, round)
    │       │
    │       ▼
    │   FastAPI GET /verify-quorum/{bank_id}/{round}
    │       │
    │       ▼
    │   cbft.go::CheckVerifyQuorum()
    │       └─ countVerifications() ≥ 2 ?
    │
    └─ (if quorum met) api_client.py::submit_commit(
           committer_id=self,
           target_bank_id=peer,
           round
       )
           │
           ▼
       FastAPI POST /submit-commit
           │
           ▼
       cbft.go::SubmitCommit()
           ├─ countVerifications() guard (double-check ≥ 2)
           └─ Write CommitRecord to ledger
```

**Quorum needed**: `CommitQuorum = 2`  
(at least 2 banks must commit before a bank is declared fully accepted)

---

### Full CBFT Component Map

```
Python Layer                  FastAPI (api-server)         Fabric Chaincode (cbft.go)
─────────────────             ────────────────────         ──────────────────────────
hq_agent.run_round()
  └─ api_client.submit_update ──► POST /submit-update ──► SubmitClusterUpdate()
                                                               └─ writes ClusterUpdate

hq_agent.verify_peer_updates()
  └─ api_client.get_cluster_update ─► GET /cluster-update ─► GetClusterUpdate()
  └─ [evaluate locally in Python]
  └─ api_client.submit_verification ─► POST /submit-verification ─► SubmitVerification()
                                                                         └─ writes VerificationVote

hq_agent.commit_peer_updates()
  └─ api_client.check_verify_quorum ─► GET /verify-quorum ─► CheckVerifyQuorum()
  └─ api_client.submit_commit ─────► POST /submit-commit ──► SubmitCommit()
                                                                  └─ writes CommitRecord

GlobalAggregator.wait_for_consensus()
  └─ api_client.check_consensus ───► GET /check-consensus ─► CheckConsensus()
                                                                  └─ reads all votes/commits
                                                                  └─ marks accepted=true
                                                                  └─ returns ["BankA","BankC"]
```

---

## 4. What is Quorum and How Does `CheckConsensus` / Polling Work?

### What is a Quorum?

A **quorum** is the **minimum number of votes required for a decision to be valid**. It prevents any single party from making a unilateral decision in a multi-party system.

In our CBFT with 3 banks:

| Threshold | Value | Meaning |
|-----------|-------|---------|
| `VerifyQuorum = 2` | 2 out of 3 | At least 2 banks must cast a `verified=True` vote before any bank can commit |
| `CommitQuorum = 2` | 2 out of 3 | At least 2 banks must submit a commit before `CheckConsensus` marks the bank as accepted |

Setting quorum to 2-of-3 provides **Byzantine fault tolerance for 1 faulty node**: even if one bank votes maliciously or is offline, the honest majority (2 banks) can still proceed.

### What does `CheckConsensus` do?

`CheckConsensus` is the chaincode function (`cbft.go`) that **tallies all votes and commits** and returns the list of banks whose models have passed all three phases:

```
GET /check-consensus/{round}
        │
        ▼
cbft.go::CheckConsensus(round)
    │
    For each bank in [BankA, BankB, BankC]:
    │
    ├─ Does a ClusterUpdate exist for this bank/round? → if not, skip
    ├─ countVerifications(bank, round) ≥ 2 ?            → if not, skip
    ├─ countCommits(bank, round) ≥ 2 ?                  → if not, skip
    │
    └─ Mark update.accepted = true on ledger
       Add bankID to accepted[] list
    │
    ▼
Return JSON: ["BankA", "BankC"]   ← these are fully CBFT-accepted this round
```

### What does "polling" CheckConsensus mean?

Because CBFT Phases 2 and 3 happen asynchronously (different HQ agents submit verifications and commits at different times), the GlobalAggregator **cannot know exactly when consensus is reached**. It must repeatedly ask the chaincode until at least one bank appears in the accepted list:

```python
# global_aggregator.py::wait_for_consensus()
while True:
    accepted = api_client.check_consensus(round)   # calls GET /check-consensus
    if accepted:                                    # non-empty list
        return accepted                             # ← stop polling
    if elapsed > consensus_timeout (60s):
        return []                                   # ← give up after timeout
    time.sleep(poll_interval)                       # wait 5s, then try again
```

**Visual timeline:**

```
t=0s   GlobalAggregator starts polling CheckConsensus → [] (no votes yet)
t=5s   Poll again → [] (phases 2 & 3 still in progress)
t=10s  BankB and BankC finish verifying + committing BankA
t=12s  Poll again → ["BankA"] ← quorum met for BankA
t=12s  Aggregator downloads BankA's model and starts aggregation
t=15s  BankA and BankC finish verifying BankB → ["BankA", "BankB"]
       (GlobalAggregator has already started — BankB may be included if
        consensus arrives before aggregation completes, depending on timing)
```

**Why not use a push/event approach?** Hyperledger Fabric does support event listening, but polling against a deterministic chaincode query is simpler, easier to test (injectable `time.sleep`), and avoids the complexity of managing long-lived event subscriptions across a network boundary.

---

*Saved: `6_proposed_framework/docs/clarifications.md`*
