# Final Evaluation — Questions & Answers

**Project**: Optimizing Communication Efficiency and Latency in Blockchain-Enabled Federated Learning Systems  
**Prepared by**: Senior Research Evaluator  
**Context**: Credit Card Fraud Detection using Hierarchical Clustered Federated Learning (HCFL) with Hyperledger Fabric and IPFS

> Questions are grouped by research dimension, from foundational motivation through to critical limitations. Answers reflect the depth expected of a final-year project defence.

---

## Part 1 — Motivation & Problem Statement

---

**Q1. Why is federated learning necessary for fraud detection? Why can't banks simply share their fraud transaction data with a central model trainer?**

**A:** Banks face three fundamental barriers to centralised data sharing:

1. **Regulatory prohibition**: GDPR (EU), PDPA (Sri Lanka), and equivalent banking regulations prohibit sharing raw customer financial records across institutional boundaries without explicit consent. A bank that exfiltrates transaction logs to a third-party trainer exposes itself to severe regulatory penalties.

2. **Competitive confidentiality**: Transaction patterns reveal customer acquisition strategies, merchant relationships, and proprietary risk models. Sharing this data gives competitors direct intelligence about a bank's business operations.

3. **Attack surface amplification**: A centralised data warehouse containing transaction records from multiple banks is an extremely high-value target. A single breach would compromise the privacy of millions of customers across all participating institutions simultaneously.

Federated Learning resolves all three by keeping raw data on-premise at each bank while allowing the mathematical gradient signal (in the form of model weights) to be shared. No individual transaction record ever leaves the originating institution.

---

**Q2. Why specifically credit card fraud detection as the application domain? What properties of this problem make it particularly well-suited to an FL study?**

**A:** Credit card fraud detection has several properties that make it an ideal FL benchmark:

- **Severe class imbalance** (~0.17% fraud rate): This reflects real-world distributions and makes naive accuracy metrics meaningless, forcing the use of PR-AUC and F1 which are sensitive to minority-class performance. This is non-trivial for FL because imbalance varies across branches (non-IID), creating genuine convergence challenges.
- **Non-IID data distribution**: Different branches serve different demographics and geographies, resulting in heterogeneous fraud patterns. This is the central challenge FL research addresses.
- **High stakes per sample**: Each missed fraud event has a quantifiable financial cost, making the business value of improved recall directly measurable.
- **Temporal structure**: Transaction data has sequential dependencies (spending patterns evolve over time), which motivates our use of LSTM over a simple MLP.
- **Established benchmark**: The Kaggle creditcard.csv dataset (284,807 transactions from European cardholders) is the de-facto benchmark in the literature, allowing our results to be compared directly against published baselines.

---

**Q3. What is the core research problem your project is trying to solve beyond just "applying FL to fraud detection"?**

**A:** The specific research problem is the **tension between communication efficiency, latency, and security in a multi-institutional federated learning deployment**.

Standard FL (e.g., vanilla FedAvg) assumes a trusted central aggregator. In a banking consortium, no single bank should be trusted as a central authority. Introducing a blockchain to replace the aggregator introduces two new costs: **communication overhead** (blockchain transactions take time to be endorsed and committed) and **storage overhead** (large model files cannot be stored on-chain).

Our research question is: *Can we design an FL system that provides Byzantine fault tolerance and cryptographic auditability without paying a prohibitive latency cost?*  

Our specific contributions are:
1. The **two-tier hierarchy** that minimises inter-bank communication by pre-aggregating within each bank.
2. **IPFS-based off-chain model storage** that keeps blockchain state lightweight.
3. The **CBFT protocol** that combines ML-level verification (cross-bank PR-AUC evaluation) with blockchain-level consensus, providing a stronger Byzantine guarantee than standard PBFT without requiring cryptographic threshold signatures.
4. The **trust-weighted global FedAvg** that incorporates historical participation quality, which to our knowledge has not been combined with CBFT-style blockchain consensus in prior art.

---

## Part 2 — Model & Data Design

---

**Q4. Why did you choose an LSTM over a simpler MLP or a more powerful Transformer for this tabular dataset?**

**A:** This was a deliberate trade-off across three dimensions:

- **Against MLP**: The Kaggle CCFD dataset, while presented as tabular, represents transactions from a temporal stream. An LSTM treats each row as a timestep in a sequence, allowing it to implicitly model recency effects and spending velocity — patterns that a memoryless MLP cannot capture. In our preliminary experiments, the LSTM achieved higher PR-AUC under our FL training conditions.

- **Against Transformer**: A Transformer encoder would be superior in terms of expressive power, but it introduces a much larger parameter count, significantly increasing the model file size (and therefore IPFS upload/download latency per round), and requires more local epochs to converge — increasing per-round communication overhead. For this problem size (30 features, binary classification), the marginal gain from attention mechanisms does not justify the communication cost, which is the primary variable we are trying to optimise.

- **Architecture stability**: Once we committed to the `LSTMTabular` definition (`input_dim=30, hidden_dim=30, num_layers=1`), all branches, HQs, and the global aggregator can perform weight arithmetic — the core operation in FedAvg — without any model-compatibility checks beyond key name matching. This is a practical FL engineering constraint that constrained the architecture choices.

---

**Q5. You use PR-AUC as your primary validation metric rather than ROC-AUC or F1. Can you explain why, and what are the limitations of PR-AUC in this context?**

**A:** 

**Why PR-AUC:**  
ROC-AUC measures the area under the curve of True Positive Rate vs. False Positive Rate. On a dataset where 99.83% of samples are negative, a model that outputs "not fraud" for every sample achieves a ROC-AUC of 0.5 — but the FPR denominator (true negatives) is so large that it masks complete failure on the positive class. This is the well-documented "imbalanced class problem" with ROC-AUC.

PR-AUC (Precision-Recall AUC) only involves the positive class and predictions about the positive class. Precision = TP/(TP+FP); Recall = TP/(TP+FN). A model that never predicts fraud gets Recall=0 and therefore PR-AUC=0 from the outset, making it far more discriminating for our use case.

**Limitations of PR-AUC:**  
- PR-AUC is threshold-agnostic but does not tell us the optimal operating threshold. In a real deployment, a bank needs to choose a score cut-off that balances fraud losses against customer friction (false positives causing declined legitimate transactions). We log F1 and recall separately for this purpose.
- PR-AUC is sensitive to label imbalance across validation sets. If one branch's validation set happens to contain unusually few fraud samples (due to random sampling), the PR-AUC estimate will have high variance. This is partially addressed by our 15% sampling strategy maintaining consistent proportional representation.
- The metric is computed over a validation set at the HQ level. If the HQ's validation data is not representative of the global distribution (which is likely in a non-IID scenario), HR-AUC at one bank may not generalise to other banks' fraud patterns, making the threshold (τ=0.20 / τ=0.7) difficult to calibrate universally.

---

**Q6. How did you handle the non-IID data distribution problem? Is FedAvg still theoretically sound under non-IID conditions?**

**A:**

**Our handling:**  
We partition the Kaggle dataset into bank-specific silos using a `3_local_silo_balancing` pipeline. Each branch receives a stratified subset that preserves the local class distribution as-is (i.e., non-IID, reflecting real-world heterogeneity). Each branch applies a local StandardScaler fitted only to its own data, preventing global statistics from being shared implicitly through the preprocessing step.

**Theoretical soundness of FedAvg under non-IID:**  
This is an active research debate. The original FedAvg paper (McMahan et al., 2017) showed convergence under IID conditions. Li et al. (2020) proved that FedAvg can diverge under highly heterogeneous (non-IID) conditions; the "client drift" phenomenon causes local models to overfit their local distributions and the global average can move away from the true global optimum.

Our mitigations:
1. **Model blending with global (`backup_logic.py`, `blend_with_global(beta=0.3)`)**: After intra-bank FedAvg, we blend the result with the previous global model. This is mathematically equivalent to adding a proximal term that keeps local aggregates from drifting too far from the global model — similar to FedProx in effect.
2. **Trust-weighted global FedAvg**: Banks with historically stable PR-AUC (i.e., consistent with the global distribution) earn higher trust scores and therefore higher weight in the global average, partially correcting for distribution shift.
3. **Hierarchical aggregation**: By pre-aggregating within a bank first, we reduce the effective heterogeneity at the global level — three bank-level aggregates are "closer" to the global optimum than nine individual branch models.

We acknowledge that full convergence guarantees under our non-IID partitioning are not derived in this project; such an analysis would require additional theoretical work.

---

## Part 3 — Federated Learning Architecture

---

**Q7. Walk me through exactly what happens in a single federation round, step by step. Who initiates what?**

**A:** A single round proceeds as follows:

**Step 1 — Round Initialization (Blockchain):**  
The `HQAgent` calls `GET /global-model/{round-1}` via the FastAPI server. Fabric chaincode (`GetGlobalModel`) returns the CID and SHA-256 hash of the previous round's global model. The HQ downloads the weight file from IPFS and verifies the hash before loading weights.

**Step 2 — Local Training (Branches):**  
Each branch (in a Docker container) loads the global model and trains for 1 epoch over its local dataset using DP-SGD (gradient clip at L2-norm=1.0, Gaussian noise σ=0.05). The resulting `state_dict` is written to a shared volume mount. For Round 1, branches initialise with random weights (no prior global model exists).

**Step 3 — Deadline Collection (HQ):**  
The `RoundCoordinator` (`wait_for_submissions`) polls for branch `state_dict` files until the deadline (`deadline_seconds=5.0`) or all expected branches have submitted. Updates arriving after the deadline are discarded for this round.

**Step 4 — Intra-Bank Aggregation (HQ):**  
`HQAgent.run_round()` calls `fedavg(branch_updates)` producing a cluster-level `avg_sd`. This is blended with the previous global model: `blend_with_global(avg_sd, global_sd, beta=0.3)`.

**Step 5 — Validation Gate (HQ):**  
The cluster model is evaluated on 15% of the HQ's validation set. If `PR-AUC < 0.20`, the model is not submitted this round.

**Step 6 — IPFS Upload & Blockchain Submit (HQ → CBFT Phase 1):**  
`torch.save` → bytes → `SHA-256` hash computed → `ipfs_upload(bytes)` → CID received. `POST /submit-update` → Fabric chaincode `SubmitClusterUpdate` records the `(bank_id, round, modelCID, modelHash, valScore)` tuple. Chaincode independently enforces `valScore ≥ 0.7`.

**Step 7 — CBFT Phase 2 — Cross-Bank Verification:**  
Each HQ downloads peer banks' models from IPFS, recalculates SHA-256, evaluates PR-AUC on its own validation data, and calls `POST /submit-verification`. Votes are recorded on-chain.

**Step 8 — CBFT Phase 3 — Commit:**  
Each HQ checks if a peer bank has ≥2 positive verification votes (`GET /verify-quorum/{bank_id}/{round}`), then calls `POST /submit-commit`. Chaincode guards commits with `VerifyQuorum ≥ 2` pre-check.

**Step 9 — Global Aggregation (BankA's GlobalAggregator):**  
`GlobalAggregator.run_full_aggregation()` polls `GET /check-consensus/{round}` until at least one bank appears in the accepted list. It fetches trust scores via `GET /trust-scores`, downloads each accepted bank's model from IPFS, verifies hashes, computes trust-weighted FedAvg, uploads global model to IPFS, and calls `POST /store-global-model` → Fabric `StoreGlobalModel`.

**Step 10 — Trust Score Update:**  
`POST /update-trust-score` adjusts each bank's on-chain score (±α/β based on whether their update improved or degraded the global model).

**Step 11 — Next Round:**  
`latest_round` pointer on ledger increments. All HQs and branches start Step 1 for round `t+1`.

---

**Q8. The global aggregation is performed by BankA. Doesn't that create a centralisation problem — if BankA is malicious or goes down, everything breaks?**

**A:** This is a valid and important critique. Our current design does centralise the final aggregation step on BankA. We address this at two levels:

**Availability**: The Backup HQ (`ActivateBackup` in chaincode, `peer1` of each organisation) handles BankA's HQ failure. However, if all of BankA's infrastructure fails, global aggregation indeed stalls in the current implementation.

**Trust / Byzantine resistance**: Even if BankA's `GlobalAggregator` is the only one computing the trust-weighted average, the inputs it uses are:
- Model CIDs that are on-chain and verifiable by any participant
- Trust scores that are on-chain and tamper-evident
- IPFS models that are content-addressed (CID = hash of content)

A malicious BankA cannot forge the inputs. However, it could refuse to call `StoreGlobalModel` or compute an incorrect weighted average using the correct inputs while submitting a fraudulent result on-chain, since the chaincode does not re-verify the arithmetic.

**Our acknowledged limitation and proposed fix**: A more robust design would rotate the global aggregator role based on the round number (e.g., `aggregator = banks[round % len(banks)]`). Alternatively, a more sophisticated approach would implement verifiable computation — the chaincode could store the individual model hashes and trust score snapshot, allowing any participant to independently verify the arithmetic of the global aggregation. This is identified as future work.

---

**Q9. Why did you choose Raft consensus for the Hyperledger Fabric orderer rather than BFT (Byzantine Fault Tolerant) consensus?**

**A:** The orderer consensus and the application-level CBFT protocol address different threat models:

- **Raft** is a **crash-fault-tolerant (CFT)** consensus protocol. It tolerates orderer nodes that crash or become unreachable, but it does **not** tolerate a malicious orderer that actively sends conflicting messages.
- **BFT ordering** (e.g., SmartBFT, available from Fabric 2.4) would tolerate a malicious orderer — but it requires a minimum of `3f+1` orderer nodes to tolerate `f` Byzantine nodes, making it at least 4 nodes for 1 failure.

**Our rationale for choosing Raft:**  
1. In our banking consortium, the orderer nodes are operated jointly by the participating banks. If a bank is already participating in the consortium and operating an orderer, their incentive to corrupt the ordering layer is limited — they would damage their own business relationship.  
2. The critical Byzantine threat in our model is at the **application layer** — a bank submitting a low-quality or poisoned model and trying to get it accepted. This is handled by **CBFT at the chaincode level**, not at the orderer level.  
3. Raft is significantly simpler to configure and operate than SmartBFT, and the additional complexity of BFT ordering is not justified given that the ordering function itself does not see private data.

The acknowledged trade-off: if an orderer node is compromised, transaction ordering could be manipulated. In a production deployment, BFT ordering would be the appropriate upgrade.

---

## Part 4 — Blockchain & Security

---

**Q10. How does your system detect and prevent a bank from submitting a poisoned (backdoored) model?**

**A:** We have three layers of detection:

**Layer 1 — Self-reported quality gate (chaincode):**  
`SubmitClusterUpdate` enforces `valScore ≥ ClusterValThreshold (0.7)`. A bank cannot submit a model that it self-reports as poor quality. However, this is bypassable — a malicious bank could compute and submit a fake `valScore`.

**Layer 2 — Cross-bank empirical verification (CBFT Phase 2):**  
Each peer bank independently downloads the submitted model from IPFS, recalculates the SHA-256 hash (catching any transmission tampering), and evaluates its PR-AUC on their own local validation data. If the model is poisoned (e.g., deliberately misclassifying a particular merchant category), the peer banks' divergent validation data will likely expose the low performance, triggering a `verified=False` vote.

This is the most meaningful anti-poisoning layer. A poisoned model that happens to look good on the submitting bank's validation data (because the poison is tailored to their local distribution) may still fail on other banks' validation data.

**Layer 3 — Trust score long-term accountability:**  
Even if a poisoned model passes one round of CBFT (e.g., because the poison is subtle enough to evade a single PR-AUC check), repeated poor performance across rounds will drive the bank's trust score toward `ScoreMin=0.1`, effectively marginalising their contribution to the global model.

**Known limitation:**  
We do not implement formal Byzantine robustness proofs or advanced defences such as gradient inspection, Shapley value attribution, or machine learning–based anomaly detection on submitted weights. A sophisticated attacker who understands the CBFT protocol and the validation procedure could craft a model that passes the PR-AUC threshold while still encoding a backdoor trigger. This is an acknowledged limitation and is a rich area for future work.

---

**Q11. IPFS Content Identifiers (CIDs) are derived from file content using SHA-256. You also separately store a SHA-256 hash on the blockchain. Isn't that redundant?**

**A:** The two hashes serve different security roles, so they are not redundant:

| Property | IPFS CID | On-chain `model_hash` |
|----------|----------|-----------------------|
| **Computed by** | IPFS daemon on upload | Python `hashlib.sha256` before upload |
| **Stored in** | IPFS DHT / content routing | Hyperledger Fabric world state |
| **Attested by** | IPFS network | All Fabric endorsing peers (≥2 in our setup) |
| **Role** | Content address (find the file) | Integrity witness (was the file tampered?) |

The critical case this protects against: if an IPFS node serving the content is compromised and returns modified bytes for a given CID (which should not happen with correct CID verification but could happen with a buggy or malicious IPFS gateway), the SHA-256 computed independently from the returned bytes would not match the on-chain `model_hash`. Our verification code (`compute_sha256(data) != stored_hash`) would catch this.

In practice, a correct IPFS implementation makes this scenario impossible because the CID is computed from the content itself. But the redundant hash adds a defence-in-depth layer against IPFS client bugs, caching anomalies, or future protocol weaknesses.

---

**Q12. How does your system handle a situation where two banks simultaneously submit honest but conflicting models that each pass the validation threshold?**

**A:** This is actually the expected and designed-for case, not a conflict. The CBFT protocol is not trying to choose between two competing proposals (as in a traditional BFT protocol like PBFT where only one value can be decided). Instead:

- Both BankA and BankB submit their independently trained cluster models.
- Both are independently evaluated by peer banks in Phase 2.
- Both can achieve `VerifyQuorum=2` and `CommitQuorum=2` simultaneously.
- `CheckConsensus` returns **both** bank IDs as accepted.
- The `GlobalAggregator` downloads **both** models and computes a **trust-weighted average** of all accepted models.

The "consensus" in CBFT is not "which one model wins" but rather "which models are trusted enough to contribute to the global average." This is a fundamentally different semantic than classical BFT, and it is the key design insight that makes CBFT appropriate for federated learning — multiple different-but-honest models should all contribute, weighted by their quality, rather than one being elected the winner.

---

**Q13. Your replay attack protection stores every CID that has ever been submitted. As the system runs for many rounds, doesn't this grow the ledger's world state unboundedly?**

**A:** Yes, this is a real operational concern we acknowledge. Every submitted `modelCID` is permanently recorded under a `cid~{cid}` key in the world state. Over N rounds with B banks, this creates `N×B` additional keys.

**Practical mitigation strategies** (not currently implemented, identified as future work):
1. **Round-scoped CID keys**: Storing `cid~{round}~{cid}` instead of `cid~{cid}` would allow old keys to be compositely expired. Fabric's ledger does not support native TTL, but a chaincode function `PurgeOldCIDs(olderThanRound)` could be authorised and run periodically by the consortium.
2. **Bloom filter**: A Bloom filter of previously seen CIDs could be maintained in a single world state entry, trading a small false-positive rate for O(1) space in the number of rounds.
3. **CID namespace by round**: The chaincode already encodes the round in the `ClusterUpdate` struct. Cross-round replay protection could be enforced logically — a CID submitted in round N is automatically invalid in round N+m for m>0, because the chaincode would check that the submitted round matches the current active round.

In the current prototype with 10 rounds and 3 banks, this generates 30 extra ledger entries — a negligible overhead. In a 5-year production deployment with weekly rounds (~260 rounds × 3 banks), it would create ~780 anti-replay entries, still trivially small.

---

## Part 5 — Communication Efficiency & Latency

---

**Q14. You claim your two-tier hierarchy reduces communication overhead. Can you quantify this reduction with concrete numbers?**

**A:** Let's compare directly for our setup: 3 banks × 3 branches each = 9 branch nodes.

**Flat FL (all branches → global aggregator):**
- Round: 9 local models uploaded globally
- Per model: ~10 MB
- **Total global communication per round: 9 × 10 MB = 90 MB inbound + 10 MB broadcast = 100 MB**
- Number of global aggregation operations: 1 (over 9 models)

**Our two-tier HCFL:**
- Tier 1 (intra-bank, private network): 3 × 3 models uploaded to HQ = 9 × 10 MB = **90 MB, but stays within each bank's private network** — this is zero inter-bank communication.
- Tier 2 (inter-bank, public consortium): 3 HQ models uploaded to IPFS and distributed = 3 × 10 MB = **30 MB inter-bank communication per round**
- CBFT verification: each bank downloads 2 peer models (Phase 2) = 2 × 10 MB × 3 banks = 60 MB, but this is concurrent and overlapping with Phase 1 processing.

**Net inter-bank communication reduction: 90 MB → 30 MB = 66.7% reduction in cross-organisational bandwidth.**

Additionally, each Tier-2 model is a pre-aggregated summary of 3 branches, so the global aggregation operates on 3 inputs instead of 9 — this reduces the computation at the aggregator by a factor of 3.

**Latency reduction** comes from parallelism: all three Tier-1 aggregations run concurrently (each bank processes its own branches independently), so the latency of Tier-1 is `max(T_bank1, T_bank2, T_bank3)` rather than `T_bank1 + T_bank2 + T_bank3`.

---

**Q15. The CBFT verification phase (Phase 2) requires each bank to download and evaluate every other bank's model. Doesn't this add significant latency that cancels out the communication savings?**

**A:** This is a genuine trade-off, and we are transparent about it. The CBFT Phase 2 verification does add latency:

- Each of the 3 banks downloads 2 peer models (2 × 10 MB = 20 MB per bank)
- Evaluates each on 15% of its validation data (~1,500 samples) ≈ <1 second per model on CPU

However, this cost is **paid in exchange for Byzantine fault tolerance** — without Phase 2, any bank could submit a poisoned model and it would be included in the global aggregation unchallenged.

**Latency can be partially overlapped**: Phase 2 verification can begin as soon as a bank has submitted its own Phase 1 model and uploaded it to IPFS. Banks don't need to wait for each other before starting verification — they can immediately start downloading and evaluating models that have already been submitted.

**Comparison to the alternative**: The alternative — a trusted third-party aggregator who performs all verification — would have lower latency but completely destroys the decentralisation property. Our CBFT latency overhead is the cost of operating without a trusted aggregator.

**Quantification**: In our lab experiments with CPU-only hardware, a single FL round including CBFT phases takes approximately 2–5 minutes (dominated by local training, not CBFT). The CBFT phases (model download + evaluation) add approximately 5–15 seconds — a small fraction of total round time.

---

**Q16. Why did you use IPFS instead of a more conventional CDN or object storage (AWS S3, MinIO) for off-chain model storage?**

**A:** The key distinction is **content addressing vs. location addressing**:

- AWS S3 / MinIO use location-based addressing: `s3://bucket/path/to/model.pt`. The same URL could serve different content if the file is overwritten. A bank claiming to have uploaded model version X at that URL can silently replace it after verification votes are cast.
- IPFS uses content-based addressing: the CID is a cryptographic hash of the content. The URL is the guarantee. It is computationally infeasible to produce a different file with the same CID (SHA-256 preimage resistance). Once a CID is submitted to the blockchain, the model bytes it refers to are permanently and unambiguously identified.

Additionally:
- **No central operator**: IPFS operates as a peer-to-peer network. No single bank controls the storage layer. With S3, the bank that controls the bucket controls the content.
- **Resilience**: Content pinned by multiple IPFS nodes survives any one node going offline. With S3, bucket owner controls availability.
- **Cost**: In a lab/prototype setting, IPFS runs as a local daemon at zero cost. In production, pinning services (Pinata, Infura) provide commercial IPFS availability without running private nodes.

**Acknowledged weakness**: IPFS content that is not pinned by any node can become unavailable (garbage-collected). Our current setup runs a single local IPFS node. In production, each bank should independently pin all models for the current and previous N rounds, providing redundancy.

---

## Part 6 — Resilience & Fault Tolerance

---

**Q17. Explain the model blending strategy. Why did you change the beta value from 0.7 in the prototype to 0.3 in the production system?**

**A:** The blending formula is: `w_recovered = β × w_global + (1 - β) × w_branch`

- **Prototype (4_CCFD_Model), β=0.7**: The blended model is 70% global and 30% local. This is very conservative — it heavily anchors the output to the previous global model and contributes relatively little of the local branch's learning.
- **Production (6_proposed_framework), β=0.3**: The blended model is 30% global and 70% local. This is more aggressive — it retains most of the local learning signal while using the global model as a stabilisation anchor.

**Why the change:** The prototype β=0.7 was chosen defensively, before empirical evidence about local training quality was available. In the prototype experiments, we observed that after several rounds, the intra-bank FedAvg output was already a reasonable model, and blending it with 70% of the old global model created excessive conservative dampening — the local branch's progress was being over-corrected.

The production β=0.3 reflects the insight that **the local cluster model is usually the better model** (it has seen new data the global has not), and the global model should serve as a regularisation term rather than dominating the blend. This is conceptually aligned with FedProx's proximal term, where the local model dominates but is penalised for drifting too far from the global.

The exact value of β=0.3 was chosen empirically from a small sweep in the prototype; a more rigorous hyperparameter optimisation with cross-validation would be part of future work.

---

**Q18. Your system has a minimum of 2 branch models required for aggregation. What happens if a bank consistently has only one branch available? Does trust score penalise it even for infrastructure failures?**

**A:** This is a nuanced point that reveals a weakness in the current incentive design.

**Current behaviour**: If fewer than `min_branches_required=2` branch models arrive before the deadline, the HQ does not perform intra-bank aggregation and does not submit to the blockchain. The bank's trust score is **not automatically penalised by the deadline mechanism** — the penalty is applied explicitly after global evaluation, not for failure to submit.

However, if the bank consistently fails to submit (because it can only aggregate 1 model), it will have zero accepted rounds, and other banks that do submit will gain relative trust. Over time, the absent bank's relative contribution weight diminishes, which achieves a similar effect to explicit penalisation even without a formal deduction.

**The design tension**: We want to penalise low-quality model submissions (malicious behaviour) but not penalise infrastructure failures (unavoidable hardware/network issues). These are difficult to distinguish from the outside. Our current system:
- Does NOT penalise for missing a threshold collection (infrastructure failure interpretation)
- DOES penalise (or simply exclude) for submitting a model that fails the validation gate (quality failure interpretation)

A more sophisticated system would implement a Bayesian reputation model that distinguishes patterns of hardware failure (random, uncorrelated) from patterns of strategic behaviour (suspiciously correlated with specific rounds or data patterns). This is acknowledged as future work.

---

## Part 7 — Experimental Validity & Limitations

---

**Q19. Your experimental setup simulates 3 banks × 3 branches on a single machine. To what extent can the results be generalised to a real 3-bank deployment across separate data centres?**

**A:** Our results should be interpreted as **proof-of-concept with known generalisation caveats**:

**Where results generalise:**
- **Model convergence behaviour** (PR-AUC, F1 trajectories over rounds) should generalise, since the FL algorithm, data distribution, and model architecture are the same.
- **CBFT correctness** (consensus quorum logic) is deterministic regardless of network topology.
- **Relative overhead comparisons** (two-tier vs. flat topology) generalise in terms of number of messages, though absolute latencies will differ.

**Where results may not generalise:**
- **Absolute latency**: IPFS transfers between local processes have sub-millisecond latency; real inter-bank WAN transfers over HTTPS would add 50–500ms per model download. This changes the CBFT Phase 2 overhead significantly.
- **Network failures**: Our prototype does not model packet loss, variable bandwidth, or WAN interruptions. Real deployments would need more robust retry logic and longer deadlines.
- **Fabric network performance**: A single-machine Fabric network avoids TLS round-trips between peers across different organisations. In a real multi-site deployment, endorsement latency increases substantially.
- **Data heterogeneity**: Our non-IID splits are from a single dataset partitioned by index. Real bank transaction distributions would be more heterogeneous due to geographic, demographic, and product line differences.

**Honesty about scale**: Three banks × three branches is the minimum topology that demonstrates the two-tier hierarchy and CBFT quorum properties. It does not constitute a large-scale FL study. Scaling experiments with 10+ banks and geographically distributed data are needed to validate performance claims at production scale.

---

**Q20. What would you do differently if you had another 6 months to extend this project?**

**A:** Prioritised list with rationale:

**1. Formal privacy accounting (ε-δ DP budget tracking):**  
Our current DP implementation adds noise and clips gradients but does not track the cumulative privacy budget across rounds. After T rounds, the effective ε grows roughly as O(√T). Publishing ε values per round and total would make the privacy claims formally defensible and comparable to published DP-FL literature.

**2. Rotate the global aggregator role:**  
Currently BankA always performs global aggregation, creating centralisation. Implementing a round-robin aggregator (or threshold secret sharing of the aggregation function) would eliminate this single point of both failure and trust.

**3. Real WAN deployment & latency measurement:**  
Deploy the three bank nodes on separate physical machines or cloud VMs in different regions. Measure actual CBFT round latency and compare against centralised FL and flat FL baselines on the same infrastructure. This would make the communication efficiency claims rigorous rather than theoretical.

**4. Adaptive trust score and threshold calibration:**  
The current trust score parameters (α=0.1, β=0.2, ScoreMin=0.1) and validation thresholds (τ=0.7 on-chain, τ=0.20 local) were set empirically. A principled study sweeping these parameters and their interaction with convergence rate and Byzantine robustness would significantly strengthen the work.

**5. Verifiable computation for global aggregation:**  
Implement a zero-knowledge proof or at minimum a commit-reveal scheme where the global aggregation arithmetic is verifiable on-chain, removing the trust assumption on BankA's aggregator.

---

## Part 8 — Rapid-Fire Technical Questions

---

**Q21. What is the difference between `SubmitVerification` and `SubmitCommit` in your CBFT chaincode? Why have both?**

**A:** `SubmitVerification` (Phase 2) is a bank asserting *"I have evaluated this peer's model and I believe it is high/low quality"* — it is an opinion vote. `SubmitCommit` (Phase 3) is a bank asserting *"I have seen enough verification votes (≥VerifyQuorum) and I formally commit to accepting this update into the global ledger"* — it is an execution vote. The two-phase design prevents a bank from committing before seeing quorum: `SubmitCommit` internally calls `countVerifications` and rejects the transaction if `VerifyQuorum` is not yet met. This mirrors the Prepare/Commit phases of PBFT and ensures that commits are only made when enough independent opinions exist.

---

**Q22. Why is `BCEWithLogitsLoss` used instead of `BCELoss` with a sigmoid activation?**

**A:** `BCEWithLogitsLoss` combines the sigmoid and binary cross-entropy loss into a single numerically stable operation using the log-sum-exp trick. Separately computing `sigmoid(logit)` then `BCELoss` can suffer from floating-point underflow or overflow in the sigmoid when logits are very large or small (common with class-imbalanced data where the model learns extreme logit values for the majority class). Numerical stability is especially important during DP-SGD where gradient noise can push weights to unusual magnitudes.

---

**Q23. What is "gradient client drift" in non-IID FL, and how does your system mitigate it?**

**A:** Client drift refers to the phenomenon where, under non-IID data, multiple local SGD steps cause each client's model to converge toward its local optimum rather than the global optimum. When these locally-drifted models are averaged in FedAvg, the resulting global model can oscillate or diverge from the true global minimum. Our mitigations: (1) only 1 local epoch per round (limiting the number of local steps that can cause drift), (2) global model blending in `blend_with_global` which acts as a proximal regulariser, and (3) trust-weighted aggregation which down-weights banks that consistently drift toward poor-performing local optima.

---

**Q24. Can a bank join the consortium after it has already started? How would it bootstrap its trust score?**

**A:** In the current implementation, the consortium is static — `InitLedger` bootstraps exactly `BankA`, `BankB`, `BankC` with `InitialTrustScore=1.0`. A new bank joining mid-run is not supported without a chaincode upgrade. In a production system, a governance mechanism (multi-org chaincode upgrade approval, as Fabric requires ≥majority of org admins to approve upgrades) would need to add the new bank with a provisional `InitialTrustScore=1.0`. The new bank would then build trust organically over subsequent rounds. A cold-start problem exists: a new bank gets equal initial weight as established banks, which could be exploited by a malicious entrant. A conservative production design would start new banks at `ScoreMin` and require several observed rounds before raising their score.

---

**Q25. Your `fedavg.py` explicitly copies non-float tensors from the first model rather than averaging them. What tensors are these and why is copying correct?**

**A:** PyTorch LSTM (and BatchNorm if it were used) maintains integer/long integer tracking tensors — most commonly `num_batches_tracked` in BatchNorm layers and certain internal counter tensors. These are counters for internal running statistics, not learnable parameters. Averaging counters from different models would produce a meaningless fractional value and could corrupt running mean/variance estimates. Copying from the first model is a pragmatic choice — it assumes the first model's counter state is representative enough for downstream inference. The correct approach for BatchNorm specifically is to re-compute running statistics on a representative global dataset after aggregation, but this requires a validation dataset pass outside the FedAvg step. Since our LSTM does not use BatchNorm in its main path, this is a correctness concern for future-proofing rather than a current bug.

---

*End of Q&A document.*  
*Total: 25 questions across 8 research dimensions.*
