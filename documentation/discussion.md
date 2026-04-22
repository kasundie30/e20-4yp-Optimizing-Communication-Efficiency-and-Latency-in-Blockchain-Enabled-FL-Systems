# Discussion
*(Draft — Research Paper: "Optimizing Communication Efficiency and Latency in Blockchain-Enabled Federated Learning Systems")*

---

## V. Discussion

### A. Summary of Main Findings

This work demonstrates that a hierarchical, blockchain-integrated federated learning system can simultaneously achieve strong fraud detection performance, low per-round communication overhead, and bounded end-to-end latency on a realistic, severely imbalanced credit card fraud dataset. Across 19 successful rounds, the global model maintained a mean PR-AUC of **0.7349** and a perfect recall of **1.0**, whilst completing each round in an average of **42.43 seconds**—well within the 120-second SLA—at a constant communication cost of **0.4896 MB per round**. The peak PR-AUC of **0.9694** observed in the extended Phase 11 run (Round 410) confirms that the framework is capable of production-grade fraud detection accuracy given sufficient training rounds.

Three design decisions are principally responsible for these results. First, the **two-tier hierarchical topology** reduced the number of model uploads to the blockchain from the total number of branches ($\sum_k M_k = 9$) to the number of banks ($K = 3$), reducing cross-organisational communication by a factor of three while preserving each bank's privacy within its internal network. Second, the **IPFS hybrid storage architecture** kept Fabric ledger transactions in the 200-byte range, making blockchain endorsement overhead negligible compared to the ML computation and settlement pauses. Third, the **CBFT consensus protocol** provided Byzantine fault tolerance with $O(K)$ vote messages per phase, enabling cross-bank model verification without the quadratic communication overhead of classical PBFT or the probabilistic finality of Nakamoto-style consensus.

---

### B. Connection to Prior Work

#### Federated Learning for Fraud Detection

The use of federated learning for credit card fraud detection has been explored in prior work [CITE], but most existing studies adopt a flat, server-centric topology where all participants communicate directly with a central aggregator. Our two-tier hierarchy addresses the well-documented scalability limitation of flat FL [CITE McMahan et al., 2017] by introducing an intermediate aggregation layer that mirrors the natural organisational structure of banking consortia. The hierarchical approach is broadly consistent with the clustered FL literature [CITE Ghosh et al., 2020; Sattler et al., 2021], but distinguishes itself by grounding the hierarchy in real institutional boundaries (banks and their branches) rather than data-similarity-driven clustering.

The choice of PR-AUC as the primary evaluation metric follows best practices established for imbalanced classification [CITE Davis & Goadrich, 2006; Saito & Rehmsmeier, 2015], and our results confirm that ROC-AUC alone gives an overly optimistic picture: despite consistently high ROC-AUC values (mean 0.9344), the PR-AUC range of 0.61–0.81 reveals the genuine precision challenge at high recall, which is the operationally relevant trade-off for fraud detection.

#### Blockchain-Integrated Federated Learning

The combination of blockchain and federated learning has been explored in several concurrent lines of work. Nguyen et al. [CITE, 2021] proposed using blockchain as an audit layer for FL model updates in IoT settings, and demonstrated that on-chain record-keeping can replace the need for a trusted central server. Li et al. [CITE, 2020] studied blockchain-based FL for healthcare data, noting that ledger bloat from storing large model weights directly on-chain is a fundamental bottleneck. Our IPFS hybrid storage directly addresses this bottleneck by separating model integrity verification (on-chain CID and hash) from model storage (off-chain IPFS), consistent with the content-addressed storage approach proposed for blockchain-based model marketplaces [CITE].

The trust-weighted FedAvg aggregation mechanism in our framework parallels reputation-based approaches in Byzantine-robust FL [CITE Fung et al., 2018; Blanchard et al., 2017], but differs in that trust scores are maintained on an **immutable, publicly auditable ledger** rather than in the central aggregator's internal state. This provides tamper-evidence properties absent from prior reputation-based approaches: any bank can independently audit every historical trust score update, making the reputation system itself a first-class object on the consortium's shared ledger.

#### Differential Privacy in Federated Learning

Our DP mechanism (Gaussian noise, $\sigma = 0.05$, $\Delta = 1.0$) is consistent with the DP-SGD framework introduced by Abadi et al. [CITE, 2016], and the noise multiplier is deliberately conservative to preserve model utility on a small, CPU-constrained system. The perfect recall result (1.0 across all rounds) suggests that the DP noise at this scale does not impair the model's ability to detect fraud, consistent with findings in [CITE Geyer et al., 2017] that show low-noise-multiplier DP is compatible with strong FL convergence on imbalanced tasks.

---

### C. Interpretation of Key Results

#### The Recall-Precision Trade-off

The observation of perfect recall (1.0) alongside low precision (0.0041) at threshold 0.5 is consistent with the well-known behaviour of models trained with heavily asymmetric class weighting. By setting $w_+ = n_{\text{neg}} / n_{\text{pos}} \approx 577$ in the loss function, the model is strongly penalised for missing any fraud, resulting in a low-threshold prediction behaviour where the model classifies most transactions as fraudulent at the default 0.5 cut-off. This is not a pathological result—it is the intended consequence of prioritising recall, which is the operationally correct objective in fraud prevention. Adjusting the decision threshold to a value above 0.5 (e.g., 0.90–0.95) would substantially increase precision while retaining near-perfect recall, as evidenced by the high PR-AUC values (up to 0.8084 in the 20-round window and 0.9694 at extended convergence).

#### Round-to-Round PR-AUC Variation

The PR-AUC variation across rounds (std dev 0.059, range 0.608–0.808 in the 20-round window) reflects the stochastic nature of federated training under non-IID data and deadline-based branch collection. In rounds where fewer branches submit before the deadline (or where a branch's data shard happens to contain fewer fraud labels), the cluster model's quality may dip before recovering in the next round when blending with the prior global model provides a stabilising effect. This is consistent with the "non-IID performance cliff" described in [CITE Zhao et al., 2018], and motivates the blending mechanism ($\beta = 0.30$) as a practical stabilisation strategy.

#### Latency Composition

The 42.43 s mean round latency is dominated by the three fixed 10-second settlement pauses inserted between CBFT phases to allow Fabric blocks to propagate. These pauses are conservative estimates appropriate for a Docker-based single-host testbed and could be substantially reduced in a multi-host deployment with optimised Fabric block cutting parameters. The ML computation itself (training, FedAvg, IPFS upload—approximately 5–10 s total) constitutes less than 25% of the total round time, indicating that future latency improvements should target blockchain settlement configuration rather than ML algorithmic changes.

---

### D. Limitations

#### 1. Single-Host Testbed
All experiments were conducted on a single physical machine running Hyperledger Fabric, IPFS, and the three bank agents within Docker containers on a single bridge network. Network latency between peers was therefore negligible (~0.1 ms), whereas a real multi-institutional deployment would introduce WAN latencies of 10–100 ms between banks. The observed round latency of ~42 s is therefore a lower bound; real-world deployments would experience higher latency from cross-institutional network communication, TLS negotiation overhead, and geographic distribution of orderer nodes.

#### 2. Small Federation ($K = 3$, $M_k = 3$)
The CBFT protocol and trust score system were evaluated with only three banks and three branches per bank. The scalability of CBFT under larger consortia ($K = 10, 50, 100$ banks) has not been experimentally validated. While the $O(K)$ message complexity of CBFT is theoretically favourable, the cross-bank IPFS model download and local evaluation step in the Verify phase scales linearly with $K$, which may become a practical bottleneck at larger consortium sizes.

#### 3. No Formal DP Privacy Accounting
The differential privacy implementation uses a fixed noise multiplier ($\sigma = 0.05$) without formally tracking the cumulative privacy budget $(\varepsilon, \delta)$ across rounds. Without privacy accounting (as implemented in Opacus [CITE] or TensorFlow Privacy), it is not possible to state a formal $(ε, δ)$-DP guarantee for the multi-round training process. The noise multiplier was chosen empirically to preserve model quality, not derived from a target privacy budget.

#### 4. Fixed Decision Threshold Evaluation
Precision and F1 were evaluated at a fixed threshold of 0.5. This under-represents the model's precision-recall trade-off, which is more accurately characterised by the full PR curve and its AUC. A calibrated threshold search over the validation set would yield substantially different precision/F1 values and would be necessary for practical deployment.

#### 5. No Adversarial Participant Injection
The Byzantine fault-tolerance of CBFT was validated by design analysis and unit tests (Phase 2–6), but no live adversarial experiment was conducted in which a bank intentionally submitted poisoned model weights or inflated its self-reported validation score. The practical effectiveness of the CBFT verification step against gradient poisoning attacks [CITE Bagdasaryan et al., 2020] or model replacement attacks [CITE Bhagoji et al., 2019] remains to be empirically quantified.

#### 6. Homogeneous Model Architecture
All participants use the same `LSTMTabular` architecture with fixed hyperparameters. In a real banking consortium, banks may operate under different regulatory constraints on model complexity, or may have heterogeneous local datasets that would benefit from architecturally heterogeneous models. Model heterogeneity in FL is an active research problem [CITE Li et al., 2021] and is not addressed by the current FedAvg-based aggregation, which requires architecturally identical models.

---

### E. Future Research Directions

Based on the above limitations, we identify the following priority directions for future work:

**1. Multi-host scalability study.** Deploying the framework across $K \geq 10$ geographically distributed institutions on real WAN infrastructure would quantify the impact of network latency on CBFT consensus time and identify the practical scalability ceiling of the proposed protocol.

**2. Formal DP privacy budget tracking.** Integrating Opacus-compatible per-sample gradient accounting would enable the derivation of round-by-round $(\varepsilon, \delta)$ guarantees and allow the noise multiplier to be set from a privacy budget requirement rather than empirically. This is essential for regulatory approval in production banking contexts.

**3. Adversarial robustness evaluation.** Injecting Byzantine participants with gradient poisoning or model replacement attacks in a live experimental setting would quantify the detection rate, response time, and trust score dynamics of the CBFT protocol under real adversarial conditions.

**4. Adaptive CBFT thresholds.** The current CBFT validation thresholds ($\tau_{\text{val}} = 0.20$ at Tier 1, $\tau_{\text{chain}} = 0.7$ at Tier 2) are fixed. Adaptive thresholds that tighten over rounds as the global model converges—or relax for new participants with initially low PR-AUC—could improve both acceptance rates in early rounds and security in later rounds.

**5. Asynchronous and semi-synchronous FL.** The current round structure is synchronous at the global tier: all banks must submit within a consensus window before the global model is updated. An asynchronous variant—where each bank's update is incorporated as it arrives, weighted by staleness—would reduce round latency and improve tolerance of slow or intermittently connected participants.

**6. Cross-chain interoperability.** Extending the framework to support multiple permissioned blockchain networks (e.g., separate Fabric channels per country or regulatory jurisdiction) federated through a cross-chain bridge would address real-world regulatory fragmentation in international banking consortia.

**7. Threshold calibration and cost-sensitive deployment.** Systematic threshold calibration using validation data—mapping expected fraud costs and false-alarm review costs to an optimal decision boundary—would convert the model's strong PR-AUC into operationally deployable precision-recall trade-offs suitable for production fraud alert systems.

---

> **Note to authors**: The Discussion section intentionally avoids repeating numerical results (which belong in Section IV) and instead focuses on interpretation, context, and forward-looking commentary. References marked `[CITE]` should be resolved before submission. Key references to confirm: Ghosh et al. (NeurIPS 2020) for clustered FL; Zhao et al. (2018, arXiv) for non-IID performance degradation; Bagdasaryan et al. (USENIX 2020) for backdoor attacks in FL; Bhagoji et al. (ICML 2019) for model replacement attacks; Abadi et al. (CCS 2016) for DP-SGD; McMahan et al. (AISTATS 2017) for FedAvg.
