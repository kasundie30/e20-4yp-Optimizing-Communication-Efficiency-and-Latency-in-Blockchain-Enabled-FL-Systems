
package main

import (
	"encoding/json"
	"fmt"
	"log"
	"strconv"

	"github.com/hyperledger/fabric-contract-api-go/contractapi"
)

// ---------------------------------------------------------------------------
//  Data structures stored on the ledger
// ---------------------------------------------------------------------------

// TrustScore holds the dynamic reputation score of a bank cluster.
// Score increases by α when the update improves global validation,
// and decreases by β when it degrades performance. Score ≥ ScoreMin.
type TrustScore struct {
	BankID    string  `json:"bankID"`
	Score     float64 `json:"score"`
	UpdatedAt int     `json:"updatedAt"` // round number of last update
}

// ClusterUpdate represents the intra-cluster aggregated model submitted
// by a bank's HQ after Level-1 (intra-cluster) aggregation.
type ClusterUpdate struct {
	BankID     string  `json:"bankID"`
	Round      int     `json:"round"`
	ModelCID   string  `json:"modelCID"`   // IPFS/off-chain content ID
	ModelHash  string  `json:"modelHash"`  // SHA-256 of the model file
	ValScore   float64 `json:"valScore"`   // validation metric (e.g., F1)
	Accepted   bool    `json:"accepted"`   // set by CheckConsensus
	BackupActive bool  `json:"backupActive"` // true if Backup HQ is acting
}

// VerificationVote is cast by a verifier bank in CBFT Phase 2.
type VerificationVote struct {
	VerifierID   string `json:"verifierID"`
	TargetBankID string `json:"targetBankID"`
	Round        int    `json:"round"`
	Verified     bool   `json:"verified"`
}

// CommitRecord is cast by a committer bank in CBFT Phase 3.
type CommitRecord struct {
	CommitterID  string `json:"committerID"`
	TargetBankID string `json:"targetBankID"`
	Round        int    `json:"round"`
}

// GlobalModel records the trust-weighted aggregated global model for a round.
type GlobalModel struct {
	Round      int    `json:"round"`
	GlobalCID  string `json:"globalCID"`
	GlobalHash string `json:"globalHash"`
}

// ---------------------------------------------------------------------------
//  Constants
// ---------------------------------------------------------------------------

const (
	// Initial trust score assigned to each bank at genesis
	InitialTrustScore = 1.0

	// Minimum trust score — prevents permanent exclusion
	ScoreMin = 0.1

	// Reward and penalty deltas (can be overridden via chaincode args)
	Alpha = 0.1 // reward for improving global validation
	Beta  = 0.2 // penalty for degrading global validation

	// CBFT consensus thresholds
	VerifyQuorum = 2 // number of verification votes needed
	CommitQuorum = 2 // number of commit votes needed

	// Cluster-level validation threshold τ
	ClusterValThreshold = 0.7 // e.g., F1-score ≥ 0.7 required

	// Key prefixes for composite CouchDB keys
	TrustKeyPrefix   = "trust~"
	UpdateKeyPrefix  = "update~"
	VerifyKeyPrefix  = "verify~"
	CommitKeyPrefix  = "commit~"
	GlobalKeyPrefix  = "global~"
	LatestRoundKey   = "latest_round"
)

// ---------------------------------------------------------------------------
//  SmartContract — implements contractapi.ContractInterface
// ---------------------------------------------------------------------------

// CBFTContract is the main chaincode struct.
type CBFTContract struct {
	contractapi.Contract
}

// ---------------------------------------------------------------------------
//  InitLedger — bootstrap trust scores for BankA, BankB, BankC
// ---------------------------------------------------------------------------

// InitLedger initialises the world state with a default TrustScore for each
// bank. This is called once when the chaincode is first instantiated.
func (c *CBFTContract) InitLedger(ctx contractapi.TransactionContextInterface) error {
	log.Println("[InitLedger] Bootstrapping trust scores for BankA, BankB, BankC")

	banks := []string{"BankA", "BankB", "BankC"}
	for _, bankID := range banks {
		ts := TrustScore{
			BankID:    bankID,
			Score:     InitialTrustScore,
			UpdatedAt: 0,
		}
		tsBytes, err := json.Marshal(ts)
		if err != nil {
			return fmt.Errorf("InitLedger: failed to marshal TrustScore for %s: %w", bankID, err)
		}
		key := TrustKeyPrefix + bankID
		if err := ctx.GetStub().PutState(key, tsBytes); err != nil {
			return fmt.Errorf("InitLedger: PutState failed for %s: %w", bankID, err)
		}
		log.Printf("[InitLedger] Trust score initialised for %s = %.2f\n", bankID, InitialTrustScore)
	}
	return nil
}

// ---------------------------------------------------------------------------
//  SubmitClusterUpdate — Level-1 HQ submits intra-cluster aggregated model
// ---------------------------------------------------------------------------

// SubmitClusterUpdate is called by an HQ peer after completing intra-cluster
// (Level-1) FedAvg aggregation.  The model is identified by an off-chain CID
// (e.g., IPFS hash) and a SHA-256 hash for integrity verification.
//
// The submission is accepted on ledger only if valScore ≥ ClusterValThreshold.
// If the current HQ has failed and the Backup HQ is acting, pass backupActive=true.
//
// Parameters:
//   bankID     — e.g., "BankA"
//   round      — current FL training round (string → int)
//   modelCID   — IPFS content identifier of the aggregated cluster model
//   modelHash  — SHA-256 hex digest of the model file
//   valScore   — validation metric (AUC-ROC, F1) achieved on validation set
func (c *CBFTContract) SubmitClusterUpdate(
	ctx contractapi.TransactionContextInterface,
	bankID, roundStr, modelCID, modelHash, valScoreStr string,
) error {
	log.Printf("[SubmitClusterUpdate] bankID=%s round=%s valScore=%s\n", bankID, roundStr, valScoreStr)

	round, err := strconv.Atoi(roundStr)
	if err != nil {
		return fmt.Errorf("SubmitClusterUpdate: invalid round '%s': %w", roundStr, err)
	}
	valScore, err := strconv.ParseFloat(valScoreStr, 64)
	if err != nil {
		return fmt.Errorf("SubmitClusterUpdate: invalid valScore '%s': %w", valScoreStr, err)
	}

	// Enforce cluster-level validation threshold τ
	if valScore < ClusterValThreshold {
		return fmt.Errorf("SubmitClusterUpdate: valScore %.4f below threshold %.4f — submission rejected", valScore, ClusterValThreshold)
	}

	// Replay Attack Protection: Ensure modelCID has not been submitted before
	cidKey := "cid~" + modelCID
	cidBytes, err := ctx.GetStub().GetState(cidKey)
	if err != nil {
		return fmt.Errorf("SubmitClusterUpdate: GetState error checking CID: %w", err)
	}
	if cidBytes != nil {
		return fmt.Errorf("SubmitClusterUpdate: replay attack detected, modelCID %s already exists", modelCID)
	}
	// Mark CID as used (e.g., store bankID and round)
	if err := ctx.GetStub().PutState(cidKey, []byte(fmt.Sprintf("%s~%d", bankID, round))); err != nil {
		return fmt.Errorf("SubmitClusterUpdate: PutState error for CID: %w", err)
	}

	update := ClusterUpdate{
		BankID:    bankID,
		Round:     round,
		ModelCID:  modelCID,
		ModelHash: modelHash,
		ValScore:  valScore,
		Accepted:  false, // will be set after CBFT consensus
	}
	updateBytes, err := json.Marshal(update)
	if err != nil {
		return fmt.Errorf("SubmitClusterUpdate: marshal error: %w", err)
	}
	key := fmt.Sprintf("%s%s~%d", UpdateKeyPrefix, bankID, round)
	return ctx.GetStub().PutState(key, updateBytes)
}

// ---------------------------------------------------------------------------
//  SubmitVerification — CBFT Phase 2: peer verification vote
// ---------------------------------------------------------------------------

// SubmitVerification allows a verifier bank (HQ) to cast a verification vote
// on another bank's cluster update. In the CBFT protocol this is the
// "Prepare" phase — N-of-N or quorum-of-N verifiers must approve before commit.
//
// Parameters:
//   verifierID   — bank casting the vote, e.g., "BankB"
//   targetBankID — bank whose update is being verified, e.g., "BankA"
//   round        — FL training round
//   verified     — "true" / "false"
func (c *CBFTContract) SubmitVerification(
	ctx contractapi.TransactionContextInterface,
	verifierID, targetBankID, roundStr, verifiedStr string,
) error {
	log.Printf("[SubmitVerification] verifier=%s target=%s round=%s verified=%s\n",
		verifierID, targetBankID, roundStr, verifiedStr)

	round, err := strconv.Atoi(roundStr)
	if err != nil {
		return fmt.Errorf("SubmitVerification: invalid round: %w", err)
	}
	verified, err := strconv.ParseBool(verifiedStr)
	if err != nil {
		return fmt.Errorf("SubmitVerification: invalid verified flag: %w", err)
	}

	vote := VerificationVote{
		VerifierID:   verifierID,
		TargetBankID: targetBankID,
		Round:        round,
		Verified:     verified,
	}
	voteBytes, err := json.Marshal(vote)
	if err != nil {
		return fmt.Errorf("SubmitVerification: marshal error: %w", err)
	}
	// Composite key: verify~<targetBankID>~<round>~<verifierID>
	key := fmt.Sprintf("%s%s~%d~%s", VerifyKeyPrefix, targetBankID, round, verifierID)
	return ctx.GetStub().PutState(key, voteBytes)
}

// ---------------------------------------------------------------------------
//  SubmitCommit — CBFT Phase 3: commit confirmation
// ---------------------------------------------------------------------------

// SubmitCommit represents the final "Commit" phase of CBFT.
// A bank commits to accepting the target bank's update after VerifyQuorum
// positive verifications have been observed on the ledger.
//
// Parameters:
//   committerID  — e.g., "BankC"
//   targetBankID — bank whose model is being committed
//   round        — FL training round
func (c *CBFTContract) SubmitCommit(
	ctx contractapi.TransactionContextInterface,
	committerID, targetBankID, roundStr string,
) error {
	log.Printf("[SubmitCommit] committer=%s target=%s round=%s\n", committerID, targetBankID, roundStr)

	round, err := strconv.Atoi(roundStr)
	if err != nil {
		return fmt.Errorf("SubmitCommit: invalid round: %w", err)
	}

	// Verify that enough verification votes exist before accepting commit
	verifyOK, err := c.countVerifications(ctx, targetBankID, round)
	if err != nil {
		return fmt.Errorf("SubmitCommit: error counting verifications: %w", err)
	}
	if verifyOK < VerifyQuorum {
		return fmt.Errorf("SubmitCommit: insufficient verifications (%d/%d) for %s round %d",
			verifyOK, VerifyQuorum, targetBankID, round)
	}

	record := CommitRecord{
		CommitterID:  committerID,
		TargetBankID: targetBankID,
		Round:        round,
	}
	recBytes, err := json.Marshal(record)
	if err != nil {
		return fmt.Errorf("SubmitCommit: marshal error: %w", err)
	}
	key := fmt.Sprintf("%s%s~%d~%s", CommitKeyPrefix, targetBankID, round, committerID)
	return ctx.GetStub().PutState(key, recBytes)
}

// countVerifications returns the number of positive verification votes
// recorded for a given (targetBankID, round) pair.
func (c *CBFTContract) countVerifications(
	ctx contractapi.TransactionContextInterface,
	targetBankID string, round int,
) (int, error) {
	prefix := fmt.Sprintf("%s%s~%d~", VerifyKeyPrefix, targetBankID, round)
	iter, err := ctx.GetStub().GetStateByRange(prefix, prefix+"~")
	if err != nil {
		return 0, err
	}
	defer iter.Close()

	count := 0
	for iter.HasNext() {
		kv, err := iter.Next()
		if err != nil {
			return 0, err
		}
		var vote VerificationVote
		if err := json.Unmarshal(kv.Value, &vote); err != nil {
			continue
		}
		if vote.Verified {
			count++
		}
	}
	return count, nil
}

// ---------------------------------------------------------------------------
//  CheckConsensus — returns list of bankIDs accepted this round
// ---------------------------------------------------------------------------

// CheckConsensus evaluates all cluster updates for the given round.
// A bank's update is considered accepted (CBFT-committed) when:
//   1. It has been submitted (SubmitClusterUpdate) and valScore ≥ τ
//   2. It received ≥ VerifyQuorum positive verifications
//   3. It received ≥ CommitQuorum commit records
//
// Returns a JSON array of accepted bankIDs, e.g., ["BankA","BankC"]
func (c *CBFTContract) CheckConsensus(
	ctx contractapi.TransactionContextInterface,
	roundStr string,
) (string, error) {
	round, err := strconv.Atoi(roundStr)
	if err != nil {
		return "", fmt.Errorf("CheckConsensus: invalid round: %w", err)
	}

	banks := []string{"BankA", "BankB", "BankC"}
	accepted := []string{}

	for _, bankID := range banks {
		// Check update exists
		updateKey := fmt.Sprintf("%s%s~%d", UpdateKeyPrefix, bankID, round)
		updateBytes, err := ctx.GetStub().GetState(updateKey)
		if err != nil || updateBytes == nil {
			continue
		}

		// Count verifications
		vCount, _ := c.countVerifications(ctx, bankID, round)
		if vCount < VerifyQuorum {
			continue
		}

		// Count commits
		cCount, _ := c.countCommits(ctx, bankID, round)
		if cCount < CommitQuorum {
			continue
		}

		// Mark update as accepted on ledger
		var update ClusterUpdate
		if err := json.Unmarshal(updateBytes, &update); err == nil {
			update.Accepted = true
			if b, err := json.Marshal(update); err == nil {
				_ = ctx.GetStub().PutState(updateKey, b)
			}
		}
		accepted = append(accepted, bankID)
	}

	result, err := json.Marshal(accepted)
	if err != nil {
		return "", fmt.Errorf("CheckConsensus: marshal error: %w", err)
	}
	log.Printf("[CheckConsensus] round=%d accepted=%s\n", round, string(result))
	return string(result), nil
}

// countCommits returns the number of commit records for (targetBankID, round).
func (c *CBFTContract) countCommits(
	ctx contractapi.TransactionContextInterface,
	targetBankID string, round int,
) (int, error) {
	prefix := fmt.Sprintf("%s%s~%d~", CommitKeyPrefix, targetBankID, round)
	iter, err := ctx.GetStub().GetStateByRange(prefix, prefix+"~")
	if err != nil {
		return 0, err
	}
	defer iter.Close()
	count := 0
	for iter.HasNext() {
		if _, err := iter.Next(); err == nil {
			count++
		}
	}
	return count, nil
}

// ---------------------------------------------------------------------------
//  StoreGlobalModel — record trust-weighted aggregated global model
// ---------------------------------------------------------------------------

// StoreGlobalModel persists the CID and hash of the trust-weighted global
// model produced by Level-2 (inter-cluster) aggregation.
//
// Parameters:
//   round      — FL training round
//   globalCID  — IPFS content identifier of the global model
//   globalHash — SHA-256 hex digest of the global model file
func (c *CBFTContract) StoreGlobalModel(
	ctx contractapi.TransactionContextInterface,
	roundStr, globalCID, globalHash string,
) error {
	round, err := strconv.Atoi(roundStr)
	if err != nil {
		return fmt.Errorf("StoreGlobalModel: invalid round: %w", err)
	}

	gm := GlobalModel{
		Round:      round,
		GlobalCID:  globalCID,
		GlobalHash: globalHash,
	}
	gmBytes, err := json.Marshal(gm)
	if err != nil {
		return fmt.Errorf("StoreGlobalModel: marshal error: %w", err)
	}
	key := fmt.Sprintf("%s%d", GlobalKeyPrefix, round)
	if err := ctx.GetStub().PutState(key, gmBytes); err != nil {
		return fmt.Errorf("StoreGlobalModel: PutState error: %w", err)
	}

	// Update latest round pointer if this is the highest round seen
	latestBytes, _ := ctx.GetStub().GetState(LatestRoundKey)
	currentLatest := 0
	if latestBytes != nil {
		currentLatest, _ = strconv.Atoi(string(latestBytes))
	}
	if round > currentLatest {
		_ = ctx.GetStub().PutState(LatestRoundKey, []byte(strconv.Itoa(round)))
	}

	log.Printf("[StoreGlobalModel] Stored global model for round %d CID=%s\n", round, globalCID)
	return nil
}

// ---------------------------------------------------------------------------
//  GetLatestRound — return the highest completed round number
// ---------------------------------------------------------------------------

func (c *CBFTContract) GetLatestRound(ctx contractapi.TransactionContextInterface) (int, error) {
	latestBytes, err := ctx.GetStub().GetState(LatestRoundKey)
	if err != nil {
		return 0, fmt.Errorf("GetLatestRound: GetState error: %w", err)
	}
	if latestBytes == nil {
		return 0, nil
	}
	round, _ := strconv.Atoi(string(latestBytes))
	return round, nil
}

// ---------------------------------------------------------------------------
//  GetGlobalModel — fetch the stored global model for a round
// ---------------------------------------------------------------------------

// GetGlobalModel retrieves the GlobalModel record for the specified round.
// Returns JSON string of the GlobalModel struct.
func (c *CBFTContract) GetGlobalModel(
	ctx contractapi.TransactionContextInterface,
	roundStr string,
) (string, error) {
	round, err := strconv.Atoi(roundStr)
	if err != nil {
		return "", fmt.Errorf("GetGlobalModel: invalid round: %w", err)
	}

	key := fmt.Sprintf("%s%d", GlobalKeyPrefix, round)
	gmBytes, err := ctx.GetStub().GetState(key)
	if err != nil {
		return "", fmt.Errorf("GetGlobalModel: GetState error: %w", err)
	}
	if gmBytes == nil {
		return "", fmt.Errorf("GetGlobalModel: no model found for round %d", round)
	}
	log.Printf("[GetGlobalModel] round=%d\n", round)
	return string(gmBytes), nil
}

// ---------------------------------------------------------------------------
//  GetClusterUpdate — fetch a specific cluster update
// ---------------------------------------------------------------------------

// GetClusterUpdate retrieves the ClusterUpdate for a specific bank and round.
func (c *CBFTContract) GetClusterUpdate(
	ctx contractapi.TransactionContextInterface,
	bankID, roundStr string,
) (string, error) {
	round, err := strconv.Atoi(roundStr)
	if err != nil {
		return "", fmt.Errorf("GetClusterUpdate: invalid round: %w", err)
	}

	key := fmt.Sprintf("%s%s~%d", UpdateKeyPrefix, bankID, round)
	updateBytes, err := ctx.GetStub().GetState(key)
	if err != nil {
		return "", fmt.Errorf("GetClusterUpdate: GetState error: %w", err)
	}
	if updateBytes == nil {
		return "", fmt.Errorf("GetClusterUpdate: no update found for %s round %d", bankID, round)
	}
	return string(updateBytes), nil
}

// ---------------------------------------------------------------------------
//  CheckVerifyQuorum — returns true if a bank has enough verification votes
// ---------------------------------------------------------------------------

// CheckVerifyQuorum checks if a target bank has >= VerifyQuorum votes for a round.
func (c *CBFTContract) CheckVerifyQuorum(
	ctx contractapi.TransactionContextInterface,
	targetBankID, roundStr string,
) (bool, error) {
	round, err := strconv.Atoi(roundStr)
	if err != nil {
		return false, fmt.Errorf("CheckVerifyQuorum: invalid round: %w", err)
	}

	count, err := c.countVerifications(ctx, targetBankID, round)
	if err != nil {
		return false, fmt.Errorf("CheckVerifyQuorum: %w", err)
	}
	return count >= VerifyQuorum, nil
}

// ---------------------------------------------------------------------------
//  UpdateTrustScore — reward or penalise a cluster after global validation
// ---------------------------------------------------------------------------

// UpdateTrustScore adjusts the trust score of a bank cluster based on
// whether its update improved or degraded the global model performance.
//
// Positive delta (α) rewards; negative delta (-β) penalises.
// Score is clamped to ScoreMin to prevent permanent exclusion.
//
// Parameters:
//   bankID — e.g., "BankA"
//   delta  — signed float, e.g., "+0.1" or "-0.2"
func (c *CBFTContract) UpdateTrustScore(
	ctx contractapi.TransactionContextInterface,
	bankID, deltaStr string,
) error {
	delta, err := strconv.ParseFloat(deltaStr, 64)
	if err != nil {
		return fmt.Errorf("UpdateTrustScore: invalid delta '%s': %w", deltaStr, err)
	}

	key := TrustKeyPrefix + bankID

	// Read existing score
	tsBytes, err := ctx.GetStub().GetState(key)
	if err != nil || tsBytes == nil {
		return fmt.Errorf("UpdateTrustScore: trust score not found for %s: %w", bankID, err)
	}
	var ts TrustScore
	if err := json.Unmarshal(tsBytes, &ts); err != nil {
		return fmt.Errorf("UpdateTrustScore: unmarshal error: %w", err)
	}

	// Apply delta and clamp to ScoreMin
	ts.Score += delta
	if ts.Score < ScoreMin {
		ts.Score = ScoreMin
		log.Printf("[UpdateTrustScore] %s score clamped to minimum %.2f\n", bankID, ScoreMin)
	}

	updatedBytes, err := json.Marshal(ts)
	if err != nil {
		return fmt.Errorf("UpdateTrustScore: marshal error: %w", err)
	}
	log.Printf("[UpdateTrustScore] %s delta=%.4f newScore=%.4f\n", bankID, delta, ts.Score)
	return ctx.GetStub().PutState(key, updatedBytes)
}

// ---------------------------------------------------------------------------
//  GetTrustScores — return all current trust scores
// ---------------------------------------------------------------------------

// GetTrustScores returns a JSON map of bankID → trust score for all banks.
// Used by the Level-2 aggregator to compute trust-weighted global model:
//   w_global = Σ (S_b / Σ S_k) * w_b
func (c *CBFTContract) GetTrustScores(
	ctx contractapi.TransactionContextInterface,
) (string, error) {
	banks := []string{"BankA", "BankB", "BankC"}
	scores := make(map[string]float64)

	for _, bankID := range banks {
		key := TrustKeyPrefix + bankID
		tsBytes, err := ctx.GetStub().GetState(key)
		if err != nil || tsBytes == nil {
			scores[bankID] = 0.0
			continue
		}
		var ts TrustScore
		if err := json.Unmarshal(tsBytes, &ts); err != nil {
			scores[bankID] = 0.0
			continue
		}
		scores[bankID] = ts.Score
	}

	result, err := json.Marshal(scores)
	if err != nil {
		return "", fmt.Errorf("GetTrustScores: marshal error: %w", err)
	}
	log.Printf("[GetTrustScores] %s\n", string(result))
	return string(result), nil
}

// ---------------------------------------------------------------------------
//  ActivateBackup — failover from HQ to Backup HQ for a given bank
// ---------------------------------------------------------------------------

// ActivateBackup is invoked when a bank's HQ (peer0) is unavailable.
// It records the failover event on ledger so that other participants know the
// Backup HQ (peer1) is now acting as the committee representative.
//
// The subsequent SubmitClusterUpdate call should set backupActive=true.
//
// Parameters:
//   bankID — e.g., "BankA"
//   round  — FL training round when the failover occurs
func (c *CBFTContract) ActivateBackup(
	ctx contractapi.TransactionContextInterface,
	bankID, roundStr string,
) error {
	round, err := strconv.Atoi(roundStr)
	if err != nil {
		return fmt.Errorf("ActivateBackup: invalid round: %w", err)
	}

	// Read existing cluster update if any and flag backup as active
	updateKey := fmt.Sprintf("%s%s~%d", UpdateKeyPrefix, bankID, round)
	updateBytes, err := ctx.GetStub().GetState(updateKey)

	var update ClusterUpdate
	if err == nil && updateBytes != nil {
		if err2 := json.Unmarshal(updateBytes, &update); err2 == nil {
			update.BackupActive = true
		}
	} else {
		// No update yet — create a placeholder so failover is recorded
		update = ClusterUpdate{
			BankID:       bankID,
			Round:        round,
			BackupActive: true,
		}
	}

	b, err := json.Marshal(update)
	if err != nil {
		return fmt.Errorf("ActivateBackup: marshal error: %w", err)
	}
	log.Printf("[ActivateBackup] Backup HQ activated for %s round %d\n", bankID, round)
	return ctx.GetStub().PutState(updateKey, b)
}

// ---------------------------------------------------------------------------
//  main — register the chaincode
// ---------------------------------------------------------------------------

func main() {
	cc, err := contractapi.NewChaincode(&CBFTContract{})
	if err != nil {
		log.Panicf("Error creating CBFT chaincode: %v", err)
	}
	if err := cc.Start(); err != nil {
		log.Panicf("Error starting CBFT chaincode: %v", err)
	}
}
