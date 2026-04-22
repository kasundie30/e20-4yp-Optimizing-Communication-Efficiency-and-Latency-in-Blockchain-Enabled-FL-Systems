package main

import (
	"encoding/json"
	"testing"

	"github.com/hyperledger/fabric-chaincode-go/pkg/cid"
	"github.com/hyperledger/fabric-chaincode-go/shim"
	"github.com/hyperledger/fabric-chaincode-go/shimtest"
	"github.com/hyperledger/fabric-contract-api-go/contractapi"
	"github.com/stretchr/testify/assert"
)

// We need a mock implementation of contractapi.TransactionContextInterface
type MockTransactionContext struct {
	stub shim.ChaincodeStubInterface
}

func (c *MockTransactionContext) GetStub() shim.ChaincodeStubInterface {
	return c.stub
}

func (c *MockTransactionContext) GetClientIdentity() cid.ClientIdentity {
	return nil
}

// Helper to create a new mock stub
func getMockContext() (*MockTransactionContext, *shimtest.MockStub) {
	cc, _ := contractapi.NewChaincode(new(CBFTContract))
	stub := shimtest.NewMockStub("mockStub", cc)
	ctx := &MockTransactionContext{stub: stub}
	return ctx, stub
}

func TestInitLedger(t *testing.T) {
	contract := new(CBFTContract)
	ctx, stub := getMockContext()
	stub.MockTransactionStart("tx1")
	defer stub.MockTransactionEnd("tx1")

	// Initialise
	err := contract.InitLedger(ctx)
	assert.NoError(t, err)

	// Verify all 3 banks have trust score 1.0
	for _, bank := range []string{"BankA", "BankB", "BankC"} {
		bytes, err := stub.GetState(TrustKeyPrefix + bank)
		assert.NoError(t, err)
		assert.NotNil(t, bytes)

		var ts TrustScore
		err = json.Unmarshal(bytes, &ts)
		assert.NoError(t, err)
		assert.Equal(t, 1.0, ts.Score)
		assert.Equal(t, bank, ts.BankID)
	}
}

func TestSubmitClusterUpdate(t *testing.T) {
	contract := new(CBFTContract)
	ctx, stub := getMockContext()
	stub.MockTransactionStart("tx1")
	defer stub.MockTransactionEnd("tx1")

	// Valid submission
	err := contract.SubmitClusterUpdate(ctx, "BankA", "1", "QmTestCID", "testXHash", "0.85")
	assert.NoError(t, err)

	bytes, err := stub.GetState(UpdateKeyPrefix + "BankA~1")
	assert.NoError(t, err)
	assert.NotNil(t, bytes)

	var update ClusterUpdate
	err = json.Unmarshal(bytes, &update)
	assert.NoError(t, err)
	assert.Equal(t, "BankA", update.BankID)
	assert.Equal(t, 1, update.Round)
	assert.Equal(t, "QmTestCID", update.ModelCID)
	assert.Equal(t, 0.85, update.ValScore)

	// Below threshold submission
	err = contract.SubmitClusterUpdate(ctx, "BankB", "1", "QmTestCID2", "testXHash2", "0.65")
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "below threshold")
}

func TestSubmitVerification(t *testing.T) {
	contract := new(CBFTContract)
	ctx, stub := getMockContext()
	stub.MockTransactionStart("tx1")
	defer stub.MockTransactionEnd("tx1")

	err := contract.SubmitVerification(ctx, "BankB", "BankA", "1", "true")
	assert.NoError(t, err)

	bytes, err := stub.GetState(VerifyKeyPrefix + "BankA~1~BankB")
	assert.NoError(t, err)
	assert.NotNil(t, bytes)

	var vote VerificationVote
	err = json.Unmarshal(bytes, &vote)
	assert.NoError(t, err)
	assert.Equal(t, "BankB", vote.VerifierID)
	assert.Equal(t, "BankA", vote.TargetBankID)
	assert.Equal(t, 1, vote.Round)
	assert.True(t, vote.Verified)
}

func TestSubmitCommit(t *testing.T) {
	contract := new(CBFTContract)
	ctx, stub := getMockContext()
	stub.MockTransactionStart("tx1")
	defer stub.MockTransactionEnd("tx1")

	// No verifications yet — should fail
	err := contract.SubmitCommit(ctx, "BankC", "BankA", "1")
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "insufficient verifications")

	// Add 2 verifications (quorum met)
	_ = contract.SubmitVerification(ctx, "BankB", "BankA", "1", "true")
	_ = contract.SubmitVerification(ctx, "BankC", "BankA", "1", "true")

	// Now it should pass
	err = contract.SubmitCommit(ctx, "BankC", "BankA", "1")
	assert.NoError(t, err)

	bytes, err := stub.GetState(CommitKeyPrefix + "BankA~1~BankC")
	assert.NoError(t, err)
	assert.NotNil(t, bytes)

	var commit CommitRecord
	err = json.Unmarshal(bytes, &commit)
	assert.NoError(t, err)
	assert.Equal(t, "BankC", commit.CommitterID)
	assert.Equal(t, "BankA", commit.TargetBankID)
}

func TestCheckConsensus(t *testing.T) {
	contract := new(CBFTContract)
	ctx, stub := getMockContext()
	stub.MockTransactionStart("tx1")
	defer stub.MockTransactionEnd("tx1")

	// Add a valid update
	_ = contract.SubmitClusterUpdate(ctx, "BankA", "1", "QmCID", "hash", "0.85")

	// Without verify/commit quorum, nothing is accepted
	res, err := contract.CheckConsensus(ctx, "1")
	assert.NoError(t, err)
	assert.Equal(t, "[]", res)

	// Add 2 verifications
	_ = contract.SubmitVerification(ctx, "BankB", "BankA", "1", "true")
	_ = contract.SubmitVerification(ctx, "BankC", "BankA", "1", "true")

	// Add 2 commits
	_ = contract.SubmitCommit(ctx, "BankB", "BankA", "1")
	_ = contract.SubmitCommit(ctx, "BankC", "BankA", "1")

	// Now it should be accepted
	res, err = contract.CheckConsensus(ctx, "1")
	assert.NoError(t, err)

	var accepted []string
	err = json.Unmarshal([]byte(res), &accepted)
	assert.NoError(t, err)
	assert.Contains(t, accepted, "BankA")

	// Check ledger update
	bytes, _ := stub.GetState(UpdateKeyPrefix + "BankA~1")
	var update ClusterUpdate
	json.Unmarshal(bytes, &update)
	assert.True(t, update.Accepted)
}

func TestUpdateTrustScore(t *testing.T) {
	contract := new(CBFTContract)
	ctx, stub := getMockContext()
	stub.MockTransactionStart("tx1")
	defer stub.MockTransactionEnd("tx1")

	// Init ledger to get baseline
	_ = contract.InitLedger(ctx)

	// Reward BankA
	err := contract.UpdateTrustScore(ctx, "BankA", "0.1")
	assert.NoError(t, err)

	bytes, _ := stub.GetState(TrustKeyPrefix + "BankA")
	var ts TrustScore
	json.Unmarshal(bytes, &ts)
	assert.Equal(t, 1.1, ts.Score)

	// Penalise BankB heavily to trigger clamp
	err = contract.UpdateTrustScore(ctx, "BankB", "-2.0")
	assert.NoError(t, err)

	bytes, _ = stub.GetState(TrustKeyPrefix + "BankB")
	json.Unmarshal(bytes, &ts)
	assert.Equal(t, ScoreMin, ts.Score)
}

func TestStoreAndGetGlobalModel(t *testing.T) {
	contract := new(CBFTContract)
	ctx, stub := getMockContext()
	stub.MockTransactionStart("tx1")
	defer stub.MockTransactionEnd("tx1")

	// Store
	err := contract.StoreGlobalModel(ctx, "1", "QmGlobal", "hashGlobal")
	assert.NoError(t, err)

	// Get
	res, err := contract.GetGlobalModel(ctx, "1")
	assert.NoError(t, err)

	var gm GlobalModel
	json.Unmarshal([]byte(res), &gm)
	assert.Equal(t, 1, gm.Round)
	assert.Equal(t, "QmGlobal", gm.GlobalCID)
}
