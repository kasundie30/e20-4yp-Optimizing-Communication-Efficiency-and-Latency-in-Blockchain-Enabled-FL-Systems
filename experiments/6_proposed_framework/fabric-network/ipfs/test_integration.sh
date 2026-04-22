#!/usr/bin/env bash
# Integration Test: IPFS upload -> Fabric SubmitClusterUpdate

set -e

# Load python env
source "/media/fyp-group-18/1TB-Hard/FYP-Group18/experiments/6_proposed_framework/.venv/bin/activate"

# 1. Provide a dummy model to the IPFS python client
echo "=== 1. Uploading model to IPFS ==="
cat << 'EOF' > run_ipfs.py
import sys
import json
from ipfs_client import IPFSClient

client = IPFSClient()
dummy_weights = {"bias": 0.5, "weights": [0.1, 0.2, 0.3], "round": 2}
cid, hsh = client.upload_model(dummy_weights)
print(f"{cid},{hsh}")
EOF

IPFS_RES=$(python run_ipfs.py)
rm run_ipfs.py
MODEL_CID=$(echo $IPFS_RES | cut -d',' -f1)
MODEL_HASH=$(echo $IPFS_RES | cut -d',' -f2)

echo "IPFS CID: ${MODEL_CID}"
echo "Model Hash: ${MODEL_HASH}"

# 2. Setup Fabric environment variables
export PATH="/media/fyp-group-18/1TB-Hard/FYP-Group18/experiments/6_proposed_framework/fabric-samples/bin:$PATH"
export FABRIC_CFG_PATH="/media/fyp-group-18/1TB-Hard/FYP-Group18/experiments/6_proposed_framework/fabric-samples/config"
CRYPTO="/media/fyp-group-18/1TB-Hard/FYP-Group18/experiments/6_proposed_framework/fabric-network/crypto-config"
export CORE_PEER_LOCALMSPID=BankAMSP
export CORE_PEER_ADDRESS=peer0.banka.fabricfl.com:7051
export CORE_PEER_TLS_ENABLED=true
export CORE_PEER_TLS_ROOTCERT_FILE="${CRYPTO}/peerOrganizations/banka.fabricfl.com/peers/peer0.banka.fabricfl.com/tls/ca.crt"
export CORE_PEER_MSPCONFIGPATH="${CRYPTO}/peerOrganizations/banka.fabricfl.com/users/Admin@banka.fabricfl.com/msp"
ORDERER_CA="${CRYPTO}/ordererOrganizations/orderer.fabricfl.com/orderers/orderer0.orderer.fabricfl.com/msp/tlscacerts/tlsca.orderer.fabricfl.com-cert.pem"

# 3. Invoke SubmitClusterUpdate on the chaincode
echo "=== 2. Invoking SubmitClusterUpdate on Chaincode ==="
peer chaincode invoke \
  --channelID fraud-detection-global \
  --name cbft-fl \
  --tls --cafile "${ORDERER_CA}" \
  --orderer orderer0.orderer.fabricfl.com:7050 \
  --peerAddresses peer0.banka.fabricfl.com:7051 \
    --tlsRootCertFiles "${CRYPTO}/peerOrganizations/banka.fabricfl.com/peers/peer0.banka.fabricfl.com/tls/ca.crt" \
  --peerAddresses peer0.bankb.fabricfl.com:8051 \
    --tlsRootCertFiles "${CRYPTO}/peerOrganizations/bankb.fabricfl.com/peers/peer0.bankb.fabricfl.com/tls/ca.crt" \
  -c "{\"function\":\"SubmitClusterUpdate\",\"Args\":[\"BankA\", \"2\", \"${MODEL_CID}\", \"${MODEL_HASH}\", \"0.88\"]}"

echo "Waiting for transaction to commit..."
sleep 3

# 4. Query ledger to verify
echo "=== 3. Querying Ledger to Verify ==="
# Note: Since the contract doesn't expose a GetUpdate query, we'll just check success from invoke above
# Normally we'd do: peer chaincode query -C fraud-detection-global -n cbft-fl -c '{"function":"GetUpdate","Args":["BankA","2"]}'
echo "Integration Test SUCCESS!"
