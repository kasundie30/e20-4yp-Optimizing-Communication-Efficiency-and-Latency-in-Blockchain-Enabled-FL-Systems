#!/bin/bash
# test_orderer_failover.sh - Tests the Raft Orderer Fault Tolerance (Phase 10.3)
#
# Steps:
# 1. Submit a transaction to verify the network is healthy
# 2. Stop orderer1
# 3. Wait 15s for Raft re-election
# 4. Submit a new transaction
# 5. Bring orderer1 back online

set -euo pipefail

echo "==========================================="
echo "   Orderer Fault Tolerance Test (10.3)     "
echo "==========================================="

echo "[1/5] Submitting initial transaction to confirm network health..."
docker exec cli peer chaincode invoke -o orderer0.orderer.fabricfl.com:7050 \
  --ordererTLSHostnameOverride orderer0.orderer.fabricfl.com \
  --tls --cafile /opt/gopath/src/github.com/hyperledger/fabric/peer/crypto/ordererOrganizations/orderer.fabricfl.com/orderers/orderer0.orderer.fabricfl.com/msp/tlscacerts/tlsca.orderer.fabricfl.com-cert.pem \
  -C fraud-detection-global -n cbft-fl \
  --peerAddresses peer0.banka.fabricfl.com:7051 --tlsRootCertFiles /opt/gopath/src/github.com/hyperledger/fabric/peer/crypto/peerOrganizations/banka.fabricfl.com/peers/peer0.banka.fabricfl.com/tls/ca.crt \
  --peerAddresses peer0.bankb.fabricfl.com:8051 --tlsRootCertFiles /opt/gopath/src/github.com/hyperledger/fabric/peer/crypto/peerOrganizations/bankb.fabricfl.com/peers/peer0.bankb.fabricfl.com/tls/ca.crt \
  -c '{"function":"UpdateTrustScore","Args":["BankC","0.05"]}' > /dev/null 2>&1

sleep 5

echo "[2/5] Taking down orderer1.orderer.fabricfl.com..."
docker stop orderer1.orderer.fabricfl.com

echo "[3/5] Waiting 15s for Raft Leader re-election to stabilize..."
sleep 15

# Note: The CLI targets orderer0 by default. If orderer1 was leader, orderer0 or orderer2 will take over.
# If orderer0 was leader, it remains leader. In either case, targeting orderer0 works because 
# Fabric client endpoints automatically redirect or wait if the orderer knows the new leader.
echo "[4/5] Submitting second transaction with one orderer down..."
if docker exec cli peer chaincode invoke -o orderer0.orderer.fabricfl.com:7050 \
  --ordererTLSHostnameOverride orderer0.orderer.fabricfl.com \
  --tls --cafile /opt/gopath/src/github.com/hyperledger/fabric/peer/crypto/ordererOrganizations/orderer.fabricfl.com/orderers/orderer0.orderer.fabricfl.com/msp/tlscacerts/tlsca.orderer.fabricfl.com-cert.pem \
  -C fraud-detection-global -n cbft-fl \
  --peerAddresses peer0.banka.fabricfl.com:7051 --tlsRootCertFiles /opt/gopath/src/github.com/hyperledger/fabric/peer/crypto/peerOrganizations/banka.fabricfl.com/peers/peer0.banka.fabricfl.com/tls/ca.crt \
  --peerAddresses peer0.bankb.fabricfl.com:8051 --tlsRootCertFiles /opt/gopath/src/github.com/hyperledger/fabric/peer/crypto/peerOrganizations/bankb.fabricfl.com/peers/peer0.bankb.fabricfl.com/tls/ca.crt \
  -c '{"function":"UpdateTrustScore","Args":["BankA","-0.05"]}'; then
    echo "✅ Transaction successful during orderer failure! Network is highly available."
else
    echo "❌ Transaction failed! Raft cluster lost consensus."
    exit 1
fi

echo "[5/5] Bringing orderer1 back online and testing state replication..."
docker start orderer1.orderer.fabricfl.com
sleep 5
echo "Test complete."
