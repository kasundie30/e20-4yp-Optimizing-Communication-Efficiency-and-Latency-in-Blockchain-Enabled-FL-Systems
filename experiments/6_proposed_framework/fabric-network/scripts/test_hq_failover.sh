#!/bin/bash
# test_hq_failover.sh - Tests the HQ failover mechanism (Phase 10.2)
#
# Steps:
# 1. Stop peer0.banka
# 2. Re-point CLI to peer1.banka
# 3. Invoke ActivateBackup("BankA", 1)
# 4. Verify backupActive state in chaincode

set -euo pipefail

echo "==========================================="
echo "   HQ Failover Testing (Phase 10.2)        "
echo "==========================================="

echo "[1/4] Stopping primary HQ peer (peer0.banka)..."
docker stop peer0.banka.fabricfl.com

echo "[2/4] Triggering ActivateBackup via peer1.banka..."
docker exec \
  -e CORE_PEER_ADDRESS=peer1.banka.fabricfl.com:7061 \
  -e CORE_PEER_TLS_CERT_FILE=/opt/gopath/src/github.com/hyperledger/fabric/peer/crypto/peerOrganizations/banka.fabricfl.com/peers/peer1.banka.fabricfl.com/tls/server.crt \
  -e CORE_PEER_TLS_KEY_FILE=/opt/gopath/src/github.com/hyperledger/fabric/peer/crypto/peerOrganizations/banka.fabricfl.com/peers/peer1.banka.fabricfl.com/tls/server.key \
  cli \
  peer chaincode invoke -o orderer0.orderer.fabricfl.com:7050 \
  --ordererTLSHostnameOverride orderer0.orderer.fabricfl.com \
  --tls --cafile /opt/gopath/src/github.com/hyperledger/fabric/peer/crypto/ordererOrganizations/orderer.fabricfl.com/orderers/orderer0.orderer.fabricfl.com/msp/tlscacerts/tlsca.orderer.fabricfl.com-cert.pem \
  -C fraud-detection-global -n cbft-fl \
  --peerAddresses peer1.banka.fabricfl.com:7061 --tlsRootCertFiles /opt/gopath/src/github.com/hyperledger/fabric/peer/crypto/peerOrganizations/banka.fabricfl.com/peers/peer1.banka.fabricfl.com/tls/ca.crt \
  --peerAddresses peer0.bankb.fabricfl.com:8051 --tlsRootCertFiles /opt/gopath/src/github.com/hyperledger/fabric/peer/crypto/peerOrganizations/bankb.fabricfl.com/peers/peer0.bankb.fabricfl.com/tls/ca.crt \
  -c '{"function":"ActivateBackup","Args":["BankA","1"]}'

sleep 20

echo "[3/4] Fetching Trust Scores or Update to verify BackupActive is true..."
docker exec \
  -e CORE_PEER_ADDRESS=peer1.banka.fabricfl.com:7061 \
  -e CORE_PEER_TLS_CERT_FILE=/opt/gopath/src/github.com/hyperledger/fabric/peer/crypto/peerOrganizations/banka.fabricfl.com/peers/peer1.banka.fabricfl.com/tls/server.crt \
  -e CORE_PEER_TLS_KEY_FILE=/opt/gopath/src/github.com/hyperledger/fabric/peer/crypto/peerOrganizations/banka.fabricfl.com/peers/peer1.banka.fabricfl.com/tls/server.key \
  cli \
  peer chaincode query -C fraud-detection-global -n cbft-fl -c '{"function":"GetClusterUpdate","Args":["BankA","1"]}' > failover_result.json || true

cat failover_result.json
if grep -q "backupActive\\\":true" failover_result.json; then
    echo "✅ HQ Failover verified successfully!"
else
    echo "❌ HQ Failover failed! Backup flag not active."
fi

# Cleanup
rm -f failover_result.json
echo "[4/4] Restarting peer0.banka to restore network state..."
docker start peer0.banka.fabricfl.com
sleep 5
echo "Failover test complete."
