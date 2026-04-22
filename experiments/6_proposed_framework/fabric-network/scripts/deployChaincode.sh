#!/bin/bash
# =============================================================================
#  deployChaincode.sh — Package, Install, Approve, and Commit CBFT Chaincode
#  Hierarchical Clustered FL | Fabric 2.5 Lifecycle
#
#  Chaincode: cbft-fl  v1.0  sequence 1
#  Endorsement policy: OutOf(2, BankAMSP, BankBMSP, BankCMSP)
#
#  Steps:
#    1. Package chaincode from ./chaincode/cbft/
#    2. Install on all 6 peers
#    3. Query package ID on each peer
#    4. Approve chaincode definition for each org (uses one peer per org)
#    5. Check commit readiness
#    6. Commit chaincode on the channel
#    7. Invoke InitLedger to bootstrap trust scores
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FABRIC_DIR="$(dirname "$SCRIPT_DIR")"
source "${FABRIC_DIR}/.env"
export FABRIC_CFG_PATH  # peer CLI needs this to find core.yaml

# ── Paths ────────────────────────────────────────────────────────────────────
CRYPTO="${FABRIC_DIR}/crypto-config"
CC_SRC_PATH="${FABRIC_DIR}/chaincode/cbft"
CC_PKG_FILE="${FABRIC_DIR}/${CHAINCODE_NAME}.tar.gz"
ORDERER_ADDRESS="orderer0.orderer.fabricfl.com:7050"
ORDERER_CA="${CRYPTO}/ordererOrganizations/orderer.fabricfl.com/orderers/orderer0.orderer.fabricfl.com/msp/tlscacerts/tlsca.orderer.fabricfl.com-cert.pem"

# ── Color helpers ─────────────────────────────────────────────────────────
GREEN='\033[0;32m'; RED='\033[0;31m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*" >&2; exit 1; }

# ── setPeerEnv: set CORE_PEER_* env for a given org and peer number ──────────
setPeerEnv() {
  local ORG="$1"
  local PEER="$2"
  local MSP_ID
  case "$ORG" in
    banka) MSP_ID="BankAMSP" ;;
    bankb) MSP_ID="BankBMSP" ;;
    bankc) MSP_ID="BankCMSP" ;;
    *) error "Unknown org: $ORG" ;;
  esac
  local BASE_PORT
  case "$ORG" in
    banka) BASE_PORT=7051 ;;
    bankb) BASE_PORT=8051 ;;
    bankc) BASE_PORT=9051 ;;
  esac
  local PEER_PORT=$(( BASE_PORT + PEER * 10 ))

  export CORE_PEER_LOCALMSPID="$MSP_ID"
  export CORE_PEER_ADDRESS="peer${PEER}.${ORG}.fabricfl.com:${PEER_PORT}"
  export CORE_PEER_TLS_ENABLED=true
  export CORE_PEER_TLS_ROOTCERT_FILE="${CRYPTO}/peerOrganizations/${ORG}.fabricfl.com/peers/peer${PEER}.${ORG}.fabricfl.com/tls/ca.crt"
  export CORE_PEER_MSPCONFIGPATH="${CRYPTO}/peerOrganizations/${ORG}.fabricfl.com/users/Admin@${ORG}.fabricfl.com/msp"
}

# =============================================================================
#  STEP 1 — Package the chaincode
#  Creates a .tar.gz file conforming to Fabric 2.x lifecycle format
# =============================================================================
info "== STEP 1: Packaging chaincode '${CHAINCODE_NAME}' from ${CC_SRC_PATH} =="

peer lifecycle chaincode package "${CC_PKG_FILE}" \
  --path "${CC_SRC_PATH}" \
  --lang golang \
  --label "${CHAINCODE_NAME}_${CHAINCODE_VERSION}"

info "Chaincode packaged: ${CC_PKG_FILE}"

# =============================================================================
#  STEP 2 — Install on all 6 peers
#  Each peer must have the chaincode installed before it can endorse
# =============================================================================
info "== STEP 2: Installing chaincode on all 6 peers =="

installOnPeer() {
  local ORG="$1"
  local PEER="$2"
  setPeerEnv "$ORG" "$PEER"
  info "  Installing on peer${PEER}.${ORG}..."
  peer lifecycle chaincode install "${CC_PKG_FILE}" || info "  Already installed on peer${PEER}.${ORG}"
  info "  Installation step finished for peer${PEER}.${ORG}"
}

installOnPeer banka 0   # BankA HQ
installOnPeer banka 1   # BankA Backup HQ
installOnPeer bankb 0   # BankB HQ
installOnPeer bankb 1   # BankB Backup HQ
installOnPeer bankc 0   # BankC HQ
installOnPeer bankc 1   # BankC Backup HQ

# =============================================================================
#  STEP 3 — Query the package ID from peer0 of BankA
#  The package ID is <label>:<sha256-hash> and is needed for approval
# =============================================================================
info "== STEP 3: Querying installed chaincode to get package ID =="

setPeerEnv banka 0
CC_PACKAGE_ID=$(peer lifecycle chaincode queryinstalled \
  --output json | \
  python3 -c "
import json, sys
data = json.load(sys.stdin)
for cc in data.get('installed_chaincodes', []):
    if cc['label'] == '${CHAINCODE_NAME}_${CHAINCODE_VERSION}':
        print(cc['package_id'])
        break
")

if [[ -z "$CC_PACKAGE_ID" ]]; then
  error "Could not retrieve package ID. Ensure chaincode was installed successfully."
fi
info "Package ID: ${CC_PACKAGE_ID}"

# =============================================================================
#  IDEMPOTENCY CHECK — skip approve/commit if chaincode already committed
#  at this sequence. This makes the script safe to re-run on a live network.
# =============================================================================
setPeerEnv banka 0
COMMITTED_SEQ=$(peer lifecycle chaincode querycommitted \
  --channelID "${CHANNEL_NAME}" \
  --name "${CHAINCODE_NAME}" \
  --tls \
  --cafile "${ORDERER_CA}" \
  --output json 2>/dev/null | python3 -c "
import json, sys
try:
    data = json.load(sys.stdin)
    print(data.get('sequence', 0))
except:
    print(0)
") || COMMITTED_SEQ=0

if [[ "${COMMITTED_SEQ}" -ge "${CHAINCODE_SEQUENCE}" ]]; then
  info "Chaincode '${CHAINCODE_NAME}' already committed at sequence ${COMMITTED_SEQ} — skipping approve/commit/InitLedger."
  exit 0
fi

# =============================================================================
#  STEP 4 — Approve chaincode definition for each org
#  Approval must happen on at least one peer per org.
#  We use peer0 (HQ) of each org for the approval transaction.
#  Endorsement policy: OutOf(2, BankAMSP, BankBMSP, BankCMSP)
# =============================================================================
info "== STEP 4: Approving chaincode definition for each org =="

ENDORSE_POLICY="OutOf(2, 'BankAMSP.peer', 'BankBMSP.peer', 'BankCMSP.peer')"

approveForOrg() {
  local ORG="$1"
  setPeerEnv "$ORG" 0   # HQ peer approves on behalf of the org

  info "  Approving for ${ORG}..."
  peer lifecycle chaincode approveformyorg \
    --channelID "${CHANNEL_NAME}" \
    --name "${CHAINCODE_NAME}" \
    --version "${CHAINCODE_VERSION}" \
    --package-id "${CC_PACKAGE_ID}" \
    --sequence "${CHAINCODE_SEQUENCE}" \
    --signature-policy "${ENDORSE_POLICY}" \
    --tls \
    --cafile "${ORDERER_CA}" \
    --orderer "${ORDERER_ADDRESS}"
  info "  Approved for ${ORG}"
}

approveForOrg banka
approveForOrg bankb
approveForOrg bankc

# =============================================================================
#  STEP 5 — Check commit readiness (all 3 orgs must show approved=true)
# =============================================================================
info "== STEP 5: Checking commit readiness =="

setPeerEnv banka 0
peer lifecycle chaincode checkcommitreadiness \
  --channelID "${CHANNEL_NAME}" \
  --name "${CHAINCODE_NAME}" \
  --version "${CHAINCODE_VERSION}" \
  --sequence "${CHAINCODE_SEQUENCE}" \
  --signature-policy "${ENDORSE_POLICY}" \
  --tls \
  --cafile "${ORDERER_CA}" \
  --output json

# =============================================================================
#  STEP 6 — Commit chaincode definition on the channel
#  Commit requires endorsement from the signing peers of a quorum of orgs.
#  We provide --peerAddresses for all 3 HQ peers to satisfy the policy.
# =============================================================================
info "== STEP 6: Committing chaincode definition on channel '${CHANNEL_NAME}' =="

peer lifecycle chaincode commit \
  --channelID "${CHANNEL_NAME}" \
  --name "${CHAINCODE_NAME}" \
  --version "${CHAINCODE_VERSION}" \
  --sequence "${CHAINCODE_SEQUENCE}" \
  --signature-policy "${ENDORSE_POLICY}" \
  --tls \
  --cafile "${ORDERER_CA}" \
  --orderer "${ORDERER_ADDRESS}" \
  --peerAddresses peer0.banka.fabricfl.com:7051 \
    --tlsRootCertFiles "${CRYPTO}/peerOrganizations/banka.fabricfl.com/peers/peer0.banka.fabricfl.com/tls/ca.crt" \
  --peerAddresses peer0.bankb.fabricfl.com:8051 \
    --tlsRootCertFiles "${CRYPTO}/peerOrganizations/bankb.fabricfl.com/peers/peer0.bankb.fabricfl.com/tls/ca.crt" \
  --peerAddresses peer0.bankc.fabricfl.com:9051 \
    --tlsRootCertFiles "${CRYPTO}/peerOrganizations/bankc.fabricfl.com/peers/peer0.bankc.fabricfl.com/tls/ca.crt"

info "Chaincode '${CHAINCODE_NAME}' v${CHAINCODE_VERSION} committed successfully!"

# =============================================================================
#  STEP 7 — Invoke InitLedger to bootstrap trust scores on the ledger
# =============================================================================
info "== STEP 7: Invoking InitLedger to bootstrap trust scores =="

setPeerEnv banka 0
peer chaincode invoke \
  --channelID "${CHANNEL_NAME}" \
  --name "${CHAINCODE_NAME}" \
  --tls \
  --cafile "${ORDERER_CA}" \
  --orderer "${ORDERER_ADDRESS}" \
  --peerAddresses peer0.banka.fabricfl.com:7051 \
    --tlsRootCertFiles "${CRYPTO}/peerOrganizations/banka.fabricfl.com/peers/peer0.banka.fabricfl.com/tls/ca.crt" \
  --peerAddresses peer0.bankb.fabricfl.com:8051 \
    --tlsRootCertFiles "${CRYPTO}/peerOrganizations/bankb.fabricfl.com/peers/peer0.bankb.fabricfl.com/tls/ca.crt" \
  -c '{"function":"InitLedger","Args":[]}'

info "== Chaincode deployment complete. Trust scores bootstrapped on ledger. =="
