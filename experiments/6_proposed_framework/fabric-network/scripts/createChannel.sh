#!/bin/bash
# =============================================================================
#  createChannel.sh — Create and join the fraud-detection-global channel
#  Hierarchical Clustered FL | Fabric 2.5  (no system channel)
#
#  Fabric 2.5 uses the channel-participation API (osnadmin).
#  There is NO orderer system channel.
#
#  Steps executed:
#    1. Generate channel genesis block via configtxgen (FraudDetectionChannel)
#    2. Join all 3 orderers to the channel via osnadmin channel join
#    3. Join all 6 peers (peer0 + peer1 of BankA, BankB, BankC) via peer channel join
#    4. Update anchor peers for each org (peer0 = HQ is the anchor)
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FABRIC_DIR="$(dirname "$SCRIPT_DIR")"
source "${FABRIC_DIR}/.env"
export FABRIC_CFG_PATH  # peer CLI needs this to find core.yaml

# ── Ensure fabric binaries (configtxgen, osnadmin, peer) are on PATH ──────────
FABRIC_BIN="$(dirname "${FABRIC_DIR}")/fabric-samples/bin"
export PATH="${FABRIC_BIN}:${PATH}"

# ── Paths ────────────────────────────────────────────────────────────────────
CRYPTO="${FABRIC_DIR}/crypto-config"
CHANNEL_ARTIFACTS="${FABRIC_DIR}/channel-artifacts"

# Orderer TLS CA cert (used for peer channel join / update)
ORDERER_CA="${CRYPTO}/ordererOrganizations/orderer.fabricfl.com/orderers/orderer0.orderer.fabricfl.com/msp/tlscacerts/tlsca.orderer.fabricfl.com-cert.pem"
ORDERER_ADDRESS="orderer0.orderer.fabricfl.com:7050"

# Orderer Admin user TLS credentials (used for osnadmin — mutual TLS)
ORDERER_ADMIN_TLS_DIR="${CRYPTO}/ordererOrganizations/orderer.fabricfl.com/users/Admin@orderer.fabricfl.com/tls"
ORDERER_ADMIN_CA="${ORDERER_ADMIN_TLS_DIR}/ca.crt"
ORDERER_ADMIN_CERT="${ORDERER_ADMIN_TLS_DIR}/client.crt"
ORDERER_ADMIN_KEY="${ORDERER_ADMIN_TLS_DIR}/client.key"

# ── Color helpers ─────────────────────────────────────────────────────────
GREEN='\033[0;32m'; RED='\033[0;31m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*" >&2; exit 1; }

# ── Helper: set peer context env vars ────────────────────────────────────────
# setPeerEnv <org> <peerNum>
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
#  STEP 1 — Generate channel genesis block using configtxgen
#           (FraudDetectionChannel profile — no system channel needed)
# =============================================================================
info "== STEP 1: Generating channel genesis block for '${CHANNEL_NAME}' =="

CHANNEL_GENESIS="${CHANNEL_ARTIFACTS}/${CHANNEL_NAME}_genesis.block"

# configtxgen must find configtx.yaml AND resolve relative cert paths from FABRIC_DIR
# We temporarily override FABRIC_CFG_PATH for configtxgen only, then restore it.
FABRIC_CFG_PATH_ORIG="${FABRIC_CFG_PATH}"
export FABRIC_CFG_PATH="${FABRIC_DIR}"

configtxgen \
  -profile FraudDetectionChannel \
  -outputBlock "${CHANNEL_GENESIS}" \
  -channelID "${CHANNEL_NAME}"

export FABRIC_CFG_PATH="${FABRIC_CFG_PATH_ORIG}"

info "Channel genesis block written to: ${CHANNEL_GENESIS}"

# =============================================================================
#  STEP 2 — Join all 3 orderers to the channel via osnadmin channel join
#           Each orderer exposes its admin port (7053, 8053, 9053)
# =============================================================================
info "== STEP 2: Joining orderers to channel '${CHANNEL_NAME}' via osnadmin =="

osnadminJoin() {
  local ORDERER_HOST="$1"     # e.g. orderer0.orderer.fabricfl.com
  local ADMIN_PORT="$2"       # e.g. 7053

  info "  Joining ${ORDERER_HOST} (admin port ${ADMIN_PORT})..."
  local OUT
  OUT=$(osnadmin channel join \
    --channelID "${CHANNEL_NAME}" \
    --config-block "${CHANNEL_GENESIS}" \
    -o "${ORDERER_HOST}:${ADMIN_PORT}" \
    --ca-file     "${ORDERER_ADMIN_CA}" \
    --client-cert "${ORDERER_ADMIN_CERT}" \
    --client-key  "${ORDERER_ADMIN_KEY}" 2>&1)
  local HTTP_STATUS
  HTTP_STATUS=$(echo "$OUT" | grep -oP 'Status: \K[0-9]+' | head -1)
  if [[ "$HTTP_STATUS" == "201" || "$HTTP_STATUS" == "200" ]]; then
    info "  ${ORDERER_HOST} successfully joined the channel."
  elif [[ "$HTTP_STATUS" == "405" ]]; then
    info "  ${ORDERER_HOST} already a member of channel (skipping)."
  else
    echo "$OUT"
    error "  Failed to join ${ORDERER_HOST}: HTTP ${HTTP_STATUS:-unknown}"
  fi
}

osnadminJoin "orderer0.orderer.fabricfl.com" 7053
osnadminJoin "orderer1.orderer.fabricfl.com" 8053
osnadminJoin "orderer2.orderer.fabricfl.com" 9053

info "All orderers joined channel '${CHANNEL_NAME}'."

# =============================================================================
#  STEP 3 — Join all 6 peers to the channel
# =============================================================================
info "== STEP 3: Joining all 6 peers to channel '${CHANNEL_NAME}' =="

joinPeer() {
  local ORG="$1"
  local PEER="$2"
  setPeerEnv "$ORG" "$PEER"
  # Check if peer already has the channel
  if peer channel list 2>/dev/null | grep -q "^${CHANNEL_NAME}$"; then
    info "  peer${PEER}.${ORG} already joined channel (skipping)."
    return 0
  fi
  info "  Joining peer${PEER}.${ORG}..."
  peer channel join \
    --blockpath "${CHANNEL_GENESIS}"
  info "  peer${PEER}.${ORG} successfully joined."
}

joinPeer banka 0
joinPeer banka 1
joinPeer bankb 0
joinPeer bankb 1
joinPeer bankc 0
joinPeer bankc 1

# =============================================================================
#  STEP 4 — Update anchor peers for each org (Fabric 2.5 configtxlator method)
#  peer0 of each bank is the HQ and hence the anchor peer.
#  We use the modern approach: fetch live channel config → parse → modify →
#  compute delta → submit channel update. The old configtxgen
#  -outputAnchorPeersUpdate tx files are version-locked and fail on live channels.
# =============================================================================
info "== STEP 4: Updating anchor peers (peer0 = HQ anchor for each org) =="

FABRIC_BIN="$(dirname "${FABRIC_DIR}")/fabric-samples/bin"

updateAnchor() {
  local ORG="$1"
  local MSP_ID="$2"
  local ANCHOR_HOST="$3"
  local ANCHOR_PORT="$4"

  setPeerEnv "$ORG" 0

  info "  Setting anchor peer ${ANCHOR_HOST}:${ANCHOR_PORT} for ${MSP_ID}..."

  local TMPDIR
  TMPDIR=$(mktemp -d)

  # Fetch current channel config block and decode to JSON
  peer channel fetch config "${TMPDIR}/config_block.pb" \
    --channelID "${CHANNEL_NAME}" \
    --orderer "${ORDERER_ADDRESS}" \
    --tls --cafile "${ORDERER_CA}" 2>/dev/null

  "${FABRIC_BIN}/configtxlator" proto_decode \
    --input "${TMPDIR}/config_block.pb" \
    --type common.Block \
    --output "${TMPDIR}/config_block.json"

  # Extract the channel config portion
  python3 -c "
import json, sys
with open('${TMPDIR}/config_block.json') as f:
    b = json.load(f)
cfg = b['data']['data'][0]['payload']['data']['config']
with open('${TMPDIR}/config.json','w') as f:
    json.dump(cfg, f)
"

  # Build modified config with anchor peer injected
  python3 - <<PYEOF
import json, copy

with open('${TMPDIR}/config.json') as f:
    config = json.load(f)

modified = copy.deepcopy(config)
app_groups = modified['channel_group']['groups']['Application']['groups']

if '${MSP_ID}' in app_groups:
    msp_group = app_groups['${MSP_ID}']
    anchor = {'host': '${ANCHOR_HOST}', 'port': ${ANCHOR_PORT}}
    existing = msp_group.get('values', {}).get('AnchorPeers', {}).get('value', {}).get('anchor_peers', [])
    if anchor not in existing:
        msp_group.setdefault('values', {})['AnchorPeers'] = {
            'mod_policy': 'Admins',
            'value': {'anchor_peers': [anchor]},
            'version': '0'
        }

with open('${TMPDIR}/modified_config.json', 'w') as f:
    json.dump(modified, f)
PYEOF

  # Encode both config JSONs to protobuf
  "${FABRIC_BIN}/configtxlator" proto_encode \
    --input "${TMPDIR}/config.json" \
    --type common.Config \
    --output "${TMPDIR}/config.pb"

  "${FABRIC_BIN}/configtxlator" proto_encode \
    --input "${TMPDIR}/modified_config.json" \
    --type common.Config \
    --output "${TMPDIR}/modified_config.pb"

  # Compute the config update delta
  "${FABRIC_BIN}/configtxlator" compute_update \
    --channel_id "${CHANNEL_NAME}" \
    --original "${TMPDIR}/config.pb" \
    --updated "${TMPDIR}/modified_config.pb" \
    --output "${TMPDIR}/config_update.pb" 2>/dev/null || {
      info "  Anchor peer already set for ${MSP_ID} — skipping."
      rm -rf "${TMPDIR}"
      return 0
    }

  # Wrap the update in an envelope
  "${FABRIC_BIN}/configtxlator" proto_decode \
    --input "${TMPDIR}/config_update.pb" \
    --type common.ConfigUpdate \
    --output "${TMPDIR}/config_update.json"

  python3 -c "
import json
with open('${TMPDIR}/config_update.json') as f:
    update = json.load(f)
envelope = {
    'payload': {
        'header': {'channel_header': {
            'channel_id': '${CHANNEL_NAME}',
            'type': 2
        }},
        'data': {'config_update': update}
    }
}
with open('${TMPDIR}/config_update_envelope.json','w') as f:
    json.dump(envelope, f)
"

  "${FABRIC_BIN}/configtxlator" proto_encode \
    --input "${TMPDIR}/config_update_envelope.json" \
    --type common.Envelope \
    --output "${TMPDIR}/config_update_in_envelope.pb"

  # Submit the channel update
  peer channel update \
    --channelID "${CHANNEL_NAME}" \
    --file "${TMPDIR}/config_update_in_envelope.pb" \
    --orderer "${ORDERER_ADDRESS}" \
    --tls --cafile "${ORDERER_CA}"

  info "  Anchor peer set for ${MSP_ID} ✔"
  rm -rf "${TMPDIR}"
}

updateAnchor banka BankAMSP peer0.banka.fabricfl.com 7051
updateAnchor bankb BankBMSP peer0.bankb.fabricfl.com 8051
updateAnchor bankc BankCMSP peer0.bankc.fabricfl.com 9051

info "== Channel setup complete. All 6 peers joined, anchor peers configured. =="
info "Next step: run scripts/deployChaincode.sh"
