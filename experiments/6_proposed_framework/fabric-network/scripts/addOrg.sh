#!/bin/bash
# =============================================================================
#  addOrg.sh — Dynamically Add a New Organization to fraud-detection-global
#  Hierarchical Clustered FL | Fabric 2.5
#
#  Usage:
#    ./addOrg.sh BankD
#
#  Steps:
#    1. Validate input and set environment
#    2. Generate crypto material for the new org (cryptogen extend)
#    3. Print the new org's MSP JSON config (configtxlator)
#    4. Fetch current channel config block
#    5. Decode config block to JSON
#    6. Embed new org MSP into the config JSON
#    7. Compute the config delta (config update envelope)
#    8. Encode and wrap as channel update envelope
#    9. Collect signatures from existing org admins (BankA, BankB, BankC)
#   10. Submit the channel update transaction
# =============================================================================

set -euo pipefail

# ── Argument validation ───────────────────────────────────────────────────────
if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <NewOrgName>   e.g.:  $0 BankD"
  exit 1
fi

NEW_ORG="$1"                              # e.g., BankD
NEW_ORG_LOWER="$(echo "$NEW_ORG" | tr '[:upper:]' '[:lower:]')"  # bankd
NEW_ORG_DOMAIN="${NEW_ORG_LOWER}.fabricfl.com"
NEW_ORG_MSP="${NEW_ORG}MSP"

# ── Resolve paths ────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FABRIC_DIR="$(dirname "$SCRIPT_DIR")"
source "${FABRIC_DIR}/.env"
export FABRIC_CFG_PATH  # peer CLI needs this to find core.yaml
export PATH=${FABRIC_DIR}/../fabric-samples/bin:$PATH

CRYPTO="${FABRIC_DIR}/crypto-config"
CHANNEL_ARTIFACTS="${FABRIC_DIR}/channel-artifacts"
ORDERER_CA="${CRYPTO}/ordererOrganizations/orderer.fabricfl.com/orderers/orderer0.orderer.fabricfl.com/msp/tlscacerts/tlsca.orderer.fabricfl.com-cert.pem"
ORDERER_ADDRESS="orderer0.orderer.fabricfl.com:7050"

# Temp directory for this operation
TMPDIR="$(mktemp -d)"
trap "rm -rf $TMPDIR" EXIT

# ── Color helpers ─────────────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*" >&2; exit 1; }

# ── setPeerEnv helper (same as other scripts) ─────────────────────────────
setPeerEnv() {
  local ORG="$1"; local PEER="$2"
  local MSP_ID BASE_PORT PEER_PORT
  case "$ORG" in
    banka) MSP_ID="BankAMSP"; BASE_PORT=7051 ;;
    bankb) MSP_ID="BankBMSP"; BASE_PORT=8051 ;;
    bankc) MSP_ID="BankCMSP"; BASE_PORT=9051 ;;
    *) error "Unknown existing org: $ORG" ;;
  esac
  PEER_PORT=$(( BASE_PORT + PEER * 10 ))
  export CORE_PEER_LOCALMSPID="$MSP_ID"
  export CORE_PEER_ADDRESS="peer${PEER}.${ORG}.fabricfl.com:${PEER_PORT}"
  export CORE_PEER_TLS_ENABLED=true
  export CORE_PEER_TLS_ROOTCERT_FILE="${CRYPTO}/peerOrganizations/${ORG}.fabricfl.com/peers/peer${PEER}.${ORG}.fabricfl.com/tls/ca.crt"
  export CORE_PEER_MSPCONFIGPATH="${CRYPTO}/peerOrganizations/${ORG}.fabricfl.com/users/Admin@${ORG}.fabricfl.com/msp"
}

# =============================================================================
#  STEP 1 — Check that the new org doesn't already exist
# =============================================================================
info "== STEP 1: Checking if ${NEW_ORG} crypto material already exists =="

if [[ -d "${CRYPTO}/peerOrganizations/${NEW_ORG_DOMAIN}" ]]; then
  warn "${NEW_ORG} crypto material already exists. Skipping crypto generation."
else
  # ==========================================================================
  #  STEP 2 — Generate crypto material for the new org using cryptogen extend
  # ==========================================================================
  info "== STEP 2: Generating crypto material for ${NEW_ORG} =="

  # Write a minimal crypto-config snippet for the new org
  cat > "${TMPDIR}/crypto-config-${NEW_ORG_LOWER}.yaml" <<EOF
PeerOrgs:
  - Name: ${NEW_ORG}
    Domain: ${NEW_ORG_DOMAIN}
    EnableNodeOUs: true
    Template:
      Count: 2
      SANS:
        - localhost
    Users:
      Count: 1
EOF

  # cryptogen extend adds new orgs to an existing crypto-config tree
  cryptogen extend \
    --config="${TMPDIR}/crypto-config-${NEW_ORG_LOWER}.yaml" \
    --input="${CRYPTO}"

  info "Crypto material created at ${CRYPTO}/peerOrganizations/${NEW_ORG_DOMAIN}"
fi

# =============================================================================
#  STEP 3 — Generate JSON MSP config for the new org using configtxgen
# =============================================================================
info "== STEP 3: Generating MSP config JSON for ${NEW_ORG} =="

# Write a minimal configtx.yaml for the new org's MSP
cat > "${TMPDIR}/configtx.yaml" <<EOF
Organizations:
  - &${NEW_ORG}
    Name: ${NEW_ORG_MSP}
    ID: ${NEW_ORG_MSP}
    MSPDir: ${CRYPTO}/peerOrganizations/${NEW_ORG_DOMAIN}/msp
    Policies:
      Readers:
        Type: Signature
        Rule: "OR('${NEW_ORG_MSP}.admin', '${NEW_ORG_MSP}.peer', '${NEW_ORG_MSP}.client')"
      Writers:
        Type: Signature
        Rule: "OR('${NEW_ORG_MSP}.admin', '${NEW_ORG_MSP}.client')"
      Admins:
        Type: Signature
        Rule: "OR('${NEW_ORG_MSP}.admin')"
      Endorsement:
        Type: Signature
        Rule: "OR('${NEW_ORG_MSP}.peer')"
    AnchorPeers:
      - Host: peer0.${NEW_ORG_DOMAIN}
        Port: 10051
EOF

# configtxgen -printOrg outputs the new org's MSP definition as JSON
FABRIC_CFG_PATH="${TMPDIR}" configtxgen \
  -printOrg "${NEW_ORG_MSP}" \
  -configPath "${TMPDIR}" \
  > "${TMPDIR}/${NEW_ORG_LOWER}.json"

info "New org MSP JSON written to ${TMPDIR}/${NEW_ORG_LOWER}.json"

# =============================================================================
#  STEP 4 — Fetch the latest channel config block
# =============================================================================
info "== STEP 4: Fetching current channel configuration block =="

setPeerEnv banka 0
peer channel fetch config "${TMPDIR}/config_block.pb" \
  --channelID "${CHANNEL_NAME}" \
  --orderer "${ORDERER_ADDRESS}" \
  --tls \
  --cafile "${ORDERER_CA}"

info "Config block fetched."

# =============================================================================
#  STEP 5 — Decode channel config block to JSON
# =============================================================================
info "== STEP 5: Decoding config block to JSON using configtxlator =="

configtxlator proto_decode \
  --input "${TMPDIR}/config_block.pb" \
  --type common.Block \
  | python3 -c "
import json, sys
block = json.load(sys.stdin)
config = block['data']['data'][0]['payload']['data']['config']
print(json.dumps(config, indent=2))
" > "${TMPDIR}/config.json"

info "Current channel config decoded to ${TMPDIR}/config.json"

# =============================================================================
#  STEP 6 — Embed the new org's MSP into the channel config JSON
# =============================================================================
info "== STEP 6: Injecting ${NEW_ORG} MSP into channel config =="

python3 - <<PYEOF
import json

with open("${TMPDIR}/config.json") as f:
    config = json.load(f)

with open("${TMPDIR}/${NEW_ORG_LOWER}.json") as f:
    new_org = json.load(f)

# Inject new org into channel application groups
groups = config["channel_group"]["groups"]["Application"]["groups"]
groups["${NEW_ORG_MSP}"] = new_org

# Save updated config
with open("${TMPDIR}/modified_config.json", "w") as f:
    json.dump(config, f, indent=2)

print("Modified config written with ${NEW_ORG} injected.")
PYEOF

# =============================================================================
#  STEP 7 — Compute config update delta (original vs modified)
# =============================================================================
info "== STEP 7: Computing config delta (channel update envelope) =="

# Encode both original and modified configs to protobuf
configtxlator proto_encode \
  --input "${TMPDIR}/config.json" \
  --type common.Config \
  --output "${TMPDIR}/config.pb"

configtxlator proto_encode \
  --input "${TMPDIR}/modified_config.json" \
  --type common.Config \
  --output "${TMPDIR}/modified_config.pb"

# Compute the delta — only the changed portions need to be sent
configtxlator compute_update \
  --channel_id "${CHANNEL_NAME}" \
  --original "${TMPDIR}/config.pb" \
  --updated "${TMPDIR}/modified_config.pb" \
  --output "${TMPDIR}/config_update.pb"

# Wrap the delta as a full channel update envelope
configtxlator proto_decode \
  --input "${TMPDIR}/config_update.pb" \
  --type common.ConfigUpdate \
  | python3 -c "
import json, sys
update = json.load(sys.stdin)
envelope = {
    'payload': {
        'header': {
            'channel_header': {
                'channel_id': '${CHANNEL_NAME}',
                'type': 2
            }
        },
        'data': {
            'config_update': update
        }
    }
}
print(json.dumps(envelope))
" > "${TMPDIR}/config_update_envelope.json"

configtxlator proto_encode \
  --input "${TMPDIR}/config_update_envelope.json" \
  --type common.Envelope \
  --output "${CHANNEL_ARTIFACTS}/${NEW_ORG_LOWER}_update_in_envelope.pb"

info "Config update envelope: ${CHANNEL_ARTIFACTS}/${NEW_ORG_LOWER}_update_in_envelope.pb"

# =============================================================================
#  STEP 8 — Collect signatures from all existing org admins
#  Fabric channel update requires a majority-admin signature
# =============================================================================
info "== STEP 8: Collecting signatures from BankA, BankB, BankC admins =="

for ORG in banka bankb bankc; do
  setPeerEnv "$ORG" 0
  info "  Signing with ${ORG} Admin..."
  peer channel signconfigtx \
    --file "${CHANNEL_ARTIFACTS}/${NEW_ORG_LOWER}_update_in_envelope.pb"
  info "  Signed by ${ORG}."
done

# =============================================================================
#  STEP 9 — Submit the channel update transaction
# =============================================================================
info "== STEP 9: Submitting channel update to add ${NEW_ORG} =="

# Use BankA Admin to submit (last signer auto-submits via 'update')
setPeerEnv banka 0
peer channel update \
  --channelID "${CHANNEL_NAME}" \
  --file "${CHANNEL_ARTIFACTS}/${NEW_ORG_LOWER}_update_in_envelope.pb" \
  --orderer "${ORDERER_ADDRESS}" \
  --tls \
  --cafile "${ORDERER_CA}"

info "== ${NEW_ORG} successfully added to channel '${CHANNEL_NAME}' =="
info ""
info "Next steps for ${NEW_ORG}:"
info "  1. Start ${NEW_ORG} peer containers (add to docker-compose.yaml)"
info "  2. Fetch the channel genesis block and join peers to the channel"
info "  3. Install and approve chaincode on ${NEW_ORG} peers"
info "  4. Update endorsement policy if required"
