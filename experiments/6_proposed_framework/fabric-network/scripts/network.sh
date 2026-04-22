#!/bin/bash
# =============================================================================
#  network.sh — Fabric Network Lifecycle Manager
#  Hierarchical Clustered FL | BankA, BankB, BankC | 3-node Raft Orderer
#
#  Usage:
#    ./network.sh up        — generate crypto, genesis block, start containers
#    ./network.sh down      — stop containers (keep volumes and crypto material)
#    ./network.sh teardown  — stop containers and DELETE volumes + crypto
#    ./network.sh status    — list running fabric containers
# =============================================================================

set -euo pipefail

# ── Resolve script directory so the script works from any CWD ──────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FABRIC_DIR="$(dirname "$SCRIPT_DIR")"   # fabric-network/
CHANNEL_ARTIFACTS="${FABRIC_DIR}/channel-artifacts"
export FABRIC_CFG_PATH  # configtxgen/cryptogen need this

# ── Load .env for IMAGE_TAG etc. ────────────────────────────────────────────
source "${FABRIC_DIR}/.env"

# ── Add Fabric binaries to PATH (cryptogen, configtxgen, peer, osnadmin) ────
FABRIC_BIN="$(dirname "${FABRIC_DIR}")/fabric-samples/bin"
export PATH="${FABRIC_BIN}:${PATH}"

# ── Color helpers ─────────────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*" >&2; }

# =============================================================================
#  up — full network bootstrap
# =============================================================================
networkUp() {
  info "== STEP 1: Generating crypto material with cryptogen =="
  # cryptogen reads crypto-config.yaml and writes all certs/keys under
  # crypto-config/ (one sub-tree per org). Skip if already generated.
  if [ -d "${FABRIC_DIR}/crypto-config/peerOrganizations" ]; then
    info "Crypto material already exists — skipping cryptogen."
  else
    cryptogen generate \
      --config="${FABRIC_DIR}/crypto-config.yaml" \
      --output="${FABRIC_DIR}/crypto-config"
    info "Crypto material generated at ${FABRIC_DIR}/crypto-config"
  fi

  info "== STEP 2: Creating channel-artifacts directory =="
  mkdir -p "${CHANNEL_ARTIFACTS}"

  info "== STEP 3: Starting all Docker containers =="
  # NOTE: This network uses Fabric 2.5 channel-participation API (osnadmin).
  # There is NO system channel. The orderers use BOOTSTRAPMETHOD=none.
  # Channel genesis block + peer joins are handled by createChannel.sh.
  docker compose \
    -f "${FABRIC_DIR}/docker-compose.yaml" \
    --env-file "${FABRIC_DIR}/.env" \
    up -d

  info "== Network is UP. Run scripts/createChannel.sh next. =="
}

# =============================================================================
#  down — stop containers but KEEP volumes and crypto material
# =============================================================================
networkDown() {
  warn "Stopping containers (volumes and crypto-config are preserved)..."
  docker compose \
    -f "${FABRIC_DIR}/docker-compose.yaml" \
    --env-file "${FABRIC_DIR}/.env" \
    down
  info "Containers stopped."
}

# =============================================================================
#  teardown — stop + delete everything (volumes, crypto, channel artifacts)
# =============================================================================
networkTeardown() {
  warn "Tearing down — ALL volumes, crypto material, and channel artifacts will be deleted."
  read -rp "Are you sure? (yes/no): " CONFIRM
  if [[ "$CONFIRM" != "yes" ]]; then
    info "Teardown aborted."
    exit 0
  fi

  # Stop containers and remove named volumes defined in docker-compose.yaml
  docker compose \
    -f "${FABRIC_DIR}/docker-compose.yaml" \
    --env-file "${FABRIC_DIR}/.env" \
    down --volumes --remove-orphans

  # Also stop CA containers if running
  docker compose \
    -f "${FABRIC_DIR}/docker-compose-ca.yaml" \
    --env-file "${FABRIC_DIR}/.env" \
    down --volumes --remove-orphans 2>/dev/null || true

  # Remove generated crypto material
  rm -rf "${FABRIC_DIR}/crypto-config"
  info "Removed crypto-config/"

  # Remove channel artifacts (genesis block, channel tx files)
  rm -rf "${CHANNEL_ARTIFACTS}"
  info "Removed channel-artifacts/"

  # Remove any chaincode packages that were created during deployment
  rm -f "${FABRIC_DIR}"/*.tar.gz

  info "Teardown complete. Network is clean."
}

# =============================================================================
#  status — list running fabric-related containers
# =============================================================================
networkStatus() {
  info "Running Fabric containers:"
  docker ps --filter "network=fabric_network" \
    --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
}

# =============================================================================
#  Entry point
# =============================================================================
case "${1:-}" in
  up)       networkUp       ;;
  down)     networkDown     ;;
  teardown) networkTeardown ;;
  status)   networkStatus   ;;
  *)
    echo "Usage: $0 {up|down|teardown|status}"
    exit 1
    ;;
esac
