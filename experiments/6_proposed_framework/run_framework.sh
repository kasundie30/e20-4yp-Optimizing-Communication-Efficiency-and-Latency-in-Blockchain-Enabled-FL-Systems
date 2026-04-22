#!/bin/bash
# ============================================================
#  run_framework.sh — End-to-End HCFL Framework Runner
#
#  Stages:
#    1. Start Fabric network + IPFS + API server
#    2. Wait for API to be healthy
#    3. Broadcast initial global model (Round 0)
#    4. Run FL training rounds (intra-cluster + CBFT + global agg)
#    5. Print final benchmark summary
#
#  Usage:
#    ./run_framework.sh              # uses NUM_ROUNDS default below
#    NUM_ROUNDS=3 ./run_framework.sh # override from command line
# ============================================================

set -e

# ─── USER CONFIGURATION ──────────────────────────────────────
# Change this number to control how many FL training rounds to run
NUM_ROUNDS=${NUM_ROUNDS:-20}
# ─────────────────────────────────────────────────────────────

WORKSPACE_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_PYTHON="${WORKSPACE_DIR}/.venv/bin/python"
API_URL="http://localhost:8000"
RESULTS_FILE="${WORKSPACE_DIR}/fl-integration/scripts/benchmark_results.json"
LOG_FILE="${WORKSPACE_DIR}/run_framework.log"

# ─── Colour helpers ──────────────────────────────────────────
GREEN="\033[0;32m"; YELLOW="\033[1;33m"; RED="\033[0;31m"; BLUE="\033[1;34m"; NC="\033[0m"
info()    { echo -e "${BLUE}[INFO]${NC}  $*" | tee -a "$LOG_FILE"; }
success() { echo -e "${GREEN}[OK]${NC}    $*" | tee -a "$LOG_FILE"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*" | tee -a "$LOG_FILE"; }
error()   { echo -e "${RED}[ERROR]${NC} $*" | tee -a "$LOG_FILE"; }

# ─── Cleanup trap on failure ─────────────────────────────────
teardown() {
    error "Script failed or interrupted. Running teardown..."
    cd "$WORKSPACE_DIR"
    ./stop_system.sh 2>>"$LOG_FILE" || true
    error "Teardown complete. Check ${LOG_FILE} for details."
}
trap teardown ERR INT TERM

# ─── Pre-flight checks ────────────────────────────────────────
if [ ! -f "$VENV_PYTHON" ]; then
    error "Python virtual environment not found at ${VENV_PYTHON}"
    error "Please run: python3 -m venv .venv && source .venv/bin/activate && pip install -r fl-layer/requirements.txt"
    exit 1
fi

if ! command -v docker &>/dev/null; then
    error "Docker is not installed or not in PATH."
    exit 1
fi

if ! command -v ipfs &>/dev/null; then
    error "IPFS (kubo) is not installed or not in PATH."
    exit 1
fi

# ─── Data Validation ─────────────────────────────────────────
DATA_DIR="${WORKSPACE_DIR}/data/splits/fl_clients"
for BANK in BankA BankB BankC; do
    if [ ! -e "${DATA_DIR}/${BANK}" ]; then
        error "Expected data split not found: ${DATA_DIR}/${BANK}"
        error "Please ensure the preprocessed data splits exist in data/splits/fl_clients/"
        exit 1
    fi
done
success "Data splits verified for BankA, BankB, BankC."

# ─────────────────────────────────────────────────────────────
echo -e "${BLUE}"
echo "╔══════════════════════════════════════════════════════╗"
echo "║         HCFL Framework — End-to-End Runner          ║"
echo "╠══════════════════════════════════════════════════════╣"
echo "║  Training Rounds  : ${NUM_ROUNDS}                                  ║"
echo "║  Workspace        : ${WORKSPACE_DIR:0:45}...  ║"
echo "║  Log file         : run_framework.log                ║"
echo "╚══════════════════════════════════════════════════════╝"
echo -e "${NC}"

# ─────────────────────────────────────────────────────────────
# PHASE 1: Start Infrastructure
# ─────────────────────────────────────────────────────────────
echo ""
info "══════════════════════════════════════════════"
info " PHASE 1 — Starting Backend Infrastructure    "
info "══════════════════════════════════════════════"
cd "$WORKSPACE_DIR"
./start_system.sh 2>&1 | tee -a "$LOG_FILE"
success "Infrastructure started."

# ─────────────────────────────────────────────────────────────
# PHASE 2: Health Check — Wait for API server
# ─────────────────────────────────────────────────────────────
echo ""
info "══════════════════════════════════════════════"
info " PHASE 2 — Waiting for API Server to be Ready"
info "══════════════════════════════════════════════"
MAX_WAIT=60
ELAPSED=0
POLL=5
API_READY=false

while [ $ELAPSED -lt $MAX_WAIT ]; do
    if curl -sf "${API_URL}/health" >/dev/null 2>&1; then
        API_READY=true
        break
    fi
    warn "API not ready yet... retrying in ${POLL}s (${ELAPSED}/${MAX_WAIT}s)"
    sleep $POLL
    ELAPSED=$((ELAPSED + POLL))
done

if [ "$API_READY" != "true" ]; then
    error "API server did not become healthy within ${MAX_WAIT}s."
    exit 1
fi
success "API server is healthy at ${API_URL}."

# ─────────────────────────────────────────────────────────────
# PHASE 3: Broadcast Initial Global Model (Round 0)
# ─────────────────────────────────────────────────────────────
echo ""
info "══════════════════════════════════════════════"
info " PHASE 3 — Broadcasting Initial Global Model  "
info "           (Round 0 — Random Seed Init)       "
info "══════════════════════════════════════════════"
cd "$WORKSPACE_DIR/fl-integration"
"$VENV_PYTHON" scripts/init_round_zero.py --seed 42 2>&1 | tee -a "$LOG_FILE"
success "Round 0 model registered on blockchain."

# Wait a moment for the ledger to settle
info "Waiting 5s for ledger to settle before starting training..."
sleep 5

# ─────────────────────────────────────────────────────────────
# PHASE 4: FL Training Rounds
#   - Intra-cluster: Local branch training → HQ FedAvg
#   - Inter-cluster: CBFT cross-verification → Global aggregation
#   - Per-round evaluation on held-out test set
# ─────────────────────────────────────────────────────────────
echo ""
info "══════════════════════════════════════════════"
info " PHASE 4 — Federated Learning Training Loop   "
info "           Rounds          : ${NUM_ROUNDS}            "
info "           Data            : data/splits/fl_clients  "
info "           Test set        : data/splits/test/        "
info "══════════════════════════════════════════════"
echo ""
info "Each round performs:"
info "  [A] Intra-cluster training  — branches train locally with Differential Privacy"
info "  [B] HQ FedAvg              — branches averaged → Cluster Model"
info "  [C] IPFS upload            — Cluster Model uploaded; CID hash stored on Blockchain"
info "  [D] CBFT cross-verification — Banks verify each other's models (SHA-256 + PR-AUC)"
info "  [E] Global aggregation      — Trust-weighted FedAvg across accepted banks"
info "  [F] Evaluation              — Global model evaluated (F1, PR-AUC, ROC-AUC)"
echo ""

cd "$WORKSPACE_DIR/fl-integration"
"$VENV_PYTHON" scripts/run_10_rounds.py --num-rounds "$NUM_ROUNDS" 2>&1 | tee -a "$LOG_FILE"

# ─────────────────────────────────────────────────────────────
# PHASE 5: Final Summary
# ─────────────────────────────────────────────────────────────
echo ""
info "══════════════════════════════════════════════"
info " PHASE 5 — Final Summary                      "
info "══════════════════════════════════════════════"

if [ -f "$RESULTS_FILE" ]; then
    success "Benchmark results saved to:"
    echo "          ${RESULTS_FILE}"
    echo ""

    # Print a quick summary from the JSON using the venv Python
    # Pass RESULTS_FILE via env so the heredoc doesn't depend on cwd
    HCFL_RESULTS_FILE="$RESULTS_FILE" "$VENV_PYTHON" - <<'PYEOF' 2>/dev/null | tee -a "$LOG_FILE"
import json, os, sys

results_path = os.environ.get("HCFL_RESULTS_FILE", "")
if not results_path:
    sys.exit(0)

try:
    with open(results_path) as f:
        data = json.load(f)
except FileNotFoundError:
    sys.exit(0)

completed = [r for r in data if "f1" in r]
if not completed:
    print("No completed round metrics found.")
    sys.exit(0)

import statistics
f1_scores     = [r["f1"]      for r in completed]
prauc_scores  = [r["pr_auc"]  for r in completed]
rocauc_scores = [r["roc_auc"] for r in completed]
latencies     = [r["latency_sec"] for r in data]
comm_costs    = [r.get("comm_cost_mb", 0) for r in completed]

print("┌─────────────────────────────────────────────┐")
print("│          BENCHMARK SUMMARY ACROSS ALL ROUNDS │")
print("├───────────────────────────┬─────────┬────────┤")
print("│ Metric                    │  Mean   │  Best  │")
print("├───────────────────────────┼─────────┼────────┤")
print(f"│ F1 Score                  │  {statistics.mean(f1_scores):.4f} │ {max(f1_scores):.4f} │")
print(f"│ PR-AUC                    │  {statistics.mean(prauc_scores):.4f} │ {max(prauc_scores):.4f} │")
print(f"│ ROC-AUC                   │  {statistics.mean(rocauc_scores):.4f} │ {max(rocauc_scores):.4f} │")
print(f"│ E2E Latency (s)           │  {statistics.mean(latencies):.2f}  │ {min(latencies):.2f}  │")
print(f"│ Comm Cost (MB/round)      │  {statistics.mean(comm_costs):.2f}   │ {min(comm_costs):.2f}   │")
print("└───────────────────────────┴─────────┴────────┘")

sla_ok = statistics.mean(latencies) <= 120
print(f"\nLatency SLA (≤ 120s avg): {'✅ PASSED' if sla_ok else '❌ FAILED'}")
print(f"Completed rounds with metrics: {len(completed)}/{len(data)}")
PYEOF

else
    warn "benchmark_results.json not found. Check logs for errors."
fi

echo ""
echo -e "${GREEN}"
echo "╔══════════════════════════════════════════════════════╗"
echo "║          ✅  Framework Run Complete!                 ║"
echo "╠══════════════════════════════════════════════════════╣"
echo "║  Full log : ${WORKSPACE_DIR}/run_framework.log       ║"
echo "║  Results  : fl-integration/scripts/benchmark_results.json ║"
echo "╚══════════════════════════════════════════════════════╝"
echo -e "${NC}"
info "To stop all background services run: ./stop_system.sh"
info "To check service status run:         ./status.sh"
