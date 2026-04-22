#!/bin/bash
# start_system.sh - Starts the full HCFL framework

set -e

echo "========================================"
echo "    HCFL Framework Setup & Start        "
echo "========================================"

WORKSPACE_DIR=$(pwd)

# 1. Start Fabric Network
echo "[1/3] Starting Hyperledger Fabric Network..."
cd fabric-network
if [ ! -d "../fabric-samples/bin" ]; then
    echo "Fabric binaries not found in ../fabric-samples/bin. Please ensure Phase 1 setup is complete."
    exit 1
fi
# Add binaries to path so scripts can find them
export PATH=${WORKSPACE_DIR}/fabric-samples/bin:$PATH

./scripts/network.sh up
sleep 10
./scripts/createChannel.sh
./scripts/deployChaincode.sh
cd "$WORKSPACE_DIR"

# 2. Start IPFS
echo "[2/3] Starting IPFS Daemon..."
ipfs daemon > ipfs.log 2>&1 &
IPFS_PID=$!
echo $IPFS_PID > ipfs.pid
echo "IPFS started (PID: $IPFS_PID)"
sleep 3 # Give it a moment to bind ports

# 3. Start API Server
echo "[3/3] Starting REST API Server..."
if [ ! -d ".venv" ]; then
    echo "Python virtual environment (.venv) not found!"
    exit 1
fi
cd api-server
${WORKSPACE_DIR}/.venv/bin/python -m uvicorn main:app --host 0.0.0.0 --port 8000 > api_server.log 2>&1 &
API_PID=$!
echo $API_PID > ../api_server.pid
cd "$WORKSPACE_DIR"
echo "API Server started (PID: $API_PID)"
sleep 3

echo "========================================"
echo "   All systems started successfully!    "
echo "========================================"
echo "IPFS URL:  http://127.0.0.1:5001"
echo "API URL:   http://127.0.0.1:8000"
echo "Check status via: ./status.sh"
echo "Stop system via: ./stop_system.sh"
echo "========================================"
