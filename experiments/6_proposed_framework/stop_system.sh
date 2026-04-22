#!/bin/bash
# stop_system.sh - Tears down the HCFL framework

echo "========================================"
echo "    HCFL Framework Teardown             "
echo "========================================"

WORKSPACE_DIR=$(pwd)

# 1. Stop API Server
echo "[1/3] Stopping API Server..."
if [ -f "api_server.pid" ]; then
    PID=$(cat api_server.pid)
    kill $PID 2>/dev/null || true
    rm api_server.pid
    echo "API Server stopped."
else
    echo "API Server PID file not found. Skipping."
    # Fallback kill
    pkill -f "uvicorn main:app" || true
fi

# 2. Stop IPFS
echo "[2/3] Stopping IPFS Daemon..."
if [ -f "ipfs.pid" ]; then
    PID=$(cat ipfs.pid)
    kill $PID 2>/dev/null || true
    rm ipfs.pid
    echo "IPFS Daemon stopped."
else
    echo "IPFS PID file not found. Skipping."
    pkill -f "ipfs daemon" || true
fi

# 3. Stop Fabric Network
echo "[3/3] Taking down Hyperledger Fabric Network..."
cd fabric-network
if [ -x "scripts/network.sh" ]; then
    ./scripts/network.sh down
else
    echo "Cannot find fabric-network/scripts/network.sh!"
fi
cd "$WORKSPACE_DIR"

echo "========================================"
echo "       System Teardown Complete         "
echo "========================================"
