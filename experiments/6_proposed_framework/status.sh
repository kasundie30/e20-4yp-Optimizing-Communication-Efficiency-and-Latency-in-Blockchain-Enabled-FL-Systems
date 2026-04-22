#!/bin/bash
# status.sh - Checks the status of HCFL framework components

echo "========================================"
echo "    HCFL Framework Status Check         "
echo "========================================"

# 1. API Server
if [ -f api_server.pid ] && kill -0 $(cat api_server.pid) 2>/dev/null; then
  echo -e "✅ API Server:    [RUNNING] (PID: $(cat api_server.pid))"
else
  # Check via pgrep as fallback
  if pgrep -f "uvicorn main:app" > /dev/null; then
      echo -e "✅ API Server:    [RUNNING] (PID: $(pgrep -f 'uvicorn main:app' | head -1))"
  else
      echo -e "❌ API Server:    [STOPPED]"
  fi
fi

# 2. IPFS Daemon
if [ -f ipfs.pid ] && kill -0 $(cat ipfs.pid) 2>/dev/null; then
  echo -e "✅ IPFS Daemon:   [RUNNING] (PID: $(cat ipfs.pid))"
else
  if pgrep -f "ipfs daemon" > /dev/null; then
      echo -e "✅ IPFS Daemon:   [RUNNING] (PID: $(pgrep -f 'ipfs daemon' | head -1))"
  else
      echo -e "❌ IPFS Daemon:   [STOPPED]"
  fi
fi

echo "----------------------------------------"
echo "Fabric Network Containers:"
echo "----------------------------------------"
CONTAINERS=$(docker ps --format "table {{.Names}}\t{{.Status}}" | grep -E "peer|orderer|couchdb|cli|chaincode")

if [ -z "$CONTAINERS" ]; then
    echo "❌ No Fabric containers running."
else
    echo "$CONTAINERS"
fi
echo "========================================"
