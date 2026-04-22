"""
fabric_client.py — Fabric network client using subprocess calls to the peer CLI.

Since the Fabric Python Gateway SDK doesn't yet have a stable PyPI package for
Fabric 2.5, we use subprocess invocations of the `peer` CLI binary. This is
the most reliable approach for the current Fabric 2.5 + channel participation setup.
"""
import json
import logging
import os
import re
import subprocess
from typing import Any, Optional

from config import (
    BANK_CONFIGS,
    CHAINCODE_NAME,
    CHANNEL_NAME,
    DEFAULT_BANK,
    FABRIC_BIN,
    FABRIC_CFG,
    ORDERER,
    ORDERER_CA,
)

logger = logging.getLogger("fabric_client")


class FabricError(Exception):
    """Raised when the peer CLI command fails."""


def _build_env(bank_id: str) -> dict:
    """Build environment variables for a peer CLI call for the given bank."""
    cfg = BANK_CONFIGS[bank_id]
    env = os.environ.copy()
    env["PATH"] = f"{FABRIC_BIN}:{env.get('PATH', '')}"
    env["FABRIC_CFG_PATH"] = str(FABRIC_CFG)
    env["CORE_PEER_LOCALMSPID"] = cfg["msp_id"]
    env["CORE_PEER_ADDRESS"] = cfg["peer"]
    env["CORE_PEER_TLS_ENABLED"] = "true"
    env["CORE_PEER_TLS_ROOTCERT_FILE"] = cfg["tls_root_cert"]
    env["CORE_PEER_MSPCONFIGPATH"] = cfg["msp_config"]
    return env


def _run(cmd: list[str], env: dict) -> str:
    """Run a peer CLI command and return stdout. Raises FabricError on failure."""
    logger.debug("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=30)
    if result.returncode != 0:
        raise FabricError(result.stderr.strip() or result.stdout.strip())
    return result.stdout.strip()


def _extract_tx_id(output: str) -> Optional[str]:
    """Try to extract a transaction ID from peer invoke output (stderr-like logs)."""
    m = re.search(r"txid\[([a-f0-9]{64})\]", output)
    if m:
        return m.group(1)
    return None


def invoke(
    bank_id: str,
    function: str,
    args: list[str],
    endorsing_peers: Optional[list[str]] = None,
) -> dict[str, Any]:
    """
    Invoke a chaincode transaction on the live network.
    Returns a dict with tx_id and status.
    """
    if bank_id not in BANK_CONFIGS:
        raise ValueError(f"Unknown bank_id: {bank_id}")

    env = _build_env(bank_id)
    cfg = BANK_CONFIGS[bank_id]

    # Build endorsing peer flags — always include caller's peer
    peer_args: list[str] = []
    if endorsing_peers:
        for ep_bank in endorsing_peers:
            ep_cfg = BANK_CONFIGS[ep_bank]
            peer_args += ["--peerAddresses", ep_cfg["peer"],
                          "--tlsRootCertFiles", ep_cfg["tls_root_cert"]]
    else:
        # By default endorse on all 3 banks for safety
        for bk, bc in BANK_CONFIGS.items():
            peer_args += ["--peerAddresses", bc["peer"],
                          "--tlsRootCertFiles", bc["tls_root_cert"]]

    payload = json.dumps({"function": function, "Args": args})
    cmd = [
        "peer", "chaincode", "invoke",
        "--channelID", CHANNEL_NAME,
        "--name", CHAINCODE_NAME,
        "--tls", "--cafile", ORDERER_CA,
        "--orderer", ORDERER,
        *peer_args,
        "-c", payload,
    ]

    output = _run(cmd, env)
    logger.info("[invoke] bank=%s fn=%s → %s", bank_id, function, output)
    tx_id = _extract_tx_id(output)
    return {"status": "success", "tx_id": tx_id, "output": output}


def query(bank_id: str, function: str, args: list[str]) -> str:
    """
    Query the chaincode (read-only). Returns the raw JSON string from the peer.
    """
    if bank_id not in BANK_CONFIGS:
        raise ValueError(f"Unknown bank_id: {bank_id}")

    env = _build_env(bank_id)

    payload = json.dumps({"function": function, "Args": args})
    cmd = [
        "peer", "chaincode", "query",
        "--channelID", CHANNEL_NAME,
        "--name", CHAINCODE_NAME,
        "-c", payload,
    ]
    output = _run(cmd, env)
    logger.info("[query] bank=%s fn=%s → %s", bank_id, function, output)
    return output
