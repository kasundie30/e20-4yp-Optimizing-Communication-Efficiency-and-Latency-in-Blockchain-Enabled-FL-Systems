"""
config.py — Per-bank Fabric configuration for the API server.

Each bank's HQ is identified by its MSPID, peer address, and certificate paths.
The server reads the BANK_ID env variable at startup to load the correct config.
"""
import os
from pathlib import Path

FABRIC_NET = Path("/media/fyp-group-18/1TB-Hard/FYP-Group18/experiments/6_proposed_framework/fabric-network")
CRYPTO = FABRIC_NET / "crypto-config"
FABRIC_BIN = Path("/media/fyp-group-18/1TB-Hard/FYP-Group18/experiments/6_proposed_framework/fabric-samples/bin")
FABRIC_CFG = Path("/media/fyp-group-18/1TB-Hard/FYP-Group18/experiments/6_proposed_framework/fabric-samples/config")

CHANNEL_NAME = "fraud-detection-global"
CHAINCODE_NAME = "cbft-fl"

ORDERER = "orderer0.orderer.fabricfl.com:7050"
ORDERER_CA = str(
    CRYPTO / "ordererOrganizations/orderer.fabricfl.com/orderers/orderer0.orderer.fabricfl.com/msp/tlscacerts/tlsca.orderer.fabricfl.com-cert.pem"
)

BANK_CONFIGS = {
    "BankA": {
        "msp_id": "BankAMSP",
        "peer": "peer0.banka.fabricfl.com:7051",
        "tls_root_cert": str(CRYPTO / "peerOrganizations/banka.fabricfl.com/peers/peer0.banka.fabricfl.com/tls/ca.crt"),
        "msp_config": str(CRYPTO / "peerOrganizations/banka.fabricfl.com/users/Admin@banka.fabricfl.com/msp"),
    },
    "BankB": {
        "msp_id": "BankBMSP",
        "peer": "peer0.bankb.fabricfl.com:8051",
        "tls_root_cert": str(CRYPTO / "peerOrganizations/bankb.fabricfl.com/peers/peer0.bankb.fabricfl.com/tls/ca.crt"),
        "msp_config": str(CRYPTO / "peerOrganizations/bankb.fabricfl.com/users/Admin@bankb.fabricfl.com/msp"),
    },
    "BankC": {
        "msp_id": "BankCMSP",
        "peer": "peer0.bankc.fabricfl.com:9051",
        "tls_root_cert": str(CRYPTO / "peerOrganizations/bankc.fabricfl.com/peers/peer0.bankc.fabricfl.com/tls/ca.crt"),
        "msp_config": str(CRYPTO / "peerOrganizations/bankc.fabricfl.com/users/Admin@bankc.fabricfl.com/msp"),
    },
}

# Default bank: can be overridden by BANK_ID env var
DEFAULT_BANK = os.getenv("BANK_ID", "BankA")
