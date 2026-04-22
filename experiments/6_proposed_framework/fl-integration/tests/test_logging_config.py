"""
Tests for structured JSON logging (Sub-task 7.3.1)
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import logging
import json
import pytest
from io import StringIO

from config.logging_config import setup_logging


def test_setup_logging_outputs_json(capsys):
    """Confirm the logging setup outputs valid JSON with required fields."""
    logger = setup_logging(bank_id="BankB", component="test_agent")
    
    # Clear any previous pytest stdout captures
    capsys.readouterr()
    
    logger.info("This is a structured log test", extra={"round_num": 5})
    
    # Capture standard output
    captured = capsys.readouterr()
    out = captured.out.strip()
    
    assert out != ""
    
    try:
        data = json.loads(out)
    except json.JSONDecodeError:
        pytest.fail(f"Output is not valid JSON: {out}")
        
    assert data["message"] == "This is a structured log test"
    assert data["bank_id"] == "BankB"
    assert data["component"] == "test_agent"
    assert data["level"] == "INFO"
    assert data["round_num"] == 5
    assert "timestamp" in data

    # Clean up the handler so it doesn't affect other tests
    logger.handlers.clear()
