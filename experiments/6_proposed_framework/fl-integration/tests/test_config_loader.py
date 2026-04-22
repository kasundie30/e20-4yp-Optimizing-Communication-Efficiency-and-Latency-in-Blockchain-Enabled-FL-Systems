"""
Tests for unified configuration system (Sub-task 7.1.1 and 7.1.2)
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from unittest.mock import patch, mock_open
import yaml

from config.config_loader import load_config, FLConfig, ConfigValidationError


VALID_YAML = """
fl:
  epochs: 2
  learning_rate: 0.05
  l2_norm_clip: 1.0
  noise_multiplier: 0.1
  deadline_seconds: 10.0
  min_branches_required: 2
  validation_threshold: 0.20
  backup_beta: 0.3
blockchain:
  api_url: "http://test:8000"
  bank_id: "BankA"
  round_timeout_seconds: 120.0
ipfs:
  api_url: "http://localhost:5001"
data:
  base_path: "/tmp/data"
  partition_prefix: "branch"
"""


INVALID_MISSING_KEY_YAML = """
fl:
  epochs: 2
  learning_rate: 0.05
  l2_norm_clip: 1.0
  noise_multiplier: 0.1
  deadline_seconds: 10.0
  min_branches_required: 2
  validation_threshold: 0.20
  backup_beta: 0.3
blockchain:
  api_url: "http://test:8000"
  # bank_id is deliberately missing
  round_timeout_seconds: 120.0
ipfs:
  api_url: "http://localhost:5001"
data:
  base_path: "/tmp/data"
  partition_prefix: "branch"
"""


INVALID_TYPE_YAML = """
fl:
  epochs: "not an int"
  learning_rate: 0.05
  l2_norm_clip: 1.0
  noise_multiplier: 0.1
  deadline_seconds: 10.0
  min_branches_required: 2
  validation_threshold: 0.20
  backup_beta: 0.3
blockchain:
  api_url: "http://test:8000"
  bank_id: "BankA"
  round_timeout_seconds: 120.0
ipfs:
  api_url: "http://localhost:5001"
data:
  base_path: "/tmp/data"
  partition_prefix: "branch"
"""


def test_valid_config_loading():
    """Test loading a fully valid YAML outputs correct FLConfig."""
    load_config.cache_clear()
    with patch("builtins.open", mock_open(read_data=VALID_YAML)), \
         patch("os.path.exists", return_value=True):
        config = load_config("/fake/path.yaml")
    
    assert isinstance(config, FLConfig)
    assert config.fl.epochs == 2
    assert config.blockchain.bank_id == "BankA"
    assert config.ipfs.api_url == "http://localhost:5001"


def test_missing_key_raises_validation_error():
    load_config.cache_clear()
    with patch("builtins.open", mock_open(read_data=INVALID_MISSING_KEY_YAML)), \
         patch("os.path.exists", return_value=True):
        with pytest.raises(ConfigValidationError) as exc:
            load_config("/fake/path.yaml")
        assert "bank_id" in str(exc.value)


def test_wrong_type_raises_validation_error():
    load_config.cache_clear()
    with patch("builtins.open", mock_open(read_data=INVALID_TYPE_YAML)), \
         patch("os.path.exists", return_value=True):
        with pytest.raises(ConfigValidationError) as exc:
            load_config("/fake/path.yaml")
        assert "epochs" in str(exc.value)


def test_config_loader_is_idempotent():
    load_config.cache_clear()
    with patch("builtins.open", mock_open(read_data=VALID_YAML)) as m_open, \
         patch("os.path.exists", return_value=True):
        
        c1 = load_config("/fake/path.yaml")
        c2 = load_config("/fake/path.yaml")
        
        # Idempotency achieved via @lru_cache(maxsize=1)
        assert c1 is c2
        m_open.assert_called_once()
