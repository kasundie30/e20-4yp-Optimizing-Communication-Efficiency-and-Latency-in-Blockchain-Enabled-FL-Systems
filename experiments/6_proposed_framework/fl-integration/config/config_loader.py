import os
from functools import lru_cache
from pydantic import BaseModel, Field
import yaml


class FLConfigSection(BaseModel):
    epochs: int = Field(gt=0)
    learning_rate: float = Field(gt=0.0)
    l2_norm_clip: float = Field(ge=0.0)
    noise_multiplier: float = Field(ge=0.0)
    deadline_seconds: float = Field(gt=0.0)
    min_branches_required: int = Field(gt=0)
    validation_threshold: float = Field(ge=0.0)
    backup_beta: float = Field(ge=0.0, le=1.0)

class BlockchainConfigSection(BaseModel):
    api_url: str = Field(min_length=1)
    bank_id: str = Field(min_length=1)
    round_timeout_seconds: float = Field(gt=0.0)

class IPFSConfigSection(BaseModel):
    api_url: str = Field(min_length=1)

class DataConfigSection(BaseModel):
    base_path: str = Field(min_length=1)
    partition_prefix: str = Field(min_length=1)

class FLConfig(BaseModel):
    fl: FLConfigSection
    blockchain: BlockchainConfigSection
    ipfs: IPFSConfigSection
    data: DataConfigSection

class ConfigValidationError(Exception):
    """Raised when config is missing or invalid."""
    pass

@lru_cache(maxsize=1)
def load_config(config_path: str = None) -> FLConfig:
    """Loads and validates the configuration from YAML."""
    if config_path is None:
        # Default config path relative to this script
        base_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(base_dir, "fl_config.yaml")

    if not os.path.exists(config_path):
        raise ConfigValidationError(f"Configuration file not found at {config_path}")

    with open(config_path, "r") as f:
        try:
            raw_data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ConfigValidationError(f"Invalid YAML format: {e}")
            
    if not raw_data:
        raise ConfigValidationError("Configuration file is empty")

    try:
        return FLConfig(**raw_data)
    except Exception as e:
        # pydantic ValidationError gives detailed missing/wrong type info
        raise ConfigValidationError(f"Config schema validation failed: {str(e)}")
