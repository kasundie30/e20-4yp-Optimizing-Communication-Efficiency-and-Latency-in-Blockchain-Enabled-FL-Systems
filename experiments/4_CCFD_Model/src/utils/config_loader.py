# src/utils/config_loader.py
# Loads configuration from YAML files

import yaml

def load_config(path: str = "config/config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}