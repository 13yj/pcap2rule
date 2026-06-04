"""YAML configuration loader with dot-notation access."""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional


class Config:
    """Nested configuration object with attribute and dict-style access."""

    def __init__(self, data: Dict[str, Any]):
        for key, value in data.items():
            if isinstance(value, dict):
                value = Config(value)
            setattr(self, key, value)

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any):
        setattr(self, key, value)

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def to_dict(self) -> Dict[str, Any]:
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Config):
                value = value.to_dict()
            result[key] = value
        return result

    def __repr__(self) -> str:
        return f"Config({self.to_dict()})"


def load_config(config_path: str, overrides: Optional[Dict[str, Any]] = None) -> Config:
    """Load a YAML configuration file with optional overrides.

    Args:
        config_path: Path to the YAML config file.
        overrides: Optional dictionary of key-value overrides (supports
                   dot-separated nested keys, e.g. 'llm.model_name').

    Returns:
        Config object with attribute-style access.
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)

    if overrides:
        for key, value in overrides.items():
            _set_nested(data, key, value)

    return Config(data)


def _set_nested(data: dict, key: str, value: Any):
    """Set a nested dict value using dot-separated key."""
    keys = key.split('.')
    for k in keys[:-1]:
        if k not in data:
            data[k] = {}
        data = data[k]
    data[keys[-1]] = value


def get_project_root() -> Path:
    """Return the absolute path to the project root directory."""
    return Path(__file__).parent.parent
