"""Configuration loading utilities.

Supports loading from YAML, TOML, and JSON formats using OmegaConf.
"""

from __future__ import annotations

import contextlib
import json
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load configuration from YAML, TOML, or JSON file.

    Args:
        path: Path to configuration file

    Returns:
        Configuration dictionary

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config format is unsupported
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    suffix = path.suffix.lower()

    if suffix in [".yaml", ".yml"]:
        with open(path) as f:
            return yaml.safe_load(f)
    elif suffix == ".json":
        with open(path) as f:
            return json.load(f)
    elif suffix == ".toml":
        try:
            import tomli

            with open(path, "rb") as f:
                return tomli.load(f)
        except ImportError:
            # Fallback to basic parsing for simple TOML
            config = {}
            with open(path) as f:
                section = None
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if line.startswith("[") and line.endswith("]"):
                        section = line[1:-1]
                        config[section] = {}
                    elif "=" in line and section:
                        key, value = line.split("=", 1)
                        key = key.strip()
                        value = value.strip().strip("\"'")
                        # Try to parse as number
                        with contextlib.suppress(ValueError):
                            value = float(value) if "." in value else int(value)
                        config[section][key] = value
            return config
    else:
        raise ValueError(f"Unsupported config format: {suffix}")


def save_config(config: dict[str, Any], path: str | Path) -> None:
    """Save configuration to file.

    Args:
        config: Configuration dictionary
        path: Output path
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    suffix = path.suffix.lower()

    if suffix in [".yaml", ".yml"]:
        with open(path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)
    elif suffix == ".json":
        with open(path, "w") as f:
            json.dump(config, f, indent=2)
    elif suffix == ".toml":
        # Basic TOML writer
        with open(path, "w") as f:
            for section, values in config.items():
                f.write(f"[{section}]\n")
                for key, value in values.items():
                    if isinstance(value, str):
                        f.write(f'{key} = "{value}"\n')
                    else:
                        f.write(f"{key} = {value}\n")
                f.write("\n")
    else:
        raise ValueError(f"Unsupported config format: {suffix}")


def merge_configs(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge two configuration dictionaries.

    Args:
        base: Base configuration
        override: Override configuration

    Returns:
        Merged configuration
    """

    result = base.copy()

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value

    return result
