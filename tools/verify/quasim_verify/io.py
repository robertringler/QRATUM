"""I/O utilities for verification tool."""

import hashlib
import json
import os
from typing import Any

import yaml


def load_yaml(path: str) -> dict[str, Any]:
    """Load YAML file.

    Args:
        path: Path to YAML file

    Returns:
        Parsed YAML content as dictionary
    """

    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_json(path: str) -> dict[str, Any]:
    """Load JSON file.

    Args:
        path: Path to JSON file

    Returns:
        Parsed JSON content as dictionary
    """

    with open(path, encoding="utf-8") as f:
        return json.load(f)


def sha256_file(path: str) -> str:
    """Compute SHA256 hash of a file.

    Args:
        path: Path to file

    Returns:
        Hexadecimal SHA256 hash
    """

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: str, obj: dict[str, Any]) -> None:
    """Write dictionary to JSON file.

    Args:
        path: Output file path
        obj: Dictionary to serialize
    """

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
