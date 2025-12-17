"""Artifact serialization utilities.

Provides functions to save simulation results in JSONL and NPZ formats.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def save_jsonl(data: list[dict[str, Any]], path: str | Path) -> None:
    """Save data as JSON Lines format.

    Args:
        data: List of dictionaries to save
        path: Output file path
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        for item in data:
            f.write(json.dumps(item, default=_json_default) + "\n")


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load data from JSON Lines format.

    Args:
        path: Input file path

    Returns:
        List of dictionaries
    """

    path = Path(path)
    data = []

    with open(path) as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))

    return data


def save_npz(data: dict[str, np.ndarray], path: str | Path) -> None:
    """Save numpy arrays as compressed NPZ.

    Args:
        data: Dictionary of arrays to save
        path: Output file path
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **data)


def load_npz(path: str | Path) -> dict[str, np.ndarray]:
    """Load numpy arrays from NPZ file.

    Args:
        path: Input file path

    Returns:
        Dictionary of arrays
    """

    path = Path(path)
    data = np.load(path)
    return {key: data[key] for key in data.files}


def save_metrics(metrics: dict[str, Any], path: str | Path) -> None:
    """Save metrics dictionary as JSON.

    Args:
        metrics: Metrics dictionary
        path: Output file path
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        json.dump(metrics, f, indent=2, default=_json_default)


def load_metrics(path: str | Path) -> dict[str, Any]:
    """Load metrics from JSON file.

    Args:
        path: Input file path

    Returns:
        Metrics dictionary
    """

    path = Path(path)

    with open(path) as f:
        return json.load(f)


def _json_default(obj: Any) -> Any:
    """JSON serialization helper for numpy types.

    Args:
        obj: Object to serialize

    Returns:
        Serializable representation
    """

    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif hasattr(obj, "__dict__"):
        return obj.__dict__
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
