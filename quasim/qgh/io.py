"""I/O utilities for QGH package."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def save_tensor(tensor: np.ndarray, path: str | Path) -> None:
    """Save tensor to .npy file.

    Parameters
    ----------
    tensor : np.ndarray
        Tensor to save
    path : str or Path
        Output path
    """

    np.save(path, tensor)


def load_tensor(path: str | Path) -> np.ndarray:
    """Load tensor from .npy file.

    Parameters
    ----------
    path : str or Path
        Path to .npy file

    Returns
    -------
    np.ndarray
        Loaded tensor
    """

    return np.load(path)


def export_results_to_json(results: dict[str, Any], path: str | Path) -> None:
    """Export algorithm results to JSON.

    Handles numpy array serialization.

    Parameters
    ----------
    results : dict
        Results to export
    path : str or Path
        Output path
    """

    def serialize(obj: Any) -> Any:
        """Serialize numpy arrays and special types."""

        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        if isinstance(obj, dict):
            return {k: serialize(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [serialize(item) for item in obj]
        return obj

    serialized = serialize(results)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(serialized, f, indent=2)


def load_results_from_json(path: str | Path) -> dict[str, Any]:
    """Load results from JSON file.

    Parameters
    ----------
    path : str or Path
        Path to JSON file

    Returns
    -------
    dict[str, Any]
        Loaded results
    """

    with open(path, encoding="utf-8") as f:
        return json.load(f)
