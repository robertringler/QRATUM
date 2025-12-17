"""I/O utilities for REVULTRA package."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


def load_ciphertext(path: str | Path) -> str:
    """Load ciphertext from file.

    Parameters
    ----------
    path : str or Path
        Path to ciphertext file

    Returns
    -------
    str
        Loaded ciphertext
    """

    with open(path, encoding="utf-8") as f:
        return f.read()


def save_ciphertext(text: str, path: str | Path) -> None:
    """Save ciphertext to file.

    Parameters
    ----------
    text : str
        Ciphertext to save
    path : str or Path
        Output path
    """

    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def export_to_json(data: dict[str, Any], path: str | Path) -> None:
    """Export analysis results to JSON.

    Handles numpy array serialization automatically.

    Parameters
    ----------
    data : dict
        Data to export
    path : str or Path
        Output path
    """

    def serialize(obj: Any) -> Any:
        """Serialize numpy arrays to lists."""

        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        if isinstance(obj, complex):
            return {"real": obj.real, "imag": obj.imag}
        if isinstance(obj, dict):
            return {k: serialize(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [serialize(item) for item in obj]
        return obj

    serialized = serialize(data)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(serialized, f, indent=2)


def export_to_csv(data: dict[str, float], path: str | Path) -> None:
    """Export simple key-value data to CSV.

    Parameters
    ----------
    data : dict[str, float]
        Key-value pairs to export
    path : str or Path
        Output path
    """

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["key", "value"])
        for key, value in data.items():
            writer.writerow([key, value])


def save_numpy_array(array: np.ndarray, path: str | Path) -> None:
    """Save numpy array to .npy file.

    Parameters
    ----------
    array : np.ndarray
        Array to save
    path : str or Path
        Output path
    """

    np.save(path, array)


def load_numpy_array(path: str | Path) -> np.ndarray:
    """Load numpy array from .npy file.

    Parameters
    ----------
    path : str or Path
        Path to .npy file

    Returns
    -------
    np.ndarray
        Loaded array
    """

    return np.load(path)
