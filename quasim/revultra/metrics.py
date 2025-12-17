"""Shared metrics utilities for REVULTRA package."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def find_peaks(array: NDArray[np.float64], threshold: float = 0.1) -> list[int]:
    """Find peaks in an array above a threshold.

    Parameters
    ----------
    array : NDArray[np.float64]
        Input array
    threshold : float, optional
        Minimum relative height for peaks (default: 0.1)

    Returns
    -------
    list[int]
        Indices of detected peaks

    Examples
    --------
    >>> import numpy as np
    >>> arr = np.array([0.1, 0.8, 0.2, 0.9, 0.1])
    >>> peaks = find_peaks(arr, threshold=0.5)
    >>> 1 in peaks and 3 in peaks
    True
    """

    if len(array) < 3:
        return []

    peaks = []
    max_val = np.max(array)
    threshold_val = threshold * max_val

    for i in range(1, len(array) - 1):
        if array[i] > threshold_val and array[i] > array[i - 1] and array[i] > array[i + 1]:
            peaks.append(i)

    return peaks


def normalize_array(array: NDArray[np.float64], method: str = "minmax") -> NDArray[np.float64]:
    """Normalize array values.

    Parameters
    ----------
    array : NDArray[np.float64]
        Input array
    method : str, optional
        Normalization method: 'minmax' or 'zscore' (default: 'minmax')

    Returns
    -------
    NDArray[np.float64]
        Normalized array
    """

    if method == "minmax":
        min_val, max_val = np.min(array), np.max(array)
        if max_val - min_val < 1e-10:
            return np.zeros_like(array)
        return (array - min_val) / (max_val - min_val)
    elif method == "zscore":
        mean, std = np.mean(array), np.std(array)
        if std < 1e-10:
            return np.zeros_like(array)
        return (array - mean) / std
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def compute_entropy(probabilities: NDArray[np.float64]) -> float:
    """Compute Shannon entropy from probability distribution.

    Parameters
    ----------
    probabilities : NDArray[np.float64]
        Probability distribution (should sum to 1.0)

    Returns
    -------
    float
        Entropy in bits

    Examples
    --------
    >>> import numpy as np
    >>> probs = np.array([0.5, 0.5])
    >>> abs(compute_entropy(probs) - 1.0) < 0.01
    True
    """

    # Filter out zeros to avoid log(0)
    probs = probabilities[probabilities > 0]
    return float(-np.sum(probs * np.log2(probs)))


def sliding_window_stats(
    array: NDArray[np.float64], window_size: int
) -> dict[str, NDArray[np.float64]]:
    """Compute statistics over sliding windows.

    Parameters
    ----------
    array : NDArray[np.float64]
        Input array
    window_size : int
        Size of sliding window

    Returns
    -------
    dict[str, NDArray[np.float64]]
        Dictionary with 'mean', 'std', 'min', 'max' arrays
    """

    if len(array) < window_size:
        return {
            "mean": np.array([]),
            "std": np.array([]),
            "min": np.array([]),
            "max": np.array([]),
        }

    n = len(array) - window_size + 1
    means = np.zeros(n)
    stds = np.zeros(n)
    mins = np.zeros(n)
    maxs = np.zeros(n)

    for i in range(n):
        window = array[i : i + window_size]
        means[i] = np.mean(window)
        stds[i] = np.std(window)
        mins[i] = np.min(window)
        maxs[i] = np.max(window)

    return {"mean": means, "std": stds, "min": mins, "max": maxs}
