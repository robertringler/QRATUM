"""Metrics for model evaluation."""

import time
from typing import Any

import numpy as np
from numpy.typing import NDArray


def accuracy(y_true: NDArray, y_pred: NDArray) -> float:
    """Compute classification accuracy.

    Parameters
    ----------
    y_true : NDArray
        True labels
    y_pred : NDArray
        Predicted labels

    Returns
    -------
    float
        Accuracy score
    """
    return float(np.mean(y_true == y_pred))


def mae(y_true: NDArray[np.float32], y_pred: NDArray[np.float32]) -> float:
    """Compute Mean Absolute Error.

    Parameters
    ----------
    y_true : NDArray
        True values
    y_pred : NDArray
        Predicted values

    Returns
    -------
    float
        MAE score
    """
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: NDArray[np.float32], y_pred: NDArray[np.float32]) -> float:
    """Compute Root Mean Squared Error.

    Parameters
    ----------
    y_true : NDArray
        True values
    y_pred : NDArray
        Predicted values

    Returns
    -------
    float
        RMSE score
    """
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def f1_score(y_true: NDArray, y_pred: NDArray, average: str = "weighted") -> float:
    """Compute F1 score.

    Parameters
    ----------
    y_true : NDArray
        True labels
    y_pred : NDArray
        Predicted labels
    average : str
        Averaging method (default: 'weighted')

    Returns
    -------
    float
        F1 score
    """
    from sklearn.metrics import f1_score as sklearn_f1

    return float(sklearn_f1(y_true, y_pred, average=average, zero_division=0))


def measure_latency(model: Any, X: Any, n_runs: int = 100) -> dict[str, float]:
    """Measure model inference latency.

    Parameters
    ----------
    model : Any
        Model with predict() method
    X : Any
        Input data
    n_runs : int
        Number of runs for measurement

    Returns
    -------
    dict[str, float]
        Dictionary with p50 and p95 latencies in milliseconds
    """
    latencies = []

    for _ in range(n_runs):
        start = time.perf_counter()
        _ = model.predict(X)
        end = time.perf_counter()
        latencies.append((end - start) * 1000)  # Convert to ms

    latencies = np.array(latencies)

    return {
        "p50_ms": float(np.percentile(latencies, 50)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "mean_ms": float(np.mean(latencies)),
    }


def measure_throughput(model: Any, X: Any, duration_sec: float = 1.0) -> float:
    """Measure model throughput (predictions per second).

    Parameters
    ----------
    model : Any
        Model with predict() method
    X : Any
        Input data
    duration_sec : float
        Duration to measure throughput

    Returns
    -------
    float
        Throughput in predictions per second
    """
    start = time.perf_counter()
    count = 0

    while time.perf_counter() - start < duration_sec:
        _ = model.predict(X)
        count += 1

    elapsed = time.perf_counter() - start
    return count / elapsed


def estimate_model_size_mb(model: Any) -> float:
    """Estimate model size in megabytes.

    Parameters
    ----------
    model : Any
        Model object

    Returns
    -------
    float
        Estimated size in MB
    """
    import sys

    size_bytes = sys.getsizeof(model)

    # Try to get more accurate size for sklearn models
    if hasattr(model, "__dict__"):
        for attr in model.__dict__.values():
            if isinstance(attr, np.ndarray):
                size_bytes += attr.nbytes
            else:
                size_bytes += sys.getsizeof(attr)

    return size_bytes / (1024 * 1024)


def estimate_energy_proxy(latency_ms: float, throughput: float = None) -> float:
    """Estimate energy consumption proxy based on CPU time.

    This is a simple heuristic: energy ~ latency × TDP_estimate

    Parameters
    ----------
    latency_ms : float
        Inference latency in milliseconds
    throughput : float, optional
        Throughput (not currently used)

    Returns
    -------
    float
        Energy proxy in arbitrary units
    """
    # Assume typical CPU TDP of 65W
    tdp_watts = 65.0

    # Energy = Power × Time
    # Convert latency to seconds
    time_sec = latency_ms / 1000.0

    # Energy in Joules
    energy_joules = tdp_watts * time_sec

    return energy_joules


def compute_stability_margin(scores: list[float]) -> float:
    """Compute stability margin (1 - CV) across repeated runs.

    Parameters
    ----------
    scores : list[float]
        List of scores from repeated runs

    Returns
    -------
    float
        Stability margin (higher is more stable)
    """
    if len(scores) < 2:
        return 1.0

    scores_array = np.array(scores)
    mean = np.mean(scores_array)
    std = np.std(scores_array)

    if mean < 1e-10:
        return 0.0

    cv = std / mean  # Coefficient of variation
    stability = 1.0 - min(cv, 1.0)

    return float(stability)
