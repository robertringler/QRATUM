"""Observable extraction for TERC validation tiers.

This module extracts observables from REVULTRA and QGH algorithms that can
be consumed by TERC validation tiers for robustness testing and validation.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quasim.qgh.nonspec_algorithms import (SelfConsistencyPropagator,
                                           StabilityMonitor)
from quasim.revultra.algorithms import REVULTRAAlgorithms
from quasim.revultra.metrics import find_peaks


def beta_metrics_from_cipher(text: str) -> dict[str, float]:
    """Extract beta-like metrics from ciphertext using REVULTRA.

    These metrics can be used as synthetic state surrogates for TERC validation
    when traditional quantum observables are not available.

    Parameters
    ----------
    text : str
        Ciphertext to analyze

    Returns
    -------
    dict[str, float]
        Dictionary of beta-like observables:
        - beta_entropy: normalized entropy metric
        - beta_complexity: complexity score
        - beta_coherence: inverse of IoC variance (coherence proxy)
        - beta_periodicity: strength of detected periodicities

    Examples
    --------
    >>> metrics = beta_metrics_from_cipher("HELLO" * 20)
    >>> 'beta_entropy' in metrics and 'beta_complexity' in metrics
    True
    """
    rev = REVULTRAAlgorithms()

    complexity = rev.emergent_complexity_score(text)
    rev.index_of_coincidence_tensor(text, max_period=20)
    autocorr = rev.spectral_autocorrelation(text, max_lag=30)

    # Beta entropy: normalized character entropy
    beta_entropy = complexity["entropy"] / (np.log(26) + 1e-10)  # Max entropy for 26 letters

    # Beta complexity: normalized complexity score
    beta_complexity = complexity["score"] / 100.0

    # Beta coherence: inverse of IoC variance (higher variance = lower coherence)
    ioc_var = complexity["ioc_variance"]
    beta_coherence = 1.0 / (1.0 + ioc_var * 100)

    # Beta periodicity: strength of autocorrelation peaks
    autocorr_peaks = find_peaks(autocorr, threshold=0.3)
    beta_periodicity = len(autocorr_peaks) / 10.0 if autocorr_peaks else 0.0

    return {
        "beta_entropy": float(beta_entropy),
        "beta_complexity": float(beta_complexity),
        "beta_coherence": float(beta_coherence),
        "beta_periodicity": float(min(beta_periodicity, 1.0)),
    }


def ioc_period_candidates(text: str, max_period: int = 20, threshold: float = 0.2) -> list[int]:
    """Detect period candidates using Index of Coincidence analysis.

    Parameters
    ----------
    text : str
        Ciphertext to analyze
    max_period : int, optional
        Maximum period to test (default: 20)
    threshold : float, optional
        Peak detection threshold (default: 0.2)

    Returns
    -------
    list[int]
        List of candidate period lengths (1-indexed)

    Examples
    --------
    >>> candidates = ioc_period_candidates("ABCABC" * 10, max_period=10)
    >>> 3 in candidates  # Period of 3 expected
    True
    """
    rev = REVULTRAAlgorithms()
    ioc = rev.index_of_coincidence_tensor(text, max_period=max_period)

    peaks = find_peaks(ioc, threshold=threshold)

    # Convert from 0-indexed to 1-indexed periods
    return [p + 1 for p in peaks]


def emergent_complexity(text: str) -> dict[str, float]:
    """Compute emergent complexity observables for TERC validation.

    Parameters
    ----------
    text : str
        Ciphertext to analyze

    Returns
    -------
    dict[str, float]
        Complexity metrics including score and components

    Examples
    --------
    >>> complexity = emergent_complexity("ATTACK" * 15)
    >>> 'score' in complexity and 'entropy' in complexity
    True
    """
    rev = REVULTRAAlgorithms()
    return rev.emergent_complexity_score(text)


def qgh_consensus_status(
    node_states: np.ndarray, damping: float = 0.5, max_iterations: int = 100
) -> dict[str, Any]:
    """Extract consensus status from QGH propagation for TERC robustness tests.

    This provides metrics on convergence, stability, and consensus quality
    that can feed into TERC Tier-1 and Tier-5 validation.

    Parameters
    ----------
    node_states : np.ndarray
        Initial node states, shape (num_nodes, state_dim)
    damping : float, optional
        Damping factor (default: 0.5)
    max_iterations : int, optional
        Maximum iterations (default: 100)

    Returns
    -------
    dict[str, Any]
        Consensus metrics:
        - converged: bool indicating convergence
        - iterations: number of iterations to converge
        - final_variance: variance in final states
        - stability: stability score
        - robustness: robustness score (inverse of iterations / max)

    Examples
    --------
    >>> import numpy as np
    >>> states = np.random.rand(5, 3)
    >>> status = qgh_consensus_status(states)
    >>> 'converged' in status and 'stability' in status
    True
    """
    num_nodes = node_states.shape[0]

    # Propagate to consensus
    propagator = SelfConsistencyPropagator(
        num_nodes=num_nodes, damping=damping, max_iterations=max_iterations
    )
    result = propagator.propagate(node_states)

    # Compute variance in final states
    final_variance = float(np.mean(np.var(result["states"], axis=0)))

    # Stability metric: low variance indicates stability
    stability = 1.0 / (1.0 + final_variance)

    # Robustness: converged quickly = more robust
    robustness = 1.0 - (result["iterations"] / max_iterations)

    return {
        "converged": result["converged"],
        "iterations": result["iterations"],
        "final_variance": final_variance,
        "stability": float(stability),
        "robustness": float(robustness),
    }


def stream_synchronization_metrics(
    stream_data: dict[int, list[float]], threshold: float = 0.7
) -> dict[str, Any]:
    """Extract synchronization metrics from distributed streams.

    Parameters
    ----------
    stream_data : dict[int, list[float]]
        Dictionary mapping stream IDs to data lists
    threshold : float, optional
        Synchronization correlation threshold (default: 0.7)

    Returns
    -------
    dict[str, Any]
        Synchronization metrics including sync_pairs and sync_ratio
    """
    from quasim.qgh.nonspec_algorithms import DistributedStreamMonitor

    num_streams = len(stream_data)
    monitor = DistributedStreamMonitor(num_streams=num_streams, buffer_size=10000)

    # Add data to monitor
    for stream_id, data in stream_data.items():
        for value in data:
            monitor.add_sample(stream_id, value)

    # Detect synchronization
    sync_pairs = monitor.detect_sync_patterns(threshold=threshold)
    max_pairs = num_streams * (num_streams - 1) // 2
    sync_ratio = len(sync_pairs) / max(max_pairs, 1)

    return {
        "num_streams": num_streams,
        "sync_pairs": sync_pairs,
        "sync_ratio": float(sync_ratio),
        "synchronized": sync_ratio > 0.5,
    }


def stability_assessment(metrics: list[float], window_size: int = 50) -> dict[str, Any]:
    """Assess stability of a metric time series.

    Parameters
    ----------
    metrics : list[float]
        Time series of metric values
    window_size : int, optional
        Window size for stability check (default: 50)

    Returns
    -------
    dict[str, Any]
        Stability assessment including is_stable, trend, and stats
    """
    monitor = StabilityMonitor(window_size=window_size, threshold=2.0)

    for value in metrics:
        monitor.add_metric(value)

    is_stable = monitor.is_stable()
    stats = monitor.get_stats()

    return {
        "is_stable": is_stable,
        "mean": stats["mean"],
        "std": stats["std"],
        "trend": stats["trend"],
        "assessment": "stable" if is_stable else "unstable",
    }
