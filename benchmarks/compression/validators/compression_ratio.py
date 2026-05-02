"""Compression ratio computation and validation.

This module provides utilities to compute compression metrics and statistics.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.complex128]


def compute_compression_metrics(
    original_size: int, compressed_size: int, metadata: dict[str, Any] | None = None
) -> dict[str, float]:
    """Compute comprehensive compression metrics.

    Args:
        original_size: Size of original state (number of complex elements)
        compressed_size: Size of compressed representation (number of stored values)
        metadata: Optional additional metadata from compression algorithm

    Returns:
        Dictionary containing:
            - compression_ratio: Ratio of original to compressed size
            - space_savings: Percentage of space saved
            - bits_per_element_original: Original bits per complex element
            - bits_per_element_compressed: Compressed bits per element

    Example:
        >>> metrics = compute_compression_metrics(1024, 64)
        >>> assert metrics["compression_ratio"] == 16.0
        >>> assert metrics["space_savings"] == 93.75
    """

    if compressed_size <= 0:
        raise ValueError("Compressed size must be positive")
    if original_size <= 0:
        raise ValueError("Original size must be positive")

    compression_ratio = float(original_size) / float(compressed_size)
    space_savings = 100.0 * (1.0 - float(compressed_size) / float(original_size))

    # Each complex number is 16 bytes (2 float64)
    bytes_per_complex = 16
    bits_per_element_original = 8 * bytes_per_complex
    bits_per_element_compressed = bits_per_element_original / compression_ratio

    metrics = {
        "compression_ratio": compression_ratio,
        "space_savings": space_savings,
        "bits_per_element_original": bits_per_element_original,
        "bits_per_element_compressed": bits_per_element_compressed,
        "original_size": original_size,
        "compressed_size": compressed_size,
    }

    # Add metadata if provided
    if metadata:
        for key, value in metadata.items():
            if key not in metrics:
                metrics[key] = value

    return metrics


def aggregate_compression_statistics(
    compression_ratios: list[float],
) -> dict[str, float]:
    """Compute aggregate statistics over multiple compression runs.

    Args:
        compression_ratios: List of compression ratios from multiple test cases

    Returns:
        Dictionary with statistical measures:
            - min, max, mean, median, std
            - percentiles (25th, 75th, 90th, 95th)

    Example:
        >>> ratios = [10.0, 20.0, 30.0, 40.0, 50.0]
        >>> stats = aggregate_compression_statistics(ratios)
        >>> assert stats["mean"] == 30.0
        >>> assert stats["median"] == 30.0
    """

    if not compression_ratios:
        raise ValueError("Empty compression ratio list")

    ratios_array = np.array(compression_ratios)

    statistics = {
        "min": float(np.min(ratios_array)),
        "max": float(np.max(ratios_array)),
        "mean": float(np.mean(ratios_array)),
        "median": float(np.median(ratios_array)),
        "std": float(np.std(ratios_array)),
        "percentile_25": float(np.percentile(ratios_array, 25)),
        "percentile_75": float(np.percentile(ratios_array, 75)),
        "percentile_90": float(np.percentile(ratios_array, 90)),
        "percentile_95": float(np.percentile(ratios_array, 95)),
        "count": len(compression_ratios),
    }

    return statistics


def compute_compression_efficiency(
    compression_ratio: float, fidelity: float, runtime_seconds: float
) -> dict[str, float]:
    """Compute compression efficiency metrics.

    Efficiency considers both compression ratio and quality (fidelity).

    Args:
        compression_ratio: Achieved compression ratio
        fidelity: Achieved fidelity (0 to 1)
        runtime_seconds: Time taken to compress in seconds

    Returns:
        Dictionary with efficiency metrics:
            - efficiency_score: Weighted score combining ratio and fidelity
            - throughput: Compression ratio per second
            - fidelity_adjusted_ratio: Ratio weighted by fidelity

    Example:
        >>> efficiency = compute_compression_efficiency(20.0, 0.995, 2.0)
        >>> assert efficiency["throughput"] == 10.0
    """

    # Efficiency score: geometric mean of ratio and fidelity
    # Scale fidelity to similar range as compression ratio
    fidelity_scaled = fidelity * 100  # Scale 0.99 -> 99

    if compression_ratio > 0 and fidelity > 0:
        efficiency_score = float(np.sqrt(compression_ratio * fidelity_scaled))
    else:
        efficiency_score = 0.0

    throughput = compression_ratio / max(runtime_seconds, 1e-6)
    fidelity_adjusted_ratio = compression_ratio * fidelity

    return {
        "efficiency_score": efficiency_score,
        "throughput": throughput,
        "fidelity_adjusted_ratio": fidelity_adjusted_ratio,
    }


def format_compression_ratio(ratio: float, precision: int = 2) -> str:
    """Format compression ratio for display.

    Args:
        ratio: Compression ratio value
        precision: Number of decimal places

    Returns:
        Formatted string like "24.7×"

    Example:
        >>> formatted = format_compression_ratio(24.73)
        >>> assert formatted == "24.73×"
    """

    return f"{ratio:.{precision}f}×"


def validate_compression_claim(
    measured_ratio: float,
    claimed_min: float,
    claimed_max: float,
    claimed_typical: float | None = None,
) -> dict[str, Any]:
    """Validate measured compression ratio against documented claims.

    Args:
        measured_ratio: Actual measured compression ratio
        claimed_min: Claimed minimum ratio
        claimed_max: Claimed maximum ratio
        claimed_typical: Claimed typical ratio (optional)

    Returns:
        Validation result dictionary with:
            - in_range: Boolean indicating if measured is in claimed range
            - meets_min: Boolean indicating if measured >= claimed_min
            - meets_typical: Boolean (if typical provided)
            - status: String status ("VALIDATED", "WITHIN_RANGE", "BELOW_MIN", etc.)

    Example:
        >>> result = validate_compression_claim(25.0, 10.0, 50.0, 30.0)
        >>> assert result["in_range"]
        >>> assert result["status"] == "WITHIN_RANGE"
    """

    in_range = claimed_min <= measured_ratio <= claimed_max
    meets_min = measured_ratio >= claimed_min
    meets_max = measured_ratio <= claimed_max

    # Determine status
    if in_range and claimed_typical is not None and measured_ratio >= claimed_typical:
        status = "VALIDATED"
    elif in_range:
        status = "WITHIN_RANGE"
    elif measured_ratio < claimed_min:
        status = "BELOW_MIN"
    else:
        status = "EXCEEDS_MAX"

    result = {
        "measured": measured_ratio,
        "claimed_min": claimed_min,
        "claimed_max": claimed_max,
        "in_range": in_range,
        "meets_min": meets_min,
        "meets_max": meets_max,
        "status": status,
    }

    if claimed_typical is not None:
        result["claimed_typical"] = claimed_typical
        result["meets_typical"] = measured_ratio >= claimed_typical

    return result
