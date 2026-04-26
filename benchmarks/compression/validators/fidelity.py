"""Fidelity validation for quantum state compression.

This module provides functions to compute and validate quantum state fidelity
after compression/reconstruction operations.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.complex128]


def compute_fidelity(original: Array, reconstructed: Array) -> float:
    """Compute quantum state fidelity between original and reconstructed states.

    For pure states, fidelity is computed as F(ψ, φ) = |⟨ψ|φ⟩|²

    This function normalizes the input states before computing fidelity to handle
    states that may have been scaled during compression.

    Args:
        original: Original quantum state vector
        reconstructed: Reconstructed quantum state vector

    Returns:
        Fidelity value in range [0, 1], where 1 means identical states

    Raises:
        ValueError: If states have incompatible shapes or are zero vectors

    Example:
        >>> state1 = np.array([1, 0], dtype=complex)
        >>> state2 = np.array([1, 0], dtype=complex)
        >>> fidelity = compute_fidelity(state1, state2)
        >>> assert abs(fidelity - 1.0) < 1e-10
    """

    if original.shape != reconstructed.shape:
        raise ValueError(f"State shape mismatch: {original.shape} vs {reconstructed.shape}")

    # Check for zero vectors
    original_norm = np.linalg.norm(original)
    reconstructed_norm = np.linalg.norm(reconstructed)

    if original_norm < 1e-14:
        raise ValueError("Original state is zero vector")
    if reconstructed_norm < 1e-14:
        raise ValueError("Reconstructed state is zero vector")

    # Normalize states
    original_normalized = original / original_norm
    reconstructed_normalized = reconstructed / reconstructed_norm

    # Compute overlap: ⟨ψ|φ⟩
    overlap = np.vdot(original_normalized, reconstructed_normalized)

    # Fidelity for pure states: F = |⟨ψ|φ⟩|²
    fidelity = float(np.abs(overlap) ** 2)

    return fidelity


def validate_fidelity_bound(fidelity: float, target: float) -> bool:
    """Check if fidelity meets minimum target threshold.

    Args:
        fidelity: Computed fidelity value
        target: Minimum acceptable fidelity threshold

    Returns:
        True if fidelity >= target, False otherwise

    Example:
        >>> assert validate_fidelity_bound(0.996, 0.995)
        >>> assert not validate_fidelity_bound(0.994, 0.995)
    """

    return fidelity >= target


def compute_trace_distance(original: Array, reconstructed: Array) -> float:
    """Compute trace distance between two quantum states.

    For pure states, trace distance is related to fidelity by:
    D(ρ, σ) = √(1 - F(ρ, σ))

    Trace distance ranges from 0 (identical states) to 1 (orthogonal states).

    Args:
        original: Original quantum state vector
        reconstructed: Reconstructed quantum state vector

    Returns:
        Trace distance in range [0, 1]

    Example:
        >>> state1 = np.array([1, 0], dtype=complex)
        >>> state2 = np.array([1, 0], dtype=complex)
        >>> distance = compute_trace_distance(state1, state2)
        >>> assert distance < 1e-10
    """

    fidelity = compute_fidelity(original, reconstructed)
    # For pure states: D = √(1 - F)
    trace_distance = float(np.sqrt(max(0.0, 1.0 - fidelity)))

    return trace_distance


def compute_state_overlap(state1: Array, state2: Array) -> complex:
    """Compute the complex overlap ⟨ψ|φ⟩ between two quantum states.

    Unlike fidelity (which returns |⟨ψ|φ⟩|²), this returns the full complex
    overlap including phase information.

    Args:
        state1: First quantum state
        state2: Second quantum state

    Returns:
        Complex overlap value

    Example:
        >>> state1 = np.array([1, 0], dtype=complex)
        >>> state2 = np.array([0, 1], dtype=complex)
        >>> overlap = compute_state_overlap(state1, state2)
        >>> assert abs(overlap) < 1e-10  # Orthogonal states
    """

    # Normalize states
    state1_norm = state1 / np.linalg.norm(state1)
    state2_norm = state2 / np.linalg.norm(state2)

    # Compute ⟨ψ|φ⟩
    overlap = np.vdot(state1_norm, state2_norm)

    return overlap
