"""Quantum Crystallization Module.

Implements Time-Dependent Ginzburg-Landau (TDGL) equation for density-wave ordering.
Based on Landau theory of phase transitions and order parameter dynamics.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.complex128]


def tdgl_evolution_step(
    order_parameter: Array,
    alpha: float,
    beta: float,
    gamma: float,
    dt: float,
) -> Array:
    """Evolve order parameter using Time-Dependent Ginzburg-Landau equation.

    TDGL equation:
        ∂ψ/∂t = -γ(α ψ + β |ψ|² ψ - ∇²ψ)

    Where ψ is the complex order parameter, α, β control phase transition,
    and γ is the relaxation rate.

    Args:
        order_parameter: Current order parameter field ψ(r)
        alpha: Linear coefficient (< 0 for ordered phase)
        beta: Nonlinear coefficient (> 0 for stability)
        gamma: Relaxation rate
        dt: Time step

    Returns:
        Updated order parameter field

    Example:
        >>> psi = np.random.randn(64) + 1j * np.random.randn(64)
        >>> psi_new = tdgl_evolution_step(psi, -1.0, 1.0, 0.5, 0.01)
    """
    # Compute |ψ|²
    psi_abs_sq = np.abs(order_parameter) ** 2

    # Compute Laplacian using finite differences (periodic BC)
    laplacian = np.zeros_like(order_parameter)
    n = len(order_parameter)

    for i in range(n):
        laplacian[i] = (
            order_parameter[(i + 1) % n] - 2 * order_parameter[i] + order_parameter[(i - 1) % n]
        )

    # TDGL evolution
    rhs = -gamma * (alpha * order_parameter + beta * psi_abs_sq * order_parameter - laplacian)

    order_parameter_new = order_parameter + dt * rhs

    return order_parameter_new


def compute_structure_factor(
    order_parameter: Array,
) -> Array:
    """Compute structure factor S(k) = |⟨ψ(k)⟩|² via Fourier transform.

    The structure factor reveals spatial correlations and ordering:
        S(k) = |FFT[ψ(r)]|²

    Peaks in S(k) indicate crystalline order at specific wavevectors.

    Args:
        order_parameter: Spatial order parameter field

    Returns:
        Structure factor as a function of wavevector

    Example:
        >>> psi = np.exp(1j * 2 * np.pi * np.arange(64) / 64)  # Periodic
        >>> S = compute_structure_factor(psi)
        >>> assert np.argmax(S) == 1  # Peak at k=1
    """
    # Fourier transform
    psi_k = np.fft.fft(order_parameter)

    # Structure factor
    S_k = np.abs(psi_k) ** 2

    # Normalize
    S_k /= len(order_parameter)

    return S_k


def detect_crystallization(
    structure_factor: Array,
    threshold: float = 10.0,
) -> tuple[bool, int, float]:
    """Detect crystalline ordering from structure factor peak.

    Crystallization is indicated by emergence of sharp peaks in S(k).

    Args:
        structure_factor: Structure factor S(k)
        threshold: Minimum peak height relative to background

    Returns:
        Tuple of (is_crystallized, peak_index, peak_height):
            - is_crystallized: True if peak exceeds threshold
            - peak_index: Location of dominant peak
            - peak_height: Height of dominant peak

    Example:
        >>> S = np.array([1, 0.5, 20, 0.5, 1])
        >>> crystallized, idx, height = detect_crystallization(S, 5.0)
        >>> assert crystallized and idx == 2
    """
    # Find peak (excluding DC component at k=0)
    peak_index = np.argmax(structure_factor[1:]) + 1
    peak_height = float(structure_factor[peak_index])

    # Compute background as median
    background = np.median(structure_factor)

    # Check if peak is significant
    is_crystallized = peak_height > threshold * background

    return is_crystallized, int(peak_index), peak_height


def simulate_crystallization(
    n_grid: int,
    alpha: float,
    beta: float,
    gamma: float,
    dt: float,
    n_steps: int,
    noise_amplitude: float = 0.1,
) -> tuple[Array, Array]:
    """Simulate full crystallization dynamics from noisy initial conditions.

    Evolves TDGL equation from random initial state until crystalline
    order emerges (or fails to emerge).

    Args:
        n_grid: Number of spatial grid points
        alpha: Linear TDGL coefficient
        beta: Nonlinear TDGL coefficient
        gamma: Relaxation rate
        dt: Time step
        n_steps: Number of evolution steps
        noise_amplitude: Initial noise level

    Returns:
        Tuple of (final_order_parameter, structure_factor_history):
            - final_order_parameter: Final ψ(r) field
            - structure_factor_history: S(k) at each time step

    Example:
        >>> psi_final, S_history = simulate_crystallization(
        ...     64, -1.0, 1.0, 0.5, 0.01, 1000
        ... )
    """
    # Initialize with noise
    rng = np.random.RandomState(42)  # Deterministic for testing
    psi = noise_amplitude * (rng.randn(n_grid) + 1j * rng.randn(n_grid))

    structure_factor_history = []

    for step in range(n_steps):
        # Evolve TDGL
        psi = tdgl_evolution_step(psi, alpha, beta, gamma, dt)

        # Record structure factor periodically
        if step % (n_steps // 10) == 0:
            S_k = compute_structure_factor(psi)
            structure_factor_history.append(S_k)

    return psi, np.array(structure_factor_history)
