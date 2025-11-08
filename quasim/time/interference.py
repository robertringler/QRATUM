"""Temporal Interference Module.

Computes emission intensity and temporal interference patterns from
multi-state quantum superpositions.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.complex128]


def compute_emission_intensity(
    state: Array,
    dipole_operator: Array,
    time_points: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute time-dependent emission intensity from quantum state.

    Emission intensity is proportional to:
        I(t) = |⟨ψ(t)|d|ψ(t)⟩|²

    where d is the dipole operator.

    Args:
        state: Quantum state vector (can be time-dependent)
        dipole_operator: Dipole transition operator
        time_points: Array of time values

    Returns:
        Emission intensity as function of time

    Example:
        >>> psi = np.array([1/np.sqrt(2), 1/np.sqrt(2)], dtype=complex)
        >>> d = np.array([[0, 1], [1, 0]], dtype=complex)
        >>> t = np.linspace(0, 10, 100)
        >>> I = compute_emission_intensity(psi, d, t)
    """
    intensity = np.zeros(len(time_points))

    for i, t in enumerate(time_points):
        # Compute expectation value of dipole operator
        dipole_expectation = state.conj() @ dipole_operator @ state

        # Intensity is square of dipole moment
        intensity[i] = np.abs(dipole_expectation) ** 2

    return intensity


def rabi_oscillation(
    omega_0: float,
    omega_rabi: float,
    time_points: NDArray[np.float64],
    detuning: float = 0.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute Rabi oscillations for a two-level atom.

    For a two-level system driven by resonant field:
        P_e(t) = (Ω_R² / (Ω_R² + δ²)) sin²(√(Ω_R² + δ²) t / 2)

    where Ω_R is the Rabi frequency and δ is the detuning.

    Args:
        omega_0: Atomic transition frequency
        omega_rabi: Rabi frequency (coupling strength)
        time_points: Array of time values
        detuning: Detuning δ = ω_drive - ω_0

    Returns:
        Tuple of (population_excited, population_ground):
            - population_excited: P_e(t)
            - population_ground: P_g(t) = 1 - P_e(t)

    Example:
        >>> t = np.linspace(0, 10, 100)
        >>> P_e, P_g = rabi_oscillation(1.0, 0.5, t, detuning=0.0)
        >>> assert np.allclose(P_e + P_g, 1.0)

    Reference:
        Rabi, I.I. (1937). "Space Quantization in a Gyrating Magnetic Field"
    """
    # Generalized Rabi frequency
    omega_eff = np.sqrt(omega_rabi**2 + detuning**2)

    # Excited state population
    population_excited = (omega_rabi**2 / (omega_rabi**2 + detuning**2)) * np.sin(
        omega_eff * time_points / 2
    ) ** 2

    # Ground state population (conservation)
    population_ground = 1.0 - population_excited

    return population_excited, population_ground


def interference_pattern(
    state_1: Array,
    state_2: Array,
    relative_phase: float,
    observable: Array,
) -> float:
    """Compute interference between two quantum states.

    For superposition |ψ⟩ = (|1⟩ + e^(iφ)|2⟩)/√2,
    interference term is: I = 2Re[⟨1|O|2⟩ e^(iφ)]

    Args:
        state_1: First state vector |1⟩
        state_2: Second state vector |2⟩
        relative_phase: Relative phase φ between states
        observable: Observable operator O

    Returns:
        Interference contribution to expectation value

    Example:
        >>> psi1 = np.array([1, 0], dtype=complex)
        >>> psi2 = np.array([0, 1], dtype=complex)
        >>> sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
        >>> I = interference_pattern(psi1, psi2, 0, sigma_x)
    """
    # Normalize states
    state_1 = state_1 / np.linalg.norm(state_1)
    state_2 = state_2 / np.linalg.norm(state_2)

    # Compute cross-term ⟨1|O|2⟩
    cross_term = state_1.conj() @ observable @ state_2

    # Include phase and extract real part (interference)
    interference = 2 * np.real(cross_term * np.exp(1j * relative_phase))

    return float(interference)


def multi_state_superposition_intensity(
    amplitudes: Array,
    energies: NDArray[np.float64],
    dipole_matrix: Array,
    time_points: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute emission intensity from multi-state superposition.

    For |ψ(t)⟩ = Σ_n c_n e^(-iE_n t)|n⟩, emission intensity includes
    all interference terms between energy eigenstates.

    Args:
        amplitudes: Complex amplitudes c_n for each eigenstate
        energies: Energy eigenvalues E_n
        dipole_matrix: Dipole transition matrix elements d_mn
        time_points: Array of time values

    Returns:
        Time-dependent emission intensity I(t)

    Example:
        >>> c = np.array([1/np.sqrt(2), 1/np.sqrt(2)], dtype=complex)
        >>> E = np.array([0.0, 1.0])
        >>> d = np.array([[0, 1], [1, 0]], dtype=complex)
        >>> t = np.linspace(0, 10, 100)
        >>> I = multi_state_superposition_intensity(c, E, d, t)
    """
    n_states = len(amplitudes)
    intensity = np.zeros(len(time_points))

    for t_idx, t in enumerate(time_points):
        # Construct time-evolved state
        state_t = amplitudes * np.exp(-1j * energies * t)

        # Compute dipole expectation
        dipole_exp = state_t.conj() @ dipole_matrix @ state_t

        # Intensity
        intensity[t_idx] = np.abs(dipole_exp) ** 2

    return intensity
