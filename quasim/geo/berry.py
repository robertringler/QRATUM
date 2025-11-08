"""Berry Phase Evolution Module.

Implements geometric phase evolution using parameter-dependent eigenvectors.
Based on Berry, M.V. (1984). "Quantal Phase Factors Accompanying Adiabatic Changes."
Proceedings of the Royal Society of London A 392: 45–57.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.complex128]


def compute_berry_phase(
    eigenvectors: Array,
    parameter_path: Array,
) -> float:
    """Compute Berry phase for a closed loop in parameter space.

    The Berry phase γ is given by:
        γ = i ∮ ⟨ψ(R)|∇_R|ψ(R)⟩ · dR

    where ψ(R) is a parameter-dependent eigenstate and R is the parameter vector.

    Args:
        eigenvectors: Array of shape (n_steps, n_dim) containing eigenvectors
                     along the parameter path
        parameter_path: Array of shape (n_steps, n_params) containing parameter values

    Returns:
        Berry phase γ in radians, normalized to [-π, π]

    Example:
        >>> n_steps = 100
        >>> theta = np.linspace(0, 2*np.pi, n_steps)
        >>> eigvecs = np.array([[np.cos(t/2), np.sin(t/2)*np.exp(1j*phi)]
        ...                     for t, phi in zip(theta, theta)])
        >>> params = np.column_stack([theta, theta])
        >>> gamma = compute_berry_phase(eigvecs, params)
    """
    n_steps = len(eigenvectors)
    if n_steps < 2:
        return 0.0

    # Ensure eigenvectors are normalized
    eigenvectors = eigenvectors / np.linalg.norm(eigenvectors, axis=1, keepdims=True)

    # Compute Berry connection: A = i⟨ψ|∇ψ⟩
    berry_connection = 0.0

    for i in range(n_steps - 1):
        psi_current = eigenvectors[i]
        psi_next = eigenvectors[i + 1]

        # Compute overlap ⟨ψ_i|ψ_{i+1}⟩
        overlap = np.vdot(psi_current, psi_next)

        # Accumulate phase: γ ≈ Σ Im[log(⟨ψ_i|ψ_{i+1}⟩)]
        berry_connection += np.angle(overlap)

    # Normalize to [-π, π]
    gamma = berry_connection
    while gamma > np.pi:
        gamma -= 2 * np.pi
    while gamma < -np.pi:
        gamma += 2 * np.pi

    return float(gamma)


def evolve_with_berry_phase(
    initial_state: Array,
    hamiltonian_func: callable,
    parameter_path: Array,
    dt: float,
) -> tuple[Array, float]:
    """Evolve quantum state along parameter path, tracking Berry phase.

    Performs adiabatic evolution with the Hamiltonian H(R(t)) while computing
    the accumulated geometric phase.

    Args:
        initial_state: Initial quantum state vector
        hamiltonian_func: Function H(R) returning Hamiltonian matrix for parameters R
        parameter_path: Array of shape (n_steps, n_params) defining the path
        dt: Time step for evolution

    Returns:
        Tuple of (final_state, berry_phase):
            - final_state: Evolved state vector
            - berry_phase: Accumulated Berry phase in radians

    Example:
        >>> initial = np.array([1.0, 0.0], dtype=complex)
        >>> def H(R): return np.array([[R[0], R[1]], [R[1], -R[0]]])
        >>> path = np.linspace([0, 0.1], [1, 0.1], 50)
        >>> final, gamma = evolve_with_berry_phase(initial, H, path, 0.01)
    """
    state = initial_state.copy()
    eigenvectors_history = []

    for params in parameter_path:
        H = hamiltonian_func(params)

        # Diagonalize Hamiltonian
        eigenvalues, eigenvectors = np.linalg.eigh(H)

        # Project state onto instantaneous eigenbasis
        coeffs = eigenvectors.conj().T @ state

        # Store dominant eigenvector
        dominant_idx = np.argmax(np.abs(coeffs))
        eigenvectors_history.append(eigenvectors[:, dominant_idx])

        # Evolve dynamical phase
        phase_factors = np.exp(-1j * eigenvalues * dt)
        coeffs *= phase_factors

        # Reconstruct state
        state = eigenvectors @ coeffs

        # Renormalize
        state /= np.linalg.norm(state)

    # Compute Berry phase from collected eigenvectors
    eigenvectors_array = np.array(eigenvectors_history)
    berry_phase = compute_berry_phase(eigenvectors_array, parameter_path)

    return state, berry_phase


def analytical_berry_phase_spin_half(solid_angle: float) -> float:
    """Compute analytical Berry phase for spin-1/2 on a sphere.

    For a spin-1/2 particle transported around a closed path enclosing
    solid angle Ω on the Bloch sphere, the Berry phase is γ = -Ω/2.

    Args:
        solid_angle: Solid angle Ω enclosed by the path (in steradians)

    Returns:
        Berry phase γ = -Ω/2 in radians

    Reference:
        Berry (1984), Section 5: "Spin in a precessing magnetic field"
    """
    return -solid_angle / 2.0
