"""Holographic Boundary Coupling Module.

Implements bulk-boundary correspondence for open quantum systems.
Based on holographic duality principles and boundary state composition.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.complex128]


def bulk_boundary_hamiltonian(
    bulk_H: Array,
    boundary_H: Array,
    coupling_strength: float,
) -> Array:
    """Construct composite Hamiltonian with bulk-boundary coupling.

    Composes bulk and boundary Hamiltonians with interaction term:
        H_total = H_bulk ⊗ I_boundary + I_bulk ⊗ H_boundary + λ H_coupling

    Args:
        bulk_H: Bulk Hamiltonian matrix (n_bulk × n_bulk)
        boundary_H: Boundary Hamiltonian matrix (n_boundary × n_boundary)
        coupling_strength: Coupling parameter λ

    Returns:
        Total Hamiltonian of shape (n_bulk*n_boundary × n_bulk*n_boundary)

    Example:
        >>> bulk = np.array([[0, 1], [1, 0]])
        >>> boundary = np.array([[1, 0], [0, -1]])
        >>> H_total = bulk_boundary_hamiltonian(bulk, boundary, 0.1)
    """
    n_bulk = bulk_H.shape[0]
    n_boundary = boundary_H.shape[0]

    # Create identity matrices
    I_bulk = np.eye(n_bulk, dtype=complex)
    I_boundary = np.eye(n_boundary, dtype=complex)

    # Construct tensor product terms
    H_total = np.kron(bulk_H, I_boundary) + np.kron(I_bulk, boundary_H)

    # Add coupling term (simplified as local interaction)
    if coupling_strength != 0:
        # Coupling between bulk and boundary degrees of freedom
        coupling = np.zeros((n_bulk * n_boundary, n_bulk * n_boundary), dtype=complex)
        for i in range(min(n_bulk, n_boundary)):
            idx = i * n_boundary + i
            if idx < len(coupling):
                coupling[idx, idx] = coupling_strength
        H_total += coupling

    return H_total


def evolve_open_boundary(
    initial_state: Array,
    hamiltonian: Array,
    dt: float,
    dissipation_rate: float = 0.0,
) -> Array:
    """Evolve state with open boundary conditions.

    Implements Lindblad-type evolution for open quantum systems:
        dρ/dt = -i[H,ρ] + Σ_k (L_k ρ L_k† - 1/2{L_k†L_k, ρ})

    Simplified to state vector evolution with dissipation.

    Args:
        initial_state: Initial state vector
        hamiltonian: System Hamiltonian
        dt: Time step
        dissipation_rate: Dissipation coefficient γ ≥ 0

    Returns:
        Evolved state vector (renormalized)

    Example:
        >>> psi0 = np.array([1.0, 0.0], dtype=complex)
        >>> H = np.array([[0, 1], [1, 0]], dtype=complex)
        >>> psi_final = evolve_open_boundary(psi0, H, 0.01, 0.1)
    """
    # Unitary evolution
    U = np.linalg.matrix_power(np.eye(len(hamiltonian)) - 1j * hamiltonian * dt, 1)
    state = U @ initial_state

    # Apply dissipation (simplified decay)
    if dissipation_rate > 0:
        decay_factor = np.exp(-dissipation_rate * dt / 2)
        state *= decay_factor

    # Renormalize
    norm = np.linalg.norm(state)
    if norm > 1e-14:
        state /= norm

    return state


def check_probability_conservation(
    evolved_state: Array,
    tolerance: float = 1e-10,
) -> tuple[bool, float]:
    """Verify probability conservation for evolved state.

    Checks that ⟨ψ|ψ⟩ = 1 within tolerance.

    Args:
        evolved_state: State vector after evolution
        tolerance: Maximum allowed deviation from unity

    Returns:
        Tuple of (is_conserved, deviation):
            - is_conserved: True if |⟨ψ|ψ⟩ - 1| < tolerance
            - deviation: Actual deviation from unity

    Example:
        >>> state = np.array([1/np.sqrt(2), 1/np.sqrt(2)], dtype=complex)
        >>> is_ok, dev = check_probability_conservation(state)
        >>> assert is_ok and dev < 1e-10
    """
    norm_squared = np.vdot(evolved_state, evolved_state).real
    deviation = abs(norm_squared - 1.0)
    is_conserved = deviation < tolerance

    return is_conserved, float(deviation)


def boundary_projection(
    bulk_state: Array,
    n_bulk: int,
    n_boundary: int,
) -> Array:
    """Project bulk+boundary state onto boundary degrees of freedom.

    Extracts boundary subsystem by tracing out bulk degrees of freedom.

    Args:
        bulk_state: Composite state vector of shape (n_bulk * n_boundary,)
        n_bulk: Dimension of bulk subsystem
        n_boundary: Dimension of boundary subsystem

    Returns:
        Boundary state (reduced density matrix diagonal or state vector)

    Example:
        >>> state = np.array([1, 0, 0, 0], dtype=complex)  # Product state
        >>> boundary = boundary_projection(state, 2, 2)
    """
    # Reshape state into bulk ⊗ boundary structure
    state_reshaped = bulk_state.reshape(n_bulk, n_boundary)

    # Compute reduced density matrix for boundary
    # ρ_boundary = Tr_bulk(|ψ⟩⟨ψ|)
    rho_boundary = np.zeros((n_boundary, n_boundary), dtype=complex)

    for i in range(n_bulk):
        rho_boundary += np.outer(state_reshaped[i], state_reshaped[i].conj())

    # Return diagonal (measurement probabilities) as a vector
    boundary_state = np.diag(rho_boundary)
    boundary_state = boundary_state / np.sum(boundary_state.real)  # Renormalize

    return boundary_state
