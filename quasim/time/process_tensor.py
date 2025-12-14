"""Process Tensor and Temporal Correlations Module.

Implements two-time correlation functions and process tensor formalism
for memory effects in open quantum systems.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.complex128]


def two_time_correlation(
    state: Array,
    observable_A: Array,
    observable_B: Array,
    hamiltonian: Array,
    time_1: float,
    time_2: float,
) -> complex:
    """Compute two-time correlation function ⟨A(t₁)B(t₂)⟩.

    Correlation function:
        C(t₁,t₂) = ⟨ψ|A(t₁)B(t₂)|ψ⟩
                 = ⟨ψ|e^{iHt₁} A e^{-iHt₁} e^{iHt₂} B e^{-iHt₂}|ψ⟩

    Args:
        state: Initial quantum state |ψ⟩
        observable_A: First observable operator A
        observable_B: Second observable operator B
        hamiltonian: System Hamiltonian H
        time_1: First measurement time t₁
        time_2: Second measurement time t₂

    Returns:
        Complex correlation value C(t₁,t₂)

    Example:
        >>> psi = np.array([1, 0], dtype=complex)
        >>> sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
        >>> sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
        >>> H = np.array([[1, 0], [0, -1]], dtype=complex)
        >>> C = two_time_correlation(psi, sigma_x, sigma_z, H, 0, 1)
    """
    # Evolution operators
    U_1 = np.linalg.matrix_power(np.eye(len(hamiltonian)) - 1j * hamiltonian * time_1, 1)
    U_2 = np.linalg.matrix_power(np.eye(len(hamiltonian)) - 1j * hamiltonian * time_2, 1)

    # Time-evolved operators
    A_t1 = U_1.conj().T @ observable_A @ U_1
    B_t2 = U_2.conj().T @ observable_B @ U_2

    # Correlation
    correlation = state.conj() @ A_t1 @ B_t2 @ state

    return complex(correlation)


def process_tensor_choi(
    hamiltonian: Array,
    dt: float,
    n_steps: int,
) -> Array:
    """Construct Choi matrix representation of temporal evolution.

    Choi matrix Λ maps initial to final states through dynamical map:
        ρ(t) = Tr_A[Λ (ρ(0) ⊗ I)]

    For unitary evolution: Λ = |U⟩⟩⟨⟨U|

    Args:
        hamiltonian: System Hamiltonian
        dt: Time step
        n_steps: Number of time steps

    Returns:
        Choi matrix in vectorized form

    Example:
        >>> H = np.array([[0, 1], [1, 0]], dtype=complex)
        >>> Λ = process_tensor_choi(H, 0.1, 10)

    Reference:
        Pollock et al. (2018), "Non-Markovian quantum processes"
    """
    dim = hamiltonian.shape[0]

    # Total evolution operator
    dt * n_steps
    U = np.linalg.matrix_power(np.eye(dim) - 1j * hamiltonian * dt, n_steps)

    # Choi matrix: Λ = |U⟩⟩⟨⟨U| where |U⟩⟩ = (I ⊗ U)|I⟩⟩
    # Vectorize U
    U_vec = U.flatten()

    # Choi matrix
    choi = np.outer(U_vec, U_vec.conj())

    return choi


def temporal_bell_inequality(
    state: Array,
    measurements: list[Array],
    hamiltonian: Array,
    times: NDArray[np.float64],
) -> float:
    """Compute temporal Bell-CHSH parameter for time-separated measurements.

    Leggett-Garg inequality for temporal correlations:
        K = |C(t₁,t₂) + C(t₂,t₃) + C(t₃,t₄) - C(t₁,t₄)| ≤ 2 (classical)

    Quantum systems can violate: K > 2

    Args:
        state: Initial quantum state
        measurements: List of measurement operators [A, B, C, D]
        hamiltonian: System Hamiltonian
        times: Measurement times [t₁, t₂, t₃, t₄]

    Returns:
        CHSH parameter K

    Example:
        >>> psi = np.array([1, 0], dtype=complex)
        >>> sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
        >>> measurements = [sigma_x] * 4
        >>> H = np.array([[1, 0], [0, -1]], dtype=complex)
        >>> times = np.array([0, 0.5, 1.0, 1.5])
        >>> K = temporal_bell_inequality(psi, measurements, H, times)

    Reference:
        Leggett & Garg (1985), "Quantum mechanics versus macroscopic realism"
    """
    if len(measurements) != 4 or len(times) != 4:
        raise ValueError("Need exactly 4 measurements and 4 times")

    A, B, C, D = measurements
    t1, t2, t3, t4 = times

    # Compute correlations
    C_12 = two_time_correlation(state, A, B, hamiltonian, t1, t2).real
    C_23 = two_time_correlation(state, B, C, hamiltonian, t2, t3).real
    C_34 = two_time_correlation(state, C, D, hamiltonian, t3, t4).real
    C_14 = two_time_correlation(state, A, D, hamiltonian, t1, t4).real

    # CHSH combination
    K = abs(C_12 + C_23 + C_34 - C_14)

    return float(K)


def memory_kernel(
    correlation_function: NDArray[np.complex128],
    times: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Extract memory kernel from two-time correlation function.

    Memory kernel K(t,t') characterizes non-Markovian dynamics:
        ∂ρ(t)/∂t = ∫₀ᵗ K(t,t') ρ(t') dt'

    Args:
        correlation_function: C(t,t') correlation matrix
        times: Time points

    Returns:
        Memory kernel K(t)

    Example:
        >>> times = np.linspace(0, 10, 100)
        >>> C = np.exp(-0.1 * np.abs(times[:, None] - times[None, :]))
        >>> K = memory_kernel(C, times)
    """
    # Time derivative of correlation function
    dt = times[1] - times[0] if len(times) > 1 else 1.0

    # Simplified: K(t) ≈ -dC(t,0)/dt
    if len(correlation_function.shape) == 2:
        # 2D correlation matrix
        kernel = -np.gradient(correlation_function[:, 0].real, dt)
    else:
        # 1D correlation function
        kernel = -np.gradient(correlation_function.real, dt)

    return kernel


def quantum_coherence_measure(
    density_matrix: Array,
) -> float:
    """Compute quantum coherence from off-diagonal density matrix elements.

    Coherence measure:
        C(ρ) = Σ_{i≠j} |ρ_ij|²

    Args:
        density_matrix: Density matrix ρ

    Returns:
        Coherence measure C(ρ) ≥ 0

    Example:
        >>> rho = np.array([[0.5, 0.4], [0.4, 0.5]], dtype=complex)
        >>> C = quantum_coherence_measure(rho)
        >>> assert C > 0
    """
    # Off-diagonal elements
    n = density_matrix.shape[0]
    coherence = 0.0

    for i in range(n):
        for j in range(n):
            if i != j:
                coherence += np.abs(density_matrix[i, j]) ** 2

    return float(coherence)
