"""Core mathematical utilities for the CIIR simulation.

Provides normalization, density-matrix construction, and
von Neumann entropy computation on a complex Hilbert space.
"""

from __future__ import annotations

import numpy as np


def normalize(psi: np.ndarray) -> np.ndarray:
    """Normalize a complex state vector to unit norm.

    Args:
        psi: Complex state vector.

    Returns:
        Unit-norm copy of *psi*.

    Raises:
        ValueError: If *psi* is the zero vector.
    """
    norm = np.linalg.norm(psi)
    if norm < 1e-15:
        raise ValueError("Cannot normalize the zero vector.")
    return psi / norm


def density_matrix(psi: np.ndarray) -> np.ndarray:
    """Compute the pure-state density matrix ρ = |ψ⟩⟨ψ|.

    Args:
        psi: Unit-norm complex state vector of shape ``(n,)``.

    Returns:
        Hermitian density matrix of shape ``(n, n)``.
    """
    psi = psi.reshape(-1, 1)
    return psi @ psi.conj().T


def von_neumann_entropy(rho: np.ndarray) -> float:
    """Compute von Neumann entropy S = −Tr(ρ log ρ).

    Args:
        rho: Density matrix (Hermitian, positive semi-definite).

    Returns:
        Non-negative real entropy value.
    """
    eigenvalues = np.linalg.eigvalsh(rho)
    eigenvalues = eigenvalues[eigenvalues > 1e-15]
    return float(-np.sum(eigenvalues * np.log(eigenvalues)))


def measurement_probability(psi: np.ndarray, operator: np.ndarray) -> float:
    """Compute P = ⟨ψ|M†M|ψ⟩ for measurement operator *operator*.

    Args:
        psi: State vector.
        operator: Measurement operator matrix.

    Returns:
        Born-rule probability (clamped to [0, 1]).
    """
    m_psi = operator @ psi
    return float(np.clip(np.real(np.vdot(m_psi, m_psi)), 0.0, 1.0))
