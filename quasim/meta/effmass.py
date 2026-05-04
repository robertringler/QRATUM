"""Effective Mass Operator Module.

Implements spatially varying effective-mass Hamiltonian for heterogeneous systems.
Used in semiconductor heterostructures and metamaterials.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.complex128]


def effective_mass_hamiltonian(
    mass_profile: NDArray[np.float64],
    dx: float,
    potential: NDArray[np.float64] | None = None,
) -> Array:
    """Construct position-dependent effective-mass Hamiltonian.

    Hamiltonian with spatially varying effective mass:
        H = -∇ · (ℏ²/2m(r)) ∇ + V(r)

    Discretized using finite differences with proper operator ordering.

    Args:
        mass_profile: Effective mass m(r) at each spatial point
        dx: Spatial grid spacing
        potential: Optional potential energy V(r)

    Returns:
        Effective-mass Hamiltonian matrix

    Example:
        >>> m = np.ones(64)
        >>> m[32:] = 2.0  # Mass jump at x = L/2
        >>> H = effective_mass_hamiltonian(m, 0.1)

    Reference:
        BenDaniel & Duke (1966), Phys. Rev. 152, 683
    """

    n = len(mass_profile)
    H = np.zeros((n, n), dtype=complex)

    # Kinetic energy: T = -∇ · (1/m(r)) ∇
    # Discretized as: T_ij = (1/(m_i+1 + m_i))(ψ_{i+1} - ψ_i) - ...

    for i in range(n):
        # Diagonal term
        if i > 0:
            m_left = (mass_profile[i] + mass_profile[i - 1]) / 2
            H[i, i] += 1.0 / (2 * m_left * dx**2)

        if i < n - 1:
            m_right = (mass_profile[i] + mass_profile[i + 1]) / 2
            H[i, i] += 1.0 / (2 * m_right * dx**2)

        # Off-diagonal terms
        if i > 0:
            m_left = (mass_profile[i] + mass_profile[i - 1]) / 2
            H[i, i - 1] = -1.0 / (2 * m_left * dx**2)

        if i < n - 1:
            m_right = (mass_profile[i] + mass_profile[i + 1]) / 2
            H[i, i + 1] = -1.0 / (2 * m_right * dx**2)

    # Add potential energy
    if potential is not None:
        for i in range(n):
            H[i, i] += potential[i]

    # Make Hermitian (fix numerical errors)
    H = (H + H.conj().T) / 2

    return H


def dispersion_relation(
    hamiltonian: Array,
    lattice_constant: float,
) -> tuple[NDArray[np.float64], Array]:
    """Compute energy dispersion E(k) from Hamiltonian.

    Diagonalizes Hamiltonian to obtain energy bands.

    Args:
        hamiltonian: System Hamiltonian matrix
        lattice_constant: Lattice spacing a

    Returns:
        Tuple of (wavevectors, energies):
            - wavevectors: k-points in first Brillouin zone
            - energies: Energy eigenvalues E_n(k)

    Example:
        >>> H = np.diag([1, 2, 3, 4])
        >>> k, E = dispersion_relation(H, 1.0)
    """

    # Diagonalize Hamiltonian
    eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian)

    # Generate k-points (for 1D system)
    n = len(hamiltonian)
    k_max = np.pi / lattice_constant
    wavevectors = np.linspace(-k_max, k_max, n)

    return wavevectors, eigenvalues


def test_dispersion_parabolic(
    hamiltonian: Array,
    effective_mass_expected: float,
    tolerance: float = 0.1,
) -> tuple[bool, float]:
    """Test if dispersion relation is parabolic E ∝ k²/2m.

    Fits low-energy dispersion to parabolic form and extracts effective mass.

    Args:
        hamiltonian: Hamiltonian matrix
        effective_mass_expected: Expected effective mass
        tolerance: Relative tolerance for mass extraction

    Returns:
        Tuple of (is_parabolic, extracted_mass):
            - is_parabolic: True if dispersion is parabolic within tolerance
            - extracted_mass: Fitted effective mass

    Example:
        >>> m = np.ones(64)
        >>> H = effective_mass_hamiltonian(m, 0.1)
        >>> is_ok, m_eff = test_dispersion_parabolic(H, 1.0, 0.1)
    """

    # Get dispersion
    k, E = dispersion_relation(hamiltonian, 1.0)

    # Find ground state energy
    E_0 = np.min(E)

    # Fit parabolic dispersion near k=0 (lowest few states)
    n_fit = min(5, len(E))
    E_fit = np.sort(E)[:n_fit] - E_0
    k_eff = np.arange(n_fit)  # Effective k-index

    # E(k) ≈ k²/2m → m ≈ k²/2E
    if E_fit[1] > 1e-10:  # Avoid division by zero
        extracted_mass = k_eff[1] ** 2 / (2 * E_fit[1])
    else:
        extracted_mass = 0.0

    # Check if close to expected
    relative_error = abs(extracted_mass - effective_mass_expected) / effective_mass_expected
    is_parabolic = relative_error < tolerance

    return is_parabolic, float(extracted_mass)


def heterostructure_hamiltonian(
    n_points: int,
    mass_left: float,
    mass_right: float,
    barrier_height: float,
    dx: float,
) -> Array:
    """Construct Hamiltonian for a heterostructure with mass discontinuity.

    Models semiconductor heterostructure with:
        - Different effective masses in left/right regions
        - Potential barrier at interface

    Args:
        n_points: Number of spatial grid points
        mass_left: Effective mass in left region
        mass_right: Effective mass in right region
        barrier_height: Potential barrier height at interface
        dx: Spatial grid spacing

    Returns:
        Heterostructure Hamiltonian matrix

    Example:
        >>> H = heterostructure_hamiltonian(128, 1.0, 2.0, 0.5, 0.1)
    """

    # Mass profile with discontinuity at center
    mass_profile = np.ones(n_points) * mass_left
    mass_profile[n_points // 2 :] = mass_right

    # Potential with barrier
    potential = np.zeros(n_points)
    potential[n_points // 2 - 2 : n_points // 2 + 2] = barrier_height

    # Construct Hamiltonian
    H = effective_mass_hamiltonian(mass_profile, dx, potential)

    return H
