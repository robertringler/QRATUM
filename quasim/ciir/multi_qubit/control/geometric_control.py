r"""Information-Geometric Control — GAP 7 closure.

Replaces Euclidean LQR/Pontryagin control with a geometry-aware
controller that respects the curved Riemannian structure of the
density-matrix manifold.

Quantum Fisher metric
---------------------
For a parameterized family ρ(θ) the quantum Fisher information metric is:

    g_{ij}(θ) = ½ Tr( ρ(θ) { L_i(θ), L_j(θ) } )

where L_k(θ) is the symmetric logarithmic derivative (SLD) defined by:

    ∂_k ρ = ½ ( ρ L_k + L_k ρ )

We use an exponential-family parameterization:

    ρ(θ) = exp(−H(θ)) / Z(θ),   H(θ) = H₀ + Σ_k θ_k H_k

where H_k are control Hamiltonians.  The SLD in this family is computed
numerically via matrix perturbation.

Natural gradient
----------------
Given a cost J(θ) (e.g. negative fidelity to target), the natural gradient
descent update is:

    θ̇ = −η · g⁻¹(θ) · ∇_θ J(θ)

which follows the steepest descent on the statistical manifold rather than
the Euclidean parameter space, leading to faster and more robust
convergence.

References
----------
* Amari (1998) — Natural Gradient Works Efficiently in Learning
* Braunstein & Caves (1994) — Statistical Distance and the Geometry of
  Quantum States
* Ch21 CIIR monograph — §7 Information-Geometric Control
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from numpy.typing import NDArray

CMatrix = NDArray[np.complex128]
RVec = NDArray[np.float64]


# ---------------------------------------------------------------------------
# Utility: matrix exponential from eigendecomposition
# ---------------------------------------------------------------------------

def _matrix_exp(A: CMatrix) -> CMatrix:
    """exp(A) via eigendecomposition (A must be Hermitian)."""
    eigvals, eigvecs = np.linalg.eigh(A)
    return eigvecs @ np.diag(np.exp(eigvals)) @ eigvecs.conj().T


def _density_from_theta(
    theta: RVec,
    H0: CMatrix,
    H_ops: List[CMatrix],
    eps_reg: float = 1e-10,
) -> CMatrix:
    r"""Compute ρ(θ) = exp(−H(θ)) / Z in the exponential family.

    H(θ) = H₀ + Σ_k θ_k H_k

    Returns a valid density matrix.
    """
    H_theta = H0.copy()
    for k, hk in enumerate(H_ops):
        H_theta = H_theta + theta[k] * hk
    # ρ = exp(−H) / Z; use −H to ensure positivity when H is Hermitian
    rho = _matrix_exp(-H_theta)
    tr = float(np.trace(rho).real)
    if tr > eps_reg:
        rho /= tr
    return rho


# ---------------------------------------------------------------------------
# Symmetric logarithmic derivative (SLD)
# ---------------------------------------------------------------------------

def sld(rho: CMatrix, drho: CMatrix, eps: float = 1e-8) -> CMatrix:
    r"""Compute the SLD L such that ∂ρ = ½(ρL + Lρ).

    Solved via the Lyapunov-like equation:
        ρL + Lρ = 2 ∂ρ

    Using the eigenbasis of ρ:  L_{ij} = 2 ∂ρ_{ij} / (λ_i + λ_j)
    for λ_i + λ_j > eps.
    """
    eigvals, eigvecs = np.linalg.eigh(rho)
    # Transform ∂ρ to eigenbasis
    drho_basis = eigvecs.conj().T @ drho @ eigvecs
    d = rho.shape[0]
    L_basis = np.zeros((d, d), dtype=complex)
    for i in range(d):
        for j in range(d):
            denom = eigvals[i] + eigvals[j]
            if abs(denom) > eps:
                L_basis[i, j] = 2.0 * drho_basis[i, j] / denom
    # Transform back
    return eigvecs @ L_basis @ eigvecs.conj().T


# ---------------------------------------------------------------------------
# Quantum Fisher metric
# ---------------------------------------------------------------------------

def quantum_fisher_metric(
    theta: RVec,
    H0: CMatrix,
    H_ops: List[CMatrix],
    eps_fd: float = 1e-4,
) -> CMatrix:
    r"""Compute the (p×p) quantum Fisher information metric numerically.

    g_{ij}(θ) = ½ Tr( ρ(θ) { L_i, L_j } )

    where L_k is the SLD for parameter direction k, computed via
    finite differences:

        ∂_k ρ ≈ [ρ(θ + ε e_k) − ρ(θ − ε e_k)] / (2ε)

    Parameters
    ----------
    theta   : parameter vector (length p)
    H0      : drift Hamiltonian
    H_ops   : list of p control Hamiltonians
    eps_fd  : finite-difference step size

    Returns
    -------
    (p, p) real positive-semidefinite matrix
    """
    p = len(theta)
    rho0 = _density_from_theta(theta, H0, H_ops)

    # Compute ∂_k ρ for each k via central finite differences
    d_rho = []
    for k in range(p):
        theta_p = theta.copy()
        theta_m = theta.copy()
        theta_p[k] += eps_fd
        theta_m[k] -= eps_fd
        rho_p = _density_from_theta(theta_p, H0, H_ops)
        rho_m = _density_from_theta(theta_m, H0, H_ops)
        d_rho.append((rho_p - rho_m) / (2.0 * eps_fd))

    # Compute SLDs
    slds = [sld(rho0, drk) for drk in d_rho]

    # Fisher metric g_{ij} = ½ Re Tr(ρ {L_i, L_j})
    G = np.zeros((p, p), dtype=float)
    for i in range(p):
        for j in range(i, p):
            anticomm = slds[i] @ slds[j] + slds[j] @ slds[i]
            g_ij = 0.5 * float(np.real(np.trace(rho0 @ anticomm)))
            G[i, j] = g_ij
            G[j, i] = g_ij

    return G


# ---------------------------------------------------------------------------
# Cost function: negative fidelity
# ---------------------------------------------------------------------------

def fidelity_cost(
    theta: RVec,
    H0: CMatrix,
    H_ops: List[CMatrix],
    rho_target: CMatrix,
) -> float:
    """J(θ) = 1 − F(ρ(θ), ρ_target)  where F = Tr(ρ ρ_target)."""
    rho = _density_from_theta(theta, H0, H_ops)
    F = float(np.real(np.trace(rho @ rho_target)))
    return 1.0 - F


def fidelity_gradient(
    theta: RVec,
    H0: CMatrix,
    H_ops: List[CMatrix],
    rho_target: CMatrix,
    eps_fd: float = 1e-4,
) -> RVec:
    """Numerical gradient of J(θ) = 1 − Tr(ρ(θ) ρ_target)."""
    p = len(theta)
    grad = np.zeros(p)
    for k in range(p):
        theta_p = theta.copy()
        theta_m = theta.copy()
        theta_p[k] += eps_fd
        theta_m[k] -= eps_fd
        grad[k] = (fidelity_cost(theta_p, H0, H_ops, rho_target)
                   - fidelity_cost(theta_m, H0, H_ops, rho_target)) / (2.0 * eps_fd)
    return grad


# ---------------------------------------------------------------------------
# Natural gradient controller
# ---------------------------------------------------------------------------

@dataclass
class GeometricControlResult:
    """Output of one natural-gradient control step."""

    theta: RVec            # updated parameters
    rho: CMatrix           # updated density matrix
    cost: float            # J(θ)
    fidelity: float        # F(ρ, ρ_target)
    grad_norm: float       # ‖∇J‖
    nat_grad_norm: float   # ‖g⁻¹ ∇J‖


class NaturalGradientController:
    r"""Quantum control via natural gradient on the density-matrix manifold.

    At each step:
        1. Compute ρ(θ) = exp(−H(θ)) / Z
        2. Compute Fisher metric G(θ)
        3. Compute gradient ∇_θ J(θ)
        4. Update: θ ← θ − η · G⁻¹ ∇_θ J(θ)

    Parameters
    ----------
    H0      : drift (bare) Hamiltonian
    H_ops   : list of p control Hamiltonians
    eta     : learning rate
    reg     : Tikhonov regularization added to G before inversion
    eps_fd  : finite-difference step for gradient/metric
    """

    def __init__(
        self,
        H0: CMatrix,
        H_ops: List[CMatrix],
        eta: float = 0.05,
        reg: float = 1e-4,
        eps_fd: float = 1e-4,
    ) -> None:
        self.H0 = H0
        self.H_ops = H_ops
        self.eta = eta
        self.reg = reg
        self.eps_fd = eps_fd
        self.p = len(H_ops)

    def step(
        self,
        theta: RVec,
        rho_target: CMatrix,
    ) -> GeometricControlResult:
        """Perform one natural-gradient descent step."""
        # Current state and cost
        rho = _density_from_theta(theta, self.H0, self.H_ops)
        cost = fidelity_cost(theta, self.H0, self.H_ops, rho_target)
        fid = float(np.real(np.trace(rho @ rho_target)))

        # Euclidean gradient
        grad = fidelity_gradient(theta, self.H0, self.H_ops, rho_target, self.eps_fd)
        grad_norm = float(np.linalg.norm(grad))

        # Fisher metric (regularized)
        G = quantum_fisher_metric(theta, self.H0, self.H_ops, self.eps_fd)
        G_reg = G + self.reg * np.eye(self.p)

        # Natural gradient: G⁻¹ ∇J
        try:
            nat_grad = np.linalg.solve(G_reg, grad)
        except np.linalg.LinAlgError:
            nat_grad = grad  # fallback to Euclidean gradient

        nat_grad_norm = float(np.linalg.norm(nat_grad))

        # Update parameters
        theta_new = theta - self.eta * nat_grad

        return GeometricControlResult(
            theta=theta_new,
            rho=_density_from_theta(theta_new, self.H0, self.H_ops),
            cost=cost,
            fidelity=fid,
            grad_norm=grad_norm,
            nat_grad_norm=nat_grad_norm,
        )

    def optimize(
        self,
        theta0: RVec,
        rho_target: CMatrix,
        n_steps: int = 100,
        tol: float = 1e-4,
    ) -> Tuple[RVec, List[float], List[float]]:
        """Run natural-gradient optimization.

        Returns
        -------
        (theta_opt, cost_history, fidelity_history)
        """
        theta = theta0.copy()
        costs = []
        fidelities = []

        for _ in range(n_steps):
            result = self.step(theta, rho_target)
            theta = result.theta
            costs.append(result.cost)
            fidelities.append(result.fidelity)
            if result.cost < tol:
                break

        return theta, costs, fidelities

    def H_ctrl_from_theta(self, theta: RVec) -> CMatrix:
        """Return H_ctrl = H₀ + Σ_k θ_k H_k corresponding to current θ."""
        H = self.H0.copy()
        for k, hk in enumerate(self.H_ops):
            H = H + theta[k] * hk
        return H
