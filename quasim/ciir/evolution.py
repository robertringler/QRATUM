r"""State evolution: projected gradient dynamics with operator splitting.

Implements the CIIR dynamical evolution from Ch7 (D7.1–D7.17, T7.1–T7.7)
in numerically executable form.

Update Rules
-------------
Basic gradient descent (§3 of formalization):

    ρ_{t+1} = ρ_t − η ∇L(ρ_t)

Projected dynamics (D7.4, Ax4.2 constraint satisfaction):

    ρ_{t+1} = Π_C(ρ_t − η ∇L(ρ_t))

where Π_C is the projection onto the constraint-compatible density
operator cone D(H_C) ∩ {ρ : C_i(ρ) ≈ 0}.

Operator Splitting (T7.3 — Strang splitting):

    1. Half-step unitary:  ρ ← e^{-iH dt/2} ρ e^{iH dt/2}
    2. Full-step dissipative: ρ ← ρ + dt · L[ρ]  (Lindblad, D7.4)
    3. Half-step unitary:  ρ ← e^{-iH dt/2} ρ e^{iH dt/2}

Stability is guaranteed by T7.2 (CPTP semigroup).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from quasim.ciir.loss import CIIRLoss
from quasim.ciir.theory import RealTensor


class IntegrationMethod(Enum):
    """Supported integration methods."""

    GRADIENT = "gradient"
    PROJECTED_GRADIENT = "projected_gradient"
    STRANG_SPLITTING = "strang_splitting"


# ================================================================
# Projection operators  (D7.4, Ax4.2)
# ================================================================


def project_density(rho: RealTensor) -> RealTensor:
    r"""Project onto the density operator cone D(H_C).

    Enforces: ρ ≥ 0, Tr(ρ) = 1 via eigenvalue clipping and
    trace renormalisation (D3.14).

    Parameters
    ----------
    rho : array, shape (B, D, D)

    Returns
    -------
    rho_proj : array, shape (B, D, D)
    """
    B = rho.shape[0]
    out = np.zeros_like(rho)
    for b in range(B):
        # Symmetrise
        r = 0.5 * (rho[b] + rho[b].T)
        eigvals, eigvecs = np.linalg.eigh(r)
        # Clip negative eigenvalues (positivity, D3.14)
        eigvals = np.clip(eigvals, 0, None)
        total = eigvals.sum()
        if total > 1e-12:
            eigvals /= total
        else:
            eigvals = np.ones_like(eigvals) / len(eigvals)
        out[b] = eigvecs @ np.diag(eigvals) @ eigvecs.T
    return out


# ================================================================
# Gradient step
# ================================================================


def gradient_step(rho: RealTensor, grad: RealTensor, lr: float) -> RealTensor:
    r"""ρ_{t+1} = ρ_t − η ∇L(ρ_t)."""
    return rho - lr * grad


def projected_gradient_step(rho: RealTensor, grad: RealTensor, lr: float) -> RealTensor:
    r"""ρ_{t+1} = Π_C(ρ_t − η ∇L(ρ_t))."""
    rho_new = gradient_step(rho, grad, lr)
    return project_density(rho_new)


# ================================================================
# Operator splitting  (T7.3 Strang)
# ================================================================


def _unitary_half_step(
    rho: RealTensor, hamiltonian: RealTensor, dt: float
) -> RealTensor:
    r"""e^{-iH dt/2} ρ e^{iH dt/2}.

    Uses eigendecomposition for matrix exponentiation.
    """
    eigvals, eigvecs = np.linalg.eigh(hamiltonian)
    phase = np.exp(-1j * eigvals * dt / 2)
    U = eigvecs @ np.diag(phase) @ np.conj(eigvecs).T
    Ud = np.conj(U).T
    B = rho.shape[0]
    out = np.zeros_like(rho)
    for b in range(B):
        r_complex = rho[b].astype(complex)
        out[b] = np.real(U @ r_complex @ Ud)
    return out


def strang_splitting_step(
    rho: RealTensor,
    grad: RealTensor,
    lr: float,
    hamiltonian: RealTensor | None = None,
) -> RealTensor:
    r"""Strang splitting: unitary → dissipative → unitary.

    Parameters
    ----------
    rho : array, shape (B, D, D)
    grad : array, shape (B, D, D)
        Gradient of the dissipative (Lindblad) part.
    lr : float
        Learning rate / time step.
    hamiltonian : array, shape (D, D), optional
        Constraint Hamiltonian H_C (D5.7).  If None, skip unitary steps.
    """
    if hamiltonian is not None:
        rho = _unitary_half_step(rho, hamiltonian, lr)
    rho = projected_gradient_step(rho, grad, lr)
    if hamiltonian is not None:
        rho = _unitary_half_step(rho, hamiltonian, lr)
    return rho


# ================================================================
# Evolution loop
# ================================================================


@dataclass
class EvolutionResult:
    """Container for evolution trajectory."""

    states: list[RealTensor]
    losses: list[float]
    loss_components: list[dict[str, float]]
    converged: bool
    n_steps: int


def evolve(
    loss_fn: CIIRLoss,
    rho_init: RealTensor,
    n_steps: int = 100,
    lr: float = 0.01,
    method: IntegrationMethod = IntegrationMethod.PROJECTED_GRADIENT,
    hamiltonian: RealTensor | None = None,
    tol: float = 1e-6,
    record_every: int = 1,
) -> EvolutionResult:
    r"""Run CIIR state evolution.

    Parameters
    ----------
    loss_fn : CIIRLoss
        The CIIR loss functional.
    rho_init : array, shape (B, D, D)
        Initial density matrices.
    n_steps : int
        Maximum number of evolution steps.
    lr : float
        Learning rate / time step η.
    method : IntegrationMethod
        Integration scheme.
    hamiltonian : array, shape (D, D), optional
        Constraint Hamiltonian for Strang splitting.
    tol : float
        Convergence tolerance on |ΔL|.
    record_every : int
        Record state every N steps.

    Returns
    -------
    EvolutionResult
        Trajectory, losses, convergence flag.
    """
    rho = rho_init.copy()
    states: list[RealTensor] = [rho.copy()]
    losses: list[float] = []
    components: list[dict[str, float]] = []
    prev_loss = float("inf")

    for t in range(n_steps):
        loss_val, grad = loss_fn(rho)
        losses.append(loss_val)
        components.append(loss_fn.loss_components(rho))

        # Check convergence
        if abs(prev_loss - loss_val) < tol and t > 0:
            return EvolutionResult(
                states=states,
                losses=losses,
                loss_components=components,
                converged=True,
                n_steps=t + 1,
            )
        prev_loss = loss_val

        # Step
        if method == IntegrationMethod.GRADIENT:
            rho = gradient_step(rho, grad, lr)
        elif method == IntegrationMethod.PROJECTED_GRADIENT:
            rho = projected_gradient_step(rho, grad, lr)
        elif method == IntegrationMethod.STRANG_SPLITTING:
            rho = strang_splitting_step(rho, grad, lr, hamiltonian)

        if (t + 1) % record_every == 0:
            states.append(rho.copy())

    return EvolutionResult(
        states=states,
        losses=losses,
        loss_components=components,
        converged=False,
        n_steps=n_steps,
    )
