r"""Variational formulation: CIIR axioms → loss functional.

Derives the explicit CIIR loss functional from the axiom system:

    L(X) = Σ_i λ_i C_i(X)²  +  Σ_j μ_j |O_j(X) − y_j|²  +  γ · Ω(X)

Justification of each term from CIIR axioms
--------------------------------------------
Term 1 — Constraint penalty  Σ λ_i C_i(ρ)²:
    Follows from Ax4.2 (Constraint Geometry).  The constraint
    manifold M induces hard constraints C_i(ρ) = 0 on admissible
    cognitive states.  The quadratic penalty relaxes these into a
    differentiable loss, with λ_i as Lagrange multiplier weights
    (cf. D5.2, constraint operator algebra).

Term 2 — Observer fit  Σ μ_j |⟨O_j⟩_ρ − y_j|²:
    Follows from Ax4.4 (Observable Algebra) and T8.2 (Born Rule).
    The predicted observable ⟨O_j⟩_ρ = Tr(O_j ρ) must match the
    observed data y_j.  The Born rule (T8.2) supplies the
    prediction; the L² loss measures departure.

Term 3 — Regularisation  γ · Ω(X):
    Follows from Ax4.3 (Epistemic Constraint) and T8.4 (Distortion
    Contraction).  The interface map contracts information; the
    regulariser encodes the entropy bound from T7.6 (Entropy
    Monotonicity):
        Ω(X) = −Tr(ρ log ρ)    (von Neumann entropy)
    ensuring the state does not collapse to a degenerate point.

All gradients are provided in closed form for tensor-compatible
numerical optimisation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from quasim.ciir.theory import (
    CIIRTheory,
    RealTensor,
)


@dataclass
class CIIRLoss:
    r"""CIIR variational loss functional.

    L(ρ) = Σ_i λ_i C_i(ρ)² + Σ_j μ_j |Tr(O_j ρ) − y_j|² + γ · Ω(ρ)

    Parameters
    ----------
    theory : CIIRTheory
        The underlying CIIR theory tuple.
    constraint_weights : array, shape (n_constraints,)
        Lagrange multipliers λ_i.
    observer_weights : array, shape (n_observers,)
        Observer loss weights μ_j.
    observer_targets : array, shape (n_observers,)
        Target values y_j for observers.
    entropy_weight : float
        Regularisation coefficient γ.
    """

    theory: CIIRTheory
    constraint_weights: RealTensor | None = None
    observer_weights: RealTensor | None = None
    observer_targets: RealTensor | None = None
    entropy_weight: float = 0.01

    def __post_init__(self) -> None:
        nc = len(self.theory.constraints)
        no = len(self.theory.observers)
        if self.constraint_weights is None:
            self.constraint_weights = np.ones(nc)
        if self.observer_weights is None:
            self.observer_weights = np.ones(no)
        if self.observer_targets is None:
            self.observer_targets = np.zeros(no)

    # ----------------------------------------------------------
    # Term 1: Constraint penalty  (Ax4.2)
    # ----------------------------------------------------------

    def constraint_loss(self, rho: RealTensor) -> tuple[float, RealTensor]:
        r"""L_C(ρ) = Σ_i λ_i C_i(ρ)².

        Returns
        -------
        loss : float
            Scalar loss summed over batch and constraints.
        grad : array, shape (B, D, D)
            ∂L_C/∂ρ.
        """
        grad = np.zeros_like(rho)
        total = 0.0
        for i, c in enumerate(self.theory.constraints):
            v = c.violation(rho)  # (B,)
            total += float(self.constraint_weights[i] * v.sum())
            grad += self.constraint_weights[i] * c.gradient_rho(rho)
        return total, grad

    # ----------------------------------------------------------
    # Term 2: Observer fit  (Ax4.4, T8.2 Born Rule)
    # ----------------------------------------------------------

    def observer_loss(self, rho: RealTensor) -> tuple[float, RealTensor]:
        r"""L_O(ρ) = Σ_j μ_j |Tr(O_j ρ) − y_j|².

        The prediction Tr(O_j ρ) is the Born-rule expectation (T8.2).

        Returns
        -------
        loss : float
        grad : array, shape (B, D, D)
            ∂L_O/∂ρ = Σ_j 2 μ_j (⟨O_j⟩ − y_j) O_j.
        """
        grad = np.zeros_like(rho)
        total = 0.0
        for j, o in enumerate(self.theory.observers):
            exp_val = o.expectation(rho)  # (B,)
            residual = exp_val - self.observer_targets[j]  # (B,)
            total += float(self.observer_weights[j] * (residual**2).sum())
            # gradient w.r.t. ρ
            grad += (
                2.0
                * self.observer_weights[j]
                * residual[:, None, None]
                * np.real(o.matrix)[None, :, :]
            )
        return total, grad

    # ----------------------------------------------------------
    # Term 3: Entropy regularisation  (Ax4.3, T7.6, T8.4)
    # ----------------------------------------------------------

    @staticmethod
    def _von_neumann_entropy(rho: RealTensor) -> tuple[RealTensor, RealTensor]:
        r"""Ω(ρ) = −Tr(ρ log ρ).

        Uses eigendecomposition for numerical stability.

        Returns
        -------
        S : array, shape (B,)
            Entropy per batch element.
        grad : array, shape (B, D, D)
            ∂Ω/∂ρ = −(log ρ + I).
        """
        B = rho.shape[0]
        S = np.zeros(B)
        grad = np.zeros_like(rho)
        for b in range(B):
            eigvals, eigvecs = np.linalg.eigh(rho[b])
            eigvals = np.clip(eigvals, 1e-12, None)
            S[b] = -np.sum(eigvals * np.log(eigvals))
            # ∂S/∂ρ = −(log ρ + I) in eigenbasis
            log_diag = -(np.log(eigvals) + 1.0)
            grad[b] = eigvecs @ np.diag(log_diag) @ eigvecs.T
        return S, grad

    def entropy_loss(self, rho: RealTensor) -> tuple[float, RealTensor]:
        r"""L_Ω(ρ) = −γ · Σ_b Ω(ρ_b).

        Minimising −entropy ≡ maximising entropy (regularisation).
        """
        S, grad_S = self._von_neumann_entropy(rho)
        loss = -self.entropy_weight * float(S.sum())
        grad = -self.entropy_weight * grad_S
        return loss, grad

    # ----------------------------------------------------------
    # Full loss and gradient
    # ----------------------------------------------------------

    def __call__(self, rho: RealTensor) -> tuple[float, RealTensor]:
        r"""L(ρ) = L_C + L_O + L_Ω.

        Parameters
        ----------
        rho : array, shape (B, D, D)
            Batch of density matrices.

        Returns
        -------
        loss : float
            Total scalar loss.
        grad : array, shape (B, D, D)
            ∂L/∂ρ.
        """
        lc, gc = self.constraint_loss(rho)
        lo, go = self.observer_loss(rho)
        le, ge = self.entropy_loss(rho)
        return lc + lo + le, gc + go + ge

    def loss_components(self, rho: RealTensor) -> dict[str, float]:
        """Return individual loss components for logging."""
        lc, _ = self.constraint_loss(rho)
        lo, _ = self.observer_loss(rho)
        le, _ = self.entropy_loss(rho)
        return {
            "constraint": lc,
            "observer": lo,
            "entropy": le,
            "total": lc + lo + le,
        }
