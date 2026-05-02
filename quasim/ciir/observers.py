r"""Non-commutative observer formalism with sequential measurement effects.

Implements §7 of the CIIR→QuASIM formalization:

    [O_i, O_j] = O_i O_j − O_j O_i ≠ 0

Sequential measurement effects (T8.3, T8.6, T8.7):
    - State collapse / projection operators (PVM, D8.2)
    - POVM generalisation (D8.10)
    - Measurement update: ρ → P_o ρ P_o / Tr(P_o ρ)  (T8.3)
    - Interference from non-commutativity (T8.6)
    - Contextuality: outcome probabilities depend on context (T8.7)

References
----------
CIIR Monograph:
    D3.20 (Observable Algebra), D8.1–D8.22 (Measurement Theory),
    T8.1–T8.9, T5.1 (Commutation Relations)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from quasim.ciir.theory import ComplexTensor, ObserverOperator, RealTensor

# ================================================================
# Projection-Valued Measure (PVM)  — D8.2, T8.1
# ================================================================


@dataclass
class PVM:
    r"""Projection-valued measure constructed from an observable.

    From T8.1: every self-adjoint A ∈ A(H_C) admits a spectral
    decomposition A = Σ_o a_o P_o where {P_o} is a PVM.

    Parameters
    ----------
    eigenvalues : array, shape (n_outcomes,)
        Distinct eigenvalues {a_o}.
    projectors : array, shape (n_outcomes, D, D)
        Orthogonal projectors {P_o} with P_o² = P_o, Σ P_o = I.
    """

    eigenvalues: RealTensor
    projectors: RealTensor

    @classmethod
    def from_observable(cls, observable: ObserverOperator) -> PVM:
        """Construct PVM from spectral decomposition of an observable (T8.1)."""
        O = np.real(observable.matrix) if np.isrealobj(observable.matrix) else observable.matrix
        eigvals, eigvecs = np.linalg.eigh(O)

        # Group degenerate eigenvalues
        unique_vals = []
        projectors = []
        tol = 1e-10
        i = 0
        while i < len(eigvals):
            val = eigvals[i]
            # Find degenerate subspace
            j = i
            while j < len(eigvals) and abs(eigvals[j] - val) < tol:
                j += 1
            vecs = eigvecs[:, i:j]
            P = vecs @ vecs.T
            unique_vals.append(val)
            projectors.append(P)
            i = j

        return cls(
            eigenvalues=np.array(unique_vals),
            projectors=np.stack(projectors, axis=0),
        )

    def validate(self) -> dict[str, bool]:
        """Verify PVM axioms: idempotence, orthogonality, completeness."""
        checks = {}
        n = len(self.eigenvalues)
        D = self.projectors.shape[1]

        # Idempotence: P_o² = P_o
        for o in range(n):
            P = self.projectors[o]
            checks[f"idempotent_{o}"] = bool(np.allclose(P @ P, P, atol=1e-10))

        # Orthogonality: P_o P_{o'} = 0 for o ≠ o'
        for o in range(n):
            for p in range(o + 1, n):
                prod = self.projectors[o] @ self.projectors[p]
                checks[f"orthogonal_{o}_{p}"] = bool(np.allclose(prod, 0, atol=1e-10))

        # Completeness: Σ P_o = I
        total = self.projectors.sum(axis=0)
        checks["complete"] = bool(np.allclose(total, np.eye(D), atol=1e-10))

        return checks


# ================================================================
# Born Rule  — T8.2
# ================================================================


def born_probability(rho: RealTensor, projector: RealTensor) -> RealTensor:
    r"""p(o) = Tr(P_o ρ).

    Derived in T8.2 from interface map + trace structure + positivity.

    Parameters
    ----------
    rho : array, shape (B, D, D) or (D, D)
    projector : array, shape (D, D)

    Returns
    -------
    p : array, shape (B,) or scalar
    """
    if rho.ndim == 2:
        return float(np.trace(projector @ rho))
    return np.einsum("de,bde->b", projector, rho)


def born_distribution(rho: RealTensor, pvm: PVM) -> RealTensor:
    r"""Full probability distribution p(o) = Tr(P_o ρ) for all outcomes.

    Parameters
    ----------
    rho : array, shape (B, D, D) or (D, D)
    pvm : PVM

    Returns
    -------
    probs : array, shape (B, n_outcomes) or (n_outcomes,)
    """
    if rho.ndim == 2:
        return np.array([born_probability(rho, pvm.projectors[o])
                         for o in range(len(pvm.eigenvalues))])
    return np.stack([born_probability(rho, pvm.projectors[o])
                     for o in range(len(pvm.eigenvalues))], axis=1)


# ================================================================
# Measurement Update (Collapse)  — T8.3
# ================================================================


def measurement_update(rho: RealTensor, projector: RealTensor) -> RealTensor:
    r"""Post-measurement state: ρ → P_o ρ P_o / Tr(P_o ρ).

    This update is:
    - Completely positive (P_o · P_o is a single Kraus operator)
    - Trace-preserving (conditional on outcome o)
    - Consistent with CIIR dynamics (T8.8)

    Parameters
    ----------
    rho : array, shape (B, D, D) or (D, D)
    projector : array, shape (D, D)

    Returns
    -------
    rho_post : same shape as rho
    """
    if rho.ndim == 2:
        numerator = projector @ rho @ projector
        denom = np.trace(numerator)
        if denom < 1e-12:
            return rho  # outcome has zero probability; no update
        return numerator / denom

    B = rho.shape[0]
    numerator = np.einsum("de,bef,fg->bdg", projector, rho, projector)
    denom = np.einsum("bdd->b", numerator)
    denom = np.maximum(denom, 1e-12)
    return numerator / denom[:, None, None]


def sequential_measurement(
    rho: RealTensor, observables: Sequence[ObserverOperator]
) -> tuple[RealTensor, list[RealTensor]]:
    r"""Apply sequential measurements, demonstrating non-commutative effects.

    When [O_i, O_j] ≠ 0, the order of measurement matters (T8.7).

    Parameters
    ----------
    rho : array, shape (D, D)
        Initial state.
    observables : sequence of ObserverOperator
        Observables to measure sequentially.

    Returns
    -------
    rho_final : array, shape (D, D)
        State after all measurements (collapsed into most probable outcome
        at each step).
    distributions : list of array
        Probability distribution at each step.
    """
    distributions = []
    state = rho.copy()
    for obs in observables:
        pvm = PVM.from_observable(obs)
        probs = born_distribution(state, pvm)
        distributions.append(probs)
        # Collapse to most probable outcome
        best = int(np.argmax(probs))
        state = measurement_update(state, pvm.projectors[best])
    return state, distributions


# ================================================================
# Interference  — T8.6
# ================================================================


def interference_term(
    psi1: RealTensor, psi2: RealTensor, O: RealTensor | ComplexTensor
) -> float:
    r"""Compute Re⟨ψ₁|O|ψ₂⟩.

    From T8.6: E_{ψ±}[O] = ½(E_{ψ₁} + E_{ψ₂}) ± Re⟨ψ₁|O|ψ₂⟩

    Parameters
    ----------
    psi1, psi2 : array, shape (D,)
        Pure state vectors.
    O : array, shape (D, D)
        Observable matrix.

    Returns
    -------
    interference : float
    """
    return float(np.real(psi1.conj() @ O @ psi2))


def superposition_expectation(
    psi1: RealTensor, psi2: RealTensor, O: RealTensor | ComplexTensor, sign: int = 1
) -> float:
    r"""E_{ψ±}[O] for ψ± = (ψ₁ ± ψ₂)/√2.

    Implements T8.6:
        E_{ψ±}[O] = ½(⟨ψ₁|O|ψ₁⟩ + ⟨ψ₂|O|ψ₂⟩) ± Re⟨ψ₁|O|ψ₂⟩

    Parameters
    ----------
    psi1, psi2 : array, shape (D,)
    O : array, shape (D, D)
    sign : +1 or -1

    Returns
    -------
    expectation : float
    """
    e1 = float(np.real(psi1.conj() @ O @ psi1))
    e2 = float(np.real(psi2.conj() @ O @ psi2))
    cross = interference_term(psi1, psi2, O)
    return 0.5 * (e1 + e2) + sign * cross


# ================================================================
# Contextuality  — T8.7
# ================================================================


@dataclass
class MeasurementContext:
    r"""Measurement context: a maximal set of compatible observables.

    From D8.15: a context C = {O₁, …, O_k} with [O_i, O_j] = 0 for all i, j.

    Parameters
    ----------
    observables : list of ObserverOperator
        Pairwise-commuting observables defining this context.
    """

    observables: list[ObserverOperator]

    def is_compatible(self, tol: float = 1e-10) -> bool:
        """Check all pairs commute: [O_i, O_j] = 0."""
        for i, oi in enumerate(self.observables):
            for j in range(i + 1, len(self.observables)):
                oj = self.observables[j]
                comm = oi.commutator(oj)
                if not np.allclose(comm, 0, atol=tol):
                    return False
        return True

    def joint_pvm(self) -> list[PVM]:
        """Construct PVM for each observable in the context."""
        return [PVM.from_observable(o) for o in self.observables]


def contextuality_witness(
    rho: RealTensor,
    context1: MeasurementContext,
    context2: MeasurementContext,
    shared_observable: ObserverOperator,
) -> float:
    r"""Measure context-dependence of outcome probabilities.

    From T8.7: if O belongs to two contexts C₁ and C₂ with
    non-commuting members, the full joint distribution
    may differ even though the marginal on O is the same.

    Returns the total variation distance between the marginal
    distributions on shared_observable in both contexts.
    """
    pvm = PVM.from_observable(shared_observable)

    # In context 1: first measure other observables, then shared
    rho1 = rho.copy()
    for obs in context1.observables:
        if obs is not shared_observable:
            pvm_other = PVM.from_observable(obs)
            probs = born_distribution(rho1, pvm_other)
            best = int(np.argmax(probs))
            rho1 = measurement_update(rho1, pvm_other.projectors[best])
    p1 = born_distribution(rho1, pvm)

    # In context 2
    rho2 = rho.copy()
    for obs in context2.observables:
        if obs is not shared_observable:
            pvm_other = PVM.from_observable(obs)
            probs = born_distribution(rho2, pvm_other)
            best = int(np.argmax(probs))
            rho2 = measurement_update(rho2, pvm_other.projectors[best])
    p2 = born_distribution(rho2, pvm)

    # Total variation distance
    return float(0.5 * np.sum(np.abs(p1 - p2)))
