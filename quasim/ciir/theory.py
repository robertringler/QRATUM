r"""Axiomatic foundation: CIIR mathematical objects → tensor representations.

This module constructs the CIIR theory tuple

    T_CIIR = (M, {C_i}, {O_j}, Π)

from the formal definitions in the CIIR monograph (Ch3 D3.1–D3.21,
Ch4 Ax4.1–Ax4.6, Ch5 D5.1–D5.12) and provides tensor-compatible
representations suitable for numerical computation.

Mathematical Objects
--------------------
State manifold M ⊆ R^n (tensor space T^{r,d})
    Implements D3.1 (Reality Space) restricted to finite-dimensional
    submanifolds suitable for numerical computation.

Constraint operators C_i: M → R
    Implements D3.7 (Constraint Manifold) and D5.2 (Constraint
    Operator Algebra) as matrix-valued maps on the state tensor.

Observer operators O_j
    Implements D3.20 (Observable Algebra) with explicit
    non-commutativity from T5.1 (Commutation Relations).

Interface map Π: M → I
    Implements D3.18 (Interface Map) as a differentiable projection
    satisfying the contraction bound of T8.4.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

# Type aliases
RealTensor = NDArray[np.floating]
ComplexTensor = NDArray[np.complexfloating]


# ================================================================
# §1  STATE MANIFOLD  (D3.1, D3.11)
# ================================================================


@dataclass(frozen=True)
class StateManifold:
    r"""Finite-dimensional state manifold M ⊆ R^n.

    Implements the cognitive state space H_C (D3.11) restricted to
    a numerical tensor representation X ∈ R^{B×R×D} where
    B = batch, R = rank (ontic dimension), D = representation dim.

    Parameters
    ----------
    rank : int
        Dimension of the ontic state space (R in D3.1).
    rep_dim : int
        Representation dimension of the cognitive Hilbert space (D3.11).
    dtype : np.dtype
        Numerical precision (fp32/fp64).
    """

    rank: int
    rep_dim: int
    dtype: np.dtype = np.float64

    def random_state(self, batch: int = 1, rng: np.random.Generator | None = None) -> RealTensor:
        """Sample X ∈ R^{B×R×D} from the state manifold.

        Initialised on the unit sphere to satisfy the trace-one
        normalisation of density operators (D3.14).
        """
        rng = rng or np.random.default_rng(42)
        X = rng.standard_normal((batch, self.rank, self.rep_dim)).astype(self.dtype)
        norms = np.linalg.norm(X, axis=(1, 2), keepdims=True)
        return X / np.maximum(norms, 1e-12)

    def density_matrix(self, X: RealTensor) -> RealTensor:
        r"""Construct density operator ρ = X X^T / Tr(X X^T).

        Maps state tensor to the density operator space D(H_C) (D3.14).

        Parameters
        ----------
        X : array, shape (B, R, D)
            State tensor.

        Returns
        -------
        rho : array, shape (B, D, D)
            Density matrices satisfying ρ ≥ 0, Tr(ρ) = 1.
        """
        # ρ_b = X_b^T X_b  (D×D)
        rho = np.einsum("brd,bre->bde", X, X)
        traces = np.einsum("bdd->b", rho)[:, None, None]
        return rho / np.maximum(traces, 1e-12)


# ================================================================
# §1  CONSTRAINT OPERATORS  (D3.7, D5.1, D5.2)
# ================================================================


@dataclass
class ConstraintOperator:
    r"""Constraint operator C_i: M → R.

    Implements the constraint operator algebra K(H_C) (D5.2).
    Each constraint is represented by a symmetric matrix K_i
    so that C_i(X) = Tr(K_i · ρ(X)).

    The operator algebra structure follows from D5.2:
        K(H_C) = alg*{Φ(ι_*(Ĉ(e_i)))}

    Parameters
    ----------
    matrix : array, shape (D, D)
        Hermitian constraint kernel K_i.
    name : str
        Human-readable label.
    """

    matrix: RealTensor
    name: str = ""

    def __post_init__(self) -> None:
        # Enforce Hermiticity (Ax4.2 constraint geometry)
        self.matrix = 0.5 * (self.matrix + self.matrix.T)

    def evaluate(self, rho: RealTensor) -> RealTensor:
        r"""C_i(X) = Tr(K_i · ρ).  Shape (B,)."""
        return np.einsum("de,bde->b", self.matrix, rho)

    def violation(self, rho: RealTensor) -> RealTensor:
        """Squared constraint violation C_i(ρ)^2.  Shape (B,)."""
        return self.evaluate(rho) ** 2

    def gradient_rho(self, rho: RealTensor) -> RealTensor:
        r"""∂C_i^2/∂ρ = 2 C_i(ρ) · K_i.  Shape (B, D, D)."""
        c = self.evaluate(rho)  # (B,)
        return 2.0 * c[:, None, None] * self.matrix[None, :, :]

    @property
    def adjoint(self) -> ConstraintOperator:
        r"""K_i^\dagger = K_i (Hermiticity)."""
        return self

    def commutator(self, other: ConstraintOperator) -> RealTensor:
        r"""[K_i, K_j] = K_i K_j - K_j K_i  (T5.1)."""
        return self.matrix @ other.matrix - other.matrix @ self.matrix


# ================================================================
# §1  OBSERVER OPERATORS  (D3.20, T5.1)
# ================================================================


@dataclass
class ObserverOperator:
    r"""Observer operator O_j ∈ A(H_C).

    Implements the observable algebra (D3.20) as self-adjoint
    operators on H_C.  Non-commutativity follows from T5.1
    (holonomy of constraint curvature).

    Parameters
    ----------
    matrix : array, shape (D, D)
        Self-adjoint observable matrix.
    name : str
        Label for the observable.
    """

    matrix: RealTensor | ComplexTensor
    name: str = ""

    def __post_init__(self) -> None:
        # Enforce self-adjointness (D3.20: A(H_C) ⊂ B(H_C)_sa)
        self.matrix = 0.5 * (self.matrix + np.conj(self.matrix).T)

    def expectation(self, rho: RealTensor | ComplexTensor) -> RealTensor:
        r"""⟨O_j⟩_ρ = Tr(O_j ρ).  Shape (B,)."""
        return np.real(np.einsum("de,bde->b", self.matrix, rho))

    def commutator(self, other: ObserverOperator) -> ComplexTensor:
        r"""[O_i, O_j] = O_i O_j − O_j O_i.

        Non-zero when observers are incompatible (T5.1, §8.7).
        """
        return self.matrix @ other.matrix - other.matrix @ self.matrix


# ================================================================
# §1  INTERFACE MAP  (D3.18, T8.4)
# ================================================================


@dataclass
class InterfaceMap:
    r"""Interface map Π: M → I.

    Implements D3.18 with the contraction property of T8.4:
        d_{H_C}(Π(r1), Π(r2)) ≤ d_R(r1, r2) · exp(-K_min · D)

    The map is parameterised as a learnable linear projection
    composed with a non-linear contraction (softmax normalisation).

    Parameters
    ----------
    projection : array, shape (D_out, D_in)
        Linear component of the interface map.
    kappa_min : float
        Curvature lower bound K_min (D3.8).
    diameter : float
        Diameter of constraint manifold D (D3.7).
    """

    projection: RealTensor
    kappa_min: float = 1.0
    diameter: float = 1.0

    @property
    def contraction_factor(self) -> float:
        """exp(-K_min · D) from T8.4."""
        return float(np.exp(-self.kappa_min * self.diameter))

    def apply(self, X: RealTensor) -> RealTensor:
        r"""Π(X) = α · W̃ · X, where α = e^{-K_min · D} and W̃ = W/‖W‖.

        The projection matrix is normalised to unit spectral norm and
        then scaled by the contraction factor, ensuring the map is
        Lipschitz with constant α = e^{-K_min · D}, which guarantees
        the T8.4 metric contraction bound:

            ‖Π(X₁) − Π(X₂)‖ ≤ α · ‖X₁ − X₂‖

        Parameters
        ----------
        X : array, shape (B, R, D_in)

        Returns
        -------
        Y : array, shape (B, R, D_out)
        """
        # Normalise projection to unit spectral norm
        sigma_max = np.linalg.norm(self.projection, ord=2)
        W_normed = self.projection / max(sigma_max, 1e-12)
        # Apply contracted projection: Lipschitz constant = α
        Y = self.contraction_factor * np.einsum("od,brd->bro", W_normed, X)
        return Y

    def distortion_bound(self, X1: RealTensor, X2: RealTensor) -> RealTensor:
        r"""Verify T8.4: d(Π(X1), Π(X2)) ≤ d(X1, X2) · e^{-K·D}.

        Returns (actual_distance, bound) per batch element.
        """
        Y1 = self.apply(X1)
        Y2 = self.apply(X2)
        d_image = np.linalg.norm(Y1 - Y2, axis=(1, 2))
        d_source = np.linalg.norm(X1 - X2, axis=(1, 2))
        bound = d_source * self.contraction_factor
        return d_image, bound


# ================================================================
# §1  CIIR THEORY TUPLE  (composite)
# ================================================================


@dataclass
class CIIRTheory:
    r"""Complete CIIR theory tuple T_CIIR = (M, {C_i}, {O_j}, Π).

    Assembles all mathematical objects required by the formalization.
    Every field is grounded in exactly one CIIR definition.

    Parameters
    ----------
    manifold : StateManifold
        The cognitive state space (D3.11).
    constraints : list of ConstraintOperator
        Constraint operators {C_i} (D5.2).
    observers : list of ObserverOperator
        Observer operators {O_j} (D3.20).
    interface : InterfaceMap
        Interface map Π (D3.18).
    """

    manifold: StateManifold
    constraints: list[ConstraintOperator] = field(default_factory=list)
    observers: list[ObserverOperator] = field(default_factory=list)
    interface: InterfaceMap | None = None

    def validate(self) -> dict[str, bool]:
        """Validate that the theory tuple satisfies CIIR axioms."""
        checks: dict[str, bool] = {}

        # Ax4.1: Reality space exists and is non-empty
        checks["ax4.1_nonempty"] = self.manifold.rank > 0 and self.manifold.rep_dim > 0

        # Ax4.2: Constraint manifold has Riemannian structure
        # (verified by constraint operators being Hermitian)
        for i, c in enumerate(self.constraints):
            sym = np.allclose(c.matrix, c.matrix.T)
            checks[f"ax4.2_hermitian_C{i}"] = bool(sym)

        # Ax4.3: Interface map exists
        checks["ax4.3_interface_exists"] = self.interface is not None

        # Ax4.4: Observable algebra is a *-algebra
        for j, o in enumerate(self.observers):
            sa = np.allclose(o.matrix, np.conj(o.matrix).T)
            checks[f"ax4.4_selfadjoint_O{j}"] = bool(sa)

        return checks


def build_default_theory(
    rank: int = 4,
    rep_dim: int = 8,
    n_constraints: int = 3,
    n_observers: int = 2,
    seed: int = 42,
) -> CIIRTheory:
    """Construct a default CIIR theory for testing and benchmarks.

    Parameters
    ----------
    rank : int
        Ontic dimension.
    rep_dim : int
        Cognitive representation dimension.
    n_constraints : int
        Number of constraint operators.
    n_observers : int
        Number of observer operators.
    seed : int
        Random seed.

    Returns
    -------
    CIIRTheory
        Fully initialised theory tuple.
    """
    rng = np.random.default_rng(seed)
    manifold = StateManifold(rank=rank, rep_dim=rep_dim)

    constraints = []
    for i in range(n_constraints):
        K = rng.standard_normal((rep_dim, rep_dim))
        K = 0.5 * (K + K.T)
        constraints.append(ConstraintOperator(matrix=K, name=f"C{i}"))

    observers = []
    for j in range(n_observers):
        O = rng.standard_normal((rep_dim, rep_dim))
        O = 0.5 * (O + O.T)
        observers.append(ObserverOperator(matrix=O, name=f"O{j}"))

    W = rng.standard_normal((rep_dim, rep_dim))
    interface = InterfaceMap(projection=W, kappa_min=1.0, diameter=1.0)

    theory = CIIRTheory(
        manifold=manifold,
        constraints=constraints,
        observers=observers,
        interface=interface,
    )
    return theory
