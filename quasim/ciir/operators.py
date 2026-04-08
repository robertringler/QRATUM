"""Operator system for the CIIR → QRATUM adaptive measurement framework.

Stores and evolves a set of non-commuting measurement operators according
to the QRATUM recursive update rule:

    M_i^{t+1} = M_i^{t} + η (ρ_t M_i^{t} − M_i^{t} ρ_t)

After each update operators are re-normalized to Frobenius unit norm.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def commutator(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute [A, B] = AB − BA."""
    return a @ b - b @ a


def frobenius_norm(m: np.ndarray) -> float:
    """Frobenius norm ‖M‖_F."""
    return float(np.linalg.norm(m, "fro"))


def _normalize_operator(m: np.ndarray) -> np.ndarray:
    """Normalize an operator to Frobenius unit norm."""
    nrm = frobenius_norm(m)
    if nrm < 1e-15:
        return m
    return m / nrm


@dataclass
class OperatorSystem:
    """Container for a set of adaptive measurement operators.

    Attributes:
        operators: Mapping from label to complex matrix.
        eta: QRATUM learning rate.
    """

    operators: dict[str, np.ndarray] = field(default_factory=dict)
    eta: float = 0.05

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def random(
        cls,
        labels: list[str],
        dim: int,
        eta: float = 0.05,
        rng: np.random.Generator | None = None,
    ) -> OperatorSystem:
        """Create an operator set from random complex matrices.

        Args:
            labels: Operator labels (e.g. ``["A", "B", "C"]``).
            dim: Hilbert-space dimension.
            eta: Learning rate.
            rng: NumPy random generator.

        Returns:
            Initialized :class:`OperatorSystem`.
        """
        if rng is None:
            rng = np.random.default_rng(42)
        ops: dict[str, np.ndarray] = {}
        for label in labels:
            real = rng.standard_normal((dim, dim))
            imag = rng.standard_normal((dim, dim))
            ops[label] = _normalize_operator(real + 1j * imag)
        return cls(operators=ops, eta=eta)

    # ------------------------------------------------------------------
    # QRATUM update
    # ------------------------------------------------------------------

    def qratum_update(self, rho: np.ndarray) -> dict[str, float]:
        """Apply one QRATUM operator update step.

        Args:
            rho: Current density matrix.

        Returns:
            Dictionary mapping each label to its Frobenius drift.
        """
        drifts: dict[str, float] = {}
        for label, m in self.operators.items():
            delta = self.eta * (rho @ m - m @ rho)
            new_m = _normalize_operator(m + delta)
            drifts[label] = frobenius_norm(new_m - m)
            self.operators[label] = new_m
        return drifts

    # ------------------------------------------------------------------
    # Commutator norms
    # ------------------------------------------------------------------

    def commutator_norms(self) -> dict[tuple[str, str], float]:
        """Compute pairwise commutator Frobenius norms."""
        labels = sorted(self.operators)
        result: dict[tuple[str, str], float] = {}
        for i, li in enumerate(labels):
            for lj in labels[i + 1 :]:
                c = commutator(self.operators[li], self.operators[lj])
                result[(li, lj)] = frobenius_norm(c)
        return result

    # ------------------------------------------------------------------
    # Snapshot
    # ------------------------------------------------------------------

    def snapshot(self) -> dict[str, np.ndarray]:
        """Return a deep copy of the current operators."""
        return {k: v.copy() for k, v in self.operators.items()}
