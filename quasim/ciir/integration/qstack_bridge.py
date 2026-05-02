"""qstack bridge: constraint weighting via alignment layer.

Maps CIIR constraint weights λ_i to the qstack alignment/scheduling
system, enabling dynamic reweighting during distributed execution.

QRATUM subsystem: qstack (orchestration & alignment)
CIIR mapping: λ_i ↔ constraint weighting / priority
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from quasim.ciir.theory import CIIRTheory, RealTensor


@dataclass
class QStackConstraintWeighter:
    """Bridge CIIR constraint weights to qstack alignment layer.

    The qstack scheduler assigns priorities and resources to
    computational tasks.  This bridge maps CIIR constraint weights
    λ_i to scheduling priorities, and observer weights μ_j to
    observation importance scores.

    Parameters
    ----------
    theory : CIIRTheory
        The CIIR theory tuple.
    base_priority : float
        Base scheduling priority for CIIR tasks.
    """

    theory: CIIRTheory
    base_priority: float = 1.0

    def constraint_priorities(self, weights: RealTensor | None = None) -> dict[str, float]:
        """Map constraint weights λ_i to scheduling priorities.

        Higher-weighted constraints get higher priority in the
        qstack scheduler.
        """
        if weights is None:
            weights = np.ones(len(self.theory.constraints))

        priorities = {}
        total = float(weights.sum()) or 1.0
        for i, c in enumerate(self.theory.constraints):
            name = c.name or f"C{i}"
            priorities[name] = self.base_priority * float(weights[i]) / total
        return priorities

    def observer_importance(self, weights: RealTensor | None = None) -> dict[str, float]:
        """Map observer weights μ_j to importance scores."""
        if weights is None:
            weights = np.ones(len(self.theory.observers))

        importance = {}
        total = float(weights.sum()) or 1.0
        for j, o in enumerate(self.theory.observers):
            name = o.name or f"O{j}"
            importance[name] = float(weights[j]) / total
        return importance

    def to_qstack_config(self) -> dict[str, Any]:
        """Export as qstack-compatible configuration dict."""
        return {
            "task_type": "ciir_evolution",
            "constraint_priorities": self.constraint_priorities(),
            "observer_importance": self.observer_importance(),
            "base_priority": self.base_priority,
            "n_constraints": len(self.theory.constraints),
            "n_observers": len(self.theory.observers),
        }
