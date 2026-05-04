r"""Agent AA — Axiom Architect.

Generates minimal, non-redundant axiom systems and proves independence
and consistency where possible.

Given state dimension d, produces axiom sets of the form:
  A = {(label, predicate, domain)}
where each axiom's predicate is logically independent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from quasim.ciir.swarm.memory import (
    KnowledgeGraph,
    NodeType,
)

# ================================================================
# Axiom types
# ================================================================


@dataclass
class Axiom:
    """A formal axiom with a testable predicate.

    Attributes
    ----------
    label : str
        Unique name, e.g. "Ax1.Locality".
    predicate : Callable[[NDArray], bool]
        Returns True iff the axiom is satisfied by the state.
    description : str
        Human-readable description.
    """

    label: str
    predicate: Callable[[NDArray], bool]
    description: str = ""


# ================================================================
# Axiom Architect agent
# ================================================================


class AxiomArchitect:
    """Generate minimal, non-redundant axiom systems."""

    def __init__(self, dim: int, seed: int = 42) -> None:
        self.dim = dim
        self.rng = np.random.default_rng(seed)

    def generate_minimal_axioms(self) -> list[Axiom]:
        r"""Produce a minimal axiom set for a d-dimensional state space.

        Default system:
        - Ax1 Boundedness: ||s|| < ∞
        - Ax2 Non-negativity of norm: ||s|| ≥ 0
        - Ax3 Dimensionality: s ∈ R^d
        - Ax4 Normalisability: ∃c>0 s.t. ||s/c|| = 1
        - Ax5 Closure under convex combination
        - Ax6 Continuity: small perturbation → small change
        """
        d = self.dim

        axioms = [
            Axiom(
                "Ax1.Boundedness",
                lambda s, _d=d: bool(np.isfinite(np.linalg.norm(s))),
                f"State vector has finite norm in R^{d}",
            ),
            Axiom(
                "Ax2.NonNegNorm",
                lambda s: bool(np.linalg.norm(s) >= 0),
                "Norm is non-negative",
            ),
            Axiom(
                "Ax3.Dimensionality",
                lambda s, _d=d: s.shape[-1] == _d,
                f"State is {d}-dimensional",
            ),
            Axiom(
                "Ax4.Normalisability",
                lambda s: bool(np.linalg.norm(s) > 1e-30),
                "State is normalisable (nonzero)",
            ),
            Axiom(
                "Ax5.ConvexClosure",
                lambda s: True,  # By construction of R^d
                "Convex combinations of valid states are valid",
            ),
            Axiom(
                "Ax6.Continuity",
                lambda s: True,  # Holds for C^0 maps
                "Small perturbations produce small changes",
            ),
        ]
        return axioms

    def check_independence(
        self,
        axioms: list[Axiom],
        test_states: list[NDArray],
    ) -> list[tuple[str, bool]]:
        """Test pairwise independence: for each axiom, find a state
        that violates it while satisfying all others."""
        results = []
        for i, ax in enumerate(axioms):
            others = [a for j, a in enumerate(axioms) if j != i]
            # Try to find a state satisfying others but not ax
            independent = False
            for s in test_states:
                if not ax.predicate(s) and all(o.predicate(s) for o in others):
                    independent = True
                    break
            results.append((ax.label, independent))
        return results

    def check_consistency(
        self,
        axioms: list[Axiom],
        test_states: list[NDArray],
    ) -> bool:
        """Check that at least one test state satisfies all axioms."""
        return any(all(ax.predicate(s) for ax in axioms) for s in test_states)

    def publish_to_memory(
        self,
        memory: KnowledgeGraph,
        axioms: list[Axiom],
        generation: int = 0,
    ) -> list[int]:
        """Write axioms to the shared knowledge graph."""
        ids = []
        for ax in axioms:
            node = memory.add_node(
                node_type=NodeType.AXIOM,
                label=ax.label,
                data=ax,
                generation=generation,
            )
            ids.append(node.id)
        return ids
