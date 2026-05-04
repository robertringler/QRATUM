r"""Agent FZ — Formalizer.

Converts all outputs into strict symbolic mathematics using algebraic,
geometric, or category-theoretic structures.

Constructs formal representations:
- Algebraic: operator algebras, group actions
- Geometric: metric tensors, connection forms
- Category-theoretic: functors, natural transformations
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from quasim.ciir.swarm.axiom_architect import Axiom
from quasim.ciir.swarm.memory import (
    KnowledgeGraph,
    NodeType,
)
from quasim.ciir.swarm.physics_lang import RuleDecl

# ================================================================
# Formal structures
# ================================================================


@dataclass
class FormalStructure:
    """A formalized mathematical structure.

    Attributes
    ----------
    name : str
    structure_type : str
        One of 'algebra', 'metric', 'category', 'group', 'manifold'.
    components : dict
        Named mathematical components (matrices, tensors, maps).
    properties : list of str
        Verified formal properties.
    """

    name: str
    structure_type: str
    components: dict = field(default_factory=dict)
    properties: list[str] = field(default_factory=list)


# ================================================================
# Formalizer agent
# ================================================================


class Formalizer:
    """Convert computational objects into strict symbolic structures."""

    def __init__(self, dim: int) -> None:
        self.dim = dim

    def formalize_axiom_system(
        self,
        axioms: list[Axiom],
    ) -> FormalStructure:
        """Map axiom set to an algebraic structure.

        The axiom system defines a convex state space which we formalize
        as a positive cone in a C*-algebra.
        """
        return FormalStructure(
            name="AxiomAlgebra",
            structure_type="algebra",
            components={
                "dimension": self.dim,
                "axiom_count": len(axioms),
                "axiom_labels": [a.label for a in axioms],
                "identity": np.eye(self.dim),
            },
            properties=[
                "convex",
                "bounded",
                "normalisable",
            ],
        )

    def construct_metric_tensor(
        self,
        states: list[NDArray],
    ) -> NDArray:
        r"""Construct Fisher information metric from state ensemble.

        g_{ij} = E[∂_i log p · ∂_j log p]

        Approximated via covariance of normalized log-states.
        """
        if not states:
            return np.eye(self.dim)

        mat = np.array(states)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms = np.where(norms < 1e-30, 1.0, norms)
        normalized = mat / norms

        # log-state covariance as Fisher approximation
        log_states = np.log(np.abs(normalized) + 1e-30)
        mean = log_states.mean(axis=0)
        centered = log_states - mean
        g = (centered.T @ centered) / max(len(states) - 1, 1)
        return g

    def formalize_rules(
        self,
        rules: list[RuleDecl],
    ) -> FormalStructure:
        """Convert a rule set into an operator algebra description."""
        return FormalStructure(
            name="RuleOperatorAlgebra",
            structure_type="algebra",
            components={
                "rule_count": len(rules),
                "rule_names": [r.name for r in rules],
                "mode_types": [r.mode.name for r in rules],
            },
            properties=[
                "compositional",
                "associative",
            ],
        )

    def derive_category_structure(
        self,
        axiom_struct: FormalStructure,
        rule_struct: FormalStructure,
    ) -> FormalStructure:
        r"""Construct the category whose objects are states and
        morphisms are rule applications.

        Objects: States satisfying axioms
        Morphisms: Rule-induced state transitions
        Composition: Sequential rule application
        Identity: No-op rule
        """
        return FormalStructure(
            name="PhysicsCategory",
            structure_type="category",
            components={
                "objects": axiom_struct.name,
                "morphisms": rule_struct.name,
                "composition": "sequential",
                "identity": "identity_rule",
            },
            properties=[
                "well_defined_composition",
                "identity_morphism",
                "associative",
            ],
        )

    def publish_to_memory(
        self,
        memory: KnowledgeGraph,
        structures: list[FormalStructure],
        generation: int = 0,
    ) -> list[int]:
        """Write formal structures to knowledge graph."""
        ids = []
        for s in structures:
            node = memory.add_node(
                node_type=NodeType.THEORY,
                label=s.name,
                data=s,
                generation=generation,
            )
            ids.append(node.id)
        return ids
