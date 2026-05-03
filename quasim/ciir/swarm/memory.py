r"""Shared Memory Structure — typed knowledge graph M = (N, E, τ).

Every agent reads/writes to this global structure.

Node types: Axiom, Rule, Theory, SimulationOutput, Invariant, Observable
Edge types: Derivation, Dependency, Contradiction, Equivalence
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any

# ================================================================
# Type enumerations
# ================================================================


class NodeType(Enum):
    """Knowledge graph node types."""

    AXIOM = auto()
    RULE = auto()
    THEORY = auto()
    SIMULATION_OUTPUT = auto()
    INVARIANT = auto()
    OBSERVABLE = auto()
    PREDICTION = auto()


class EdgeType(Enum):
    """Knowledge graph edge types."""

    DERIVATION = auto()
    DEPENDENCY = auto()
    CONTRADICTION = auto()
    EQUIVALENCE = auto()
    GENERATES = auto()
    VALIDATES = auto()


# ================================================================
# Primitives
# ================================================================


@dataclass
class KNode:
    """Knowledge graph node.

    Attributes
    ----------
    id : int
        Unique identifier.
    node_type : NodeType
        Semantic type of this knowledge item.
    label : str
        Human-readable label.
    data : Any
        Typed payload (axiom dict, rule object, simulation array, etc.).
    confidence : float
        Confidence score ∈ [0, 1].
    generation : int
        Evolution cycle in which this node was created.
    """

    id: int
    node_type: NodeType
    label: str
    data: Any = None
    confidence: float = 1.0
    generation: int = 0


@dataclass
class KEdge:
    """Knowledge graph directed edge.

    Attributes
    ----------
    source : int
        Source node id.
    target : int
        Target node id.
    edge_type : EdgeType
        Semantic relation.
    weight : float
        Strength/confidence of the relation.
    """

    source: int
    target: int
    edge_type: EdgeType
    weight: float = 1.0


# ================================================================
# Knowledge graph
# ================================================================


class KnowledgeGraph:
    r"""Global typed knowledge graph M = (N, E, τ).

    Thread-safe read/write interface for all swarm agents.
    """

    def __init__(self) -> None:
        self.nodes: dict[int, KNode] = {}
        self.edges: list[KEdge] = []
        self._next_id: int = 0

    # ---- CRUD operations ----------------------------------------

    def add_node(
        self,
        node_type: NodeType,
        label: str,
        data: Any = None,
        confidence: float = 1.0,
        generation: int = 0,
    ) -> KNode:
        """Create and insert a new node."""
        node = KNode(
            id=self._next_id,
            node_type=node_type,
            label=label,
            data=data,
            confidence=confidence,
            generation=generation,
        )
        self.nodes[node.id] = node
        self._next_id += 1
        return node

    def add_edge(
        self,
        source: int,
        target: int,
        edge_type: EdgeType,
        weight: float = 1.0,
    ) -> KEdge:
        """Create and insert a new edge."""
        edge = KEdge(source=source, target=target, edge_type=edge_type, weight=weight)
        self.edges.append(edge)
        return edge

    def get_nodes_by_type(self, node_type: NodeType) -> list[KNode]:
        """Return all nodes of a given type."""
        return [n for n in self.nodes.values() if n.node_type == node_type]

    def get_edges_from(self, node_id: int) -> list[KEdge]:
        """Return all edges originating from node_id."""
        return [e for e in self.edges if e.source == node_id]

    def get_edges_to(self, node_id: int) -> list[KEdge]:
        """Return all edges targeting node_id."""
        return [e for e in self.edges if e.target == node_id]

    def has_contradiction(self, node_a: int, node_b: int) -> bool:
        """Check whether a contradiction edge exists between two nodes."""
        return any(
            e
            for e in self.edges
            if e.edge_type == EdgeType.CONTRADICTION
            and (
                (e.source == node_a and e.target == node_b)
                or (e.source == node_b and e.target == node_a)
            )
        )

    @property
    def size(self) -> int:
        """Number of nodes."""
        return len(self.nodes)

    def summary(self) -> dict[str, int]:
        """Return counts of each node type."""
        from collections import Counter

        counts = Counter(n.node_type.name for n in self.nodes.values())
        return dict(counts)
