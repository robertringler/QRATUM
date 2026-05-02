"""AION-SIR: Semantic Intermediate Representation.

This module implements the typed attributed hypergraph that forms
the core of the AION semantic IR.

Version: 1.0.0
Status: Production
"""

from __future__ import annotations

from .edges import EdgeType, HyperEdge
from .hypergraph import GraphBuilder, HyperGraph
from .vertices import Vertex, VertexType

__all__ = [
    "Vertex",
    "VertexType",
    "HyperEdge",
    "EdgeType",
    "HyperGraph",
    "GraphBuilder",
]
