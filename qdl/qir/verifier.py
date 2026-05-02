"""Verifier for QIR graphs."""

from __future__ import annotations

from .ir_nodes import Graph


def verify(graph: Graph) -> None:
    for node in graph.nodes():
        if node.name == "safety_barrier" and len(node.inputs) < 2:
            raise ValueError("Safety barrier missing mandatory edges")
