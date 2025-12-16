"""Optimizer for QIR graphs."""

from __future__ import annotations

from .ir_nodes import Constant, Graph, Operation


def constant_fold(graph: Graph) -> Graph:
    for node in graph.nodes():
        if isinstance(node, Operation) and node.op_type in {"add", "sub", "mul", "div"}:
            if all(isinstance(inp, Constant) for inp in node.inputs):
                values = [inp.value for inp in node.inputs]
                result = None
                if node.op_type == "add":
                    result = values[0] + values[1]
                elif node.op_type == "sub":
                    result = values[0] - values[1]
                elif node.op_type == "mul":
                    result = values[0] * values[1]
                elif node.op_type == "div":
                    result = values[0] / values[1]
                node.inputs.clear()
                node.__class__ = Constant
                node.__dict__.update(
                    {"name": "const", "value": result, "attributes": {"value": result}}
                )
    return graph


def fuse(graph: Graph) -> Graph:
    # placeholder deterministic fusion preserving order
    return graph


def optimize(graph: Graph) -> Graph:
    graph = constant_fold(graph)
    graph = fuse(graph)
    return graph
