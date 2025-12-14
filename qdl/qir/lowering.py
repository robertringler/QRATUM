"""Lowering QIR graphs to deterministic Python callables."""

from __future__ import annotations

from typing import Callable

from .ir_nodes import Constant, Graph, Operation


def lower_to_python(graph: Graph) -> Callable:
    def execute():
        values = {}
        for node in graph.nodes():
            if isinstance(node, Constant):
                values[id(node)] = node.value
            elif isinstance(node, Operation):
                inputs = [values[id(inp)] for inp in node.inputs]
                if node.op_type == "add":
                    values[id(node)] = inputs[0] + inputs[1]
                elif node.op_type == "sub":
                    values[id(node)] = inputs[0] - inputs[1]
                elif node.op_type == "mul":
                    values[id(node)] = inputs[0] * inputs[1]
                elif node.op_type == "div":
                    values[id(node)] = inputs[0] / inputs[1]
                else:
                    values[id(node)] = inputs
        outputs = [values.get(id(out), None) for out in graph.outputs]
        return outputs[0] if len(outputs) == 1 else outputs

    return execute
