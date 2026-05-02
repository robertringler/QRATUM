"""Compiler pipeline for QDL to QIR."""

from __future__ import annotations

from .ast import (BinaryOp, EconomicPrimitive, Number, Program, SafetyGuard,
                  SimulationKernel, WorldModelCall)
from .parser import Parser
from .qir import lowering, optimizer, verifier
from .qir.ir_nodes import Constant, Graph, Operation, SafetyBarrier


class Compiler:
    def __init__(self, source: str):
        self.source = source
        self.program = Parser.parse(source)

    def compile(self) -> Graph:
        outputs = [self._compile_node(stmt) for stmt in self.program.statements]
        graph = Graph(outputs)
        verifier.verify(graph)
        return optimizer.optimize(graph)

    def lower(self):
        graph = self.compile()
        return lowering.lower_to_python(graph)

    def _compile_node(self, node):
        if isinstance(node, Number):
            return Constant(node.value)
        if isinstance(node, BinaryOp):
            left = self._compile_node(node.left)
            right = self._compile_node(node.right)
            op_map = {
                "+": "add",
                "-": "sub",
                "*": "mul",
                "/": "div",
                "==": "eq",
                "<": "lt",
                ">": "gt",
            }
            op_type = op_map.get(node.op, node.op)
            return Operation(op_type, [left, right])
        if isinstance(node, SimulationKernel):
            params = {k: self._compile_node(v) for k, v in node.params.items()}
            op = Operation(
                "simulate", list(params.values()), {"kernel": node.kernel, "params": params}
            )
            return op
        if isinstance(node, WorldModelCall):
            args = [self._compile_node(a) for a in node.args]
            return Operation("worldmodel", args, {"target": node.target})
        if isinstance(node, EconomicPrimitive):
            amount = self._compile_node(node.amount)
            return Operation("economic", [amount], {"primitive": node.primitive})
        if isinstance(node, SafetyGuard):
            cond = self._compile_node(node.condition)
            action = self._compile_node(node.action)
            failure = self._compile_node(node.failure) if node.failure else None
            return SafetyBarrier(cond, action, failure)
        if isinstance(node, Program):
            compiled = [self._compile_node(s) for s in node.statements]
            return Graph(compiled)
        raise TypeError(f"Unsupported node type: {type(node)}")
