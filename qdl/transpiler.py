"""Transpiler from QDL to Python for deterministic execution."""
from __future__ import annotations

from .parser import Parser
from .ast import BinaryOp, EconomicPrimitive, Number, SafetyGuard, SimulationKernel, WorldModelCall


class Transpiler:
    def __init__(self, source: str):
        self.program = Parser.parse(source)

    def to_python(self) -> str:
        lines = ["def qdl_entry(ctx):"]
        for idx, stmt in enumerate(self.program.statements):
            expr = self._emit(stmt)
            lines.append(f"    _v{idx} = {expr}")
        lines.append(f"    return _v{len(self.program.statements)-1}" if self.program.statements else "    return None")
        return "\n".join(lines)

    def _emit(self, node):
        if isinstance(node, Number):
            return repr(node.value)
        if isinstance(node, BinaryOp):
            return f"({self._emit(node.left)} {node.op} {self._emit(node.right)})"
        if isinstance(node, WorldModelCall):
            args = ", ".join(self._emit(a) for a in node.args)
            return f"ctx['worldmodel'].{node.target}({args})"
        if isinstance(node, SimulationKernel):
            params = ", ".join(f"{k}={self._emit(v)}" for k, v in node.params.items())
            return f"ctx['simulation_kernels']['{node.kernel}']({params})"
        if isinstance(node, EconomicPrimitive):
            return f"ctx['economics']['{node.primitive}']({self._emit(node.amount)})"
        if isinstance(node, SafetyGuard):
            cond = self._emit(node.condition)
            action = self._emit(node.action)
            failure = self._emit(node.failure) if node.failure else "None"
            return f"({action} if {cond} else {failure})"
        return "None"
