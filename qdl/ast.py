"""AST node definitions for QDL deterministic language."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


class ASTNode:
    def evaluate(self, context: Dict[str, Any]) -> Any:
        raise NotImplementedError

    def children(self) -> List[ASTNode]:
        return []


@dataclass
class Number(ASTNode):
    value: float

    def evaluate(self, context: Dict[str, Any]) -> Any:
        return self.value


@dataclass
class Identifier(ASTNode):
    name: str

    def evaluate(self, context: Dict[str, Any]) -> Any:
        return context.get(self.name)


@dataclass
class BinaryOp(ASTNode):
    op: str
    left: ASTNode
    right: ASTNode

    def evaluate(self, context: Dict[str, Any]) -> Any:
        lval = self.left.evaluate(context)
        rval = self.right.evaluate(context)
        if self.op == "+":
            return lval + rval
        if self.op == "-":
            return lval - rval
        if self.op == "*":
            return lval * rval
        if self.op == "/":
            return lval / rval
        if self.op == "==":
            return lval == rval
        if self.op == "<":
            return lval < rval
        if self.op == ">":
            return lval > rval
        raise ValueError(f"Unknown operator {self.op}")

    def children(self) -> List[ASTNode]:
        return [self.left, self.right]


@dataclass
class WorldModelCall(ASTNode):
    target: str
    args: List[ASTNode] = field(default_factory=list)

    def evaluate(self, context: Dict[str, Any]) -> Any:
        worldmodel = context.get("worldmodel")
        if worldmodel is None or not hasattr(worldmodel, self.target):
            raise ValueError(f"Worldmodel call {self.target} undefined")
        func = getattr(worldmodel, self.target)
        evaluated_args = [a.evaluate(context) for a in self.args]
        return func(*evaluated_args)

    def children(self) -> List[ASTNode]:
        return list(self.args)


@dataclass
class SimulationKernel(ASTNode):
    kernel: str
    params: Dict[str, ASTNode]

    def evaluate(self, context: Dict[str, Any]) -> Any:
        kernel_fn = context.get("simulation_kernels", {}).get(self.kernel)
        if kernel_fn is None:
            raise ValueError(f"Simulation kernel {self.kernel} not available")
        evaluated_params = {k: v.evaluate(context) for k, v in self.params.items()}
        return kernel_fn(**evaluated_params)

    def children(self) -> List[ASTNode]:
        return list(self.params.values())


@dataclass
class EconomicPrimitive(ASTNode):
    primitive: str
    amount: ASTNode

    def evaluate(self, context: Dict[str, Any]) -> Any:
        econ = context.get("economics", {})
        primitive_fn = econ.get(self.primitive)
        if primitive_fn is None:
            raise ValueError(f"Economic primitive {self.primitive} missing")
        return primitive_fn(self.amount.evaluate(context))

    def children(self) -> List[ASTNode]:
        return [self.amount]


@dataclass
class SafetyGuard(ASTNode):
    condition: ASTNode
    action: ASTNode
    failure: Optional[ASTNode] = None

    def evaluate(self, context: Dict[str, Any]) -> Any:
        if self.condition.evaluate(context):
            return self.action.evaluate(context)
        if self.failure:
            return self.failure.evaluate(context)
        raise ValueError("Safety guard triggered without fallback")

    def children(self) -> List[ASTNode]:
        nodes = [self.condition, self.action]
        if self.failure:
            nodes.append(self.failure)
        return nodes


@dataclass
class Program(ASTNode):
    statements: List[ASTNode]

    def evaluate(self, context: Dict[str, Any]) -> Any:
        result = None
        for stmt in self.statements:
            result = stmt.evaluate(context)
        return result

    def children(self) -> List[ASTNode]:
        return list(self.statements)
