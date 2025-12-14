"""IR nodes for QDL/QIR."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class IRNode:
    name: str
    inputs: List[IRNode] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)

    def add_input(self, node: IRNode):
        self.inputs.append(node)


@dataclass
class Constant(IRNode):
    value: Any = None

    def __init__(self, value: Any):
        super().__init__("const", [], {"value": value})
        self.value = value


@dataclass
class Operation(IRNode):
    op_type: str = ""

    def __init__(
        self,
        op_type: str,
        inputs: Optional[List[IRNode]] = None,
        attributes: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(op_type, inputs or [], attributes or {})
        self.op_type = op_type


@dataclass
class SafetyBarrier(IRNode):
    def __init__(self, condition: IRNode, on_success: IRNode, on_fail: Optional[IRNode] = None):
        inputs = [condition, on_success]
        if on_fail:
            inputs.append(on_fail)
        super().__init__("safety_barrier", inputs, {})


@dataclass
class Graph:
    outputs: List[IRNode]

    def nodes(self) -> List[IRNode]:
        visited = []

        def walk(node: IRNode):
            if node in visited:
                return
            visited.append(node)
            for child in node.inputs:
                walk(child)

        for out in self.outputs:
            walk(out)
        return visited
