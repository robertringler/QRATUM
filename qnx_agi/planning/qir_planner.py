"""QIR-driven planner."""

from __future__ import annotations

from typing import List

from qdl.qir.ir_nodes import Graph


class QIRPlanner:
    def __init__(self):
        self.constraints = []

    def plan(self, graphs: List[Graph]) -> List[Graph]:
        return graphs


def plan_qir_graph(graphs: List[Graph]) -> List[Graph]:
    return QIRPlanner().plan(graphs)
