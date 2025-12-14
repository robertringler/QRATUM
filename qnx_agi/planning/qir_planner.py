"""QIR-driven planner."""
from __future__ import annotations


from qdl.qir.ir_nodes import Graph


class QIRPlanner:
    def __init__(self):
        self.constraints = []

    def plan(self, graphs: list[Graph]) -> list[Graph]:
        return graphs


def plan_qir_graph(graphs: list[Graph]) -> list[Graph]:
    return QIRPlanner().plan(graphs)
