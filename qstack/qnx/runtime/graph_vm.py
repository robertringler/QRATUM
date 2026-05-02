"""Graph-based deterministic execution for operator DAGs with replay and checkpoints."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

from .safety import SafetyEnvelope
from .tracing import TraceRecorder


@dataclass
class OperatorGraph:
    """DAG of operator names with explicit dependencies."""

    edges: dict[str, list[str]] = field(default_factory=dict)
    reverse: dict[str, list[str]] = field(default_factory=dict)

    def add_edge(self, src: str, dst: str) -> None:
        self.edges.setdefault(src, []).append(dst)
        self.reverse.setdefault(dst, []).append(src)
        self.edges.setdefault(dst, [])
        self.reverse.setdefault(src, [])

    def topological(self) -> list[str]:
        indegree: dict[str, int] = {node: len(parents) for node, parents in self.reverse.items()}
        queue = deque(sorted([n for n, deg in indegree.items() if deg == 0]))
        order: list[str] = []
        while queue:
            node = queue.popleft()
            order.append(node)
            for neighbor in self.edges.get(node, []):
                indegree[neighbor] -= 1
                if indegree[neighbor] == 0:
                    queue.append(neighbor)
        if len(order) != len(indegree):
            raise ValueError("cycle detected in operator graph")
        return order


@dataclass
class FaultIsolationZone:
    name: str
    operators: list[str]

    def contains(self, op_name: str) -> bool:
        return op_name in self.operators


class GraphVM:
    """Executes operators following a DAG ordering with deterministic tracing."""

    def __init__(self, operators, graph: OperatorGraph, envelope: SafetyEnvelope | None = None):
        self._operators = operators
        self._graph = graph
        self._trace = TraceRecorder()
        self._tick = 0
        self._envelope = envelope
        self._fault_zones: list[FaultIsolationZone] = []
        self._checkpoints: list[tuple[int, dict[str, Any]]] = []

    def add_fault_zone(self, zone: FaultIsolationZone) -> None:
        self._fault_zones.append(zone)

    def _zone_for(self, op_name: str) -> FaultIsolationZone | None:
        for zone in self._fault_zones:
            if zone.contains(op_name):
                return zone
        return None

    def run(self, state: Any, goal: Any) -> list[dict[str, Any]]:
        trace: list[dict[str, Any]] = []
        for name in self._graph.topological():
            if self._envelope and not self._envelope.inside(state):
                record = {"tick": self._tick, "op": name, "error": "safety_envelope_violation"}
                trace.append(record)
                self._trace.record("safety_violation", record)
                break
            op = self._operators.available().get(name)
            if op is None:
                raise KeyError(f"operator {name} not registered")
            zone = self._zone_for(name)
            result = op.execute(state, goal)
            record: dict[str, Any] = {"tick": self._tick, "op": name, "result": result}
            if zone:
                record["fault_zone"] = zone.name
            self._trace.record("execute", record)
            trace.append(record)
            self._tick += 1
            self._checkpoints.append((self._tick, dict(getattr(state, "data", {}))))
        return trace

    def replay_buffer(self) -> list[dict[str, Any]]:
        return [entry["payload"] for entry in self._trace.snapshot()]

    def checkpoints(self) -> list[tuple[int, dict[str, Any]]]:
        return list(self._checkpoints)
