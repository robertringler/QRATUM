from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

from ..utils.provenance import hash_payload
from .quasim_adapter import translate_simulation_output


@dataclass(frozen=True)
class WorldEvent:
    """Structured, provenance-aware event."""

    label: str
    payload: dict[str, Any]
    digest: str


@dataclass(frozen=True)
class WorldState:
    facts: dict[str, Any]
    digest: str
    parent: Optional[str] = None

    def project(self, updates: dict[str, Any]) -> "WorldState":
        merged = dict(self.facts)
        merged.update(updates)
        digest = hash_payload({"facts": merged, "parent": self.digest})
        return WorldState(facts=merged, digest=digest, parent=self.digest)


class WorldStateGraph:
    """Minimal DAG tracking state transitions for auditability."""

    def __init__(self, genesis: WorldState):
        self._nodes: dict[str, WorldState] = {genesis.digest: genesis}
        self._edges: dict[str, list[str]] = {genesis.digest: []}

    def add_transition(self, parent: WorldState, child: WorldState) -> None:
        if parent.digest not in self._nodes:
            self._nodes[parent.digest] = parent
            self._edges.setdefault(parent.digest, [])
        self._nodes[child.digest] = child
        self._edges.setdefault(child.digest, [])
        self._edges[parent.digest].append(child.digest)

    def latest(self, digest: str) -> WorldState:
        return self._nodes[digest]

    def lineage(self, digest: str) -> list[WorldState]:
        lineage: list[WorldState] = []
        cursor = digest
        visited = set()
        while cursor and cursor not in visited:
            node = self._nodes[cursor]
            lineage.append(node)
            visited.add(cursor)
            cursor = node.parent
        return lineage

    def as_dict(self) -> dict[str, Any]:
        return {"nodes": list(self._nodes.keys()), "edges": self._edges}


class WorldModel:
    """Deterministic world model with domain dynamics and simulation hooks."""

    def __init__(self, dynamics: Optional[dict[str, Callable[[dict[str, Any]], dict[str, Any]]]] = None):
        genesis = WorldState(facts={}, digest=hash_payload({}), parent=None)
        self._history: list[WorldState] = [genesis]
        self._graph = WorldStateGraph(genesis)
        self._dynamics = dynamics or {}
        self._events: list[WorldEvent] = []

    @property
    def current(self) -> WorldState:
        return self._history[-1]

    def register_dynamics(self, domain: str, fn: Callable[[dict[str, Any]], dict[str, Any]]) -> None:
        self._dynamics[domain] = fn

    def record_event(self, label: str, payload: dict[str, Any]) -> WorldEvent:
        digest = hash_payload({"label": label, "payload": payload, "parent": self.current.digest})
        event = WorldEvent(label=label, payload=payload, digest=digest)
        self._events.append(event)
        return event

    def ingest(self, percepts: list[Any]) -> WorldState:
        updates = {f"percept_{i}": p.value for i, p in enumerate(percepts)}
        event = self.record_event("perception", updates)
        next_state = self.current.project({"percepts": updates, "event": event.digest})
        self._history.append(next_state)
        self._graph.add_transition(self.current, next_state)
        return next_state

    def simulate_step(self, domain: str, state_override: Optional[dict[str, Any]] = None) -> WorldState:
        if domain not in self._dynamics:
            raise ValueError(f"No dynamics registered for domain {domain}")
        base_state = state_override or self.current.facts
        updates = self._dynamics[domain](dict(base_state))
        event = self.record_event(f"simulate_{domain}", updates)
        next_state = self.current.project({"sim": {domain: updates}, "event": event.digest})
        self._history.append(next_state)
        self._graph.add_transition(self.current, next_state)
        return next_state

    def predict_with_quasim(self, simulation_result: dict[str, Any]) -> WorldState:
        translated = translate_simulation_output(simulation_result)
        event = self.record_event("quasim_simulation", translated)
        next_state = self.current.project({"simulation": translated, "event": event.digest})
        self._history.append(next_state)
        self._graph.add_transition(self.current, next_state)
        return next_state

    def apply_qunimbus_score(self, valuation: dict[str, Any]) -> WorldState:
        event = self.record_event("qunimbus_score", valuation)
        next_state = self.current.project({"valuation": valuation, "event": event.digest})
        self._history.append(next_state)
        self._graph.add_transition(self.current, next_state)
        return next_state

    def rewind(self, steps: int = 1) -> WorldState:
        if steps >= len(self._history):
            raise ValueError("cannot rewind beyond initial state")
        self._history = self._history[: -steps]
        return self.current

    def history(self) -> list[WorldState]:
        return list(self._history)

    def events(self) -> list[WorldEvent]:
        return list(self._events)

    def graph_snapshot(self) -> dict[str, Any]:
        return self._graph.as_dict()
