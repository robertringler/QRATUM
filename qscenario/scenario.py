"""Scenario definitions and state management."""
from __future__ import annotations

from dataclasses import dataclass, field

from qscenario.timeline import Timeline
from qscenario.events import Event
from qscenario.drivers import ScenarioDrivers


@dataclass
class ScenarioConfig:
    """Configuration for a scenario run."""

    name: str
    domains: list[str]
    parameters: dict[str, object] = field(default_factory=dict)
    policies: dict[str, object] = field(default_factory=dict)
    description: str = ""


@dataclass
class ScenarioState:
    """Mutable state during scenario execution."""

    tick: int = 0
    context: dict[str, object] = field(default_factory=dict)
    metrics: dict[str, int] = field(default_factory=dict)
    incidents: list[dict[str, object]] = field(default_factory=list)

    def increment_tick(self) -> None:
        self.tick += 1

    def record_metric(self, name: str, value: int) -> None:
        self.metrics[name] = self.metrics.get(name, 0) + value

    def record_incident(self, label: str, detail: object) -> None:
        self.incidents.append({"label": label, "detail": detail, "tick": self.tick})


class Scenario:
    """Deterministic scenario execution over a logical timeline."""

    def __init__(self, config: ScenarioConfig, timeline: Timeline, drivers: ScenarioDrivers) -> None:
        self.config = config
        self.timeline = timeline
        self.drivers = drivers
        self.state = ScenarioState()
        self.results: list[dict[str, object]] = []

    def _apply_policies(self, event: Event) -> None:
        for name, policy in sorted(self.config.policies.items()):
            decision = policy(event, self.state)
            if decision is False:
                self.state.record_incident("policy_block", {"policy": name, "event": event.kind})

    def run(self) -> ScenarioState:
        for tick, events in self.timeline.stream():
            self.state.tick = tick
            for event in events:
                self._apply_policies(event)
                outcome = self.drivers.invoke(event.domain, event, self.state)
                self.results.append({"tick": tick, "event": event, "outcome": outcome})
            self.state.increment_tick()
        return self.state

    def summary(self) -> dict[str, object]:
        return {
            "config": self.config,
            "state": self.state,
            "results": self.results,
        }
