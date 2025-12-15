"""Scenario drivers connecting to subsystems."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Dict

from qscenario.events import Event

if TYPE_CHECKING:  # pragma: no cover
    from qscenario.scenario import ScenarioState


@dataclass
class ScenarioDrivers:
    simulators: Dict[str, Callable[[Event, ScenarioState], Dict[str, object]]] = field(
        default_factory=dict
    )
    governance_hook: Callable[[Event, ScenarioState], Dict[str, object]] | None = None
    node_hook: Callable[[Event, ScenarioState], Dict[str, object]] | None = None

    def invoke(self, domain: str, event: Event, state: ScenarioState) -> Dict[str, object]:
        if self.governance_hook is not None:
            governance_view = self.governance_hook(event, state)
            state.context.setdefault("governance", []).append(governance_view)
        if self.node_hook is not None:
            node_view = self.node_hook(event, state)
            state.context.setdefault("node", []).append(node_view)
        simulator = self.simulators.get(domain)
        if simulator is None:
            return {"status": "noop", "domain": domain}
        return simulator(event, state)
