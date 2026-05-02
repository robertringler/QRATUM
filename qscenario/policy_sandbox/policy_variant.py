"""Policy variant definitions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from qscenario.events import Event
from qscenario.scenario import ScenarioState


@dataclass
class PolicyVariant:
    name: str
    rules: dict[str, Callable[[Event, ScenarioState], bool]] = field(default_factory=dict)
    parameters: dict[str, object] = field(default_factory=dict)

    def apply(
        self, base_policies: dict[str, Callable[[Event, ScenarioState], bool]]
    ) -> dict[str, Callable[[Event, ScenarioState], bool]]:
        merged = dict(base_policies)
        merged.update(self.rules)
        return merged
