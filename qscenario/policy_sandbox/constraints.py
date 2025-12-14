"""Core safety and alignment constraints for scenarios."""
from __future__ import annotations

from typing import Callable

from qscenario.events import Event
from qscenario.scenario import ScenarioState


def enforce_core_constraints(policies: dict[str, Callable[[Event, ScenarioState], bool]]) -> dict[str, Callable[[Event, ScenarioState], bool]]:
    def safety_barrier(event: Event, state: ScenarioState) -> bool:
        if event.kind.startswith("unsafe"):
            state.record_incident("safety_block", event.describe())
            return False
        return True

    combined = dict(policies)
    combined.setdefault("core_safety", safety_barrier)
    return combined
