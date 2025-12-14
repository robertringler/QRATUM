"""Scenario-level invariants."""

from __future__ import annotations

from qscenario.scenario import ScenarioState


def enforce_metric_bounds(state: ScenarioState, bounds: dict[str, int]) -> list[str]:
    violations: list[str] = []
    for name, limit in bounds.items():
        value = state.metrics.get(name, 0)
        if value > limit:
            violations.append(f"metric {name} exceeded {limit} with {value}")
    return violations
