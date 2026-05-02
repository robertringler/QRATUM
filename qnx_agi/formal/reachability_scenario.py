"""Reachability checks for scenario outcomes."""

from __future__ import annotations

from qscenario.scenario import ScenarioState


def check_bad_outcomes(state: ScenarioState, bad_states: list[str]) -> list[str]:
    findings: list[str] = []
    for incident in state.incidents:
        label = incident.get("label") or incident.get("label", "")
        if label in bad_states:
            findings.append(f"bad state reached: {label} at tick {incident.get('tick')}")
    return findings
