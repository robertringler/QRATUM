"""Energy scenario templates."""

from __future__ import annotations

from qscenario.drivers import ScenarioDrivers
from qscenario.events import SystemEvent
from qscenario.scenario import Scenario, ScenarioConfig
from qscenario.timeline import Timeline, TimelineEntry


def _energy_driver(event, state):
    state.record_metric("grid_events", 1)
    if event.kind == "outage":
        state.record_incident("grid_outage", event.payload)
    return {"status": "handled", "impact": event.payload.get("impact", 0)}


def energy_grid_instability() -> Scenario:
    config = ScenarioConfig(
        name="energy_grid_instability",
        domains=["energy"],
        description="Grid instability with cascades",
    )
    entries: list[TimelineEntry] = [
        TimelineEntry(0, [SystemEvent(0, "energy", "frequency_drift", {"hz": 49.5})]),
        TimelineEntry(1, [SystemEvent(1, "energy", "outage", {"impact": 100})]),
        TimelineEntry(2, [SystemEvent(2, "energy", "remediation", {"crews": 2})]),
    ]
    timeline = Timeline(entries)
    drivers = ScenarioDrivers(simulators={"energy": _energy_driver})
    return Scenario(config, timeline, drivers)
