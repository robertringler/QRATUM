"""Pharma scenario templates."""

from __future__ import annotations

from qscenario.drivers import ScenarioDrivers
from qscenario.events import MarketEvent
from qscenario.scenario import Scenario, ScenarioConfig
from qscenario.timeline import Timeline, TimelineEntry


def _pharma_driver(event, state):
    state.record_metric("trial_events", 1)
    if event.kind == "adverse_event":
        state.record_incident("adverse", event.payload)
    return {"status": "tracked", "signal": event.kind}


def pharma_trial_outcome() -> Scenario:
    config = ScenarioConfig(
        name="pharma_trial_outcome",
        domains=["pharma"],
        description="Trial execution with supply challenges",
    )
    entries: list[TimelineEntry] = [
        TimelineEntry(0, [MarketEvent(0, "pharma", "enrollment", {"sites": 5})]),
        TimelineEntry(1, [MarketEvent(1, "pharma", "adverse_event", {"severity": "low"})]),
        TimelineEntry(2, [MarketEvent(2, "pharma", "supply_disruption", {"days": 2})]),
        TimelineEntry(3, [MarketEvent(3, "pharma", "data_lock", {"patients": 120})]),
    ]
    timeline = Timeline(entries)
    drivers = ScenarioDrivers(simulators={"pharma": _pharma_driver})
    return Scenario(config, timeline, drivers)
