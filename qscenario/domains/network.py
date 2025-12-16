"""Network scenario templates."""

from __future__ import annotations

from typing import List

from qscenario.drivers import ScenarioDrivers
from qscenario.events import SystemEvent
from qscenario.scenario import Scenario, ScenarioConfig
from qscenario.timeline import Timeline, TimelineEntry


def _network_driver(event, state):
    state.record_metric("network_events", 1)
    if event.kind == "partition":
        state.record_incident("partition", event.payload)
    return {"status": "routed", "latency": event.payload.get("latency", 0)}


def network_partition() -> Scenario:
    config = ScenarioConfig(
        name="network_partition",
        domains=["network"],
        description="Network partition and recovery",
    )
    entries: List[TimelineEntry] = [
        TimelineEntry(0, [SystemEvent(0, "network", "latency_spike", {"latency": 120})]),
        TimelineEntry(1, [SystemEvent(1, "network", "partition", {"segments": 2})]),
        TimelineEntry(2, [SystemEvent(2, "network", "reroute", {"paths": 3})]),
        TimelineEntry(3, [SystemEvent(3, "network", "heal", {"latency": 30})]),
    ]
    timeline = Timeline(entries)
    drivers = ScenarioDrivers(simulators={"network": _network_driver})
    return Scenario(config, timeline, drivers)
