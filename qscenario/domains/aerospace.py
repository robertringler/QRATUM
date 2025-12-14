"""Aerospace scenario templates."""
from __future__ import annotations


from qscenario.events import MissionEvent
from qscenario.timeline import Timeline, TimelineEntry
from qscenario.drivers import ScenarioDrivers
from qscenario.scenario import Scenario, ScenarioConfig


def _aerospace_driver(event, state):
    state.record_metric("mission_steps", 1)
    if event.kind == "engine_issue":
        state.record_incident("engine_issue", event.payload)
    return {"status": "handled", "event": event.kind}


def aerospace_launch_anomaly() -> Scenario:
    config = ScenarioConfig(
        name="aerospace_launch_anomaly",
        domains=["aerospace"],
        description="Launch with guidance drift and comms loss recovery",
    )
    entries: list[TimelineEntry] = [
        TimelineEntry(0, [MissionEvent(0, "aerospace", "liftoff", {"thrust": 1.0})]),
        TimelineEntry(1, [MissionEvent(1, "aerospace", "guidance_drift", {"delta": 0.02})]),
        TimelineEntry(2, [MissionEvent(2, "aerospace", "comms_loss", {"duration": 3})]),
        TimelineEntry(3, [MissionEvent(3, "aerospace", "engine_issue", {"engine": "B", "severity": "medium"})]),
        TimelineEntry(4, [MissionEvent(4, "aerospace", "recovery", {"mode": "safe"})]),
    ]
    timeline = Timeline(entries)
    drivers = ScenarioDrivers(simulators={"aerospace": _aerospace_driver})
    return Scenario(config, timeline, drivers)
