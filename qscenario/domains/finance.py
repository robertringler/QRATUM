"""Finance scenario templates."""
from __future__ import annotations

from qscenario.drivers import ScenarioDrivers
from qscenario.events import MarketEvent
from qscenario.scenario import Scenario, ScenarioConfig
from qscenario.timeline import Timeline, TimelineEntry


def _finance_driver(event, state):
    state.record_metric("market_events", 1)
    if event.kind == "liquidity_crunch":
        state.record_incident("liquidity", event.payload)
    return {"status": "processed", "kind": event.kind}


def finance_liquidity_crunch() -> Scenario:
    config = ScenarioConfig(
        name="finance_liquidity_crunch",
        domains=["finance"],
        description="Liquidity crunch and contagion propagation",
    )
    entries: list[TimelineEntry] = [
        TimelineEntry(0, [MarketEvent(0, "finance", "calm", {"vol": 0.1})]),
        TimelineEntry(1, [MarketEvent(1, "finance", "liquidity_crunch", {"outflows": 5})]),
        TimelineEntry(2, [MarketEvent(2, "finance", "policy_intervention", {"tool": "lending"})]),
        TimelineEntry(3, [MarketEvent(3, "finance", "contagion", {"spread": "regional"})]),
        TimelineEntry(4, [MarketEvent(4, "finance", "stabilization", {"vol": 0.2})]),
    ]
    timeline = Timeline(entries)
    drivers = ScenarioDrivers(simulators={"finance": _finance_driver})
    return Scenario(config, timeline, drivers)
