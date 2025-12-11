"""Campaign-specific scenario wiring."""
from __future__ import annotations

from qscenario.drivers import ScenarioDrivers
from qscenario.scenario import Scenario, ScenarioConfig
from qscenario.timeline import Timeline


def build_campaign_scenario(name: str, domains: list[str], timeline_ticks: int) -> Scenario:
    config = ScenarioConfig(name=name, domains=domains)
    timeline = Timeline([(tick, []) for tick in range(timeline_ticks)])
    drivers = ScenarioDrivers(domain_handlers={domain: (lambda e, s, d=domain: {"domain": d}) for domain in domains})
    return Scenario(config, timeline, drivers)
