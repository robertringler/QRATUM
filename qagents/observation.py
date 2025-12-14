"""Observation utilities for agents."""
from __future__ import annotations

from qagents.base import AgentObservation


def filtered_observation(observation: AgentObservation, allowed_keys: dict[str, bool]) -> AgentObservation:
    view = {k: v for k, v in observation.view.items() if allowed_keys.get(k, False)}
    return AgentObservation(tick=observation.tick, view=view, provenance=observation.provenance)


def merge_observations(primary: AgentObservation, secondary: AgentObservation) -> AgentObservation:
    merged = dict(primary.view)
    merged.update(secondary.view)
    provenance = ",".join([p for p in [primary.provenance, secondary.provenance] if p])
    return AgentObservation(tick=max(primary.tick, secondary.tick), view=merged, provenance=provenance)
