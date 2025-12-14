"""Deterministic rollout of intervention plans."""
from __future__ import annotations

from qintervention.actions import ScheduledAction


def rollout_plan(plan: list[ScheduledAction], scenario_state) -> list[dict[str, object]]:
    applied: list[dict[str, object]] = []
    for scheduled in sorted(plan, key=lambda a: a.key()):
        scenario_state.tick = scheduled.tick
        scenario_state.record_metric("interventions", 1)
        applied.append(
            {
                "tick": scheduled.tick,
                "action": scheduled.action.describe(),
                "allowed": True,
                "impact": scheduled.action.params.get("impact", 0.0),
            }
        )
    return applied
