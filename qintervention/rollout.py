"""Deterministic rollout of intervention plans."""

from __future__ import annotations

from typing import Dict, List

from qintervention.actions import ScheduledAction


def rollout_plan(plan: List[ScheduledAction], scenario_state) -> List[Dict[str, object]]:
    applied: List[Dict[str, object]] = []
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
