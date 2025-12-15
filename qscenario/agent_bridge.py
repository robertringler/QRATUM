"""Bridge between scenarios, grounded world state, and agents."""

from __future__ import annotations

from typing import Dict, List

from qagents.base import AgentObservation
from qintervention.actions import InterventionAction
from qnx_agi.worldmodel.grounding import GroundedState
from qscenario.scenario import ScenarioState


def observation_from_state(state: ScenarioState, grounded: GroundedState) -> AgentObservation:
    view = dict(state.context)
    view.update(grounded.blended)
    provenance = next(iter(grounded.anchors.values())).source if grounded.anchors else "scenario"
    return AgentObservation(tick=state.tick, view=view, provenance=provenance)


def apply_actions(
    state: ScenarioState, actions: List[InterventionAction]
) -> List[Dict[str, object]]:
    applied: List[Dict[str, object]] = []
    for action in actions:
        state.record_metric("agent_actions", 1)
        applied.append({"tick": state.tick, "action": action.describe()})
    return applied
