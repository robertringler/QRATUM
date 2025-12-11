"""Multi-agent planner integrating strategies and interventions."""
from __future__ import annotations

from qagents.base import Agent
from qintervention.actions import InterventionAction
from qintervention.planner import InterventionPlanner
from qintervention.rollout import rollout_plan
from qnx_agi.worldmodel.grounding import GroundedState
from qscenario.agent_bridge import apply_actions, observation_from_state
from qscenario.scenario import ScenarioState


def plan_and_apply(
    agents: list[Agent],
    scenario_state: ScenarioState,
    grounded: GroundedState,
    planner: InterventionPlanner,
) -> list[dict[str, object]]:
    observation = observation_from_state(scenario_state, grounded)
    proposals: dict[int, list[InterventionAction]] = {scenario_state.tick: []}
    for agent in sorted(agents, key=lambda a: a.agent_id):
        decision = agent.act(observation)
        action_payload = decision.get("action", {}) if isinstance(decision, dict) else decision
        action = InterventionAction(kind=action_payload.get("kind", "generic"), target=action_payload.get("target", agent.agent_id), params=action_payload.get("params", {}))
        proposals[scenario_state.tick].append(action)
    plan = planner.build_plan(proposals)
    applied = rollout_plan(plan.ordered(), scenario_state)
    apply_actions(scenario_state, [sa.action for sa in plan.actions])
    return applied
