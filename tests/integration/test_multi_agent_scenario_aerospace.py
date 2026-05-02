from qagents.base import Agent
from qagents.strategy import PolicyAdapter, ThresholdStrategy
from qintervention.constraints import ConstraintSet, domain_whitelist
from qintervention.planner import InterventionPlanner
from qnx_agi.planning.multi_agent_planner import plan_and_apply
from qnx_agi.worldmodel.grounding import (ConfidenceWeights, GroundedState,
                                          ObservationAnchor)
from qscenario.scenario import ScenarioState


def test_multi_agent_aerospace_actions_apply():
    strategy = ThresholdStrategy(
        metric="temperature",
        threshold=5,
        above_action={"kind": "aerospace", "target": "engine", "params": {"impact": 1.0}},
        below_action={"kind": "aerospace", "target": "engine", "params": {"impact": 0.0}},
    )
    agent = Agent("pilot", PolicyAdapter(strategy))
    state = ScenarioState(context={"temperature": 6})
    anchor = ObservationAnchor(source="telemetry", tick=0, weight=1.0)
    grounded = GroundedState.reconcile(
        {"temperature": 6}, {"temperature": 6}, anchor, ConfidenceWeights(0.5, 0.5)
    )
    planner = InterventionPlanner(ConstraintSet([domain_whitelist({"aerospace"})]))

    applied = plan_and_apply([agent], state, grounded, planner)

    assert len(applied) == 1
    assert applied[0]["action"]["kind"] == "aerospace"
    assert state.metrics["agent_actions"] == 1
