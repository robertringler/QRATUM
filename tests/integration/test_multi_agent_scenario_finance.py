from qagents.base import Agent
from qagents.strategy import PolicyAdapter, ScriptedStrategy
from qintervention.constraints import ConstraintSet, domain_whitelist
from qintervention.planner import InterventionPlanner
from qnx_agi.planning.multi_agent_planner import plan_and_apply
from qnx_agi.worldmodel.grounding import ConfidenceWeights, GroundedState, ObservationAnchor
from qscenario.scenario import ScenarioState


def test_multi_agent_finance_scripted_actions():
    strategy = ScriptedStrategy(script={0: {"kind": "finance", "target": "market", "params": {"impact": 2.0}}}, default_action={"kind": "finance", "target": "market", "params": {"impact": 1.0}})
    agent = Agent("trader", PolicyAdapter(strategy))
    state = ScenarioState(context={"liquidity": 10})
    anchor = ObservationAnchor(source="market", tick=0, weight=1.0)
    grounded = GroundedState.reconcile({"liquidity": 10}, {"liquidity": 10}, anchor, ConfidenceWeights(0.5, 0.5))
    planner = InterventionPlanner(ConstraintSet([domain_whitelist({"finance"})]))

    applied = plan_and_apply([agent], state, grounded, planner)

    assert applied[0]["action"]["params"]["impact"] == 2.0
    assert state.metrics["agent_actions"] == 1
