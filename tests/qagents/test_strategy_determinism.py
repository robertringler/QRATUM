from qagents.base import AgentObservation, AgentState
from qagents.strategy import DeterminismChecker, ThresholdStrategy


def test_threshold_strategy_is_deterministic():
    strategy = ThresholdStrategy(
        metric="risk",
        threshold=5,
        above_action={"kind": "reduce", "target": "sys"},
        below_action={"kind": "grow", "target": "sys"},
    )
    checker = DeterminismChecker(strategy)
    obs = AgentObservation(tick=0, view={"risk": 6}, provenance="test")
    state = AgentState(agent_id="a1")

    assert checker.is_deterministic(obs, state)
