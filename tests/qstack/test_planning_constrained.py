from qstack.qnx.runtime import SafetyEnvelope
from qstack.qnx_agi.planning import ConstrainedAStarPlanner


def test_constrained_a_star_respects_envelope():
    def heuristic(goal, state):
        return float(sum(goal[k] != state.get(k) for k in goal))

    envelope = SafetyEnvelope({"x": (0, 2)})
    planner = ConstrainedAStarPlanner(heuristic=heuristic, envelope=envelope)
    goal = {"x": 2}
    state = {"x": 0}
    steps = planner.plan(goal, state)
    assert steps[-1].parameters["x"] == 2
    assert all(step.parameters.get("x", 0) <= 2 for step in steps)
