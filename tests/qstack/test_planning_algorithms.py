from qstack.qnx.runtime.safety import SafetyEnvelope
from qstack.qnx_agi.planning.base import AStarPlanner, BeamSearchPlanner, MPCPlanner


def heuristic(goal, state):
    return float(sum(1 for k, v in goal.items() if state.get(k) != v))


def test_a_star_planner():
    planner = AStarPlanner(heuristic)
    goal = {"x": 1, "y": 2}
    path = planner.plan(goal, state={"x": 0})
    assert path[-1].parameters["x"] == 1
    assert path[-1].parameters["y"] == 2


def test_beam_search_planner():
    planner = BeamSearchPlanner(heuristic, width=2)
    goal = {"a": 1, "b": 2, "c": 3}
    path = planner.plan(goal, state={"a": 0, "b": 0, "c": 0})
    assert path[-1].parameters["c"] == 3


def test_mpc_planner_with_envelope():
    def predict(state):
        return {"temp": state.get("temp", 0) + 1}

    def cost(goal, state):
        return abs(goal.get("temp", 0) - state.get("temp", 0))

    envelope = SafetyEnvelope(bounds={"temp": (0, 5)})
    planner = MPCPlanner(predict_fn=predict, cost_fn=cost, envelope=envelope, horizon=4)
    trajectory = planner.plan(goal={"temp": 3}, state={"temp": 0})
    assert trajectory[-1].parameters["temp"] == 3
