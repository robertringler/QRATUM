from qstack.qnx_agi.worldmodel.base import WorldModel
from qstack.qnx_agi.worldmodel.dynamics import (aerospace_step, finance_step,
                                                pharma_step)


def test_worldmodel_dynamics_transitions():
    model = WorldModel({"aero": aerospace_step, "finance": finance_step, "pharma": pharma_step})
    aero_state = model.simulate_step(
        "aero", {"altitude": 100.0, "velocity": 10.0, "acceleration": 1.0}
    )
    assert aero_state.facts["sim"]["aero"]["altitude"] > 100.0

    finance_state = model.simulate_step("finance", {"price": 10.0, "drift": 0.01, "shock": -0.02})
    assert finance_state.facts["sim"]["finance"]["price"] == 10.0 * (1.0 + 0.01 - 0.02)

    pharma_state = model.simulate_step("pharma", {"A": 1.0, "B": 0.0, "rate": 0.1})
    assert abs(pharma_state.facts["sim"]["pharma"]["A"] - 0.9) < 1e-6

    graph = model.graph_snapshot()
    assert len(graph["nodes"]) == 4  # genesis + 3 updates
