from qsk.distributed.coordinator import Coordinator


def test_coordinator_orders_messages():
    coord = Coordinator()
    plan = [
        {"id": 2, "epoch": 1},
        {"id": 1, "epoch": 0},
    ]
    result = coord.broadcast_plan(plan)
    assert result[0]["id"] == 1
    assert result[1]["state"] == "applied"
