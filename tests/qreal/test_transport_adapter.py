from qreal.transport_adapter import TransportAdapter


def test_transport_adapter_clamps_speed():
    adapter = TransportAdapter()
    raw = {
        "vehicle_id": "rail-1",
        "mode": "rail",
        "position": [0, 0],
        "speed": 200000,
        "status": "on-time",
    }
    output = adapter.process(raw, tick=4)
    assert output.normalized["speed"] == 100000.0
    assert output.percept["kind"] == "transport_state"
