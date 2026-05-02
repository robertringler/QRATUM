from qreal.telemetry_adapter import TelemetryAdapter


def test_telemetry_adapter_produces_percept():
    adapter = TelemetryAdapter()
    raw = {"vehicle": "sat-1", "position": [1, 2, 3], "velocity": [0, 0, 1], "status": "nominal"}
    output = adapter.process(raw, tick=7)
    assert output.percept["vehicle"] == "sat-1"
    assert output.percept["tick"] == 7
