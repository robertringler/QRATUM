from qreal.grid_adapter import GridAdapter


def test_grid_adapter_generates_grid_state():
    adapter = GridAdapter()
    raw = {"region": "north", "load": 1000, "generation": 1200, "frequency": 59.99}
    output = adapter.process(raw, tick=9)
    assert output.percept["kind"] == "grid_state"
    assert output.normalized["tick"] == 9
