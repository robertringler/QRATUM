from qreal.market_adapter import MarketAdapter


def test_market_adapter_normalizes_and_adds_tick():
    adapter = MarketAdapter()
    raw = {"symbol": "Q", "open": 1, "high": 2, "low": 1, "close": 2, "volume": 10}
    output = adapter.process(raw, tick=3)
    assert output.normalized["tick"] == 3
    assert output.percept["price"] == 2
    assert output.provenance.source == "market"
