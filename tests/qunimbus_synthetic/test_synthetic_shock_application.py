from qunimbus.synthetic.shocks import apply_price_shock, liquidity_shock


def test_shocks_apply_deterministically():
    prices = {"X": 10.0}
    shocked = apply_price_shock(prices, {"X": -1.0, "Y": 2.0})
    assert shocked == {"X": 9.0, "Y": 2.0}

    assert liquidity_shock(5.0, 1.5) == 3.5
    assert liquidity_shock(1.0, 5.0) == 0.0
