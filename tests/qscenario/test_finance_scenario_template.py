from qscenario.domains.finance import finance_liquidity_crunch


def test_finance_template_metrics_and_incident():
    scenario = finance_liquidity_crunch()
    state = scenario.run()
    assert state.metrics["market_events"] == 5
    assert any(inc["label"] == "liquidity" for inc in state.incidents)
