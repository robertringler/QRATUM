import pytest

from qstack.qunimbus.core.engine import (QuNimbusEngine, ValuationInput,
                                         macro_engine)


def test_qunimbus_governance_and_value():
    engine = macro_engine()
    payload = ValuationInput(metrics={"gdp": 100, "inflation": 0.1, "employment": 0.8})
    valuation = engine.evaluate(payload)
    governance = engine.governance_score(payload)
    assert governance["final"] < valuation
    assert governance["penalties"]["inflation"] == pytest.approx(0.05)


def test_custom_weights_operator():
    engine = QuNimbusEngine(weights={"alpha": 1.0}, governance_rules={"alpha": 2.0})
    op = engine.valuation_operator()
    result = op({}, {"alpha": 3.0})
    assert result == 3.0
