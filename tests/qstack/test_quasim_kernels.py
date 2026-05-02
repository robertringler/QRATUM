from qstack.quasim.core.engine import build_default_engine


def test_quasim_kernel_outputs():
    engine = build_default_engine()
    telemetry = engine.run_kernel("telemetry", {"altitude": 100.0, "velocity": 5.0})
    assert telemetry["altitude"] > 100.0

    finance = engine.run_kernel("finance", {"price": 10.0, "shock": 0.1, "liquidity": 0.9})
    assert finance["price"] == 10.0 * (1.0 + 0.1) * (1.0 - 0.1 * (1.0 - 0.9))

    pharma = engine.run_kernel("pharma", {"A": 2.0, "B": 0.0})
    assert abs(pharma["A"] - 1.9) < 1e-6
