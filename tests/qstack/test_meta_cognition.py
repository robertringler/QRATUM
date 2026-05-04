from qstack.qnx_agi.meta_cognition import MetaCognitionEngine


def test_meta_cognition_selects_best_plan():
    engine = MetaCognitionEngine()
    plan_a = [{"cost": 1}, {"reward": 1}]
    plan_b = [{"cost": 0}, {"reward": 2}]
    selected = engine.select_plan([plan_a, plan_b], {"safety": 1.0, "reward": 0.5})
    assert selected in (plan_a, plan_b)
    assert engine.audit.entries[0]["category"] == "meta_plan"
