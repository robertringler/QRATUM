from qnx_agi.worldmodel.grounding import ConfidenceWeights


def test_confidence_weights_normalize():
    weights = ConfidenceWeights(simulation=1.0, observation=3.0).normalize()
    assert round(weights.simulation, 2) == 0.25
    assert round(weights.observation, 2) == 0.75
