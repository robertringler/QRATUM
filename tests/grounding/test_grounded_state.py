from qnx_agi.worldmodel.grounding import ConfidenceWeights, GroundedState, ObservationAnchor


def test_grounded_state_blends_sim_and_observed():
    simulated = {"x": 10.0}
    observed = {"x": 0.0}
    anchor = ObservationAnchor(source="telemetry", tick=5, weight=1.0)
    weights = ConfidenceWeights(simulation=0.2, observation=0.8)
    grounded = GroundedState.reconcile(simulated, observed, anchor, weights)
    assert grounded.blended["x"] == 2.0
    divergence = grounded.divergence(simulated)
    assert divergence["x"] == grounded.blended["x"] - simulated["x"]
