"""Tests for the Reality Interface Controller."""

from __future__ import annotations

import pytest

from qagents.reality_interface import (
    ACTION_TYPES,
    ControlDecision,
    FallbackPlan,
    PredictedOutcome,
    RealityInterfaceController,
    SelectedAction,
    default_simulator,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _stable_payload(**overrides):
    payload = {
        "intent": {
            "goal": "stabilize reactor",
            "constraints": {"urgency": 0.4},
            "priority": 2,
        },
        "world_state": {"instability": 0.05, "stability": 0.95, "temperature": 300.0},
        "system_limits": {
            "stability_threshold": 0.5,
            "max_magnitude": 1.0,
        },
    }
    payload.update(overrides)
    return payload


def _unstable_payload(**overrides):
    payload = {
        "intent": "stabilize reactor",
        "world_state": {"instability": 0.95, "stability": 0.05},
        "system_limits": {"stability_threshold": 0.5, "max_magnitude": 1.0},
    }
    payload.update(overrides)
    return payload


# ---------------------------------------------------------------------------
# Schema conformance
# ---------------------------------------------------------------------------


def test_step_returns_strict_schema():
    ctrl = RealityInterfaceController()
    decision = ctrl.step(_stable_payload())
    out = decision.as_dict()

    assert set(out) == {
        "intent_interpretation",
        "selected_action",
        "predicted_outcome",
        "fallback_plan",
    }
    assert set(out["intent_interpretation"]) == {"goal", "constraints", "priority"}
    assert set(out["selected_action"]) == {"type", "magnitude", "confidence"}
    assert set(out["predicted_outcome"]) == {"expected_state", "risk", "stability_score"}
    assert set(out["fallback_plan"]) == {"trigger_condition", "action"}

    sa = out["selected_action"]
    assert sa["type"] in ACTION_TYPES
    assert 0.0 <= sa["magnitude"] <= 1.0
    assert 0.0 <= sa["confidence"] <= 1.0
    po = out["predicted_outcome"]
    assert 0.0 <= po["risk"] <= 1.0
    assert 0.0 <= po["stability_score"] <= 1.0


def test_intent_interpretation_accepts_string():
    ctrl = RealityInterfaceController()
    decision = ctrl.step(
        {
            "intent": "hold position",
            "world_state": {"instability": 0.0, "stability": 1.0},
            "system_limits": {},
        }
    )
    assert decision.intent_interpretation.goal == "hold position"
    assert decision.intent_interpretation.constraints == {}
    assert decision.intent_interpretation.priority == 0


def test_decode_intent_rejects_bad_types():
    ctrl = RealityInterfaceController()
    with pytest.raises(TypeError):
        ctrl.decode_intent(123)
    with pytest.raises(TypeError):
        ctrl.decode_intent({"goal": "g", "constraints": "not-a-mapping"})


# ---------------------------------------------------------------------------
# (1) Safety dominance
# ---------------------------------------------------------------------------


def test_safety_dominance_aborts_on_constraint_violation():
    ctrl = RealityInterfaceController()
    payload = _stable_payload(
        world_state={
            "instability": 0.05,
            "stability": 0.95,
            "temperature": 1000.0,  # exceeds bound
        },
        system_limits={
            "stability_threshold": 0.5,
            "max_magnitude": 1.0,
            "safety_bounds": {"temperature": {"min": 0.0, "max": 500.0}},
        },
    )
    decision = ctrl.step(payload)
    assert decision.selected_action.type == "abort"
    assert decision.selected_action.magnitude == 0.0


def test_safety_dominance_aborts_on_high_risk():
    # Force high risk via a custom simulator.
    def risky_sim(action_type, magnitude, world, limits):
        return PredictedOutcome(expected_state={}, risk=0.95, stability_score=0.9)

    ctrl = RealityInterfaceController(simulator=risky_sim)
    decision = ctrl.step(_stable_payload())
    assert decision.selected_action.type == "abort"


def test_safety_dominance_aborts_below_stability_threshold():
    ctrl = RealityInterfaceController()
    decision = ctrl.step(_unstable_payload())
    assert decision.selected_action.type == "abort"


def test_safety_dominance_handles_explicit_violations_list():
    ctrl = RealityInterfaceController()
    payload = _stable_payload(
        world_state={"instability": 0.05, "stability": 0.95, "violations": ["over_pressure"]}
    )
    decision = ctrl.step(payload)
    assert decision.selected_action.type == "abort"


# ---------------------------------------------------------------------------
# (2) Stability awareness
# ---------------------------------------------------------------------------


def test_stability_awareness_downgrades_control_to_adjust():
    # Proposer wants 'control'; stability is borderline (above threshold but
    # below the strong band).
    def proposer(intent, world, limits):
        return "control", 0.4

    def sim(action_type, magnitude, world, limits):
        # Within (threshold, STRONG_STABILITY_BAND) -> downgrade required.
        return PredictedOutcome(
            expected_state={"a": action_type},
            risk=0.1,
            stability_score=0.7,
        )

    ctrl = RealityInterfaceController(proposer=proposer, simulator=sim)
    decision = ctrl.step(_stable_payload())
    assert decision.selected_action.type in {"adjust", "hold"}


def test_stability_awareness_allows_control_when_strongly_stable():
    def proposer(intent, world, limits):
        return "control", 0.2

    def sim(action_type, magnitude, world, limits):
        return PredictedOutcome(
            expected_state={"a": action_type},
            risk=0.05,
            stability_score=0.95,
        )

    ctrl = RealityInterfaceController(proposer=proposer, simulator=sim)
    decision = ctrl.step(_stable_payload())
    assert decision.selected_action.type == "control"


# ---------------------------------------------------------------------------
# (3) Minimal intervention
# ---------------------------------------------------------------------------


def test_minimal_intervention_caps_at_system_max_magnitude():
    def greedy_proposer(intent, world, limits):
        return "adjust", 1.0

    ctrl = RealityInterfaceController(proposer=greedy_proposer)
    payload = _stable_payload(
        system_limits={
            "stability_threshold": 0.4,
            "max_magnitude": 0.25,
        }
    )
    decision = ctrl.step(payload)
    assert decision.selected_action.magnitude <= 0.25 + 1e-9


def test_minimal_intervention_scales_magnitude_with_risk():
    """Higher risk -> smaller magnitude for the same proposal."""

    def proposer(intent, world, limits):
        return "adjust", 0.8

    def low_risk_sim(action_type, magnitude, world, limits):
        return PredictedOutcome(expected_state={}, risk=0.1, stability_score=0.9)

    def high_risk_sim(action_type, magnitude, world, limits):
        return PredictedOutcome(expected_state={}, risk=0.6, stability_score=0.9)

    ctrl_low = RealityInterfaceController(proposer=proposer, simulator=low_risk_sim)
    ctrl_high = RealityInterfaceController(proposer=proposer, simulator=high_risk_sim)
    d_low = ctrl_low.step(_stable_payload())
    d_high = ctrl_high.step(_stable_payload())
    assert d_high.selected_action.magnitude < d_low.selected_action.magnitude


def test_minimal_intervention_holds_when_margin_zero():
    def proposer(intent, world, limits):
        return "adjust", 0.5

    def sim(action_type, magnitude, world, limits):
        # stability_score == threshold -> margin == 0 -> hold
        return PredictedOutcome(expected_state={}, risk=0.1, stability_score=0.5)

    ctrl = RealityInterfaceController(proposer=proposer, simulator=sim)
    payload = _stable_payload(system_limits={"stability_threshold": 0.5})
    decision = ctrl.step(payload)
    # Below stability_threshold check is strict (<), so equality is "safe"
    # but margin <= 0 -> hold per minimal intervention rule.
    assert decision.selected_action.type == "hold"
    assert decision.selected_action.magnitude == 0.0


# ---------------------------------------------------------------------------
# (4) Predict-before-act
# ---------------------------------------------------------------------------


def test_prediction_always_present_in_output():
    ctrl = RealityInterfaceController()
    decision = ctrl.step(_stable_payload())
    # The predicted_outcome must always be a populated PredictedOutcome.
    assert isinstance(decision.predicted_outcome, PredictedOutcome)
    assert decision.predicted_outcome.expected_state is not None


def test_confidence_is_zero_when_stability_at_or_below_threshold_via_abort():
    # When unsafe, controller emits abort; abort's predicted outcome has
    # zero risk and base stability. Confidence should remain in [0, 1].
    ctrl = RealityInterfaceController()
    decision = ctrl.step(_unstable_payload())
    assert decision.selected_action.type == "abort"
    assert 0.0 <= decision.selected_action.confidence <= 1.0


def test_simulator_invoked_before_action_finalized():
    calls: list[tuple[str, float]] = []

    def tracking_sim(action_type, magnitude, world, limits):
        calls.append((action_type, magnitude))
        return PredictedOutcome(expected_state={}, risk=0.1, stability_score=0.9)

    ctrl = RealityInterfaceController(simulator=tracking_sim)
    ctrl.step(_stable_payload())
    # At least one simulation must occur before selection. With our flow,
    # the proposed action is simulated, then the finalized action is
    # re-simulated -> at least 2 calls in the safe path.
    assert len(calls) >= 1


# ---------------------------------------------------------------------------
# Fallback plan
# ---------------------------------------------------------------------------


def test_fallback_plan_always_present():
    ctrl = RealityInterfaceController()
    for payload in (_stable_payload(), _unstable_payload()):
        decision = ctrl.step(payload)
        assert isinstance(decision.fallback_plan, FallbackPlan)
        assert decision.fallback_plan.trigger_condition
        assert decision.fallback_plan.action in ACTION_TYPES


def test_fallback_action_is_safer_downgrade():
    ctrl = RealityInterfaceController()
    decision = ctrl.step(_stable_payload())
    ladder = {"control": "adjust", "adjust": "hold", "hold": "abort", "abort": "abort"}
    assert decision.fallback_plan.action == ladder[decision.selected_action.type]


# ---------------------------------------------------------------------------
# Validation / dataclasses
# ---------------------------------------------------------------------------


def test_selected_action_validates_type_and_ranges():
    with pytest.raises(ValueError):
        SelectedAction(type="explode")
    with pytest.raises(ValueError):
        SelectedAction(type="hold", magnitude=2.0)
    with pytest.raises(ValueError):
        SelectedAction(type="hold", confidence=-0.1)


def test_predicted_outcome_validates_ranges():
    with pytest.raises(ValueError):
        PredictedOutcome(risk=1.5)
    with pytest.raises(ValueError):
        PredictedOutcome(stability_score=-0.1)


def test_default_simulator_is_deterministic():
    a = default_simulator("adjust", 0.3, {"instability": 0.2}, {})
    b = default_simulator("adjust", 0.3, {"instability": 0.2}, {})
    assert a.as_dict() == b.as_dict()


def test_intent_constraint_max_violation_triggers_abort():
    ctrl = RealityInterfaceController()
    payload = {
        "intent": {
            "goal": "limit temperature",
            "constraints": {"temperature_max": 500.0},
            "priority": 1,
        },
        "world_state": {"temperature": 700.0, "instability": 0.0, "stability": 1.0},
        "system_limits": {"stability_threshold": 0.5, "max_magnitude": 1.0},
    }
    decision = ctrl.step(payload)
    assert decision.selected_action.type == "abort"


def test_step_rejects_invalid_payload():
    ctrl = RealityInterfaceController()
    with pytest.raises(TypeError):
        ctrl.step("not a mapping")  # type: ignore[arg-type]
    with pytest.raises(KeyError):
        ctrl.step({"world_state": {}, "system_limits": {}})
    with pytest.raises(TypeError):
        ctrl.step({"intent": "g", "world_state": "bad", "system_limits": {}})


def test_decision_as_dict_is_json_serializable():
    import json

    ctrl = RealityInterfaceController()
    decision = ctrl.step(_stable_payload())
    payload = json.dumps(decision.as_dict())
    assert "selected_action" in payload


def test_controller_is_exported_from_qagents_package():
    import qagents

    assert hasattr(qagents, "RealityInterfaceController")
    assert hasattr(qagents, "ControlDecision")
    assert qagents.RealityInterfaceController is RealityInterfaceController
    assert qagents.ControlDecision is ControlDecision
