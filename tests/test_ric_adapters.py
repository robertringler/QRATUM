"""Tests for the Reality Interface Controller adapters (QRATUM, CIIR, CRS)."""

from __future__ import annotations

import pytest

from qagents.adapters import (IMMUTABLE_BOUNDARIES, PROHIBITED_GOALS,
                              ciir_proposer, ciir_simulator, crs_proposer,
                              crs_simulator, make_ciir_controller,
                              make_crs_controller, make_qratum_controller,
                              qratum_proposer, qratum_simulator)
from qagents.reality_interface import (ACTION_TYPES, ControlDecision,
                                       IntentInterpretation, PredictedOutcome)

# ---------------------------------------------------------------------------
# QRATUM adapter
# ---------------------------------------------------------------------------


def _qratum_world(**overrides):
    state = {
        "OSR": 1.05,
        "CEI": 0.4,
        "SF": 0.95,
        "HRD": 0.02,
    }
    state.update(overrides)
    return state


def test_qratum_simulator_returns_valid_predicted_outcome():
    outcome = qratum_simulator("adjust", 0.3, _qratum_world(), {})
    assert isinstance(outcome, PredictedOutcome)
    assert 0.0 <= outcome.risk <= 1.0
    assert 0.0 <= outcome.stability_score <= 1.0
    assert "OSR" in outcome.expected_state


def test_qratum_simulator_abort_is_safe():
    outcome = qratum_simulator("abort", 1.0, _qratum_world(HRD=0.5), {})
    assert outcome.risk == 0.0


def test_qratum_simulator_hold_preserves_metrics():
    world = _qratum_world(HRD=0.1)
    outcome = qratum_simulator("hold", 0.0, world, {})
    assert outcome.risk == pytest.approx(0.1)
    assert outcome.stability_score == pytest.approx(world["SF"])


def test_qratum_simulator_intervention_increases_hrd():
    world = _qratum_world(SF=0.5, HRD=0.0)
    light = qratum_simulator("adjust", 0.1, world, {})
    heavy = qratum_simulator("adjust", 0.9, world, {})
    assert heavy.risk > light.risk
    assert heavy.stability_score < light.stability_score


def test_qratum_proposer_holds_when_target_met():
    intent = IntentInterpretation(goal="meet osr", constraints={"target_OSR": 1.0})
    action_type, magnitude = qratum_proposer(intent, _qratum_world(OSR=1.5), {})
    assert action_type == "hold"
    assert magnitude == 0.0


def test_qratum_proposer_adjusts_to_close_gap():
    intent = IntentInterpretation(goal="meet osr", constraints={"target_OSR": 2.0})
    action_type, magnitude = qratum_proposer(
        intent, _qratum_world(OSR=1.0), {"max_magnitude": 1.0}
    )
    assert action_type == "adjust"
    assert 0.0 < magnitude <= 1.0


def test_qratum_controller_step_returns_strict_schema():
    ctrl = make_qratum_controller()
    decision = ctrl.step(
        {
            "intent": {
                "goal": "raise OSR",
                "constraints": {"target_OSR": 1.5},
                "priority": 1,
            },
            "world_state": _qratum_world(),
            "system_limits": {
                "max_magnitude": 0.5,
                "stability_threshold": 0.5,
            },
        }
    )
    assert isinstance(decision, ControlDecision)
    payload = decision.as_dict()
    assert set(payload) == {
        "intent_interpretation",
        "selected_action",
        "predicted_outcome",
        "fallback_plan",
    }
    assert decision.selected_action.type in ACTION_TYPES


def test_qratum_controller_aborts_on_hrd_safety_bound():
    ctrl = make_qratum_controller()
    decision = ctrl.step(
        {
            "intent": {"goal": "raise OSR", "constraints": {"target_OSR": 2.0}},
            "world_state": _qratum_world(HRD=0.95, SF=0.2),
            "system_limits": {
                "max_magnitude": 1.0,
                "stability_threshold": 0.5,
                "safety_bounds": {"HRD": {"min": 0.0, "max": 0.1}},
            },
        }
    )
    assert decision.selected_action.type == "abort"


# ---------------------------------------------------------------------------
# CIIR adapter
# ---------------------------------------------------------------------------


def _ciir_world(**overrides):
    state = {
        "loss": 0.5,
        "constraint_violation": 0.2,
        "entropy": 1.0,
        "gradient_norm": 0.4,
    }
    state.update(overrides)
    return state


def test_ciir_simulator_step_reduces_loss_and_violation():
    world = _ciir_world()
    light = ciir_simulator("adjust", 0.1, world, {})
    heavy = ciir_simulator("adjust", 0.5, world, {})
    assert heavy.expected_state["loss"] < light.expected_state["loss"]
    assert heavy.risk < light.risk  # constraint violation shrinks faster
    assert heavy.stability_score > light.stability_score


def test_ciir_simulator_hold_is_idempotent():
    world = _ciir_world()
    outcome = ciir_simulator("hold", 0.0, world, {})
    assert outcome.expected_state["loss"] == pytest.approx(world["loss"])
    assert outcome.risk == pytest.approx(world["constraint_violation"])


def test_ciir_simulator_loss_is_clamped_non_negative():
    outcome = ciir_simulator(
        "adjust", 1.0, _ciir_world(loss=0.1, gradient_norm=10.0), {}
    )
    assert outcome.expected_state["loss"] >= 0.0
    assert 0.0 <= outcome.stability_score <= 1.0


def test_ciir_proposer_smaller_step_for_high_violation():
    intent = IntentInterpretation(goal="minimise loss")
    _, low_v = ciir_proposer(intent, _ciir_world(constraint_violation=0.0), {})
    _, high_v = ciir_proposer(intent, _ciir_world(constraint_violation=0.9), {})
    assert low_v > high_v


def test_ciir_controller_step_returns_decision():
    ctrl = make_ciir_controller()
    decision = ctrl.step(
        {
            "intent": {"goal": "minimise loss", "priority": 1},
            "world_state": _ciir_world(),
            "system_limits": {
                "max_magnitude": 0.5,
                "stability_threshold": 0.3,
                "projection_strength": 1.0,
            },
        }
    )
    assert decision.selected_action.type in ACTION_TYPES
    assert 0.0 <= decision.selected_action.confidence <= 1.0


def test_ciir_controller_aborts_on_violation_safety_bound():
    ctrl = make_ciir_controller()
    decision = ctrl.step(
        {
            "intent": {"goal": "minimise loss"},
            "world_state": _ciir_world(constraint_violation=0.95, loss=5.0),
            "system_limits": {
                "max_magnitude": 1.0,
                "stability_threshold": 0.5,
                "safety_bounds": {
                    "constraint_violation": {"min": 0.0, "max": 0.1},
                },
            },
        }
    )
    assert decision.selected_action.type == "abort"


# ---------------------------------------------------------------------------
# CRS / CRSI adapter
# ---------------------------------------------------------------------------


def _crs_world(**overrides):
    state = {
        "pillar": "Q-MIND",
        "proposed_goals": ["optimise reasoning"],
        "proposed_modifications": ["weights"],
        "human_oversight_pending": False,
        "rollback_available": True,
        "tampering_score": 0.05,
    }
    state.update(overrides)
    return state


def test_crs_immutable_and_prohibited_sets_are_non_empty_and_disjoint():
    assert len(IMMUTABLE_BOUNDARIES) > 0
    assert len(PROHIBITED_GOALS) > 0
    assert IMMUTABLE_BOUNDARIES.isdisjoint(PROHIBITED_GOALS)


def test_crs_simulator_blocks_on_prohibited_goal():
    outcome = crs_simulator(
        "adjust",
        0.1,
        _crs_world(proposed_goals=["remove_human_oversight"]),
        {},
    )
    assert outcome.risk == 1.0
    assert outcome.stability_score == 0.0
    assert "safety_block" in outcome.expected_state
    assert outcome.expected_state["safety_block"].startswith("prohibited_goal:")


def test_crs_simulator_blocks_on_immutable_boundary_modification():
    outcome = crs_simulator(
        "adjust",
        0.1,
        _crs_world(proposed_modifications=["determinism_guarantee"]),
        {},
    )
    assert outcome.risk == 1.0
    assert "safety_block" in outcome.expected_state
    assert outcome.expected_state["safety_block"].startswith("immutable_boundary:")


def test_crs_simulator_blocks_when_oversight_pending_or_no_rollback():
    pending = crs_simulator("adjust", 0.1, _crs_world(human_oversight_pending=True), {})
    assert pending.risk == 1.0
    no_rb = crs_simulator("adjust", 0.1, _crs_world(rollback_available=False), {})
    assert no_rb.risk == 1.0


def test_crs_simulator_abort_is_always_safe_even_under_block():
    outcome = crs_simulator(
        "abort",
        0.0,
        _crs_world(proposed_goals=["deceive_operators"]),
        {},
    )
    assert outcome.risk == 0.0
    assert outcome.stability_score == 1.0


def test_crs_proposer_holds_when_blocked():
    intent = IntentInterpretation(goal="self-improve", constraints={"step": 0.2})
    action_type, magnitude = crs_proposer(
        intent, _crs_world(human_oversight_pending=True), {}
    )
    assert action_type == "hold"
    assert magnitude == 0.0


def test_crs_proposer_halves_step_for_safety_critical_pillar():
    intent = IntentInterpretation(goal="self-improve", constraints={"step": 0.5})
    _, mind_mag = crs_proposer(intent, _crs_world(pillar="Q-MIND"), {})
    _, will_mag = crs_proposer(intent, _crs_world(pillar="Q-WILL"), {})
    assert will_mag == pytest.approx(mind_mag / 2.0)


def test_crs_controller_aborts_on_blocked_world_state():
    ctrl = make_crs_controller()
    decision = ctrl.step(
        {
            "intent": {"goal": "self-improve", "constraints": {"step": 0.2}},
            "world_state": _crs_world(proposed_goals=["modify_safety_constraints"]),
            "system_limits": {"max_magnitude": 1.0, "stability_threshold": 0.5},
        }
    )
    assert decision.selected_action.type == "abort"


def test_crs_controller_proceeds_under_safe_world_state():
    ctrl = make_crs_controller()
    decision = ctrl.step(
        {
            "intent": {"goal": "self-improve", "constraints": {"step": 0.2}},
            "world_state": _crs_world(),
            "system_limits": {"max_magnitude": 1.0, "stability_threshold": 0.5},
        }
    )
    assert decision.selected_action.type in {"adjust", "hold"}
    assert decision.fallback_plan.action in ACTION_TYPES
