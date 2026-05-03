"""Tests for the trajectory-aware RIC v2 controller and bridge integration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from qagents.ciir_ric_bridge import CIIRRICBridge
from qagents.llm_backends import DeterministicLLM
from qagents.reality_interface import ACTION_TYPES, StaticProposer
from qagents.reality_interface_v2 import RICv2Controller, RICv2History

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _intent(priority: float = 0.5) -> dict[str, Any]:
    return {
        "goal": "stabilize",
        "constraints": {
            "loss": {"key": "loss", "min": 0.0, "max": 100.0, "target": 0.0},
        },
        "priority": priority,
    }


def _limits(risk_max: float = 0.5, sigma_min: float = 0.0) -> dict[str, Any]:
    return {
        "safety_bounds": {"loss": {"min": 0.0, "max": 10.0}},
        "risk_threshold": risk_max,
        "stability_threshold": sigma_min,
    }


# ---------------------------------------------------------------------------
# Schema and determinism
# ---------------------------------------------------------------------------


class TestRICv2Schema:
    def test_strict_four_keys(self):
        ric = RICv2Controller()
        d = ric.decide(_intent(), {"loss": 1.0}, _limits()).to_dict()
        assert set(d.keys()) == {
            "intent_interpretation",
            "selected_action",
            "predicted_outcome",
            "fallback_plan",
        }

    def test_predicted_outcome_has_v2_trajectory_fields(self):
        ric = RICv2Controller()
        d = ric.decide(_intent(), {"loss": 1.0}, _limits())
        po = d.predicted_outcome
        assert "trajectory_summary" in po
        assert "stability_projection" in po
        assert "risk_profile" in po

    def test_fallback_plan_uses_v2_keys(self):
        ric = RICv2Controller()
        d = ric.decide(_intent(), {"loss": 1.0}, _limits())
        assert "trigger_conditions" in d.fallback_plan
        assert "safe_action" in d.fallback_plan

    def test_action_type_in_allowed_set(self):
        ric = RICv2Controller()
        d = ric.decide(_intent(), {"loss": 1.0}, _limits())
        assert d.selected_action["type"] in ACTION_TYPES


class TestRICv2Determinism:
    def test_same_seed_same_output(self):
        a = RICv2Controller(seed=7).decide(_intent(), {"loss": 1.0}, _limits()).to_dict()
        b = RICv2Controller(seed=7).decide(_intent(), {"loss": 1.0}, _limits()).to_dict()
        assert a == b

    def test_different_seeds_can_differ_under_unanimity(self):
        # All actions equivalent: stability_threshold high, no constraints.
        # Provide a proposer with non-zero magnitudes so perturbations vary.
        prop = StaticProposer({"control": 0.3, "adjust": 0.3, "hold": 0.3, "abort": 0.0})
        outs = set()
        for s in range(10):
            ric = RICv2Controller(proposer=prop, seed=s, n_perturbations=4)
            d = ric.decide(
                {"goal": ""},
                {"loss": 0.0},
                {"safety_bounds": {}, "risk_threshold": 1.0, "stability_threshold": 0.0},
            )
            outs.add(d.selected_action["magnitude"])
        # At least two different magnitudes across 10 seeds (entropy injection)
        assert len(outs) >= 2


# ---------------------------------------------------------------------------
# Trajectory rollout (Rule 3)
# ---------------------------------------------------------------------------


class TestTrajectoryRollout:
    def test_rollout_horizon_respected(self):
        ric = RICv2Controller(k_horizon=4, n_perturbations=0)
        d = ric.decide(_intent(), {"loss": 1.0}, _limits())
        # Trajectory summary embeds k length
        assert "k=4" in d.predicted_outcome["trajectory_summary"]

    def test_temporal_aware_max_risk_used(self):
        # Risk estimator returns increasing risk along the rollout.
        calls = {"n": 0}

        def rising_risk(state, limits):
            calls["n"] += 1
            return min(1.0, calls["n"] * 0.1)

        ric = RICv2Controller(k_horizon=5, n_perturbations=0, risk_estimator=rising_risk)
        d = ric.decide(_intent(), {"loss": 1.0}, _limits(risk_max=1.0))
        # max_risk in profile must reflect end-of-trajectory risk
        assert d.predicted_outcome["risk"] >= 0.4


# ---------------------------------------------------------------------------
# Safety dominance (Rule 1)
# ---------------------------------------------------------------------------


class TestRICv2Safety:
    def test_safety_dominance_aborts_on_constraint(self):
        ric = RICv2Controller(seed=1)
        intent = {
            "goal": "x",
            "constraints": {"loss": {"key": "loss", "max": 10.0}},
            "priority": 1.0,
        }
        d = ric.decide(intent, {"loss": 999.0}, _limits())
        assert d.selected_action["type"] == "abort"

    def test_no_admissible_yields_abort(self):
        ric = RICv2Controller(seed=1)
        limits = {
            "safety_bounds": {"loss": {"min": 0.0, "max": 0.0}},
            "risk_threshold": 0.0,
            "stability_threshold": 0.99,
        }
        d = ric.decide(_intent(), {"loss": 1.0}, limits)
        assert d.selected_action["type"] == "abort"

    def test_abort_confidence_zero(self):
        ric = RICv2Controller(seed=1)
        intent = {
            "goal": "x",
            "constraints": {"loss": {"key": "loss", "max": 0.0}},
            "priority": 1.0,
        }
        d = ric.decide(intent, {"loss": 5.0}, _limits())
        assert d.selected_action["confidence"] == 0.0


# ---------------------------------------------------------------------------
# Rule 2: anti-deadlock pressure
# ---------------------------------------------------------------------------


class TestAntiDeadlock:
    def test_hold_streak_counted(self):
        h = RICv2History()
        for _ in range(4):
            h.record("hold")
        assert h.hold_streak() == 4

    def test_hold_streak_breaks_on_other_action(self):
        h = RICv2History()
        for _ in range(3):
            h.record("hold")
        h.record("control")
        h.record("hold")
        assert h.hold_streak() == 1

    def test_exploration_pressure_breaks_hold_lock(self):
        # Construct a scenario where hold otherwise dominates (lowest magnitude
        # and equal stability), then accumulate a hold streak.
        prop = StaticProposer(
            {
                "control": 0.4,
                "adjust": 0.2,
                "hold": 0.0,
                "abort": 0.0,
            }
        )
        ric = RICv2Controller(
            proposer=prop,
            seed=3,
            n_perturbations=2,
            hold_streak_threshold=2,
            exploration_bonus=0.5,
        )
        history = RICv2History()
        # Pre-load a hold streak.
        for _ in range(3):
            history.record("hold")
        d = ric.decide(_intent(), {"loss": 1.0}, _limits(), history=history)
        # With exploration pressure active, the controller should NOT keep
        # selecting hold indefinitely.
        assert d.selected_action["type"] != "hold"


# ---------------------------------------------------------------------------
# Rule 5: anti-unanimity entropy injection
# ---------------------------------------------------------------------------


class TestAntiUnanimity:
    def test_unanimity_triggers_alternative(self):
        # Create truly equivalent admissible candidates: all same magnitude,
        # zero perturbations, identical proposer mags. Two distinct seeds may
        # therefore pick different equivalent admissible candidates.
        prop = StaticProposer({"control": 0.3, "adjust": 0.3, "hold": 0.3, "abort": 0.0})
        types_seen = set()
        for s in range(8):
            ric = RICv2Controller(proposer=prop, seed=s, n_perturbations=0, k_horizon=2)
            d = ric.decide(
                {"goal": ""},
                {"loss": 0.0},
                {"safety_bounds": {}, "risk_threshold": 1.0, "stability_threshold": 0.0},
            )
            types_seen.add(d.selected_action["type"])
        # Across many seeds we should observe at least 2 distinct types.
        assert len(types_seen) >= 2


# ---------------------------------------------------------------------------
# Bridge integration: v1 contract preserved, v2 opt-in works
# ---------------------------------------------------------------------------


@dataclass
class StubSnapshot:
    step: int = 0
    total_loss: float = 1.0
    constraint_loss: float = 0.1
    observer_loss: float = 0.05
    entropy_loss: float = 0.01
    gradient_norm: float = 0.2
    purity: float = 0.5
    entropy: float = 1.0
    constraint_violations: list = field(default_factory=lambda: [0.01, -0.02])


@dataclass
class StubConfig:
    learning_rate: float = 0.01


class StubEngine:
    def __init__(self):
        self.config = StubConfig()
        self.running = True
        self._step = 0

    def step(self) -> StubSnapshot:
        self._step += 1
        return StubSnapshot(step=self._step, total_loss=1.0)


class TestBridgeV2:
    def test_default_uses_v1(self):
        bridge = CIIRRICBridge(engine=StubEngine(), proposer=DeterministicLLM())
        assert bridge.use_v2 is False
        assert bridge.history is None

    def test_v2_opt_in(self):
        bridge = CIIRRICBridge(
            engine=StubEngine(),
            proposer=DeterministicLLM(),
            use_v2=True,
            v2_seed=42,
        )
        assert bridge.use_v2 is True
        assert bridge.history is not None

    def test_v2_run_records_history(self):
        bridge = CIIRRICBridge(
            engine=StubEngine(),
            proposer=DeterministicLLM(),
            use_v2=True,
            v2_seed=1,
        )
        result = bridge.run(n_steps=4)
        assert result.n_steps_executed >= 1
        # History should contain at least one recorded action
        assert len(bridge.history.recent_actions) >= 1
        # All recorded actions are valid types
        assert all(a in ACTION_TYPES for a in bridge.history.recent_actions)
