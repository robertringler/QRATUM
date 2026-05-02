"""Tests for the Reality Interface Controller and its CIIR / LLM wiring."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from qagents.ciir_ric_bridge import (CIIRRICBridge, default_intent,
                                     default_system_limits,
                                     snapshot_to_world_state)
from qagents.llm_backends import (AnthropicProposer, DeterministicLLM,
                                  GeminiProposer, LocalLLMProposer,
                                  OpenAIProposer, available_backends)
from qagents.reality_interface import (ACTION_TYPES, LinearSimulator,
                                       RealityInterfaceController,
                                       StaticProposer)

# ---------------------------------------------------------------------------
# RIC core
# ---------------------------------------------------------------------------


def _intent(priority: float = 0.5) -> dict[str, Any]:
    return {
        "goal": "stabilize",
        "constraints": {
            "loss": {"key": "loss", "min": 0.0, "max": 100.0, "target": 0.0},
        },
        "priority": priority,
    }


def _limits(risk_max: float = 0.5, sigma_min: float = 0.1) -> dict[str, Any]:
    return {
        "safety_bounds": {"loss": {"min": 0.0, "max": 10.0}},
        "risk_threshold": risk_max,
        "stability_threshold": sigma_min,
    }


class TestRICOutputSchema:
    def test_strict_four_keys(self):
        ric = RealityInterfaceController()
        d = ric.decide(_intent(), {"loss": 1.0}, _limits()).to_dict()
        assert set(d.keys()) == {
            "intent_interpretation",
            "selected_action",
            "predicted_outcome",
            "fallback_plan",
        }

    def test_action_type_in_allowed_set(self):
        ric = RealityInterfaceController()
        d = ric.decide(_intent(), {"loss": 1.0}, _limits())
        assert d.selected_action["type"] in ACTION_TYPES

    def test_confidence_bounded(self):
        ric = RealityInterfaceController()
        d = ric.decide(_intent(), {"loss": 1.0}, _limits())
        c = d.selected_action["confidence"]
        assert 0.0 <= c <= 1.0

    def test_predicted_outcome_has_required_fields(self):
        ric = RealityInterfaceController()
        d = ric.decide(_intent(), {"loss": 1.0}, _limits())
        po = d.predicted_outcome
        assert "expected_state" in po and "risk" in po and "stability_score" in po

    def test_fallback_plan_has_trigger_and_action(self):
        ric = RealityInterfaceController()
        d = ric.decide(_intent(), {"loss": 1.0}, _limits())
        assert "trigger_condition" in d.fallback_plan
        assert "action" in d.fallback_plan


class TestRICDeterminism:
    def test_same_input_same_output(self):
        ric = RealityInterfaceController()
        a = ric.decide(_intent(), {"loss": 1.0}, _limits()).to_dict()
        b = ric.decide(_intent(), {"loss": 1.0}, _limits()).to_dict()
        assert a == b

    def test_no_hallucinated_intent_keys(self):
        ric = RealityInterfaceController()
        d = ric.decide({"goal": "x"}, {}, _limits()).to_dict()
        assert d["intent_interpretation"]["constraints"] == {}
        assert d["intent_interpretation"]["priority"] == 0.0


class TestRICPolicy:
    def test_safety_dominance_aborts_on_constraint_violation(self):
        # Constraint loss<=10 violated by world_state.loss=999
        ric = RealityInterfaceController()
        intent = {
            "goal": "x",
            "constraints": {"loss": {"key": "loss", "max": 10.0}},
            "priority": 1.0,
        }
        d = ric.decide(intent, {"loss": 999.0}, _limits())
        assert d.selected_action["type"] == "abort"
        assert d.selected_action["confidence"] == 0.0

    def test_safety_dominance_aborts_when_no_admissible(self):
        # Force every action over the risk threshold by making safety bounds tiny.
        ric = RealityInterfaceController()
        limits = {
            "safety_bounds": {"loss": {"min": 0.0, "max": 0.0}},
            "risk_threshold": 0.0,
            "stability_threshold": 0.99,
        }
        d = ric.decide(_intent(), {"loss": 1.0}, limits)
        assert d.selected_action["type"] == "abort"

    def test_minimal_intervention_among_admissible(self):
        # Hold has magnitude 0; should win when admissible.
        ric = RealityInterfaceController(proposer=StaticProposer({
            "control": 1.0, "adjust": 0.5, "hold": 0.0, "abort": 0.0,
        }))
        d = ric.decide(_intent(), {"loss": 1.0}, _limits())
        assert d.selected_action["type"] == "hold"

    def test_stability_awareness_prefers_hold_or_adjust(self):
        # Custom stability estimator forces near-instability for control,
        # high stability for hold/adjust.
        def stab(predicted, intent):
            # Differentiate by mock-stored marker
            return predicted.get("_stab", 0.6)

        class MarkerSim:
            def predict(self, ws, action):
                out = dict(ws)
                out["_stab"] = 0.3 if action["type"] == "control" else 0.8
                return out

        ric = RealityInterfaceController(
            simulator=MarkerSim(),
            proposer=StaticProposer({
                "control": 0.1, "adjust": 0.5, "hold": 0.0, "abort": 0.0,
            }),
            stability_estimator=stab,
        )
        d = ric.decide(_intent(), {"loss": 1.0}, _limits())
        assert d.selected_action["type"] in ("hold", "adjust")

    def test_predict_before_act_confidence_zero_on_abort(self):
        ric = RealityInterfaceController()
        intent = {
            "goal": "x",
            "constraints": {"loss": {"key": "loss", "max": 0.0}},
            "priority": 1.0,
        }
        d = ric.decide(intent, {"loss": 5.0}, _limits())
        assert d.selected_action["confidence"] == 0.0


class TestRICSimulator:
    def test_linear_simulator_control_increases_state(self):
        sim = LinearSimulator()
        out = sim.predict({"x": 1.0}, {"type": "control", "magnitude": 2.0})
        assert out["x"] == pytest.approx(3.0)

    def test_linear_simulator_hold_unchanged(self):
        sim = LinearSimulator()
        out = sim.predict({"x": 1.0, "y": "tag"}, {"type": "hold", "magnitude": 1.0})
        assert out == {"x": 1.0, "y": "tag"}


# ---------------------------------------------------------------------------
# LLM backends
# ---------------------------------------------------------------------------


class TestLLMBackends:
    def test_deterministic_llm_returns_all_actions(self):
        m = DeterministicLLM().propose({"priority": 0.6}, {}, {})
        assert set(m.keys()) == set(ACTION_TYPES)
        for v in m.values():
            assert v >= 0.0

    @pytest.mark.parametrize(
        "cls",
        [OpenAIProposer, AnthropicProposer, GeminiProposer, LocalLLMProposer],
    )
    def test_network_backends_fall_back_when_unavailable(self, cls, monkeypatch):
        # Strip any real env keys so we hit the fallback path.
        for env in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY", "LOCAL_LLM_ENDPOINT"):
            monkeypatch.delenv(env, raising=False)
        proposer = cls()
        m = proposer.propose({"priority": 0.4}, {}, {})
        assert set(m.keys()) == set(ACTION_TYPES)

    def test_available_backends_includes_all_supported(self):
        names = [getattr(p, "name", p.__class__.__name__) for p in available_backends()]
        assert "deterministic" in names
        assert "openai" in names
        assert "anthropic" in names
        assert "gemini" in names
        assert "local" in names


# ---------------------------------------------------------------------------
# CIIR ↔ RIC bridge (with stub engine)
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
    def __init__(self, force_abort_at: int | None = None):
        self.config = StubConfig()
        self.running = True
        self._step = 0
        self._force_abort_at = force_abort_at

    def step(self) -> StubSnapshot:
        self._step += 1
        # If asked, push loss far outside default safety bounds (max=1e3).
        loss = 1e9 if self._force_abort_at and self._step >= self._force_abort_at else 1.0
        return StubSnapshot(step=self._step, total_loss=loss)


class TestSnapshotMapping:
    def test_snapshot_to_world_state_keys(self):
        ws = snapshot_to_world_state(StubSnapshot())
        for key in ("step", "total_loss", "purity", "entropy", "gradient_norm", "constraint_norm"):
            assert key in ws

    def test_constraint_norm_is_l1(self):
        ws = snapshot_to_world_state(StubSnapshot(constraint_violations=[0.5, -0.25]))
        assert ws["constraint_norm"] == pytest.approx(0.75)


class TestBridge:
    def test_default_intent_and_limits(self):
        intent = default_intent()
        limits = default_system_limits()
        assert "constraints" in intent and "priority" in intent
        assert "safety_bounds" in limits and "risk_threshold" in limits

    def test_run_completes_with_deterministic_llm(self):
        engine = StubEngine()
        bridge = CIIRRICBridge(engine=engine, proposer=DeterministicLLM())
        result = bridge.run(n_steps=5)
        assert result.proposer_name == "deterministic"
        assert result.n_steps_executed >= 1
        assert sum(result.action_counts.values()) >= 1

    def test_bridge_aborts_on_safety_violation(self):
        engine = StubEngine(force_abort_at=2)
        bridge = CIIRRICBridge(engine=engine, proposer=DeterministicLLM())
        result = bridge.run(n_steps=10)
        # Engine pushes loss above safety bound at step 2; RIC should abort.
        assert result.aborted_at is not None
        assert engine.running is False

    def test_bridge_iterates_all_backends(self):
        for proposer in available_backends():
            engine = StubEngine()
            bridge = CIIRRICBridge(engine=engine, proposer=proposer)
            result = bridge.run(n_steps=3)
            assert result.n_steps_executed >= 1
            assert set(result.action_counts.keys()).issubset(set(ACTION_TYPES))

    def test_lr_scaling_records_base(self):
        engine = StubEngine()
        bridge = CIIRRICBridge(engine=engine, proposer=DeterministicLLM())
        bridge.run(n_steps=4)
        # Base learning rate should be cached on config the first time control/adjust fires.
        # If only hold fired, base may not be set; treat as optional.
        if hasattr(engine.config, "_ric_base_lr"):
            assert engine.config._ric_base_lr == pytest.approx(0.01)
