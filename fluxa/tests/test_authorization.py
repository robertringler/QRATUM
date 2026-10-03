"""Safety levels and the authorization gate.

QRADLE invariant 1 requires human authorization for SENSITIVE, CRITICAL and
EXISTENTIAL operations. These tests establish that the gate actually blocks
execution rather than merely annotating it, and that FLUXA's safety
classification is neither absent nor inflated.
"""

from __future__ import annotations

import pytest

from fluxa.engine import AUTHORIZATION_REQUIRED_LEVELS
from fluxa.events import FluxaEventType
from fluxa.state import OperationalState, classify_safety_level
from fluxa.tests.conftest import make_engine
from qradle.core.invariants import FatalInvariants, InvariantType, InvariantViolation

GATED_SCENARIOS = ("D_GENERATOR_OUTAGE", "F_COMPOUND")
UNGATED_SCENARIOS = ("A_BASELINE", "B_SOLAR_SHOCK", "C_WIND_SHOCK", "E_LOAD_SPIKE")


@pytest.mark.parametrize("scenario_id", GATED_SCENARIOS)
def test_sensitive_scenario_is_refused_without_authorization(loaded, scenarios, scenario_id):
    engine = make_engine(loaded, run_id="auth-deny", n_steps=24, authorized=False)
    with pytest.raises(InvariantViolation) as exc:
        engine.run(scenarios[scenario_id])
    assert exc.value.invariant_type is InvariantType.HUMAN_OVERSIGHT
    assert "requires human authorization" in str(exc.value)


@pytest.mark.parametrize("scenario_id", GATED_SCENARIOS)
def test_refusal_happens_before_any_physics_runs(loaded, scenarios, scenario_id):
    """The gate must be upstream of the simulation, not a post-hoc label."""
    engine = make_engine(loaded, run_id="auth-upstream", n_steps=24, authorized=False)
    with pytest.raises(InvariantViolation):
        engine.run(scenarios[scenario_id])
    assert engine.qradle.get_stats()["total_executions"] == 0


@pytest.mark.parametrize("scenario_id", GATED_SCENARIOS)
def test_refusal_is_itself_recorded_as_an_event(loaded, scenarios, scenario_id):
    """A denied operation must leave an audit trail. The ledger is created
    inside ``run``, so the denial event is captured by re-running the
    authorization step against a ledger the test owns."""
    from fluxa.events import FluxaEventLedger

    engine = make_engine(loaded, run_id="auth-event", n_steps=24, authorized=False)
    ledger = FluxaEventLedger("audit")
    with pytest.raises(InvariantViolation):
        engine.authorize(scenarios[scenario_id], ledger)
    types = [e.event_type for e in ledger]
    assert FluxaEventType.AUTHORIZATION_DENIED in types
    denial = next(e for e in ledger if e.event_type is FluxaEventType.AUTHORIZATION_DENIED)
    assert denial.safety_level == "SENSITIVE"
    assert denial.payload["authorized"] is False
    assert denial.payload["gated_perturbations"]


@pytest.mark.parametrize("scenario_id", GATED_SCENARIOS)
def test_authorized_sensitive_scenario_executes(loaded, scenarios, scenario_id):
    engine = make_engine(
        loaded, run_id="auth-allow", n_steps=24, authorized=True, authorizer_id="operator_01"
    )
    result = engine.run(scenarios[scenario_id])
    assert result.qradle_result.success
    assert result.scenario.required_safety_level == "SENSITIVE"


@pytest.mark.parametrize("scenario_id", UNGATED_SCENARIOS)
def test_routine_and_elevated_scenarios_need_no_authorization(loaded, scenarios, scenario_id):
    engine = make_engine(loaded, run_id="auth-open", n_steps=24, authorized=False)
    result = engine.run(scenarios[scenario_id])
    assert result.qradle_result.success
    assert result.scenario.required_safety_level not in AUTHORIZATION_REQUIRED_LEVELS


def test_authorization_is_recorded_on_the_scenario_event(loaded, scenarios):
    engine = make_engine(
        loaded, run_id="auth-record", n_steps=24, authorized=True, authorizer_id="operator_42"
    )
    result = engine.run(scenarios["D_GENERATOR_OUTAGE"])
    event = next(
        e for e in result.ledger if e.event_type is FluxaEventType.SCENARIO_INITIALIZED
    )
    assert event.payload["authorized"] is True
    assert event.safety_level == "SENSITIVE"
    assert result.qradle_result.success


def test_qradle_execution_context_carries_the_safety_level(loaded, scenarios):
    engine = make_engine(loaded, run_id="auth-ctx", n_steps=24, authorized=True)
    result = engine.run(scenarios["F_COMPOUND"])
    # QRADLE emits an execution_started event stamped with the safety level.
    started = [
        n.data
        for n in engine.qradle.merkle_chain.nodes
        if n.data.get("event_type") == "execution_started"
    ]
    assert started and started[0]["safety_level"] == "SENSITIVE"


# ----------------------------------------------------- per-step classification
def test_load_shedding_is_classified_critical():
    assert (
        classify_safety_level(
            operational_state=OperationalState.EMERGENCY,
            violation_count=1,
            unserved_mw=5.0,
            outage_active=False,
            major_dispatch_change=False,
            storage_dispatch_mw=0.0,
        )
        == "CRITICAL"
    )


def test_two_simultaneous_violations_are_classified_critical():
    assert (
        classify_safety_level(
            operational_state=OperationalState.EMERGENCY,
            violation_count=2,
            unserved_mw=0.0,
            outage_active=False,
            major_dispatch_change=False,
            storage_dispatch_mw=0.0,
        )
        == "CRITICAL"
    )


def test_an_active_outage_is_classified_sensitive():
    assert (
        classify_safety_level(
            operational_state=OperationalState.NORMAL,
            violation_count=0,
            unserved_mw=0.0,
            outage_active=True,
            major_dispatch_change=False,
            storage_dispatch_mw=0.0,
        )
        == "SENSITIVE"
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"operational_state": OperationalState.ALERT},
        {"major_dispatch_change": True},
        {"storage_dispatch_mw": 20.0},
    ],
)
def test_degraded_margin_or_material_action_is_elevated(kwargs):
    base = dict(
        operational_state=OperationalState.NORMAL,
        violation_count=0,
        unserved_mw=0.0,
        outage_active=False,
        major_dispatch_change=False,
        storage_dispatch_mw=0.0,
    )
    assert classify_safety_level(**{**base, **kwargs}) == "ELEVATED"


def test_quiet_normal_operation_is_routine():
    assert (
        classify_safety_level(
            operational_state=OperationalState.NORMAL,
            violation_count=0,
            unserved_mw=0.0,
            outage_active=False,
            major_dispatch_change=False,
            storage_dispatch_mw=0.0,
        )
        == "ROUTINE"
    )


def test_fluxa_never_emits_existential(loaded, scenarios):
    """An ordinary power-system contingency is not an architecture-level
    catastrophe. If FLUXA ever labelled one EXISTENTIAL the safety scale
    would be meaningless."""
    engine = make_engine(loaded, run_id="no-existential", n_steps=288, authorized=True)
    result = engine.run(scenarios["F_COMPOUND"])
    levels = {s.safety_level for s in result.states} | {e.safety_level for e in result.ledger}
    assert "EXISTENTIAL" not in levels
    assert levels <= {"ROUTINE", "ELEVATED", "SENSITIVE", "CRITICAL"}


def test_critical_states_actually_occur_under_the_compound_event(loaded, scenarios):
    """The classification must be exercised, not merely available."""
    engine = make_engine(loaded, run_id="critical-occurs", n_steps=288, authorized=True)
    result = engine.run(scenarios["F_COMPOUND"])
    assert any(s.safety_level == "CRITICAL" for s in result.states)


def test_critical_dominates_sensitive_when_both_apply():
    """Precedence is deliberate: a step that both has a unit out *and* is
    shedding load is CRITICAL, not SENSITIVE. The worse condition wins, so a
    forced outage does not mask load shedding in the audit trail."""
    level = classify_safety_level(
        operational_state=OperationalState.EMERGENCY,
        violation_count=1,
        unserved_mw=12.0,
        outage_active=True,
        major_dispatch_change=True,
        storage_dispatch_mw=50.0,
    )
    assert level == "CRITICAL"


def test_sensitive_steps_occur_when_an_outage_is_survived(loaded, scenarios):
    """SENSITIVE per-step appears only where a unit is out *and* nothing is
    being shed -- in this library that is the outage's recovery ramp, when the
    unit is partially back and the system is whole again."""
    engine = make_engine(loaded, run_id="sensitive-occurs", n_steps=288, authorized=True)
    result = engine.run(scenarios["D_GENERATOR_OUTAGE"])
    sensitive = [s for s in result.states if s.safety_level == "SENSITIVE"]
    assert sensitive, "no SENSITIVE step observed under a forced outage"
    for state in sensitive:
        assert state.unserved_mw < 1e-6
        assert state.active_perturbations


def test_all_four_levels_are_exercised_across_the_library(loaded, scenarios):
    seen: set[str] = set()
    for scenario_id in ("A_BASELINE", "D_GENERATOR_OUTAGE", "F_COMPOUND"):
        engine = make_engine(loaded, run_id=f"levels-{scenario_id}", n_steps=288, authorized=True)
        seen |= {s.safety_level for s in engine.run(scenarios[scenario_id]).states}
    assert {"ROUTINE", "ELEVATED", "SENSITIVE", "CRITICAL"} <= seen


def test_qradle_gate_itself_behaves_as_documented():
    """Direct test of the substrate's gate, independent of FLUXA."""
    for level in ("SENSITIVE", "CRITICAL", "EXISTENTIAL"):
        with pytest.raises(InvariantViolation):
            FatalInvariants.enforce_human_oversight("op", level, authorized=False)
        FatalInvariants.enforce_human_oversight("op", level, authorized=True)
    for level in ("ROUTINE", "ELEVATED"):
        FatalInvariants.enforce_human_oversight("op", level, authorized=False)
