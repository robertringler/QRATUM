"""Tests for the strict CIIR-CRS-RIC five-module package.

Covers determinism, separation of concerns, all four failure modes, and the
observer-map contract.
"""
from __future__ import annotations

import pytest

from qagents.ciir_crs_ric.ciir import (C_ACTIVE, MAX_CPU, Constraint, Inv,
                                       Observation, Omega, State,
                                       all_satisfied, phi)
from qagents.ciir_crs_ric.crs import (A_FULL, Action, admissible_actions,
                                      constrained_transition, transition)
from qagents.ciir_crs_ric.executor import print_trace, run, run_step
from qagents.ciir_crs_ric.failures import (STATUS_OK, STATUS_TYPE_I,
                                           STATUS_TYPE_II, STATUS_TYPE_III,
                                           STATUS_TYPE_IV, ConstraintViolation,
                                           IntentParseFailure,
                                           InvalidTransition,
                                           InvariantViolation, classify)
from qagents.ciir_crs_ric.ric import Intent, parse_intent, select_action

# --------------------------------------------------------------------------
# CIIR layer
# --------------------------------------------------------------------------

class TestCIIR:
    def test_state_is_immutable(self):
        s = State(0, 0, "idle")
        with pytest.raises(Exception):
            s.cpu_used = 5  # type: ignore[misc]

    def test_state_rejects_bad_status(self):
        with pytest.raises(ValueError):
            State(0, 0, "exploded")

    def test_state_rejects_non_int(self):
        with pytest.raises(TypeError):
            State("zero", 0, "idle")  # type: ignore[arg-type]

    def test_omega_does_not_expose_internal_notes(self):
        s = State(1, 2, "idle", internal_notes="SECRET")
        obs = Omega(s)
        assert isinstance(obs, Observation)
        assert "SECRET" not in repr(obs)
        assert not hasattr(obs, "internal_notes")

    def test_omega_derived_fields(self):
        s = State(3, 5, "running")
        obs = Omega(s)
        assert obs.cpu_free == MAX_CPU - 3
        assert obs.mem_free == 16 - 5

    def test_phi_pure_and_swallows_exceptions(self):
        bad = Constraint(name="bad", predicate=lambda s: 1 / 0)
        assert phi(State(0, 0, "idle"), bad) is False

    def test_all_satisfied_returns_violation_names(self):
        s = State(MAX_CPU + 1, 0, "running")
        ok, violated = all_satisfied(s, C_ACTIVE)
        assert ok is False
        assert "cpu_in_range" in violated

    def test_invariant_on_halted(self):
        assert Inv(State(0, 0, "halted")) is False

    def test_invariant_on_budget_exceed(self):
        assert Inv(State(8, 13, "running")) is False
        assert Inv(State(8, 12, "running")) is True


# --------------------------------------------------------------------------
# CRS layer
# --------------------------------------------------------------------------

class TestCRS:
    def test_transition_pure(self):
        s = State(1, 1, "idle")
        s_next = transition(s, Action.ALLOCATE_CPU)
        assert s_next is not s
        assert s.cpu_used == 1  # original unchanged
        assert s_next.cpu_used == 2

    def test_transition_noop_returns_same_value(self):
        s = State(1, 1, "idle")
        assert transition(s, Action.NOOP) == s

    def test_constrained_transition_pre_violation_returns_bottom(self):
        s = State(MAX_CPU + 1, 0, "running")  # already illegal
        s_next, violated = constrained_transition(s, Action.NOOP, C_ACTIVE)
        assert s_next is None
        assert violated  # at least one constraint named

    def test_constrained_transition_post_violation_returns_bottom(self):
        s = State(MAX_CPU, 0, "running")
        s_next, violated = constrained_transition(s, Action.ALLOCATE_CPU, C_ACTIVE)
        assert s_next is None
        assert "cpu_in_range" in violated

    def test_constrained_transition_success(self):
        s = State(0, 0, "idle")
        s_next, violated = constrained_transition(s, Action.ALLOCATE_CPU, C_ACTIVE)
        assert s_next == State(1, 0, "idle")
        assert violated == ()

    def test_admissible_actions_explicit_enumeration(self):
        s = State(MAX_CPU, 0, "idle")
        adm = admissible_actions(s, A_FULL, C_ACTIVE)
        assert Action.ALLOCATE_CPU not in adm  # would exceed bound
        assert Action.RELEASE_CPU in adm
        assert Action.NOOP in adm

    def test_admissible_empty_when_pre_violates(self):
        s = State(MAX_CPU + 1, 0, "running")
        adm = admissible_actions(s, A_FULL, C_ACTIVE)
        assert adm == frozenset()


# --------------------------------------------------------------------------
# RIC layer (no raw-State access)
# --------------------------------------------------------------------------

class TestRIC:
    def test_parse_intent_canonical(self):
        assert parse_intent("increase_cpu") is Intent.INCREASE_CPU
        assert parse_intent("MORE_CPU") is Intent.INCREASE_CPU  # case-insensitive
        assert parse_intent("  begin  ") is Intent.BEGIN

    def test_parse_intent_unknown_returns_bottom(self):
        assert parse_intent("zorg") is None
        assert parse_intent("") is None
        assert parse_intent(None) is None  # type: ignore[arg-type]

    def test_select_action_picks_from_admissible_only(self):
        obs = Omega(State(0, 0, "idle"))
        adm = frozenset({Action.NOOP, Action.START})
        # Intent prefers ALLOCATE_CPU which is NOT in adm -> fallback NOOP
        a = select_action(Intent.INCREASE_CPU, obs, adm)
        assert a == Action.NOOP

    def test_select_action_intent_bottom_returns_noop(self):
        obs = Omega(State(0, 0, "idle"))
        a = select_action(None, obs, frozenset(A_FULL))
        assert a == Action.NOOP

    def test_select_action_deterministic(self):
        obs = Omega(State(0, 0, "idle"))
        adm = frozenset(A_FULL)
        a1 = select_action(Intent.INCREASE_CPU, obs, adm)
        a2 = select_action(Intent.INCREASE_CPU, obs, adm)
        assert a1 == a2 == Action.ALLOCATE_CPU


# --------------------------------------------------------------------------
# Executor 9-step loop & trace
# --------------------------------------------------------------------------

class TestExecutor:
    def test_case_A_valid(self):
        s0 = State(0, 0, "idle")
        s1, obs, entry = run_step(s0, "increase_cpu")
        assert entry.status == STATUS_OK
        assert entry.action == Action.ALLOCATE_CPU
        assert s1 == State(1, 0, "idle")
        assert obs is not None and obs.cpu_used == 1
        assert entry.failure is None

    def test_case_B_constraint_violation(self):
        s0 = State(MAX_CPU + 1, 0, "running")
        s1, obs, entry = run_step(s0, "increase_cpu")
        assert entry.status == STATUS_TYPE_I
        assert entry.failure is not None
        assert entry.failure.code == STATUS_TYPE_I
        assert s1 == s0  # no progress
        assert obs is None

    def test_case_C_invariant_violation(self):
        s0 = State(8, 12, "running")
        s1, obs, entry = run_step(s0, "increase_mem")
        assert entry.status == STATUS_TYPE_IV
        assert entry.action == Action.ALLOCATE_MEM
        assert entry.state_t1 is None  # invariant-broken state never propagated
        assert s1 == s0
        assert obs is None

    def test_case_D_parse_failure(self):
        s0 = State(0, 0, "idle")
        s1, obs, entry = run_step(s0, "frobnicate")
        assert entry.status == STATUS_TYPE_II
        assert entry.intent is None
        assert entry.action == Action.NOOP
        assert s1 == s0

    def test_run_multi_step(self):
        s0 = State(0, 0, "idle")
        sf, log = run(s0, ["increase_cpu", "increase_cpu", "begin"])
        assert sf == State(2, 0, "running")
        assert [e.status for e in log.entries] == [STATUS_OK] * 3

    def test_determinism_identical_inputs(self):
        s0 = State(2, 3, "running")
        _, _, e1 = run_step(s0, "increase_cpu")
        _, _, e2 = run_step(s0, "increase_cpu")
        assert e1.status == e2.status
        assert e1.action == e2.action
        assert e1.state_t1 == e2.state_t1
        assert e1.admissible == e2.admissible

    def test_executor_never_raises(self):
        # Even malformed inputs should not raise out of run_step.
        for raw in ["", "???", "increase_cpu", "begin"]:
            run_step(State(0, 0, "idle"), raw)

    def test_print_trace_runs(self, capsys):
        s0 = State(0, 0, "idle")
        _, log = run(s0, ["increase_cpu", "frobnicate"])
        print_trace(log)
        out = capsys.readouterr().out
        assert "TRACE" in out
        assert "TYPE_II" in out


# --------------------------------------------------------------------------
# Failures taxonomy
# --------------------------------------------------------------------------

class TestFailures:
    def test_classify_known(self):
        rec = classify(ConstraintViolation("x"))
        assert rec is not None and rec.code == STATUS_TYPE_I

    def test_classify_unknown(self):
        assert classify(ValueError("x")) is None

    def test_failure_codes_distinct(self):
        codes = {
            ConstraintViolation.code,
            IntentParseFailure.code,
            InvalidTransition.code,
            InvariantViolation.code,
        }
        assert len(codes) == 4
        assert codes == {STATUS_TYPE_I, STATUS_TYPE_II, STATUS_TYPE_III, STATUS_TYPE_IV}

    def test_failure_record_carries_details(self):
        f = ConstraintViolation("nope", action="X", admissible=[])
        rec = f.record()
        assert rec.details["action"] == "X"
        assert rec.message == "nope"
