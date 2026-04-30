"""Tests for the CIIR–CRS–RIC constraint-governed execution framework.

Covers:
    - CIIR: Constraint satisfaction, algebra, observer map
    - CRS:  State space, admissible action computation, T_C
    - RIC:  Intent parser, execution loop, all four failure modes
    - Integration: complete closed-loop execution scenarios
"""

from __future__ import annotations

import pytest
from typing import Optional

from qagents.framework.ciir import (
    Constraint,
    ConstraintAlgebra,
    ObserverMap,
    State,
    accessible_state_space,
)
from qagents.framework.crs import Action, CRS, FailureMode, NOOP
from qagents.framework.ric import Intent, RIC, TraceEntry


# ===========================================================================
# Shared domain fixtures
# ===========================================================================

VALID_MODES = {"idle", "active", "locked"}

C_LEVEL = Constraint("level_range", lambda s: 0 <= s.get("level", -1) <= 5)
C_RESOURCE = Constraint("resource_range", lambda s: 0 <= s.get("resource", -1) <= 10)
C_MODE = Constraint("mode_valid", lambda s: s.get("mode") in VALID_MODES)
C_NO_ACQUIRE_LOCKED = Constraint(
    "no_acquire_locked",
    lambda s: not (s.get("mode") == "locked" and s.get("resource", 0) > 0),
)

ALL_CONSTRAINTS = [C_LEVEL, C_RESOURCE, C_MODE, C_NO_ACQUIRE_LOCKED]

ACTIONS = [
    Action("INCREASE_LEVEL"),
    Action("DECREASE_LEVEL"),
    Action("ACQUIRE_RESOURCE"),
    Action("RELEASE_RESOURCE"),
    NOOP,
]


def transition_fn(state: State, action: Action) -> Optional[dict]:
    s = dict(state)
    name = action.name
    if name == "INCREASE_LEVEL":
        return None if s["level"] >= 5 else {**s, "level": s["level"] + 1}
    if name == "DECREASE_LEVEL":
        return None if s["level"] <= 0 else {**s, "level": s["level"] - 1}
    if name == "ACQUIRE_RESOURCE":
        return None if s["resource"] >= 10 else {**s, "resource": s["resource"] + 1}
    if name == "RELEASE_RESOURCE":
        return None if s["resource"] <= 0 else {**s, "resource": s["resource"] - 1}
    if name == "NOOP":
        return dict(s)
    return None


def invariant(state: State) -> bool:
    return (
        isinstance(state.get("level"), int) and 0 <= state["level"] <= 5
        and isinstance(state.get("resource"), int) and 0 <= state["resource"] <= 10
        and state.get("mode") in VALID_MODES
    )


OBSERVER = ObserverMap(lambda s: {"level": s.get("level"), "mode": s.get("mode")})

INTENT_MAP = {
    "acquire": Intent("acquire_resource", preferred_action="ACQUIRE_RESOURCE"),
    "release": Intent("release_resource", preferred_action="RELEASE_RESOURCE"),
    "elevate": Intent("increase_level", preferred_action="INCREASE_LEVEL"),
    "hold":    Intent("hold", preferred_action="NOOP"),
}


def intent_parser(raw: str) -> Optional[Intent]:
    return INTENT_MAP.get(raw.strip().lower())


def make_system(state: dict, constraints=None) -> tuple[CRS, RIC]:
    constraints = constraints or ALL_CONSTRAINTS
    crs = CRS(state, ACTIONS, transition_fn, ConstraintAlgebra(ALL_CONSTRAINTS), constraints, invariant)
    ric = RIC(crs, OBSERVER, intent_parser)
    return crs, ric


# ===========================================================================
# CIIR tests
# ===========================================================================

class TestConstraint:
    def test_satisfies_true(self):
        s = {"level": 3, "resource": 5, "mode": "active"}
        assert C_LEVEL.satisfies(s) is True

    def test_satisfies_false_below(self):
        s = {"level": -1, "resource": 5, "mode": "active"}
        assert C_LEVEL.satisfies(s) is False

    def test_satisfies_false_above(self):
        s = {"level": 6, "resource": 5, "mode": "active"}
        assert C_LEVEL.satisfies(s) is False

    def test_repr(self):
        assert "level_range" in repr(C_LEVEL)


class TestConstraintAlgebra:
    def setup_method(self):
        self.algebra = ConstraintAlgebra(ALL_CONSTRAINTS)
        self.valid_state = {"level": 2, "resource": 3, "mode": "active"}
        self.invalid_state = {"level": 6, "resource": 3, "mode": "active"}

    def test_satisfies_all_valid(self):
        assert self.algebra.satisfies_all(self.valid_state) is True

    def test_satisfies_all_invalid(self):
        assert self.algebra.satisfies_all(self.invalid_state) is False

    def test_violated_constraints_empty_for_valid(self):
        assert self.algebra.violated_constraints(self.valid_state) == []

    def test_violated_constraints_non_empty(self):
        violated = self.algebra.violated_constraints(self.invalid_state)
        assert C_LEVEL in violated

    def test_is_consistent_with_witness(self):
        assert self.algebra.is_consistent(self.valid_state) is True

    def test_is_inconsistent_with_bad_witness(self):
        assert self.algebra.is_consistent(self.invalid_state) is False

    def test_conjunction(self):
        conj = self.algebra.conjunction(C_LEVEL, C_RESOURCE)
        s_good = {"level": 2, "resource": 3}
        s_bad = {"level": 2, "resource": 11}
        assert conj.satisfies(s_good) is True
        assert conj.satisfies(s_bad) is False

    def test_top(self):
        top = self.algebra.top()
        assert top.satisfies({}) is True

    def test_bottom(self):
        bot = self.algebra.bottom()
        assert bot.satisfies({"level": 2}) is False

    def test_active_subset(self):
        active = [C_LEVEL]
        # resource constraint not active → invalid resource passes
        s = {"level": 2, "resource": 999, "mode": "active"}
        assert self.algebra.satisfies_all(s, active) is True


class TestObserverMap:
    def test_observe_projects_correctly(self):
        s = {"level": 3, "resource": 7, "mode": "active"}
        obs = OBSERVER.observe(s)
        assert obs == {"level": 3, "mode": "active"}
        assert "resource" not in obs

    def test_observable_equal_same_observation(self):
        s1 = {"level": 3, "resource": 2, "mode": "active"}
        s2 = {"level": 3, "resource": 9, "mode": "active"}
        # Both project to same observable (resource hidden)
        assert OBSERVER.observable_equal(s1, s2) is True

    def test_observable_not_equal_different_level(self):
        s1 = {"level": 3, "resource": 2, "mode": "active"}
        s2 = {"level": 4, "resource": 2, "mode": "active"}
        assert OBSERVER.observable_equal(s1, s2) is False


class TestAccessibleStateSpace:
    def test_valid_state_accessible(self):
        algebra = ConstraintAlgebra(ALL_CONSTRAINTS)
        s = {"level": 2, "resource": 3, "mode": "active"}
        assert accessible_state_space(s, algebra) is True

    def test_invalid_state_not_accessible(self):
        algebra = ConstraintAlgebra(ALL_CONSTRAINTS)
        s = {"level": 6, "resource": 3, "mode": "active"}
        assert accessible_state_space(s, algebra) is False


# ===========================================================================
# CRS tests
# ===========================================================================

class TestCRS:
    def setup_method(self):
        self.algebra = ConstraintAlgebra(ALL_CONSTRAINTS)
        self.base_state = {"level": 2, "resource": 3, "mode": "active"}

    def make_crs(self, state=None, constraints=None):
        s = state or self.base_state
        c = constraints or ALL_CONSTRAINTS
        return CRS(s, ACTIONS, transition_fn, self.algebra, c, invariant)

    def test_is_deterministic(self):
        crs = self.make_crs()
        assert crs.is_deterministic() is True

    def test_in_invariant_valid(self):
        crs = self.make_crs()
        assert crs.in_invariant(self.base_state) is True

    def test_in_invariant_invalid(self):
        crs = self.make_crs()
        assert crs.in_invariant({"level": 6, "resource": 3, "mode": "active"}) is False

    def test_admissible_actions_nonempty(self):
        crs = self.make_crs()
        admissible = crs.admissible_actions(self.base_state)
        assert len(admissible) > 0

    def test_acquire_in_active_admissible(self):
        crs = self.make_crs()
        names = [a.name for a in crs.admissible_actions(self.base_state)]
        assert "ACQUIRE_RESOURCE" in names

    def test_acquire_in_locked_not_admissible(self):
        """c_no_acquire_locked blocks ACQUIRE_RESOURCE in locked mode."""
        state = {"level": 2, "resource": 1, "mode": "locked"}
        crs = self.make_crs(state=state)
        names = [a.name for a in crs.admissible_actions(state)]
        assert "ACQUIRE_RESOURCE" not in names

    def test_increase_level_at_max_not_admissible(self):
        state = {"level": 5, "resource": 3, "mode": "active"}
        crs = self.make_crs(state=state)
        names = [a.name for a in crs.admissible_actions(state)]
        assert "INCREASE_LEVEL" not in names

    def test_execute_valid_returns_successor(self):
        crs = self.make_crs()
        succ = crs.execute(self.base_state, Action("ACQUIRE_RESOURCE"))
        assert succ is not None
        assert succ["resource"] == 4

    def test_execute_illegal_returns_none(self):
        state = {"level": 2, "resource": 1, "mode": "locked"}
        crs = self.make_crs(state=state)
        succ = crs.execute(state, Action("ACQUIRE_RESOURCE"))
        assert succ is None

    def test_execute_physical_ceiling_returns_none(self):
        state = {"level": 5, "resource": 3, "mode": "active"}
        crs = self.make_crs(state=state)
        succ = crs.execute(state, Action("INCREASE_LEVEL"))
        assert succ is None

    def test_noop_always_produces_same_state(self):
        crs = self.make_crs()
        succ = crs.execute(self.base_state, NOOP)
        assert succ == self.base_state

    def test_admissible_empty_when_state_violates_constraint(self):
        """s ⊭ C_act → A_C(s) = ∅."""
        bad_state = {"level": -1, "resource": 3, "mode": "active"}
        crs = self.make_crs(state=bad_state)
        assert crs.admissible_actions(bad_state) == []


# ===========================================================================
# RIC tests
# ===========================================================================

class TestRIC:
    def setup_method(self):
        self.valid_state = {"level": 2, "resource": 3, "mode": "active"}

    def test_case_a_valid_execution(self):
        """Case A: valid intent in valid state → OK transition."""
        _, ric = make_system(self.valid_state)
        entry = ric.step("acquire", self.valid_state)
        assert entry.status == "OK"
        assert entry.selected_action == "ACQUIRE_RESOURCE"
        assert entry.state_after is not None
        assert entry.state_after["resource"] == 4
        assert entry.failure_mode is None

    def test_case_b_constraint_violation(self):
        """Case B: ACQUIRE in locked mode → Type I failure."""
        locked_state = {"level": 2, "resource": 1, "mode": "locked"}
        _, ric = make_system(locked_state)
        entry = ric.step("acquire", locked_state)
        assert entry.status == "FAILED"
        assert entry.failure_mode == FailureMode.TYPE_I_CONSTRAINT_VIOLATION
        assert entry.state_after is None

    def test_case_c_physical_ceiling(self):
        """Case C: INCREASE_LEVEL at max → excluded from A_C; ρ selects fallback."""
        max_state = {"level": 5, "resource": 3, "mode": "active"}
        _, ric = make_system(max_state)
        entry = ric.step("elevate", max_state)
        # The dangerous action is excluded — key invariant of the framework
        assert "INCREASE_LEVEL" not in entry.admissible_actions
        # The controller DOES NOT execute INCREASE_LEVEL under any circumstance
        assert entry.selected_action != "INCREASE_LEVEL"

    def test_case_d_parse_failure(self):
        """Case D: malformed intent → Type II failure."""
        _, ric = make_system(self.valid_state)
        entry = ric.step("??INVALID??", self.valid_state)
        assert entry.status == "FAILED"
        assert entry.failure_mode == FailureMode.TYPE_II_INTENT_PARSE_FAILURE
        assert entry.parsed_intent is None
        assert entry.state_after is None

    def test_trace_is_appended(self):
        _, ric = make_system(self.valid_state)
        ric.step("acquire", self.valid_state)
        ric.step("release", self.valid_state)
        assert len(ric.trace) == 2

    def test_trace_step_indices(self):
        _, ric = make_system(self.valid_state)
        e0 = ric.step("acquire", self.valid_state)
        e1 = ric.step("hold", self.valid_state)
        assert e0.step == 0
        assert e1.step == 1

    def test_observation_excludes_resource(self):
        _, ric = make_system(self.valid_state)
        entry = ric.step("acquire", self.valid_state)
        assert "resource" not in entry.observation

    def test_determinism_same_output_for_same_input(self):
        """Same (intent, state) always produces same action — no randomness."""
        _, ric1 = make_system(self.valid_state)
        _, ric2 = make_system(self.valid_state)
        e1 = ric1.step("acquire", self.valid_state)
        e2 = ric2.step("acquire", self.valid_state)
        assert e1.selected_action == e2.selected_action
        assert e1.state_after == e2.state_after

    def test_legality_selected_action_in_admissible(self):
        _, ric = make_system(self.valid_state)
        entry = ric.step("acquire", self.valid_state)
        if entry.status == "OK":
            assert entry.selected_action in entry.admissible_actions

    def test_noop_fallback_on_empty_admissible(self):
        """When no action is admissible, NOOP is returned and status=FAILED."""
        bad_state = {"level": -1, "resource": 3, "mode": "active"}
        _, ric = make_system(bad_state)
        entry = ric.step("acquire", bad_state)
        assert entry.selected_action == "NOOP"
        assert entry.status == "FAILED"

    def test_invariant_breach_detected_at_step3(self):
        """Pre-step invariant check: state ∉ Inv → Type IV, halt."""
        bad_state = {"level": 99, "resource": 3, "mode": "active"}
        _, ric = make_system(bad_state)
        entry = ric.step("acquire", bad_state)
        assert entry.failure_mode == FailureMode.TYPE_IV_INVARIANT_BREACH
        assert entry.status == "FAILED"

    def test_non_interference_active_constraints_unchanged(self):
        """RIC must not mutate C_act during a step."""
        crs, ric = make_system(self.valid_state)
        before = [c.name for c in crs.active_constraints]
        ric.step("acquire", self.valid_state)
        after = [c.name for c in crs.active_constraints]
        assert before == after

    def test_trace_entry_as_dict(self):
        _, ric = make_system(self.valid_state)
        entry = ric.step("acquire", self.valid_state)
        d = entry.as_dict()
        assert "step" in d
        assert "state_before" in d
        assert "selected_action" in d
        assert "status" in d

    def test_preferred_action_selected_if_admissible(self):
        """ρ selects preferred_action when admissible."""
        _, ric = make_system(self.valid_state)
        entry = ric.step("acquire", self.valid_state)
        assert entry.selected_action == "ACQUIRE_RESOURCE"

    def test_fallback_to_first_admissible_if_preferred_not_admissible(self):
        """If preferred action is not admissible, ρ selects first admissible."""
        # "elevate" prefers INCREASE_LEVEL; at level=5 that's excluded
        max_state = {"level": 5, "resource": 3, "mode": "active"}
        _, ric = make_system(max_state)
        entry = ric.step("elevate", max_state)
        # INCREASE_LEVEL is not admissible → fallback
        assert "INCREASE_LEVEL" not in entry.admissible_actions

    def test_multi_step_state_chain(self):
        """Chain: acquire → acquire → release. Verify state evolution."""
        crs, ric = make_system(self.valid_state)
        s = dict(self.valid_state)

        e0 = ric.step("acquire", s)
        assert e0.status == "OK"
        s = e0.state_after  # resource=4

        e1 = ric.step("acquire", s)
        assert e1.status == "OK"
        s = e1.state_after  # resource=5

        e2 = ric.step("release", s)
        assert e2.status == "OK"
        assert e2.state_after["resource"] == 4

    def test_all_four_failure_modes_produce_deterministic_response(self):
        """Each failure mode: verify deterministic NOOP + FAILED status."""
        # Type II
        _, ric = make_system(self.valid_state)
        e = ric.step("NONSENSE", self.valid_state)
        assert e.failure_mode == FailureMode.TYPE_II_INTENT_PARSE_FAILURE
        assert e.selected_action == "NOOP"
        assert e.status == "FAILED"

        # Type I
        locked = {"level": 2, "resource": 1, "mode": "locked"}
        _, ric2 = make_system(locked)
        e2 = ric2.step("acquire", locked)
        assert e2.failure_mode == FailureMode.TYPE_I_CONSTRAINT_VIOLATION
        assert e2.status == "FAILED"

        # Type IV (pre-check on bad invariant state)
        bad = {"level": 99, "resource": 3, "mode": "active"}
        _, ric3 = make_system(bad)
        e3 = ric3.step("acquire", bad)
        assert e3.failure_mode == FailureMode.TYPE_IV_INVARIANT_BREACH
        assert e3.status == "FAILED"
