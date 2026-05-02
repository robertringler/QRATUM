"""Tests for the Minimal Viable Reality Injector (MVRI)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from qagents.mvri import (EPSILON, Action, H, Inv, State, acceptance_rate,
                          causal_structure, constraint_gate, delta_entropy,
                          delta_mutual_information, delta_transfer_entropy,
                          entropy_reduction, forward_model, inject,
                          internal_consistency, make_probe_policy, make_state,
                          probe_policy, reproducibility, run_mvri_loop,
                          structural_stability, validate_action,
                          violation_count)
from qagents.mvri.demo import (build_initial_state,
                               build_invalid_initial_state, case_a, case_b,
                               case_c, case_d, determinism_check)
from qagents.mvri.state import ForwardModelError, StateError

# --------------------------------------------------------------------------- #
# State
# --------------------------------------------------------------------------- #


def test_state_immutable_and_hashable():
    s = build_initial_state()
    assert hash(s) == hash(s)
    with pytest.raises((AttributeError, Exception)):
        s.step = 99  # type: ignore[misc]


def test_state_validation_rejects_unknown_edge():
    with pytest.raises((StateError, ValueError)):
        make_state(
            activations={"a": 0.1},
            edge_weights={("a", "z"): 0.2},
        )


def test_h_non_negative_and_zero_when_concentrated():
    s = make_state(
        activations={"a": 1.0, "b": 0.0, "c": 0.0},
        edge_weights={},
    )
    assert H(s) == 0.0
    s2 = make_state(
        activations={"a": 1.0, "b": 1.0, "c": 1.0},
        edge_weights={},
    )
    # Uniform across 3 ⇒ log2(3)
    assert math.isclose(H(s2), math.log2(3), rel_tol=1e-9)


def test_inv_initial_state():
    assert Inv(build_initial_state())


def test_inv_fails_when_edges_not_subset_of_initial():
    bad = build_invalid_initial_state()
    assert not Inv(bad)


# --------------------------------------------------------------------------- #
# Action validation
# --------------------------------------------------------------------------- #


def test_validate_action_within_epsilon():
    s = build_initial_state()
    a = Action("node_activation_shift", "n1", 0.04)
    assert validate_action(a, s, EPSILON)


def test_validate_action_rejects_excess_magnitude():
    s = build_initial_state()
    a = Action("node_activation_shift", "n1", 0.5)
    assert not validate_action(a, s, EPSILON)


def test_validate_action_rejects_unknown_target():
    s = build_initial_state()
    assert not validate_action(
        Action("node_activation_shift", "ghost", 0.01), s
    )
    assert not validate_action(
        Action("edge_weight_adjust", "n1->ghost", 0.01), s
    )


def test_validate_action_requires_seed_for_noise():
    s = build_initial_state()
    assert not validate_action(
        Action("noise_injection", "n1", 0.01, seed=None), s
    )
    assert validate_action(
        Action("noise_injection", "n1", 0.01, seed=7), s
    )


def test_validate_action_rejects_nan_inf():
    s = build_initial_state()
    assert not validate_action(
        Action("node_activation_shift", "n1", float("nan")), s
    )
    assert not validate_action(
        Action("node_activation_shift", "n1", float("inf")), s
    )


def test_validate_action_rejects_unknown_type():
    s = build_initial_state()
    assert not validate_action(
        Action("not_a_type", "n1", 0.01), s  # type: ignore[arg-type]
    )


# --------------------------------------------------------------------------- #
# Forward model
# --------------------------------------------------------------------------- #


def test_forward_model_pure_no_input_mutation():
    s = build_initial_state()
    snapshot = (s.activations, s.edge_weights, s.edges)
    a = Action("node_activation_shift", "n1", -0.01)
    _ = forward_model(s, a)
    assert (s.activations, s.edge_weights, s.edges) == snapshot


def test_forward_model_deterministic_on_noise():
    s = build_initial_state()
    a = Action("noise_injection", "n2", 0.02, seed=42)
    s1 = forward_model(s, a)
    s2 = forward_model(s, a)
    assert s1 == s2


def test_forward_model_edge_adjust():
    s = build_initial_state()
    a = Action("edge_weight_adjust", "n1->n2", 0.01)
    s1 = forward_model(s, a)
    assert math.isclose(s1.weight(("n1", "n2")), 0.41, rel_tol=1e-12)


def test_forward_model_clips_to_bounds():
    s = build_initial_state()
    # Push n1 (=0.6) up; ε bound separately is enforced by validate_action,
    # but F itself must clip if asked.
    a = Action("node_activation_shift", "n1", 0.5)
    s1 = forward_model(s, a)
    assert s1.activation("n1") <= s.bounds[1]


def test_forward_model_unknown_action_raises():
    s = build_initial_state()
    with pytest.raises(ForwardModelError):
        forward_model(s, Action("bogus", "n1", 0.0))  # type: ignore[arg-type]


def test_forward_model_noise_without_seed_raises():
    s = build_initial_state()
    a = Action("noise_injection", "n1", 0.01, seed=None)
    with pytest.raises(ForwardModelError):
        forward_model(s, a)


# --------------------------------------------------------------------------- #
# Constraints
# --------------------------------------------------------------------------- #


def test_individual_constraints_callable():
    s = build_initial_state()
    a = Action("node_activation_shift", "n2", -0.02)
    s_new = forward_model(s, a)
    assert entropy_reduction(s, s_new)
    assert causal_structure(s, s_new)
    assert structural_stability(s_new)
    assert internal_consistency(s_new)
    assert reproducibility(s, a)


def test_constraint_gate_passes_clean():
    s = build_initial_state()
    a = Action("node_activation_shift", "n2", -0.02)
    s_new = forward_model(s, a)
    ok, failed = constraint_gate(s, s_new, a)
    assert ok
    assert failed == []


def test_constraint_gate_collects_er_failure():
    s = build_initial_state()
    a = Action("node_activation_shift", "n3", 0.05)  # raises entropy
    s_new = forward_model(s, a)
    ok, failed = constraint_gate(s, s_new, a)
    assert not ok
    assert "ER" in failed


# --------------------------------------------------------------------------- #
# Injector
# --------------------------------------------------------------------------- #


def test_inject_accept_path_returns_new_state():
    s = build_initial_state()
    a = Action("node_activation_shift", "n2", -0.02)
    s_new, status = inject(s, a)
    assert status.outcome == "accepted"
    assert s_new != s


def test_inject_invalid_action_keeps_state():
    s = build_initial_state()
    a = Action("node_activation_shift", "n1", 1.0)
    s_new, status = inject(s, a)
    assert status.outcome == "rejected"
    assert "INVALID_ACTION" in status.failed_constraints
    assert s_new is s


def test_inject_constraint_rejection_keeps_state():
    s = build_initial_state()
    a = Action("node_activation_shift", "n3", 0.05)
    s_new, status = inject(s, a)
    assert status.outcome == "rejected"
    assert "ER" in status.failed_constraints
    assert s_new is s


def test_inject_pure_under_repeat():
    s = build_initial_state()
    a = Action("noise_injection", "n2", 0.03, seed=99)
    s1, st1 = inject(s, a)
    s2, st2 = inject(s, a)
    assert s1 == s2
    assert st1 == st2


# --------------------------------------------------------------------------- #
# Loop
# --------------------------------------------------------------------------- #


def test_loop_runs_and_traces():
    rng = np.random.default_rng(1)
    policy = make_probe_policy(rng)
    result = run_mvri_loop(build_initial_state(), policy, n_steps=10)
    assert result.n_steps == 10
    for entry in result.trace:
        assert entry.outcome in {
            "accepted",
            "rejected",
            "invalid_action",
            "invariant_halt",
        }


def test_loop_invariant_halt_on_bad_initial_state():
    rng = np.random.default_rng(0)
    policy = make_probe_policy(rng)
    result = run_mvri_loop(build_invalid_initial_state(), policy, n_steps=5)
    assert result.halted
    assert result.n_steps == 1
    assert result.trace[0].outcome == "invariant_halt"


def test_loop_does_not_couple_to_probe():
    """Custom policy independent of probe_policy."""

    def constant_policy(s: State) -> Action:
        return Action("node_activation_shift", "n2", -0.01)

    result = run_mvri_loop(build_initial_state(), constant_policy, n_steps=3)
    assert result.n_steps == 3
    # All three identical, deterministic shifts must accept.
    assert all(e.outcome == "accepted" for e in result.trace)


def test_loop_rejection_leaves_state_unchanged():
    def bad_policy(s: State) -> Action:
        # Will fail ER for the chosen state.
        return Action("node_activation_shift", "n3", 0.05)

    result = run_mvri_loop(build_initial_state(), bad_policy, n_steps=4)
    assert all(e.outcome == "rejected" for e in result.trace)
    assert all(e.state_t1 == e.state_t for e in result.trace)
    assert result.final_state == result.initial_state


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #


def test_metrics_pure_and_signs():
    s = build_initial_state()
    a = Action("node_activation_shift", "n2", -0.02)
    s_new = forward_model(s, a)
    assert delta_entropy(s, s_new) <= 0.0
    assert delta_mutual_information(s, s_new) >= 0.0
    # No edge change ⇒ ΔTE = 0
    assert delta_transfer_entropy(s, s_new) == 0.0


def test_acceptance_rate_and_violations():
    rng = np.random.default_rng(7)
    policy = make_probe_policy(rng)
    result = run_mvri_loop(build_initial_state(), policy, n_steps=15)
    rate = acceptance_rate(result.trace)
    counts = violation_count(result.trace)
    assert 0.0 <= rate <= 1.0
    # Sum of constraint violations + accepted == n_steps for cases where
    # rejection lists exactly one constraint per entry; in general just
    # check the dict is nonnegative.
    assert all(v >= 0 for v in counts.values())


# --------------------------------------------------------------------------- #
# Demo cases
# --------------------------------------------------------------------------- #


def test_case_a_accepted():
    e = case_a()
    assert e.outcome == "accepted"
    assert not e.constraints_failed


def test_case_b_invalid_action():
    e = case_b()
    assert e.outcome == "invalid_action"
    assert e.candidate_state is None  # never reached F
    assert e.state_t == e.state_t1


def test_case_c_er_rejection():
    e = case_c()
    assert e.outcome == "rejected"
    assert "ER" in e.constraints_failed
    assert e.state_t == e.state_t1


def test_case_d_invariant_halt():
    e = case_d()
    assert e.outcome == "invariant_halt"
    assert e.state_t == e.state_t1


def test_determinism_check():
    assert determinism_check()


# --------------------------------------------------------------------------- #
# Probe policy
# --------------------------------------------------------------------------- #


def test_probe_policy_requires_explicit_rng():
    s = build_initial_state()
    with pytest.raises(TypeError):
        probe_policy(s, None)  # type: ignore[arg-type]


def test_probe_policy_within_bounds_and_targets():
    s = build_initial_state()
    rng = np.random.default_rng(3)
    for _ in range(50):
        a = probe_policy(s, rng)
        assert abs(a.magnitude) <= EPSILON
        if a.type == "edge_weight_adjust":
            from qagents.mvri.state import parse_edge

            assert parse_edge(a.target) in s.edges
        else:
            assert a.target in s.nodes
        if a.type == "noise_injection":
            assert a.seed is not None


def test_probe_policy_deterministic_under_same_seed():
    s = build_initial_state()
    rng1 = np.random.default_rng(123)
    rng2 = np.random.default_rng(123)
    seq1 = [probe_policy(s, rng1) for _ in range(10)]
    seq2 = [probe_policy(s, rng2) for _ in range(10)]
    assert seq1 == seq2
