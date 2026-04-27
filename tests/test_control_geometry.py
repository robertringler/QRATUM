"""Tests for the controllability layer (qagents/control_geometry/)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from qagents.control_geometry import (
    DEFAULT_EPS,
    DEFAULT_WEIGHTS,
    ActionMetrics,
    DisplacementCache,
    ReachabilityGraph,
    action_norm,
    action_similarity,
    action_to_displacement,
    classify_action_space,
    compute_action_metrics,
    control_policy,
    controllability_rank,
    controllability_spectrum,
    displacement_matrix,
    local_reachability,
    reachability_cost,
    reachability_graph,
    score_action,
    select_action,
    sensitivity_probe,
    state_to_vector,
    vector_dim,
)
from qagents.mvri.action_space import EPSILON, Action
from qagents.mvri.demo import build_initial_state

# --------------------------------------------------------------------------- #
# Sensitivity
# --------------------------------------------------------------------------- #


def _node_index(s, node: str) -> int:
    nodes = sorted(n for n, _ in s.activations)
    return nodes.index(node)


def test_state_to_vector_canonical_order_and_dim():
    s = build_initial_state()
    v = state_to_vector(s)
    assert v.shape == (vector_dim(s),)
    # n1=0.6 sorted first.
    assert v[_node_index(s, "n1")] == pytest.approx(0.6)


def test_sensitivity_probe_deterministic_under_repeat():
    s = build_initial_state()
    a = Action("node_activation_shift", "n2", -0.02)
    d1 = sensitivity_probe(s, a)
    d2 = sensitivity_probe(s, a)
    assert np.array_equal(d1, d2)


def test_sensitivity_probe_directional_node_shift():
    s = build_initial_state()
    a = Action("node_activation_shift", "n2", -0.02)
    d = sensitivity_probe(s, a)
    # Only n2 component should be non-zero, and roughly equal to magnitude.
    n2_idx = _node_index(s, "n2")
    assert d[n2_idx] == pytest.approx(-0.02, rel=1e-6)
    # Other components zero
    other = np.delete(d, n2_idx)
    assert np.allclose(other, 0.0)


def test_sensitivity_probe_rejects_nonpositive_eps():
    s = build_initial_state()
    a = Action("node_activation_shift", "n2", -0.02)
    with pytest.raises(ValueError):
        sensitivity_probe(s, a, eps=0.0)


def test_sensitivity_probe_noise_action_is_deterministic():
    s = build_initial_state()
    a = Action("noise_injection", "n1", 0.01, seed=7)
    d1 = sensitivity_probe(s, a)
    d2 = sensitivity_probe(s, a)
    assert np.array_equal(d1, d2)


# --------------------------------------------------------------------------- #
# Embedding / cache
# --------------------------------------------------------------------------- #


def test_displacement_cache_returns_same_value():
    s = build_initial_state()
    a = Action("node_activation_shift", "n2", -0.02)
    cache = DisplacementCache()
    d1 = action_to_displacement(a, s, cache=cache)
    d2 = action_to_displacement(a, s, cache=cache)
    assert np.array_equal(d1, d2)
    assert len(cache) == 1


def test_displacement_cache_isolation_no_external_mutation():
    """Mutating the returned vector must not affect the cache."""
    s = build_initial_state()
    a = Action("node_activation_shift", "n2", -0.02)
    cache = DisplacementCache()
    d = action_to_displacement(a, s, cache=cache)
    d[0] = 999.0
    d2 = action_to_displacement(a, s, cache=cache)
    assert d2[0] != 999.0


def test_action_similarity_self_is_one_and_zero_for_orthogonal():
    s = build_initial_state()
    a1 = Action("node_activation_shift", "n2", -0.02)
    a2 = Action("edge_weight_adjust", "n1->n2", 0.01)
    assert action_similarity(a1, a1, s) == pytest.approx(1.0)
    # Different coordinate axes ⇒ orthogonal
    assert abs(action_similarity(a1, a2, s)) < 1e-9


def test_action_similarity_zero_when_either_displacement_is_zero():
    s = build_initial_state()
    # Action with zero magnitude → zero displacement.
    a_zero = Action("node_activation_shift", "n1", 0.0)
    a = Action("node_activation_shift", "n2", -0.02)
    assert action_similarity(a_zero, a, s) == 0.0


# --------------------------------------------------------------------------- #
# Reachability
# --------------------------------------------------------------------------- #


def test_local_reachability_filters_oversized_actions():
    s = build_initial_state()
    actions = [
        Action("node_activation_shift", "n2", -0.02),  # ok
        Action("node_activation_shift", "n2", 1.0),    # oversize
    ]
    reach = local_reachability(s, actions)
    assert len(reach) == 1
    assert s not in reach  # transition was non-trivial


def test_local_reachability_constraint_gated_by_default():
    s = build_initial_state()
    # raises entropy → ER violation
    bad = Action("node_activation_shift", "n3", 0.05)
    reach_gated = local_reachability(s, [bad])
    reach_raw = local_reachability(s, [bad], constraint_gated=False)
    assert reach_gated == set()
    assert len(reach_raw) == 1


def test_reachability_graph_is_reproducible():
    s = build_initial_state()
    actions = [
        Action("node_activation_shift", "n2", -0.02),
        Action("edge_weight_adjust", "n1->n2", 0.01),
        Action("node_activation_shift", "n3", 0.05),  # rejected
    ]
    g1 = reachability_graph([s], [(s, a) for a in actions])
    g2 = reachability_graph([s], [(s, a) for a in actions])
    assert isinstance(g1, ReachabilityGraph)
    assert len(g1) == len(g2)
    # Two pass the gate, one is rejected.
    assert len(g1) == 2
    # Graph reproducibility: same nodes and same edge weights.
    assert g1.nodes == g2.nodes
    assert sorted(e.weight for e in g1.edges) == sorted(
        e.weight for e in g2.edges
    )


def test_reachability_cost_minimum_over_actions():
    s = build_initial_state()
    target = local_reachability(
        s, [Action("node_activation_shift", "n2", -0.02)]
    ).pop()
    actions = [
        Action("node_activation_shift", "n2", -0.04),  # different state
        Action("node_activation_shift", "n2", -0.02),  # exact reach
    ]
    cost = reachability_cost(s, target, actions)
    assert cost == pytest.approx(0.02)


def test_reachability_cost_unreachable_returns_inf():
    s = build_initial_state()
    fake_target = build_initial_state()  # equal to s, so trivially reachable
    # Pass a target with no producing action → +inf.
    bogus = local_reachability(
        s, [Action("node_activation_shift", "n2", -0.02)]
    ).pop()
    # Action set that does NOT produce `bogus`.
    cost = reachability_cost(
        s, bogus, [Action("node_activation_shift", "n1", -0.02)]
    )
    assert math.isinf(cost)
    assert state_to_vector(fake_target) is not None  # sanity


def test_action_norm_matches_magnitude_abs():
    a = Action("node_activation_shift", "n2", -0.03)
    assert action_norm(a) == pytest.approx(0.03)


# --------------------------------------------------------------------------- #
# Controllability rank
# --------------------------------------------------------------------------- #


def test_controllability_rank_independent_of_redundant_copies():
    s = build_initial_state()
    a = Action("node_activation_shift", "n2", -0.02)
    # Five identical actions span only 1 direction.
    assert controllability_rank(s, [a] * 5) == 1


def test_controllability_rank_increases_with_independent_directions():
    s = build_initial_state()
    actions = [
        Action("node_activation_shift", "n1", -0.02),
        Action("node_activation_shift", "n2", -0.02),
        Action("node_activation_shift", "n3", -0.02),
    ]
    assert controllability_rank(s, actions) == 3


def test_controllability_rank_consistent_across_runs():
    s = build_initial_state()
    actions = [
        Action("node_activation_shift", "n1", -0.02),
        Action("node_activation_shift", "n2", -0.02),
        Action("edge_weight_adjust", "n1->n2", 0.01),
    ]
    r1 = controllability_rank(s, actions)
    r2 = controllability_rank(s, actions)
    spec1 = controllability_spectrum(s, actions)
    spec2 = controllability_spectrum(s, actions)
    assert r1 == r2
    assert np.array_equal(spec1, spec2)


def test_displacement_matrix_empty_returns_zero_columns():
    s = build_initial_state()
    m = displacement_matrix(s, [])
    assert m.shape == (vector_dim(s), 0)
    assert controllability_rank(s, []) == 0


# --------------------------------------------------------------------------- #
# Control directions / scoring / classification
# --------------------------------------------------------------------------- #


def test_compute_action_metrics_safe_action():
    s = build_initial_state()
    a = Action("node_activation_shift", "n2", -0.02)
    m = compute_action_metrics(s, a)
    assert isinstance(m, ActionMetrics)
    assert m.constraints_passed
    assert m.stable
    assert m.delta_h <= 0.0


def test_compute_action_metrics_unsafe_action():
    s = build_initial_state()
    bad = Action("node_activation_shift", "n3", 0.05)  # ER fail
    m = compute_action_metrics(s, bad)
    assert not m.constraints_passed


def test_score_action_higher_for_safe_than_unsafe():
    s = build_initial_state()
    safe = Action("node_activation_shift", "n2", -0.02)
    unsafe = Action("node_activation_shift", "n3", 0.05)
    ms = compute_action_metrics(s, safe)
    mu = compute_action_metrics(s, unsafe)
    score_s = score_action(safe, ms.delta_s, ms)
    score_u = score_action(unsafe, mu.delta_s, mu)
    assert score_s > score_u


def test_score_action_accepts_mapping_metrics():
    s = build_initial_state()
    a = Action("node_activation_shift", "n2", -0.02)
    m = compute_action_metrics(s, a)
    via_ds = score_action(a, m.delta_s, m)
    via_map = score_action(
        a,
        m.delta_s,
        {
            "delta_h": m.delta_h,
            "constraints_passed": m.constraints_passed,
            "stable": m.stable,
        },
    )
    assert via_ds == pytest.approx(via_map)


def test_classify_action_space_categories():
    s = build_initial_state()
    actions = [
        Action("node_activation_shift", "n2", -0.02),       # productive
        Action("node_activation_shift", "n2", -0.02),       # degenerate (dup)
        Action("node_activation_shift", "n3", 0.05),        # unsafe (ER)
        Action("node_activation_shift", "n1", 0.0),         # neutral
    ]
    out = classify_action_space(s, actions)
    assert set(out.keys()) == {"productive", "neutral", "degenerate", "unsafe"}
    assert len(out["productive"]) == 1
    assert len(out["degenerate"]) == 1
    assert len(out["unsafe"]) == 1
    assert len(out["neutral"]) == 1


def test_classify_action_space_deterministic():
    s = build_initial_state()
    actions = [
        Action("node_activation_shift", "n2", -0.02),
        Action("edge_weight_adjust", "n1->n2", 0.01),
        Action("node_activation_shift", "n3", 0.05),
    ]
    a = classify_action_space(s, actions)
    b = classify_action_space(s, actions)
    assert {k: list(v) for k, v in a.items()} == {
        k: list(v) for k, v in b.items()
    }


def test_default_weights_keys():
    assert {"magnitude", "entropy_reduction", "constraint_compliance",
            "stability"} <= set(DEFAULT_WEIGHTS.keys())


# --------------------------------------------------------------------------- #
# Policy optimizer
# --------------------------------------------------------------------------- #


def test_select_action_picks_aligned_direction():
    s = build_initial_state()
    obj = np.zeros(vector_dim(s))
    obj[_node_index(s, "n2")] = -1.0  # want n2 to decrease
    candidates = [
        Action("node_activation_shift", "n1", -0.02),
        Action("node_activation_shift", "n2", -0.02),  # aligned
        Action("node_activation_shift", "n2", 0.02),   # anti-aligned (also fails ER)
    ]
    chosen = select_action(s, obj, candidates)
    assert chosen is not None
    assert chosen.target == "n2"
    assert chosen.magnitude < 0


def test_select_action_rejects_oversize_actions():
    s = build_initial_state()
    obj = np.zeros(vector_dim(s))
    obj[_node_index(s, "n2")] = -1.0
    big = Action("node_activation_shift", "n2", -0.5)
    small = Action("node_activation_shift", "n2", -0.02)
    chosen = select_action(s, obj, [big, small])
    assert chosen is small


def test_select_action_returns_none_when_all_infeasible():
    s = build_initial_state()
    obj = np.zeros(vector_dim(s))
    obj[_node_index(s, "n3")] = 1.0
    bad = Action("node_activation_shift", "n3", 0.05)
    assert select_action(s, obj, [bad]) is None


def test_select_action_objective_shape_validation():
    s = build_initial_state()
    with pytest.raises(ValueError):
        select_action(s, np.zeros(vector_dim(s) + 1), [])


def test_select_action_deterministic_tie_break():
    s = build_initial_state()
    obj = np.zeros(vector_dim(s))  # all candidates score equally (0)
    a1 = Action("node_activation_shift", "n1", -0.02)
    a2 = Action("node_activation_shift", "n2", -0.02)
    pick_ab = select_action(s, obj, [a1, a2])
    pick_ba = select_action(s, obj, [a2, a1])
    assert pick_ab == pick_ba


def test_control_policy_picks_feasible_high_score():
    s = build_initial_state()
    candidates = [
        Action("node_activation_shift", "n2", -0.02),  # safe
        Action("node_activation_shift", "n3", 0.05),   # unsafe
    ]
    chosen = control_policy(s, candidates)
    assert chosen is not None
    assert chosen.target == "n2"


def test_control_policy_returns_none_when_no_feasible():
    s = build_initial_state()
    bad = Action("node_activation_shift", "n3", 0.05)
    assert control_policy(s, [bad]) is None


def test_policy_does_not_bypass_constraint_gate():
    """Policies must never select an action that fails Φ when require_feasible=True."""
    s = build_initial_state()
    candidates = [
        Action("node_activation_shift", "n3", 0.05),   # ER fail
        Action("edge_weight_adjust", "ghost->ghost", 0.01),  # fails action validation
    ]
    # control_policy uses require_feasible=True by default
    assert control_policy(s, candidates) is None
    obj = np.zeros(vector_dim(s))
    assert select_action(s, obj, candidates) is None


def test_default_eps_is_small_positive():
    assert DEFAULT_EPS > 0.0
    assert DEFAULT_EPS < 1.0
    assert EPSILON > 0.0
