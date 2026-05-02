"""Controllability layer for the MVRI system.

This package transforms MVRI from a passive perturbation engine into a
measurable control system over a structured state space.

Modules
-------
sensitivity
    Local action → state displacement approximation.
embedding
    Action → ΔS embedding with explicit (non-global) caching.
reachability
    Local R(S), reachability graphs, and minimum-cost paths.
controllability_rank
    SVD-based rank of the local control space.
control_directions
    Per-action scoring and productive/neutral/degenerate/unsafe
    classification.
policy_optimizer
    Deterministic objective-aligned and controllability-weighted
    policies over candidate action sets.

All public functions are deterministic, side-effect free unless
explicitly stated, and integrate with the existing MVRI ``State``,
``Action``, ``forward_model``, and ``constraint_gate``.
"""

from qagents.control_geometry.control_directions import (
    DEFAULT_WEIGHTS,
    ActionMetrics,
    classify_action_space,
    compute_action_metrics,
    score_action,
)
from qagents.control_geometry.controllability_rank import (
    controllability_rank,
    controllability_spectrum,
    displacement_matrix,
)
from qagents.control_geometry.embedding import (
    DisplacementCache,
    action_similarity,
    action_to_displacement,
)
from qagents.control_geometry.policy_optimizer import (
    control_policy,
    select_action,
)
from qagents.control_geometry.reachability import (
    ReachabilityGraph,
    ReachEdge,
    action_norm,
    local_reachability,
    reachability_cost,
    reachability_graph,
)
from qagents.control_geometry.sensitivity import (
    DEFAULT_EPS,
    sensitivity_probe,
    state_to_vector,
    vector_dim,
)

__all__ = [
    # sensitivity
    "DEFAULT_EPS",
    "sensitivity_probe",
    "state_to_vector",
    "vector_dim",
    # embedding
    "DisplacementCache",
    "action_similarity",
    "action_to_displacement",
    # reachability
    "ReachEdge",
    "ReachabilityGraph",
    "action_norm",
    "local_reachability",
    "reachability_cost",
    "reachability_graph",
    # controllability_rank
    "controllability_rank",
    "controllability_spectrum",
    "displacement_matrix",
    # control_directions
    "ActionMetrics",
    "DEFAULT_WEIGHTS",
    "classify_action_space",
    "compute_action_metrics",
    "score_action",
    # policy_optimizer
    "control_policy",
    "select_action",
]
