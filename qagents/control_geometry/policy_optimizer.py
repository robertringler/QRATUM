"""Deterministic policy optimizer over a candidate action set.

This module provides two entry points:

* :func:`select_action` — pick the action whose induced ΔS best aligns
  with a given ``objective_vector``, subject to constraint feasibility.

* :func:`control_policy` — a baseline policy that picks the highest
  *controllability-weighted* score from :mod:`control_directions`.

Both are pure given fixed inputs. Tie-breaking is deterministic on the
canonical (type, target, magnitude, seed) ordering of :class:`Action`,
so identical inputs always produce identical outputs.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from qagents.control_geometry.control_directions import (
    DEFAULT_WEIGHTS, compute_action_metrics, score_action)
from qagents.control_geometry.embedding import DisplacementCache
from qagents.control_geometry.sensitivity import DEFAULT_EPS, vector_dim
from qagents.mvri.action_space import EPSILON, Action, validate_action
from qagents.mvri.state import State

# --------------------------------------------------------------------------- #
# Tie-breaking
# --------------------------------------------------------------------------- #


def _action_key(a: Action) -> tuple:
    """Canonical sort key for stable tie-breaking."""

    return (a.type, a.target, float(a.magnitude), a.seed if a.seed is not None else -1)


# --------------------------------------------------------------------------- #
# Objective-aligned selection
# --------------------------------------------------------------------------- #


def select_action(
    state: State,
    objective_vector: np.ndarray | Sequence[float],
    candidates: Iterable[Action],
    *,
    eps: float = DEFAULT_EPS,
    cache: DisplacementCache | None = None,
    require_feasible: bool = True,
    zero_tol: float = 1e-12,
) -> Action | None:
    """Select the candidate action whose ΔS best aligns with the objective.

    Alignment is measured by the dot product ``<ΔS, objective>``. When
    ``require_feasible`` is ``True`` (default), unsafe actions
    (constraint gate rejects) are filtered out; the selector therefore
    cannot bypass the gate.

    Returns ``None`` if no feasible candidate is found.
    """

    obj = np.asarray(objective_vector, dtype=float)
    if obj.shape != (vector_dim(state),):
        raise ValueError(
            f"objective_vector shape {obj.shape!r} != state vector dim "
            f"{(vector_dim(state),)!r}"
        )

    best: Action | None = None
    best_score = -np.inf
    best_key: tuple | None = None

    for a in candidates:
        if require_feasible and not validate_action(a, state, EPSILON):
            continue
        m = compute_action_metrics(state, a, eps=eps, cache=cache)
        if require_feasible and not m.constraints_passed:
            continue
        if abs(a.magnitude) > EPSILON + 1e-15:
            continue
        s = float(np.dot(m.delta_s, obj))
        # numerical robustness: treat scores within zero_tol as ties
        key = _action_key(a)
        if (
            s > best_score + zero_tol
            or (abs(s - best_score) <= zero_tol and (best_key is None or key < best_key))
        ):
            best = a
            best_score = s
            best_key = key

    return best


# --------------------------------------------------------------------------- #
# Baseline controllability policy
# --------------------------------------------------------------------------- #


def control_policy(
    state: State,
    candidates: Iterable[Action],
    *,
    eps: float = DEFAULT_EPS,
    cache: DisplacementCache | None = None,
    weights: dict[str, float] | None = None,
    require_feasible: bool = True,
    zero_tol: float = 1e-12,
) -> Action | None:
    """Baseline policy: pick the action with the highest scalar score.

    The score is :func:`control_directions.score_action`; weights
    default to :data:`control_directions.DEFAULT_WEIGHTS`. When
    ``require_feasible`` is ``True`` (default) infeasible actions are
    filtered out. Ties (within ``zero_tol``) are broken by the
    canonical action key, matching :func:`select_action` for
    consistency.
    """

    used_weights = dict(DEFAULT_WEIGHTS)
    if weights is not None:
        used_weights.update(weights)

    best: Action | None = None
    best_score = -np.inf
    best_key: tuple | None = None

    for a in candidates:
        if require_feasible and not validate_action(a, state, EPSILON):
            continue
        m = compute_action_metrics(state, a, eps=eps, cache=cache)
        if require_feasible and not m.constraints_passed:
            continue
        if abs(a.magnitude) > EPSILON + 1e-15:
            continue
        s = score_action(a, m.delta_s, m, weights=used_weights)
        key = _action_key(a)
        if (
            s > best_score + zero_tol
            or (
                abs(s - best_score) <= zero_tol
                and (best_key is None or key < best_key)
            )
        ):
            best = a
            best_score = s
            best_key = key

    return best


__all__ = [
    "control_policy",
    "select_action",
]
