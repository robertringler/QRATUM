"""Control directions: scoring and classification of actions.

This layer turns *displacement geometry* into actionable signals:

* ``score_action`` produces a single scalar per action that combines
  ΔS magnitude, entropy contribution, constraint-pass bonus, and
  stability penalty.

* ``classify_action_space`` partitions a candidate set into the four
  semantic categories used by the policy optimizer:

  - ``productive``  : nonzero ΔS, constraints pass.
  - ``neutral``     : near-zero ΔS, constraints pass.
  - ``degenerate``  : ΔS lies in (or near) the existing column span of
    other actions in the set — redundant directions.
  - ``unsafe``      : constraint gate rejects the transition.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np

from qagents.control_geometry.embedding import (DisplacementCache,
                                                action_to_displacement)
from qagents.control_geometry.sensitivity import DEFAULT_EPS
from qagents.mvri.action_space import Action
from qagents.mvri.constraint_gate import constraint_gate
from qagents.mvri.forward_model import forward_model
from qagents.mvri.metrics import delta_entropy
from qagents.mvri.state import State, is_stable

# --------------------------------------------------------------------------- #
# Default scoring weights — exposed as module constants so tests and
# downstream callers can override them deterministically.
# --------------------------------------------------------------------------- #

#: Numerical floor for relative residuals to avoid divide-by-zero on
#: vanishingly small displacement vectors. Far below any plausible
#: real ``||ΔS||``; used only as a denominator guard.
_NORM_FLOOR: float = 1e-30

DEFAULT_WEIGHTS: dict[str, float] = {
    "magnitude": 1.0,
    "entropy_reduction": 1.0,
    "constraint_compliance": 1.0,
    "stability": 0.5,
}


@dataclass(frozen=True)
class ActionMetrics:
    """Per-action metrics used by ``score_action``.

    Pure dataclass; no mutation. Constructed once per action by
    ``compute_action_metrics`` and consumed by scoring/classification.
    """

    delta_s: np.ndarray
    delta_h: float            # ΔH = H(s_new) - H(s_prev)
    constraints_passed: bool
    stable: bool


def compute_action_metrics(
    state: State,
    action: Action,
    *,
    eps: float = DEFAULT_EPS,
    cache: DisplacementCache | None = None,
) -> ActionMetrics:
    """Run the deterministic forward model + gate once and bundle results.

    The displacement vector is computed at probe scale ``eps`` for
    direction; the entropy / constraint / stability checks are
    evaluated at the *full* candidate state so they reflect the action
    that would actually be executed.
    """

    delta_s = action_to_displacement(action, state, eps=eps, cache=cache)
    full_candidate = forward_model(state, action)
    passed, _ = constraint_gate(state, full_candidate, action)
    return ActionMetrics(
        delta_s=delta_s,
        delta_h=delta_entropy(state, full_candidate),
        constraints_passed=passed,
        stable=is_stable(full_candidate),
    )


def score_action(
    action: Action,
    delta_s: np.ndarray,
    metrics: Mapping[str, float | bool] | ActionMetrics,
    *,
    weights: Mapping[str, float] | None = None,
) -> float:
    """Score an action.

    The scalar combines four normalised contributions:

    * ``+ w_mag · ||ΔS||``           — bigger moves are more informative
    * ``- w_ent · ΔH``               — favour entropy reduction
    * ``+ w_cmp · constraint_pass``  — bonus for feasibility
    * ``- w_sta · (1 - stable)``     — penalty when candidate is unstable

    ``metrics`` may either be an :class:`ActionMetrics` instance or a
    mapping with the keys ``delta_h`` / ``constraints_passed`` /
    ``stable``. ``weights`` defaults to :data:`DEFAULT_WEIGHTS`.
    """

    w = dict(DEFAULT_WEIGHTS)
    if weights is not None:
        w.update(weights)

    if isinstance(metrics, ActionMetrics):
        delta_h = metrics.delta_h
        passed = metrics.constraints_passed
        stable = metrics.stable
    else:
        delta_h = float(metrics.get("delta_h", 0.0))
        passed = bool(metrics.get("constraints_passed", False))
        stable = bool(metrics.get("stable", False))

    mag = float(np.linalg.norm(delta_s))
    score = (
        w["magnitude"] * mag
        - w["entropy_reduction"] * delta_h
        + w["constraint_compliance"] * (1.0 if passed else 0.0)
        - w["stability"] * (0.0 if stable else 1.0)
    )
    return float(score)


# --------------------------------------------------------------------------- #
# Classification
# --------------------------------------------------------------------------- #


def classify_action_space(
    state: State,
    actions: Iterable[Action],
    *,
    eps: float = DEFAULT_EPS,
    cache: DisplacementCache | None = None,
    zero_tol: float = 1e-9,
    redundancy_tol: float = 1e-6,
) -> dict[str, list[Action]]:
    """Partition ``actions`` into productive/neutral/degenerate/unsafe.

    Determinism: actions are classified in input order; degeneracy is
    detected against the running column span of *prior* productive
    actions, which makes the classification stable under repeated runs.

    The returned dict always contains all four keys (possibly empty).
    """

    out: dict[str, list[Action]] = {
        "productive": [],
        "neutral": [],
        "degenerate": [],
        "unsafe": [],
    }

    span_columns: list[np.ndarray] = []  # productive ΔS so far

    for a in actions:
        m = compute_action_metrics(state, a, eps=eps, cache=cache)
        if not m.constraints_passed:
            out["unsafe"].append(a)
            continue
        norm = float(np.linalg.norm(m.delta_s))
        if norm < zero_tol:
            out["neutral"].append(a)
            continue
        if span_columns:
            mat = np.stack(span_columns, axis=1)
            # least-squares projection residual
            coef, *_ = np.linalg.lstsq(mat, m.delta_s, rcond=None)
            residual = float(np.linalg.norm(mat @ coef - m.delta_s))
            if residual / max(norm, _NORM_FLOOR) < redundancy_tol:
                out["degenerate"].append(a)
                continue
        out["productive"].append(a)
        span_columns.append(m.delta_s)

    return out


__all__ = [
    "ActionMetrics",
    "DEFAULT_WEIGHTS",
    "classify_action_space",
    "compute_action_metrics",
    "score_action",
]
