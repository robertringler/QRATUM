"""Safety gate for the real-world bridge.

Implements four named, individually-callable checks:

    MAG  — magnitude bound
    TGT  — target exists in state
    ROC  — rate of change vs previous action
    MVR  — MVRI joint constraint gate (optional)

The joint :func:`validate_action` evaluates all four in the canonical
order ``MAG → TGT → ROC → MVR`` and **collects** every failed check,
never short-circuiting. Failed actions are *rejected* — never modified
or clipped.
"""

from __future__ import annotations

from dataclasses import dataclass

from qagents.mvri.action_space import EPSILON, Action
from qagents.mvri.constraint_gate import constraint_gate
from qagents.mvri.forward_model import forward_model
from qagents.mvri.state import State
from qagents.realworld_bridge.types import ActionValidationResult

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SafetyConfig:
    """Tunables for :func:`validate_action`."""

    epsilon: float = EPSILON
    max_rate_of_change: float = float("inf")
    mvri_constraint_check: bool = True


# --------------------------------------------------------------------------- #
# Action norm (must agree with control_geometry / mvri conventions)
# --------------------------------------------------------------------------- #


def _action_norm(a: Action) -> float:
    return abs(float(a.magnitude))


def _action_distance(a1: Action, a2: Action) -> float:
    """L1 distance between two actions in magnitude space.

    Two actions of *different* type or *different* target are deemed
    maximally distant in this simple norm: their magnitude difference is
    used, plus an additive penalty equal to ``2 * EPSILON`` so a swap
    can be flagged by rate-of-change limits if desired. This keeps the
    norm scalar and deterministic.
    """

    base = abs(float(a1.magnitude) - float(a2.magnitude))
    if a1.type != a2.type or a1.target != a2.target:
        return base + 2.0 * EPSILON
    return base


# --------------------------------------------------------------------------- #
# Standalone checks
# --------------------------------------------------------------------------- #


def check_mag(action: Action, config: SafetyConfig) -> bool:
    """MAG: ``|action.magnitude| ≤ config.epsilon``."""

    try:
        m = float(action.magnitude)
    except (TypeError, ValueError):
        return False
    if m != m:  # NaN
        return False
    return abs(m) <= float(config.epsilon)


def check_tgt(action: Action, state: State) -> bool:
    """TGT: ``action.target ∈ state.nodes ∪ state.edges``."""

    try:
        return state.has_target(action.target)
    except ValueError:
        return False


def check_roc(
    action: Action,
    previous_action: Action | None,
    config: SafetyConfig,
) -> bool:
    """ROC: ``‖action − previous_action‖ ≤ max_rate_of_change``.

    If no previous action exists, the check is trivially satisfied.
    """

    if previous_action is None:
        return True
    return _action_distance(action, previous_action) <= float(config.max_rate_of_change)


def check_mvr(
    action: Action,
    state: State,
    config: SafetyConfig,
) -> bool:
    """MVR: the MVRI constraint gate accepts ``F(state, action)``.

    Skipped (returns ``True``) when ``config.mvri_constraint_check`` is
    ``False``. If the forward model itself raises (e.g. for an action
    whose target/type is malformed), the check fails closed.
    """

    if not config.mvri_constraint_check:
        return True
    try:
        candidate = forward_model(state, action)
    except Exception:  # noqa: BLE001 - fail closed
        return False
    passed, _failed = constraint_gate(state, candidate, action)
    return bool(passed)


# --------------------------------------------------------------------------- #
# Joint validator
# --------------------------------------------------------------------------- #


def validate_action(
    action: Action,
    state: State,
    config: SafetyConfig,
    previous_action: Action | None,
) -> ActionValidationResult:
    """Run MAG, TGT, ROC, MVR in order; collect every failure."""

    failed: list[str] = []

    mag_ok = check_mag(action, config)
    if not mag_ok:
        failed.append("MAG")

    tgt_ok = check_tgt(action, state)
    if not tgt_ok:
        failed.append("TGT")

    roc_ok = check_roc(action, previous_action, config)
    if not roc_ok:
        failed.append("ROC")

    # Only run MVR when prior structural checks pass — running F on a
    # malformed action can raise; ``check_mvr`` already handles that
    # but the gate semantics are clearer when MVR runs against a
    # well-typed action. We still record an MVR failure if it would
    # have failed (fail-closed) regardless of upstream checks.
    if mag_ok and tgt_ok:
        mvr_ok = check_mvr(action, state, config)
    else:
        mvr_ok = not config.mvri_constraint_check
    if not mvr_ok:
        failed.append("MVR")

    return ActionValidationResult(
        valid=not failed,
        action=action,
        failed_checks=tuple(failed),
        mvri_gate_passed=mvr_ok,
        magnitude_passed=mag_ok,
        rate_limit_passed=roc_ok,
    )


__all__ = [
    "SafetyConfig",
    "check_mag",
    "check_mvr",
    "check_roc",
    "check_tgt",
    "validate_action",
]
