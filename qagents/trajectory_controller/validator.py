"""Trajectory validation by delegation to the real-world safety gate."""

from __future__ import annotations

from qagents.mvri.forward_model import forward_model
from qagents.mvri.state import Inv, State
from qagents.realworld_bridge.safety_gate import SafetyConfig, validate_action
from qagents.trajectory_controller.predictor import (
    TrajectoryInvariantViolation,
    predict_rollout,
)
from qagents.trajectory_controller.types import Trajectory, vector_to_action


def validate_trajectory(
    trajectory: Trajectory,
    initial_state: State,
    safety_config: SafetyConfig,
    *,
    previous_action=None,
) -> tuple[bool, tuple[str, ...]]:
    """Validate a complete trajectory under MAG/TGT/ROC/MVR and Inv checks.

    Every per-step action is decoded to the existing MVRI ``Action`` type and
    checked with :func:`realworld_bridge.safety_gate.validate_action`. Cross-step
    ROC is enforced by passing the previous step's action into the safety gate.
    Any failure invalidates the entire trajectory; all detected reasons are
    returned with deterministic ``step:{i}:CHECK`` labels.
    """

    reasons: list[str] = []
    current = initial_state
    prev = previous_action

    if not Inv(initial_state):
        reasons.append("initial_state:INV")
        return False, tuple(reasons)

    for index, vector in enumerate(trajectory.actions):
        try:
            action = vector_to_action(vector, current)
        except ValueError as exc:
            reasons.append(f"step:{index}:DECODE:{exc}")
            break

        result = validate_action(action, current, safety_config, prev)
        for failed in result.failed_checks:
            reasons.append(f"step:{index}:{failed}")
        if not result.valid:
            prev = action
            continue

        try:
            candidate = forward_model(current, action)
        except Exception as exc:  # noqa: BLE001 - explicit failure surface
            reasons.append(f"step:{index}:MVR:{exc}")
            prev = action
            continue
        if not Inv(candidate):
            reasons.append(f"step:{index}:INV")
        else:
            current = candidate
        prev = action

    try:
        predict_rollout(initial_state, trajectory)
    except TrajectoryInvariantViolation as exc:
        reasons.append(f"rollout:INV:{exc}")

    return not reasons, tuple(reasons)


__all__ = ["validate_trajectory"]
