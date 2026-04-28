"""Deterministic bounded finite-horizon trajectory planner."""

from __future__ import annotations

from dataclasses import dataclass
from math import inf

import numpy as np

from qagents.control_geometry.sensitivity import state_to_vector
from qagents.mvri.action_space import Action
from qagents.mvri.forward_model import forward_model
from qagents.mvri.state import Inv, State, format_edge
from qagents.realworld_bridge.safety_gate import SafetyConfig, validate_action
from qagents.trajectory_controller.predictor import (
    TrajectoryInvariantViolation,
    predict_rollout,
)
from qagents.trajectory_controller.types import (
    LAMBDA_ACTION,
    LAMBDA_ROC,
    LAMBDA_STATE,
    Trajectory,
    TrajectoryPlanResult,
    action_to_vector,
)

BEAM_WIDTH: int = 8
MAX_CANDIDATES_PER_STEP: int = 32
TARGET_TOL: float = 1e-9


@dataclass(frozen=True)
class _BeamNode:
    state: State
    actions: tuple[Action, ...]
    cost: float
    previous_action: Action | None


def _action_key(action: Action) -> tuple[str, str, float, int]:
    return (
        action.type,
        action.target,
        float(action.magnitude),
        action.seed if action.seed is not None else -1,
    )


def _state_distance_sq(state: State, target_state: State) -> float:
    diff = state_to_vector(state) - state_to_vector(target_state)
    return float(np.dot(diff, diff))


def _action_norm_sq(action: Action) -> float:
    return float(action.magnitude) * float(action.magnitude)


def _roc_norm_sq(action: Action, previous_action: Action | None) -> float:
    if previous_action is None:
        return 0.0
    delta = float(action.magnitude) - float(previous_action.magnitude)
    penalty = 0.0
    if action.type != previous_action.type or action.target != previous_action.target:
        penalty = 2.0 * float(abs(action.magnitude))
    return float(delta * delta + penalty * penalty)


def trajectory_cost(
    states: tuple[State, ...],
    actions: tuple[Action, ...],
    target_state: State,
    *,
    previous_action: Action | None = None,
) -> float:
    """Compute the explicit finite-horizon objective J(τ)."""

    total = 0.0
    prev = previous_action
    for state, action in zip(states, actions):
        total += LAMBDA_STATE * _state_distance_sq(state, target_state)
        total += LAMBDA_ACTION * _action_norm_sq(action)
        total += LAMBDA_ROC * _roc_norm_sq(action, prev)
        prev = action
    return float(total)


def _candidate_magnitudes(diff: float, epsilon: float) -> tuple[float, ...]:
    eps = float(epsilon)
    if eps < 0.0:
        return ()
    if abs(diff) <= TARGET_TOL:
        return (0.0,)
    step = max(-eps, min(eps, float(diff)))
    half = step / 2.0
    values = (step, half, 0.0, -half, -step)
    dedup = sorted({round(float(v), 12) for v in values}, key=lambda x: (abs(x - step), abs(x), x))
    return tuple(float(v) for v in dedup)


def _candidate_actions(
    state: State,
    target_state: State,
    config: SafetyConfig,
) -> tuple[Action, ...]:
    """Canonical deterministic grid around target-state error."""

    candidates: list[Action] = []
    acts = dict(state.activations)
    target_acts = dict(target_state.activations)
    for node in state.nodes:
        diff = float(target_acts[node]) - float(acts[node])
        for magnitude in _candidate_magnitudes(diff, config.epsilon):
            candidates.append(
                Action(
                    type="node_activation_shift",
                    target=node,
                    magnitude=magnitude,
                )
            )

    weights = dict(state.edge_weights)
    target_weights = dict(target_state.edge_weights)
    for edge in sorted(state.edges):
        diff = float(target_weights[edge]) - float(weights[edge])
        for magnitude in _candidate_magnitudes(diff, config.epsilon):
            candidates.append(
                Action(
                    type="edge_weight_adjust",
                    target=format_edge(edge),
                    magnitude=magnitude,
                )
            )

    if state.nodes:
        candidates.append(
            Action(
                type="node_activation_shift",
                target=state.nodes[0],
                magnitude=0.0,
            )
        )

    unique = {_action_key(action): action for action in candidates}
    ordered = tuple(unique[key] for key in sorted(unique))
    return ordered[:MAX_CANDIDATES_PER_STEP]


def _compatible_states(current_state: State, target_state: State, model_state: State) -> tuple[str, ...]:
    reasons: list[str] = []
    if not Inv(current_state):
        reasons.append("current_state:INV")
    if not Inv(target_state):
        reasons.append("target_state:INV")
    if not Inv(model_state):
        reasons.append("model_state:INV")
    if current_state.nodes != target_state.nodes or current_state.nodes != model_state.nodes:
        reasons.append("topology:nodes")
    if current_state.edges != target_state.edges or current_state.edges != model_state.edges:
        reasons.append("topology:edges")
    if current_state.initial_edges != model_state.initial_edges:
        reasons.append("topology:initial_edges")
    return tuple(reasons)


def plan_trajectory(
    current_state: State,
    target_state: State,
    horizon: int,
    safety_config: SafetyConfig,
    model_state: State,
) -> TrajectoryPlanResult:
    """Plan a deterministic bounded beam-search trajectory.

    The search is explicitly bounded by ``horizon * BEAM_WIDTH *
    MAX_CANDIDATES_PER_STEP`` candidate expansions.  Any invalid step is
    rejected rather than repaired.  If no complete valid trajectory exists,
    the returned result has ``valid=False`` and includes collected failure
    reasons.
    """

    if int(horizon) <= 0:
        return TrajectoryPlanResult(
            trajectory=None,
            cost=inf,
            valid=False,
            failure_reasons=("horizon:non_positive",),
        )

    reasons = list(_compatible_states(current_state, target_state, model_state))
    if reasons:
        return TrajectoryPlanResult(
            trajectory=None,
            cost=inf,
            valid=False,
            failure_reasons=tuple(reasons),
        )

    beam: tuple[_BeamNode, ...] = (
        _BeamNode(
            state=model_state,
            actions=(),
            cost=0.0,
            previous_action=None,
        ),
    )
    rejected: list[str] = []

    for depth in range(int(horizon)):
        expanded: list[_BeamNode] = []
        for node in beam:
            candidates = _candidate_actions(node.state, target_state, safety_config)
            for action in candidates:
                validation = validate_action(action, node.state, safety_config, node.previous_action)
                if not validation.valid:
                    for failed in validation.failed_checks:
                        rejected.append(f"step:{depth}:{failed}")
                    continue
                try:
                    next_state = forward_model(node.state, action)
                except Exception as exc:  # noqa: BLE001 - explicit failure surface
                    rejected.append(f"step:{depth}:MVR:{exc}")
                    continue
                if not Inv(next_state):
                    rejected.append(f"step:{depth}:INV")
                    continue
                next_cost = node.cost
                next_cost += LAMBDA_STATE * _state_distance_sq(next_state, target_state)
                next_cost += LAMBDA_ACTION * _action_norm_sq(action)
                next_cost += LAMBDA_ROC * _roc_norm_sq(action, node.previous_action)
                expanded.append(
                    _BeamNode(
                        state=next_state,
                        actions=(*node.actions, action),
                        cost=float(next_cost),
                        previous_action=action,
                    )
                )

        if not expanded:
            return TrajectoryPlanResult(
                trajectory=None,
                cost=inf,
                valid=False,
                failure_reasons=tuple(sorted(set(rejected))) or (f"step:{depth}:no_valid_action",),
            )

        expanded.sort(key=lambda item: (item.cost, tuple(_action_key(a) for a in item.actions)))
        beam = tuple(expanded[:BEAM_WIDTH])

    best = beam[0]
    trajectory = Trajectory(
        actions=tuple(action_to_vector(action, model_state) for action in best.actions),
        horizon=int(horizon),
    )

    try:
        predicted = predict_rollout(model_state, trajectory)
    except TrajectoryInvariantViolation as exc:
        return TrajectoryPlanResult(
            trajectory=None,
            cost=inf,
            valid=False,
            failure_reasons=(f"rollout:INV:{exc}",),
        )

    cost = trajectory_cost(predicted, best.actions, target_state)
    return TrajectoryPlanResult(
        trajectory=trajectory,
        cost=cost,
        valid=True,
        failure_reasons=(),
    )


__all__ = [
    "BEAM_WIDTH",
    "MAX_CANDIDATES_PER_STEP",
    "TARGET_TOL",
    "plan_trajectory",
    "trajectory_cost",
]
