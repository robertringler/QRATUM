"""Deterministic trajectory rollout prediction."""

from __future__ import annotations

from qagents.control_geometry.sensitivity import state_to_vector
from qagents.mvri.forward_model import forward_model
from qagents.mvri.state import Edge, ForwardModelError, Inv, State
from qagents.realworld_bridge.types import Observation
from qagents.realworld_bridge.world_model_sync import (DEFAULT_MAX_CORRECTION,
                                                       sync_model)
from qagents.trajectory_controller.types import Trajectory, vector_to_action


class TrajectoryInvariantViolationError(RuntimeError):
    """Raised when predicted rollout would violate MVRI invariants."""


# Backwards-compatible concise alias used by this package's public API.
TrajectoryInvariantViolation = TrajectoryInvariantViolationError


def _node_edge_ids(state: State) -> tuple[tuple[str, ...], tuple[Edge, ...]]:
    return tuple(state.nodes), tuple(edge for edge, _ in state.edge_weights)


def _state_observation(state: State, *, timestamp: float) -> Observation:
    vec = state_to_vector(state)
    return Observation(
        timestamp=timestamp,
        source_id="trajectory_predictor",
        measurements=tuple(float(x) for x in vec),
        missing_mask=(False,) * int(vec.shape[0]),
    )


def predict_rollout(
    model_state: State,
    trajectory: Trajectory,
    *,
    max_correction_magnitude: float = DEFAULT_MAX_CORRECTION,
) -> tuple[State, ...]:
    """Predict post-action states for ``trajectory`` from ``model_state``.

    The rollout uses the existing MVRI forward model to compute each local
    candidate state, then routes the candidate vector through
    :func:`realworld_bridge.world_model_sync.sync_model` with bounded
    correction. Topology is inherited from ``model_state`` and never altered.
    Any invariant violation aborts prediction with
    :class:`TrajectoryInvariantViolationError`.
    """

    if not Inv(model_state):
        raise TrajectoryInvariantViolationError("initial model_state violates Inv")

    states: list[State] = []
    current = model_state
    node_ids, edge_ids = _node_edge_ids(model_state)
    topology = (model_state.nodes, model_state.edges, model_state.initial_edges)

    for index, vector in enumerate(trajectory.actions):
        try:
            action = vector_to_action(vector, current)
            candidate = forward_model(current, action)
        except (ForwardModelError, ValueError) as exc:
            raise TrajectoryInvariantViolationError(
                f"step {index}: forward prediction failed: {exc}"
            ) from exc

        if (
            candidate.nodes != topology[0]
            or candidate.edges != topology[1]
            or candidate.initial_edges != topology[2]
        ):
            raise TrajectoryInvariantViolationError(
                f"step {index}: prediction changed topology"
            )
        if not Inv(candidate):
            raise TrajectoryInvariantViolationError(
                f"step {index}: forward candidate violates Inv"
            )

        synced = sync_model(
            current,
            _state_observation(candidate, timestamp=float(index)),
            node_ids,
            edge_ids,
            alpha=1.0,
            max_correction_magnitude=max_correction_magnitude,
        )
        if synced is current and candidate != current:
            raise TrajectoryInvariantViolationError(
                f"step {index}: bounded correction rejected by sync_model"
            )
        if not Inv(synced):
            raise TrajectoryInvariantViolationError(
                f"step {index}: synced prediction violates Inv"
            )
        if (
            synced.nodes != topology[0]
            or synced.edges != topology[1]
            or synced.initial_edges != topology[2]
        ):
            raise TrajectoryInvariantViolationError(
                f"step {index}: sync_model changed topology"
            )
        states.append(synced)
        current = synced

    return tuple(states)


__all__ = [
    "TrajectoryInvariantViolation",
    "TrajectoryInvariantViolationError",
    "predict_rollout",
]
