"""Shared typed primitives for finite-horizon trajectory control.

The trajectory controller uses numeric action vectors at its public
boundary while delegating all safety and MVRI semantics to the existing
``realworld_bridge`` and ``mvri`` packages.  Action vectors use the
canonical encoding documented by :func:`action_to_vector`:

``[type_code, target_index, magnitude, seed]``

``type_code`` values are deterministic and fixed:

* ``0`` — ``node_activation_shift``; ``target_index`` indexes ``state.nodes``
* ``1`` — ``edge_weight_adjust``; ``target_index`` indexes sorted edges
* ``2`` — ``noise_injection``; ``target_index`` indexes ``state.nodes``

Planner-generated trajectories never use noise actions, so no randomness is
introduced.  The encoding still supports noise actions with an explicit seed
for validation/execution of externally supplied trajectories.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np

from qagents.mvri.action_space import Action
from qagents.mvri.state import State, format_edge
from qagents.realworld_bridge.types import ActionValidationResult, Observation

NODE_SHIFT_CODE: Final[float] = 0.0
EDGE_ADJUST_CODE: Final[float] = 1.0
NOISE_INJECTION_CODE: Final[float] = 2.0
NO_SEED: Final[float] = -1.0
ACTION_VECTOR_DIM: Final[int] = 4

LAMBDA_STATE: Final[float] = 1.0
LAMBDA_ACTION: Final[float] = 0.1
LAMBDA_ROC: Final[float] = 0.1


def _freeze_vector(vec: np.ndarray) -> np.ndarray:
    out = np.asarray(vec, dtype=np.float64).copy()
    if out.shape != (ACTION_VECTOR_DIM,):
        raise ValueError(f"action vector must have shape {(ACTION_VECTOR_DIM,)}, got {out.shape!r}")
    out.setflags(write=False)
    return out


def _vector_equal(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(np.array_equal(np.asarray(a), np.asarray(b)))


@dataclass(frozen=True, eq=False)
class Trajectory:
    """Finite-horizon action trajectory."""

    actions: tuple[np.ndarray, ...]
    horizon: int

    def __post_init__(self) -> None:
        if self.horizon <= 0:
            raise ValueError("horizon must be positive")
        if len(self.actions) != self.horizon:
            raise ValueError(f"len(actions)={len(self.actions)} must equal horizon={self.horizon}")
        object.__setattr__(
            self,
            "actions",
            tuple(_freeze_vector(a) for a in self.actions),
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Trajectory):
            return NotImplemented
        return self.horizon == other.horizon and all(
            _vector_equal(a, b) for a, b in zip(self.actions, other.actions)
        )


@dataclass(frozen=True)
class TrajectoryPlanResult:
    """Output of deterministic finite-horizon planning."""

    trajectory: Trajectory | None
    cost: float
    valid: bool
    failure_reasons: tuple[str, ...]


@dataclass(frozen=True)
class TrajectoryStepTrace:
    """Trace for one executed trajectory step."""

    step: int
    action: np.ndarray
    validation_result: ActionValidationResult
    observation: Observation
    drift: float
    accepted: bool
    failure_reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "action", _freeze_vector(self.action))


@dataclass(frozen=True)
class TrajectoryExecutionTrace:
    """Structured execution trace preserving partial progress on abort."""

    steps: tuple[TrajectoryStepTrace, ...]
    aborted: bool
    failure_reasons: tuple[str, ...]

    @property
    def accepted_steps(self) -> int:
        return sum(1 for step in self.steps if step.accepted)


def action_to_vector(action: Action, state: State) -> np.ndarray:
    """Encode an MVRI ``Action`` as a canonical immutable numeric vector."""

    if action.type == "node_activation_shift":
        code = NODE_SHIFT_CODE
        try:
            target_index = float(tuple(state.nodes).index(action.target))
        except ValueError as exc:
            raise ValueError(f"unknown node target {action.target!r}") from exc
        seed = NO_SEED
    elif action.type == "edge_weight_adjust":
        code = EDGE_ADJUST_CODE
        edges = tuple(sorted(state.edges))
        target = tuple(action.target.split("->", 1))
        if len(target) != 2 or not target[0] or not target[1]:
            raise ValueError(f"invalid edge target {action.target!r}")
        try:
            target_index = float(edges.index((target[0], target[1])))
        except ValueError as exc:
            raise ValueError(f"unknown edge target {action.target!r}") from exc
        seed = NO_SEED
    elif action.type == "noise_injection":
        code = NOISE_INJECTION_CODE
        try:
            target_index = float(tuple(state.nodes).index(action.target))
        except ValueError as exc:
            raise ValueError(f"unknown node target {action.target!r}") from exc
        if action.seed is None:
            raise ValueError("noise_injection action vector requires explicit seed")
        seed = float(action.seed)
    else:
        raise ValueError(f"unsupported action type {action.type!r}")

    return _freeze_vector(
        np.asarray([code, target_index, float(action.magnitude), seed], dtype=np.float64)
    )


def vector_to_action(vector: np.ndarray, state: State) -> Action:
    """Decode a canonical numeric action vector against ``state`` topology."""

    vec = np.asarray(vector, dtype=np.float64)
    if vec.shape != (ACTION_VECTOR_DIM,):
        raise ValueError(f"action vector must have shape {(ACTION_VECTOR_DIM,)}, got {vec.shape!r}")
    code = int(round(float(vec[0])))
    target_index = int(round(float(vec[1])))
    magnitude = float(vec[2])

    if code == int(NODE_SHIFT_CODE):
        nodes = tuple(state.nodes)
        if target_index < 0 or target_index >= len(nodes):
            raise ValueError(f"node target index out of range: {target_index}")
        return Action(
            type="node_activation_shift",
            target=nodes[target_index],
            magnitude=magnitude,
        )

    if code == int(EDGE_ADJUST_CODE):
        edges = tuple(sorted(state.edges))
        if target_index < 0 or target_index >= len(edges):
            raise ValueError(f"edge target index out of range: {target_index}")
        return Action(
            type="edge_weight_adjust",
            target=format_edge(edges[target_index]),
            magnitude=magnitude,
        )

    if code == int(NOISE_INJECTION_CODE):
        nodes = tuple(state.nodes)
        if target_index < 0 or target_index >= len(nodes):
            raise ValueError(f"node target index out of range: {target_index}")
        seed_value = int(round(float(vec[3])))
        if seed_value < 0:
            raise ValueError("noise_injection vector requires non-negative seed")
        return Action(
            type="noise_injection",
            target=nodes[target_index],
            magnitude=magnitude,
            seed=seed_value,
        )

    raise ValueError(f"unknown action type code: {code}")


__all__ = [
    "ACTION_VECTOR_DIM",
    "EDGE_ADJUST_CODE",
    "LAMBDA_ACTION",
    "LAMBDA_ROC",
    "LAMBDA_STATE",
    "NODE_SHIFT_CODE",
    "NOISE_INJECTION_CODE",
    "NO_SEED",
    "Trajectory",
    "TrajectoryExecutionTrace",
    "TrajectoryPlanResult",
    "TrajectoryStepTrace",
    "action_to_vector",
    "vector_to_action",
]
