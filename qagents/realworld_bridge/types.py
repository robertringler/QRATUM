"""Shared primitives for the real-world control bridge.

All types are ``@dataclass(frozen=True)`` and fully annotated. The bridge
deliberately reuses MVRI primitives (``Action``, ``State``,
``InjectionStatus``) instead of redefining them.
"""

from __future__ import annotations

from dataclasses import dataclass

from qagents.mvri.action_space import Action
from qagents.mvri.injector import InjectionStatus
from qagents.mvri.state import State

# --------------------------------------------------------------------------- #
# Sensor / observation
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Observation:
    """Typed partial measurement from a sensor.

    ``measurements`` is a canonical-order vector. ``missing_mask[i] = True``
    means dimension ``i`` is absent at this timestamp; a present
    measurement at that slot must be ignored by downstream consumers.

    Schema invariant: ``len(measurements) == len(missing_mask)``. The
    constructor enforces this so a structurally invalid Observation
    cannot be created.
    """

    timestamp: float
    source_id: str
    measurements: tuple[float, ...]
    missing_mask: tuple[bool, ...]

    def __post_init__(self) -> None:  # noqa: D401
        if len(self.measurements) != len(self.missing_mask):
            raise ValueError(
                "measurements and missing_mask must have equal length "
                f"(got {len(self.measurements)} vs {len(self.missing_mask)})"
            )


# --------------------------------------------------------------------------- #
# Actuator results
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ActionResult:
    """Result of a single actuator execution."""

    success: bool
    action: Action
    latency_ms: float
    failure_reason: str | None = None


# --------------------------------------------------------------------------- #
# Safety gate result
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ActionValidationResult:
    """Result of running the safety gate on a single action."""

    valid: bool
    action: Action
    failed_checks: tuple[str, ...]
    mvri_gate_passed: bool
    magnitude_passed: bool
    rate_limit_passed: bool


# --------------------------------------------------------------------------- #
# Loop trace + run result
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class LoopTrace:
    """Full record of one control step (every field always populated)."""

    step: int
    observation: Observation
    estimated_state: State
    synced_state: State
    candidates: tuple[Action, ...]
    validation_results: tuple[ActionValidationResult, ...]
    selected_action: Action | None
    action_result: ActionResult | None
    injection_status: InjectionStatus | None
    model_state_post: State
    timestamp: float
    failure_reason: str | None = None


@dataclass(frozen=True)
class ControlRunResult:
    """Aggregate result of ``ControlLoop.run(n_steps)``."""

    n_steps: int
    traces: tuple[LoopTrace, ...]
    final_model_state: State
    acceptance_rate: float
    safety_rejection_rate: float
    observer_converged: bool


__all__ = [
    "ActionResult",
    "ActionValidationResult",
    "ControlRunResult",
    "LoopTrace",
    "Observation",
]
