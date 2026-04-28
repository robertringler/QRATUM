"""High-level plan → validate → execute trajectory controller."""

from __future__ import annotations

from qagents.mvri.state import State
from qagents.realworld_bridge.control_loop import ControlLoop
from qagents.realworld_bridge.safety_gate import SafetyConfig
from qagents.realworld_bridge.types import ControlRunResult
from qagents.trajectory_controller.executor import execute_trajectory
from qagents.trajectory_controller.planner import plan_trajectory
from qagents.trajectory_controller.validator import validate_trajectory

CONVERGENCE_TOLERANCE: float = 1e-3


def run_trajectory_control(
    initial_state: State,
    target_state: State,
    horizon: int,
    control_loop: ControlLoop,
    safety_config: SafetyConfig,
) -> ControlRunResult:
    """Plan, validate, and execute a finite-horizon trajectory.

    The returned object reuses RealWorldBridge ``ControlRunResult`` so callers
    can consume trajectory control with the same aggregate fields as a bridge
    run.  Detailed per-step trajectory traces are available from
    :func:`execute_trajectory`; this orchestration function exposes aggregate
    success/failure rates only.
    """

    plan = plan_trajectory(
        current_state=initial_state,
        target_state=target_state,
        horizon=horizon,
        safety_config=safety_config,
        model_state=control_loop.model_state,
    )
    if not plan.valid or plan.trajectory is None:
        return ControlRunResult(
            n_steps=0,
            traces=(),
            final_model_state=control_loop.model_state,
            acceptance_rate=0.0,
            safety_rejection_rate=1.0,
            observer_converged=control_loop.observer.is_converged(
                tol=CONVERGENCE_TOLERANCE
            ),
        )

    valid, _reasons = validate_trajectory(
        plan.trajectory,
        initial_state,
        safety_config,
        previous_action=control_loop.previous_action,
    )
    if not valid:
        return ControlRunResult(
            n_steps=0,
            traces=(),
            final_model_state=control_loop.model_state,
            acceptance_rate=0.0,
            safety_rejection_rate=1.0,
            observer_converged=control_loop.observer.is_converged(
                tol=CONVERGENCE_TOLERANCE
            ),
        )

    execution = execute_trajectory(control_loop, plan.trajectory)
    n_steps = len(execution.steps)
    accepted = execution.accepted_steps
    return ControlRunResult(
        n_steps=n_steps,
        traces=(),
        final_model_state=control_loop.model_state,
        acceptance_rate=(accepted / n_steps) if n_steps else 0.0,
        safety_rejection_rate=(1.0 if execution.aborted and accepted == 0 else 0.0),
        observer_converged=control_loop.observer.is_converged(
            tol=CONVERGENCE_TOLERANCE
        ),
    )


__all__ = ["CONVERGENCE_TOLERANCE", "run_trajectory_control"]
