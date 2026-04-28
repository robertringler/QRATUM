"""Stepwise execution of pre-planned trajectories through ControlLoop parts."""

from __future__ import annotations

from qagents.mvri.action_space import Action
from qagents.mvri.injector import inject
from qagents.mvri.state import State
from qagents.realworld_bridge.control_loop import ControlLoop
from qagents.realworld_bridge.observer import ObserverStateInvalid
from qagents.realworld_bridge.safety_gate import validate_action
from qagents.realworld_bridge.sensors import SensorExhausted
from qagents.realworld_bridge.types import ActionValidationResult
from qagents.realworld_bridge.world_model_sync import drift_magnitude, sync_model
from qagents.trajectory_controller.types import (
    Trajectory,
    TrajectoryExecutionTrace,
    TrajectoryStepTrace,
    vector_to_action,
)


def _rejected_result(action: Action, reasons: tuple[str, ...]) -> ActionValidationResult:
    return ActionValidationResult(
        valid=False,
        action=action,
        failed_checks=reasons,
        mvri_gate_passed="MVR" not in reasons,
        magnitude_passed="MAG" not in reasons,
        rate_limit_passed="ROC" not in reasons,
    )


def execute_trajectory(
    control_loop: ControlLoop,
    trajectory: Trajectory,
) -> TrajectoryExecutionTrace:
    """Execute ``trajectory`` through an existing RealWorldBridge loop.

    Execution is strictly sequential and aborts on the first failure.  The
    partial trace accumulated before the abort is returned; no failure is
    hidden or silently continued.
    """

    steps: list[TrajectoryStepTrace] = []
    failures: list[str] = []

    sensor = control_loop.sensor
    observer = control_loop.observer
    actuator = control_loop.actuator
    safety_config = control_loop.safety_config
    node_ids = control_loop.node_ids
    edge_ids = control_loop.edge_ids
    sync_alpha = control_loop.sync_alpha

    for index, vector in enumerate(trajectory.actions):
        model_state: State = control_loop.model_state
        previous_action = control_loop.previous_action
        try:
            planned_action = vector_to_action(vector, model_state)
        except ValueError as exc:
            failures.append(f"step:{index}:DECODE:{exc}")
            return TrajectoryExecutionTrace(
                steps=tuple(steps),
                aborted=True,
                failure_reasons=tuple(failures),
            )

        try:
            observation = sensor.read()
        except SensorExhausted as exc:
            failures.append(f"step:{index}:SensorExhausted:{exc}")
            return TrajectoryExecutionTrace(
                steps=tuple(steps),
                aborted=True,
                failure_reasons=tuple(failures),
            )

        observer.update(observation)
        try:
            observer.get_state()
        except ObserverStateInvalid as exc:
            result = _rejected_result(planned_action, ("ObserverStateInvalid",))
            trace = TrajectoryStepTrace(
                step=index,
                action=vector,
                validation_result=result,
                observation=observation,
                drift=0.0,
                accepted=False,
                failure_reasons=(f"ObserverStateInvalid:{exc}",),
            )
            steps.append(trace)
            failures.append(f"step:{index}:ObserverStateInvalid:{exc}")
            # Observer failure occurs before a valid planned action is selected
            # or injected, so only step bookkeeping advances.
            control_loop.commit_execution_step()
            return TrajectoryExecutionTrace(tuple(steps), True, tuple(failures))

        synced_state = sync_model(
            model_state,
            observation,
            node_ids,
            edge_ids,
            sync_alpha,
        )
        drift = drift_magnitude(synced_state, observation, node_ids, edge_ids)
        validation = validate_action(
            planned_action,
            synced_state,
            safety_config,
            previous_action,
        )
        if not validation.valid:
            step_reasons = tuple(validation.failed_checks)
            steps.append(
                TrajectoryStepTrace(
                    step=index,
                    action=vector,
                    validation_result=validation,
                    observation=observation,
                    drift=drift,
                    accepted=False,
                    failure_reasons=step_reasons,
                )
            )
            failures.extend(f"step:{index}:{reason}" for reason in step_reasons)
            control_loop.commit_execution_step()
            return TrajectoryExecutionTrace(tuple(steps), True, tuple(failures))

        action_result = actuator.execute(planned_action)
        if not action_result.success:
            reason = action_result.failure_reason or "actuator failure"
            steps.append(
                TrajectoryStepTrace(
                    step=index,
                    action=vector,
                    validation_result=validation,
                    observation=observation,
                    drift=drift,
                    accepted=False,
                    failure_reasons=(f"ACT:{reason}",),
                )
            )
            failures.append(f"step:{index}:ACT:{reason}")
            control_loop.commit_execution_step()
            return TrajectoryExecutionTrace(tuple(steps), True, tuple(failures))

        new_state, status = inject(synced_state, planned_action)
        if not status.accepted:
            reasons = tuple(status.failed_constraints) or (status.reason,)
            steps.append(
                TrajectoryStepTrace(
                    step=index,
                    action=vector,
                    validation_result=validation,
                    observation=observation,
                    drift=drift,
                    accepted=False,
                    failure_reasons=reasons,
                )
            )
            failures.extend(f"step:{index}:INJ:{reason}" for reason in reasons)
            control_loop.commit_execution_step(
                previous_action=planned_action,
                update_previous_action=True,
            )
            return TrajectoryExecutionTrace(tuple(steps), True, tuple(failures))

        control_loop.commit_execution_step(
            model_state=new_state,
            previous_action=planned_action,
            update_previous_action=True,
        )
        steps.append(
            TrajectoryStepTrace(
                step=index,
                action=vector,
                validation_result=validation,
                observation=observation,
                drift=drift,
                accepted=True,
                failure_reasons=(),
            )
        )

    return TrajectoryExecutionTrace(
        steps=tuple(steps),
        aborted=False,
        failure_reasons=(),
    )


__all__ = ["execute_trajectory"]
