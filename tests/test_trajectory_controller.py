"""Tests for deterministic finite-horizon trajectory controller."""

from __future__ import annotations

import math

import numpy as np
import pytest

from qagents.mvri.action_space import EPSILON, Action
from qagents.mvri.state import Inv, State, make_state
from qagents.realworld_bridge import (ControlLoop, ExternalActuatorAdapter,
                                      MockActuator, MockSensor, Observation,
                                      SafetyConfig, StateObserver)
from qagents.trajectory_controller import (ACTION_VECTOR_DIM, Trajectory,
                                           TrajectoryInvariantViolationError,
                                           action_to_vector,
                                           execute_trajectory, plan_trajectory,
                                           predict_rollout,
                                           run_trajectory_control,
                                           trajectory_cost,
                                           validate_trajectory,
                                           vector_to_action)
from qagents.trajectory_controller.planner import (BEAM_WIDTH,
                                                   MAX_CANDIDATES_PER_STEP)
from qagents.trajectory_controller.types import (EDGE_ADJUST_CODE, NO_SEED,
                                                 NODE_SHIFT_CODE,
                                                 TrajectoryExecutionTrace,
                                                 TrajectoryPlanResult,
                                                 TrajectoryStepTrace)

NODE_IDS: tuple[str, ...] = ("n1", "n2", "n3")
EDGE_IDS: tuple[tuple[str, str], ...] = (
    ("n1", "n2"),
    ("n1", "n3"),
    ("n2", "n3"),
)


def _state() -> State:
    return make_state(
        activations={"n1": 0.6, "n2": 0.3, "n3": 0.1},
        edge_weights={
            ("n1", "n2"): 0.4,
            ("n1", "n3"): 0.1,
            ("n2", "n3"): 0.2,
        },
        bounds=(-1.0, 1.0),
    )


def _target() -> State:
    return make_state(
        activations={"n1": 0.61, "n2": 0.29, "n3": 0.1},
        edge_weights={
            ("n1", "n2"): 0.4,
            ("n1", "n3"): 0.1,
            ("n2", "n3"): 0.2,
        },
        bounds=(-1.0, 1.0),
    )


def _obs(timestamp: float = 0.0) -> Observation:
    return Observation(
        timestamp=timestamp,
        source_id="traj-test",
        measurements=(0.6, 0.3, 0.1, 0.4, 0.1, 0.2),
        missing_mask=(False,) * 6,
    )


def _loop(
    observations: int,
    *,
    actuator=None,
    safety_config: SafetyConfig | None = None,
) -> ControlLoop:
    s0 = _state()
    return ControlLoop(
        sensor=MockSensor(tuple(_obs(float(i)) for i in range(observations)), "traj"),
        observer=StateObserver(s0, 0.5, NODE_IDS, EDGE_IDS),
        actuator=actuator or MockActuator(),
        safety_config=safety_config
        or SafetyConfig(epsilon=EPSILON, mvri_constraint_check=True),
        initial_state=s0,
        node_ids=NODE_IDS,
        edge_ids=EDGE_IDS,
        sync_alpha=0.5,
        n_candidates=1,
        rng=np.random.default_rng(0),
    )


def _trajectory_from_action(action: Action, horizon: int = 1) -> Trajectory:
    s = _state()
    return Trajectory(
        actions=tuple(action_to_vector(action, s) for _ in range(horizon)),
        horizon=horizon,
    )


# --------------------------------------------------------------------------- #
# types.py
# --------------------------------------------------------------------------- #


def test_trajectory_requires_positive_horizon() -> None:
    with pytest.raises(ValueError):
        Trajectory(actions=(), horizon=0)


def test_trajectory_requires_actions_equal_horizon() -> None:
    with pytest.raises(ValueError):
        Trajectory(actions=(), horizon=1)


def test_action_vectors_are_immutable_copies() -> None:
    raw = np.asarray([NODE_SHIFT_CODE, 0.0, 0.0, NO_SEED])
    traj = Trajectory(actions=(raw,), horizon=1)
    raw[2] = 0.5
    assert traj.actions[0][2] == 0.0
    with pytest.raises(ValueError):
        traj.actions[0][2] = 0.1


def test_action_vector_round_trip_node() -> None:
    action = Action(type="node_activation_shift", target="n2", magnitude=-0.01)
    vector = action_to_vector(action, _state())
    assert vector.shape == (ACTION_VECTOR_DIM,)
    assert vector_to_action(vector, _state()) == action


def test_action_vector_round_trip_edge() -> None:
    action = Action(type="edge_weight_adjust", target="n1->n3", magnitude=0.01)
    vector = action_to_vector(action, _state())
    assert int(vector[0]) == int(EDGE_ADJUST_CODE)
    assert vector_to_action(vector, _state()) == action


def test_decode_rejects_out_of_range_target() -> None:
    with pytest.raises(ValueError):
        vector_to_action(np.asarray([NODE_SHIFT_CODE, 99.0, 0.0, NO_SEED]), _state())


# --------------------------------------------------------------------------- #
# planner.py
# --------------------------------------------------------------------------- #


def test_plan_valid_trajectory_generation() -> None:
    result = plan_trajectory(
        _state(), _target(), 2, SafetyConfig(epsilon=EPSILON), _state()
    )
    assert result.valid
    assert result.trajectory is not None
    assert result.trajectory.horizon == 2
    assert result.failure_reasons == ()


def test_plan_rejects_non_positive_horizon() -> None:
    result = plan_trajectory(
        _state(), _target(), 0, SafetyConfig(epsilon=EPSILON), _state()
    )
    assert not result.valid
    assert result.trajectory is None
    assert result.failure_reasons == ("horizon:non_positive",)


def test_plan_full_rejection_on_invalid_state() -> None:
    s = _state()
    invalid = State(
        nodes=s.nodes,
        activations=s.activations,
        edges=s.edges,
        edge_weights=s.edge_weights,
        bounds=s.bounds,
        initial_edges=frozenset({("n1", "n2")}),
        step=s.step,
        metadata=s.metadata,
    )
    result = plan_trajectory(invalid, _target(), 1, SafetyConfig(), invalid)
    assert not result.valid
    assert any("INV" in reason for reason in result.failure_reasons)


def test_plan_rejects_unreachable_topology() -> None:
    target = make_state(
        activations={"n1": 0.6, "n2": 0.3, "n3": 0.1},
        edge_weights={("n1", "n2"): 0.4},
    )
    result = plan_trajectory(_state(), target, 2, SafetyConfig(), _state())
    assert not result.valid
    assert "topology:edges" in result.failure_reasons


def test_plan_is_deterministic() -> None:
    args = (_state(), _target(), 3, SafetyConfig(epsilon=EPSILON), _state())
    r1 = plan_trajectory(*args)
    r2 = plan_trajectory(*args)
    assert r1 == r2


def test_plan_cost_is_finite_when_valid() -> None:
    result = plan_trajectory(_state(), _state(), 1, SafetyConfig(), _state())
    assert result.valid
    assert math.isfinite(result.cost)


def test_planner_bounds_are_explicit() -> None:
    assert BEAM_WIDTH > 0
    assert MAX_CANDIDATES_PER_STEP > 0


def test_trajectory_cost_zero_for_zero_action_at_target() -> None:
    s = _state()
    action = Action(type="node_activation_shift", target="n1", magnitude=0.0)
    assert trajectory_cost((s,), (action,), s) == 0.0


# --------------------------------------------------------------------------- #
# predictor.py
# --------------------------------------------------------------------------- #


def test_predict_rollout_returns_state_per_action() -> None:
    traj = _trajectory_from_action(
        Action(type="node_activation_shift", target="n1", magnitude=0.0),
        horizon=3,
    )
    states = predict_rollout(_state(), traj)
    assert len(states) == 3
    assert all(Inv(state) for state in states)


def test_predict_rollout_preserves_topology() -> None:
    s = _state()
    traj = _trajectory_from_action(
        Action(type="edge_weight_adjust", target="n1->n2", magnitude=0.0)
    )
    predicted = predict_rollout(s, traj)[0]
    assert predicted.nodes == s.nodes
    assert predicted.edges == s.edges
    assert predicted.initial_edges == s.initial_edges


def test_predict_rollout_rejects_invalid_initial_state() -> None:
    s = _state()
    invalid = State(
        nodes=s.nodes,
        activations=s.activations,
        edges=s.edges,
        edge_weights=s.edge_weights,
        bounds=s.bounds,
        initial_edges=frozenset({("n1", "n2")}),
    )
    traj = _trajectory_from_action(
        Action(type="node_activation_shift", target="n1", magnitude=0.0)
    )
    with pytest.raises(TrajectoryInvariantViolationError):
        predict_rollout(invalid, traj)


def test_predict_rollout_rejects_decode_failure() -> None:
    traj = Trajectory(
        actions=(np.asarray([NODE_SHIFT_CODE, 99.0, 0.0, NO_SEED]),),
        horizon=1,
    )
    with pytest.raises(TrajectoryInvariantViolationError):
        predict_rollout(_state(), traj)


def test_predict_rollout_is_deterministic() -> None:
    traj = _trajectory_from_action(
        Action(type="node_activation_shift", target="n1", magnitude=0.0),
        horizon=2,
    )
    assert predict_rollout(_state(), traj) == predict_rollout(_state(), traj)


# --------------------------------------------------------------------------- #
# validator.py
# --------------------------------------------------------------------------- #


def test_validate_trajectory_accepts_valid_zero_action() -> None:
    traj = _trajectory_from_action(
        Action(type="node_activation_shift", target="n1", magnitude=0.0)
    )
    valid, reasons = validate_trajectory(traj, _state(), SafetyConfig())
    assert valid
    assert reasons == ()


def test_validate_trajectory_mag_violation() -> None:
    traj = _trajectory_from_action(
        Action(type="node_activation_shift", target="n1", magnitude=0.5)
    )
    valid, reasons = validate_trajectory(traj, _state(), SafetyConfig(epsilon=0.05))
    assert not valid
    assert any(reason.endswith(":MAG") for reason in reasons)


def test_validate_trajectory_roc_violation() -> None:
    s = _state()
    a1 = Action(type="node_activation_shift", target="n1", magnitude=0.0)
    a2 = Action(type="node_activation_shift", target="n1", magnitude=0.04)
    traj = Trajectory(actions=(action_to_vector(a1, s), action_to_vector(a2, s)), horizon=2)
    valid, reasons = validate_trajectory(
        traj,
        s,
        SafetyConfig(epsilon=0.05, max_rate_of_change=0.001),
    )
    assert not valid
    assert any(reason.endswith(":ROC") for reason in reasons)


def test_validate_trajectory_mvr_violation() -> None:
    traj = _trajectory_from_action(
        Action(type="node_activation_shift", target="n1", magnitude=-0.01)
    )
    valid, reasons = validate_trajectory(traj, _state(), SafetyConfig())
    assert not valid
    assert any(reason.endswith(":MVR") for reason in reasons)


def test_validate_trajectory_decode_violation() -> None:
    traj = Trajectory(actions=(np.asarray([NODE_SHIFT_CODE, 999.0, 0.0, NO_SEED]),), horizon=1)
    valid, reasons = validate_trajectory(traj, _state(), SafetyConfig())
    assert not valid
    assert any(":DECODE:" in reason for reason in reasons)


def test_validate_trajectory_cross_step_target_change_roc() -> None:
    s = _state()
    a1 = Action(type="node_activation_shift", target="n1", magnitude=0.0)
    a2 = Action(type="node_activation_shift", target="n2", magnitude=0.0)
    traj = Trajectory(actions=(action_to_vector(a1, s), action_to_vector(a2, s)), horizon=2)
    valid, reasons = validate_trajectory(
        traj,
        s,
        SafetyConfig(epsilon=0.05, max_rate_of_change=0.001, mvri_constraint_check=False),
    )
    assert not valid
    assert any(reason.endswith(":ROC") for reason in reasons)


def test_validate_trajectory_mvri_check_can_be_disabled() -> None:
    traj = _trajectory_from_action(
        Action(type="node_activation_shift", target="n1", magnitude=-0.01)
    )
    valid, reasons = validate_trajectory(
        traj, _state(), SafetyConfig(mvri_constraint_check=False)
    )
    assert valid
    assert not any(reason.endswith(":MVR") for reason in reasons)


# --------------------------------------------------------------------------- #
# executor.py
# --------------------------------------------------------------------------- #


def test_execute_trajectory_success_trace() -> None:
    traj = _trajectory_from_action(
        Action(type="node_activation_shift", target="n1", magnitude=0.0),
        horizon=2,
    )
    trace = execute_trajectory(_loop(2), traj)
    assert isinstance(trace, TrajectoryExecutionTrace)
    assert not trace.aborted
    assert trace.accepted_steps == 2
    assert len(trace.steps) == 2


def test_execute_trajectory_aborts_on_invalid_validation() -> None:
    traj = _trajectory_from_action(
        Action(type="node_activation_shift", target="n1", magnitude=0.5)
    )
    trace = execute_trajectory(_loop(1), traj)
    assert trace.aborted
    assert len(trace.steps) == 1
    assert not trace.steps[0].accepted
    assert any("MAG" in reason for reason in trace.failure_reasons)


def test_execute_partial_trace_correctness() -> None:
    s = _state()
    good = Action(type="node_activation_shift", target="n1", magnitude=0.0)
    bad = Action(type="node_activation_shift", target="n1", magnitude=0.5)
    traj = Trajectory(actions=(action_to_vector(good, s), action_to_vector(bad, s)), horizon=2)
    trace = execute_trajectory(_loop(2), traj)
    assert trace.aborted
    assert len(trace.steps) == 2
    assert trace.steps[0].accepted
    assert not trace.steps[1].accepted


def test_execute_aborts_on_sensor_exhausted_preserves_prior_steps() -> None:
    traj = _trajectory_from_action(
        Action(type="node_activation_shift", target="n1", magnitude=0.0),
        horizon=2,
    )
    trace = execute_trajectory(_loop(1), traj)
    assert trace.aborted
    assert trace.accepted_steps == 1
    assert any("SensorExhausted" in reason for reason in trace.failure_reasons)


def test_execute_mid_execution_actuator_failure() -> None:
    actuator = ExternalActuatorAdapter(lambda _a: False, "bad")
    traj = _trajectory_from_action(
        Action(type="node_activation_shift", target="n1", magnitude=0.0)
    )
    trace = execute_trajectory(_loop(1, actuator=actuator), traj)
    assert trace.aborted
    assert len(trace.steps) == 1
    assert any("ACT" in reason for reason in trace.failure_reasons)


def test_execute_does_not_call_actuator_for_invalid_action() -> None:
    actuator = MockActuator()
    traj = _trajectory_from_action(
        Action(type="node_activation_shift", target="n1", magnitude=0.5)
    )
    execute_trajectory(_loop(1, actuator=actuator), traj)
    assert actuator.get_log() == ()


def test_execute_trace_step_action_is_immutable() -> None:
    traj = _trajectory_from_action(
        Action(type="node_activation_shift", target="n1", magnitude=0.0)
    )
    trace = execute_trajectory(_loop(1), traj)
    step = trace.steps[0]
    assert isinstance(step, TrajectoryStepTrace)
    with pytest.raises(ValueError):
        step.action[2] = 0.1


# --------------------------------------------------------------------------- #
# controller.py and integration
# --------------------------------------------------------------------------- #


def test_run_trajectory_control_returns_control_run_result() -> None:
    result = run_trajectory_control(
        _state(), _state(), 2, _loop(2), SafetyConfig(epsilon=EPSILON)
    )
    assert result.n_steps == 2
    assert result.acceptance_rate == 1.0


def test_run_trajectory_control_invalid_plan_returns_rejection() -> None:
    result = run_trajectory_control(
        _state(),
        make_state(activations={"n1": 0.6}, edge_weights={}),
        2,
        _loop(2),
        SafetyConfig(),
    )
    assert result.n_steps == 0
    assert result.safety_rejection_rate == 1.0


def test_horizon_one_edge_case() -> None:
    result = plan_trajectory(_state(), _state(), 1, SafetyConfig(), _state())
    assert result.valid
    assert result.trajectory is not None
    assert result.trajectory.horizon == 1


def test_execution_trace_is_deterministic() -> None:
    traj = _trajectory_from_action(
        Action(type="node_activation_shift", target="n1", magnitude=0.0),
        horizon=2,
    )

    def signature():
        trace = execute_trajectory(_loop(2), traj)
        return tuple((s.step, s.accepted, round(s.drift, 12)) for s in trace.steps)

    assert signature() == signature()


def test_planner_deterministic_without_rng() -> None:
    # The planner has no RNG parameter and produces identical value output.
    r1 = plan_trajectory(_state(), _target(), 2, SafetyConfig(), _state())
    r2 = plan_trajectory(_state(), _target(), 2, SafetyConfig(), _state())
    assert r1 == r2


def test_planner_returns_typed_result() -> None:
    result = plan_trajectory(_state(), _state(), 1, SafetyConfig(), _state())
    assert isinstance(result, TrajectoryPlanResult)


def test_execution_trace_failure_reasons_are_structured_tuple() -> None:
    traj = _trajectory_from_action(
        Action(type="node_activation_shift", target="n1", magnitude=0.5)
    )
    trace = execute_trajectory(_loop(1), traj)
    assert isinstance(trace.failure_reasons, tuple)
    assert all(isinstance(reason, str) for reason in trace.failure_reasons)


def test_validated_trajectory_matches_predictor_length() -> None:
    plan = plan_trajectory(_state(), _state(), 2, SafetyConfig(), _state())
    assert plan.trajectory is not None
    valid, reasons = validate_trajectory(plan.trajectory, _state(), SafetyConfig())
    assert valid, reasons
    assert len(predict_rollout(_state(), plan.trajectory)) == plan.trajectory.horizon
