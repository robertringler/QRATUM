"""Example usage for the deterministic trajectory controller."""

from __future__ import annotations

import numpy as np

from qagents.mvri.action_space import EPSILON
from qagents.mvri.state import make_state
from qagents.realworld_bridge import (
    ControlLoop,
    MockActuator,
    MockSensor,
    Observation,
    SafetyConfig,
    StateObserver,
)
from qagents.trajectory_controller import (
    execute_trajectory,
    plan_trajectory,
    validate_trajectory,
)

NODE_IDS = ("n1", "n2", "n3")
EDGE_IDS = (("n1", "n2"), ("n1", "n3"), ("n2", "n3"))


def build_state():
    return make_state(
        activations={"n1": 0.6, "n2": 0.3, "n3": 0.1},
        edge_weights={
            ("n1", "n2"): 0.4,
            ("n1", "n3"): 0.1,
            ("n2", "n3"): 0.2,
        },
    )


def observation(timestamp: float) -> Observation:
    return Observation(
        timestamp=timestamp,
        source_id="trajectory-demo",
        measurements=(0.6, 0.3, 0.1, 0.4, 0.1, 0.2),
        missing_mask=(False,) * 6,
    )


def main() -> None:
    initial = build_state()
    target = make_state(
        activations={"n1": 0.6, "n2": 0.3, "n3": 0.1},
        edge_weights={
            ("n1", "n2"): 0.4,
            ("n1", "n3"): 0.1,
            ("n2", "n3"): 0.2,
        },
    )
    safety = SafetyConfig(epsilon=EPSILON, mvri_constraint_check=True)
    plan = plan_trajectory(initial, target, 3, safety, initial)
    print(f"PLAN valid={plan.valid} cost={plan.cost:.6f} reasons={plan.failure_reasons}")
    if plan.trajectory is None:
        return
    valid, reasons = validate_trajectory(plan.trajectory, initial, safety)
    print(f"VALIDATION valid={valid} reasons={reasons}")

    sensor = MockSensor(tuple(observation(float(i)) for i in range(3)), "trajectory-demo")
    loop = ControlLoop(
        sensor=sensor,
        observer=StateObserver(initial, 0.5, NODE_IDS, EDGE_IDS),
        actuator=MockActuator(),
        safety_config=safety,
        initial_state=initial,
        node_ids=NODE_IDS,
        edge_ids=EDGE_IDS,
        sync_alpha=0.5,
        n_candidates=1,
        rng=np.random.default_rng(0),
    )
    trace = execute_trajectory(loop, plan.trajectory)
    print(
        f"EXECUTION aborted={trace.aborted} accepted={trace.accepted_steps} "
        f"reasons={trace.failure_reasons}"
    )


if __name__ == "__main__":
    main()


__all__ = ["main"]
