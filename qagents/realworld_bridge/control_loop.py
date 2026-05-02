"""Closed-loop controller wiring sensor → observer → safety → actuator → MVRI.

The single source of truth for the control sequence is
:meth:`ControlLoop.step`, which executes the canonical 9-step pipeline
in order with no reordering and **no silent shortcuts**.

The only mutable field on a :class:`ControlLoop` is ``_model_state``;
every other piece of state is immutable, passed in explicitly, or
recreated each step.
"""

from __future__ import annotations

import time

import numpy as np

from qagents.mvri.action_space import Action
from qagents.mvri.injector import inject
from qagents.mvri.policy import probe_policy
from qagents.mvri.state import Edge, State
from qagents.realworld_bridge.actuators import Actuator
from qagents.realworld_bridge.observer import (
    ObserverStateInvalid,
    StateObserver,
)
from qagents.realworld_bridge.safety_gate import (
    SafetyConfig,
    validate_action,
)
from qagents.realworld_bridge.sensors import Sensor, SensorExhausted
from qagents.realworld_bridge.types import (
    ActionResult,
    ActionValidationResult,
    ControlRunResult,
    LoopTrace,
)
from qagents.realworld_bridge.world_model_sync import sync_model


class ControlLoop:
    """Closed-loop controller for the real-world bridge."""

    def __init__(
        self,
        sensor: Sensor,
        observer: StateObserver,
        actuator: Actuator,
        safety_config: SafetyConfig,
        initial_state: State,
        node_ids: tuple[str, ...],
        edge_ids: tuple[Edge, ...],
        sync_alpha: float,
        n_candidates: int,
        rng: np.random.Generator,
    ) -> None:
        if not isinstance(rng, np.random.Generator):
            raise TypeError(
                "ControlLoop requires an explicit numpy.random.Generator"
            )
        if int(n_candidates) <= 0:
            raise ValueError("n_candidates must be positive")
        if not (0.0 < float(sync_alpha) <= 1.0):
            raise ValueError("sync_alpha must be in (0, 1]")

        self._sensor = sensor
        self._observer = observer
        self._actuator = actuator
        self._safety_config = safety_config
        self._node_ids = tuple(node_ids)
        self._edge_ids = tuple(edge_ids)
        self._sync_alpha = float(sync_alpha)
        self._n_candidates = int(n_candidates)
        self._rng = rng

        # Mutable controller state (the only mutable members).
        self._model_state: State = initial_state
        self._previous_action: Action | None = None
        self._step_index: int = 0

    # ------------------------------------------------------------------ #
    # Public read-only views
    # ------------------------------------------------------------------ #

    @property
    def model_state(self) -> State:
        return self._model_state

    @property
    def sensor(self) -> Sensor:
        return self._sensor

    @property
    def observer(self) -> StateObserver:
        return self._observer

    @property
    def actuator(self) -> Actuator:
        return self._actuator

    @property
    def safety_config(self) -> SafetyConfig:
        return self._safety_config

    @property
    def node_ids(self) -> tuple[str, ...]:
        return self._node_ids

    @property
    def edge_ids(self) -> tuple[Edge, ...]:
        return self._edge_ids

    @property
    def sync_alpha(self) -> float:
        return self._sync_alpha

    @property
    def previous_action(self) -> Action | None:
        return self._previous_action

    @property
    def step_index(self) -> int:
        return self._step_index

    def commit_execution_step(
        self,
        *,
        model_state: State | None = None,
        previous_action: Action | None = None,
        update_previous_action: bool = False,
    ) -> None:
        """Advance loop bookkeeping for external planned-action execution."""

        if model_state is not None:
            self._model_state = model_state
        if update_previous_action:
            self._previous_action = previous_action
        self._step_index += 1

    # ------------------------------------------------------------------ #
    # Single step — strict 9-step pipeline
    # ------------------------------------------------------------------ #

    def step(self) -> LoopTrace:
        """Execute one full control step. ``SensorExhausted`` propagates."""

        ts = float(time.time())
        idx = self._step_index

        # 1. sensor.read()
        observation = self._sensor.read()

        # 2. observer.update()
        self._observer.update(observation)

        # 3. observer.get_state()
        try:
            estimated_state = self._observer.get_state()
            failure_reason: str | None = None
        except ObserverStateInvalid as exc:
            # Step is aborted; loop continues. We log a trace pinned to
            # the current model_state and a synthetic failure_reason.
            self._step_index += 1
            return LoopTrace(
                step=idx,
                observation=observation,
                estimated_state=self._model_state,
                synced_state=self._model_state,
                candidates=(),
                validation_results=(),
                selected_action=None,
                action_result=None,
                injection_status=None,
                model_state_post=self._model_state,
                timestamp=ts,
                failure_reason=f"ObserverStateInvalid: {exc}",
            )

        # 4. sync_model()
        synced_state = sync_model(
            self._model_state,
            observation,
            self._node_ids,
            self._edge_ids,
            self._sync_alpha,
        )

        # 5. candidate generation via probe_policy (deterministic given rng)
        candidates: tuple[Action, ...] = tuple(
            probe_policy(synced_state, self._rng)
            for _ in range(self._n_candidates)
        )

        # 6. validate every candidate; collect ALL results
        validation_results: list[ActionValidationResult] = [
            validate_action(
                a, synced_state, self._safety_config, self._previous_action
            )
            for a in candidates
        ]

        # 7. first valid action wins (deterministic — input order)
        selected_action: Action | None = None
        for vr in validation_results:
            if vr.valid:
                selected_action = vr.action
                break

        # 8. execute + inject if a valid candidate exists.
        #    model_state advances ONLY on accepted injection; on
        #    rejection or absence the prior model_state is preserved.
        action_result: ActionResult | None = None
        injection_status = None
        post_state: State = self._model_state
        if selected_action is not None:
            action_result = self._actuator.execute(selected_action)
            new_state, injection_status = inject(
                synced_state, selected_action
            )
            if injection_status.accepted:
                post_state = new_state

        # Update mutable controller state.
        self._model_state = post_state
        self._previous_action = selected_action
        self._step_index += 1

        # 9. log
        return LoopTrace(
            step=idx,
            observation=observation,
            estimated_state=estimated_state,
            synced_state=synced_state,
            candidates=candidates,
            validation_results=tuple(validation_results),
            selected_action=selected_action,
            action_result=action_result,
            injection_status=injection_status,
            model_state_post=post_state,
            timestamp=ts,
            failure_reason=failure_reason,
        )

    # ------------------------------------------------------------------ #
    # Multi-step run
    # ------------------------------------------------------------------ #

    def run(self, n_steps: int) -> ControlRunResult:
        """Run up to ``n_steps``; terminates early on ``SensorExhausted``."""

        if int(n_steps) < 0:
            raise ValueError("n_steps must be non-negative")

        traces: list[LoopTrace] = []
        for _ in range(int(n_steps)):
            try:
                traces.append(self.step())
            except SensorExhausted:
                break

        accepted = sum(
            1
            for t in traces
            if t.injection_status is not None
            and t.injection_status.accepted
        )
        no_valid = sum(
            1
            for t in traces
            if t.selected_action is None and t.failure_reason is None
        )
        n = len(traces)
        acceptance_rate = (accepted / n) if n > 0 else 0.0
        safety_rejection_rate = (no_valid / n) if n > 0 else 0.0

        return ControlRunResult(
            n_steps=n,
            traces=tuple(traces),
            final_model_state=self._model_state,
            acceptance_rate=acceptance_rate,
            safety_rejection_rate=safety_rejection_rate,
            observer_converged=self._observer.is_converged(tol=1e-3),
        )


__all__ = ["ControlLoop"]
