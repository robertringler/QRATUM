"""Core MVRI operator: ``inject(state, action) -> (state, status)``.

Pure, deterministic, rejection-complete:

- Validates the action first (cheap precondition).
- Runs the deterministic forward model.
- Evaluates the joint constraint gate ``Φ``.
- Returns the *original* state unchanged on any failure path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from qagents.mvri.action_space import EPSILON, Action, validate_action
from qagents.mvri.constraint_gate import constraint_gate
from qagents.mvri.forward_model import forward_model
from qagents.mvri.state import State

InjectionOutcome = Literal["accepted", "rejected"]


@dataclass(frozen=True)
class InjectionStatus:
    """Outcome of a single injection attempt."""

    outcome: InjectionOutcome
    action: Action
    step: int
    failed_constraints: tuple[str, ...] = field(default_factory=tuple)
    reason: str = ""

    @property
    def accepted(self) -> bool:
        return self.outcome == "accepted"


def inject(
    state: State,
    action: Action,
    *,
    epsilon: float = EPSILON,
    step: int | None = None,
) -> tuple[State, InjectionStatus]:
    """Apply the MVRI operator once.

    Parameters
    ----------
    state:
        Current system state ``s``.
    action:
        Proposed perturbation ``a``.
    epsilon:
        Magnitude bound passed to ``validate_action``.
    step:
        Optional explicit step counter for the returned status. If
        ``None`` the value is taken from ``state.step``.

    Returns
    -------
    (new_state, status):
        On acceptance, ``new_state`` is ``F(state, action)`` and
        ``status.outcome == "accepted"``. On rejection, ``new_state is state``
        (returned unchanged) and ``status.failed_constraints`` is populated.
    """

    s = step if step is not None else state.step

    if not validate_action(action, state, epsilon):
        return state, InjectionStatus(
            outcome="rejected",
            action=action,
            step=s,
            failed_constraints=("INVALID_ACTION",),
            reason="action failed validate_action precondition",
        )

    candidate = forward_model(state, action)
    passed, failed = constraint_gate(state, candidate, action)

    if not passed:
        return state, InjectionStatus(
            outcome="rejected",
            action=action,
            step=s,
            failed_constraints=tuple(failed),
            reason="constraint_gate rejected candidate",
        )

    return candidate, InjectionStatus(
        outcome="accepted",
        action=action,
        step=s,
        failed_constraints=(),
        reason="all constraints satisfied",
    )


__all__ = ["InjectionOutcome", "InjectionStatus", "inject"]
