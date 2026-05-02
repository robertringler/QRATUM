"""Closed MVRI execution loop.

Implements the mandated 9-step iteration:

1. Validate ``Inv(s_current)`` → halt (Type IV) on failure.
2. Generate action: ``a = policy(s)``.
3. Validate action: ``validate_action(a, s, ε)`` → skip step if invalid.
4. Compute candidate: ``F(s, a)``.
5. Evaluate ``Φ(s, F(s, a))`` → reject if False.
6. If accepted: ``s = pipeline(s_new)``.
7. If rejected: ``s`` unchanged.
8. Log full trace entry.
9. Advance step counter.

The loop is *not coupled* to any specific policy implementation: the
``policy`` argument is an arbitrary ``Callable[[State], Action]``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal

from qagents.mvri.action_space import EPSILON, Action, validate_action
from qagents.mvri.constraint_gate import constraint_gate
from qagents.mvri.forward_model import forward_model
from qagents.mvri.metrics import (
    acceptance_rate,
    delta_entropy,
    delta_mutual_information,
    delta_transfer_entropy,
    violation_count,
)
from qagents.mvri.state import Inv, State

LoopOutcome = Literal[
    "accepted", "rejected", "invalid_action", "invariant_halt"
]


@dataclass(frozen=True)
class MVRITraceEntry:
    """One iteration of the loop, logged regardless of outcome."""

    step: int
    state_t: State
    action: Action | None
    action_valid: bool
    candidate_state: State | None
    constraints_passed: tuple[str, ...]
    constraints_failed: tuple[str, ...]
    outcome: LoopOutcome
    state_t1: State
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass
class LoopResult:
    """Final output of ``run_mvri_loop``."""

    initial_state: State
    final_state: State
    trace: list[MVRITraceEntry]
    halted: bool
    halt_reason: str = ""

    @property
    def n_steps(self) -> int:
        return len(self.trace)

    def acceptance_rate(self) -> float:
        return acceptance_rate(self.trace)

    def violation_count(self) -> dict[str, int]:
        return violation_count(self.trace)


# Default identity pipeline so callers may opt out cleanly.
def identity_pipeline(s: State) -> State:
    return s


_ALL_CONSTRAINTS: tuple[str, ...] = ("ER", "CS", "SS", "IC", "AUD")


def run_mvri_loop(
    initial_state: State,
    policy: Callable[[State], Action],
    pipeline: Callable[[State], State] = identity_pipeline,
    n_steps: int = 1,
    epsilon: float = EPSILON,
) -> LoopResult:
    """Run the closed MVRI loop and return the full trace.

    ``pipeline`` is applied to the new state only on *accepted* steps and
    only after the constraint gate has passed. It is therefore subject to
    the standing invariant ``Inv`` in the next iteration.
    """

    s_current = initial_state
    trace: list[MVRITraceEntry] = []

    if not Inv(s_current):
        # Type IV halt before any action.
        entry = MVRITraceEntry(
            step=0,
            state_t=s_current,
            action=None,
            action_valid=False,
            candidate_state=None,
            constraints_passed=(),
            constraints_failed=("INVARIANT_HALT",),
            outcome="invariant_halt",
            state_t1=s_current,
            metrics={"delta_H": 0.0, "delta_MI": 0.0, "delta_TE": 0.0},
        )
        trace.append(entry)
        return LoopResult(
            initial_state=initial_state,
            final_state=s_current,
            trace=trace,
            halted=True,
            halt_reason="initial state violates Inv (Type IV halt)",
        )

    for step in range(n_steps):
        # Step 1: invariant check at top of every iteration.
        if not Inv(s_current):
            entry = MVRITraceEntry(
                step=step,
                state_t=s_current,
                action=None,
                action_valid=False,
                candidate_state=None,
                constraints_passed=(),
                constraints_failed=("INVARIANT_HALT",),
                outcome="invariant_halt",
                state_t1=s_current,
                metrics={"delta_H": 0.0, "delta_MI": 0.0, "delta_TE": 0.0},
            )
            trace.append(entry)
            return LoopResult(
                initial_state=initial_state,
                final_state=s_current,
                trace=trace,
                halted=True,
                halt_reason=f"Inv violated at step {step} (Type IV halt)",
            )

        # Step 2: policy generates an action.
        action = policy(s_current)

        # Step 3: validate.
        valid = validate_action(action, s_current, epsilon)
        if not valid:
            entry = MVRITraceEntry(
                step=step,
                state_t=s_current,
                action=action,
                action_valid=False,
                candidate_state=None,
                constraints_passed=(),
                constraints_failed=("INVALID_ACTION",),
                outcome="invalid_action",
                state_t1=s_current,  # unchanged
                metrics={"delta_H": 0.0, "delta_MI": 0.0, "delta_TE": 0.0},
            )
            trace.append(entry)
            continue

        # Step 4: compute candidate via deterministic F.
        candidate = forward_model(s_current, action)

        # Step 5: joint constraint gate.
        passed, failed = constraint_gate(s_current, candidate, action)

        if passed:
            # Step 6: pipeline application on accepted candidate.
            s_new = pipeline(candidate)
            outcome: LoopOutcome = "accepted"
            state_t1 = s_new
            constraints_passed = _ALL_CONSTRAINTS
            constraints_failed_t: tuple[str, ...] = ()
        else:
            # Step 7: rejection — state unchanged.
            outcome = "rejected"
            state_t1 = s_current
            constraints_failed_t = tuple(failed)
            constraints_passed = tuple(
                c for c in _ALL_CONSTRAINTS if c not in failed
            )

        # Step 8: log full trace entry (observational metrics only).
        metrics = {
            "delta_H": delta_entropy(s_current, state_t1),
            "delta_MI": delta_mutual_information(s_current, state_t1),
            "delta_TE": delta_transfer_entropy(s_current, state_t1),
        }
        trace.append(
            MVRITraceEntry(
                step=step,
                state_t=s_current,
                action=action,
                action_valid=True,
                candidate_state=candidate,
                constraints_passed=constraints_passed,
                constraints_failed=constraints_failed_t,
                outcome=outcome,
                state_t1=state_t1,
                metrics=metrics,
            )
        )

        # Step 9: advance.
        s_current = state_t1

    return LoopResult(
        initial_state=initial_state,
        final_state=s_current,
        trace=trace,
        halted=False,
    )


__all__ = [
    "LoopOutcome",
    "LoopResult",
    "MVRITraceEntry",
    "identity_pipeline",
    "run_mvri_loop",
]
