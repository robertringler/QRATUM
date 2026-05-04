"""crs.py — Constrained Reality Substrate.

Implements:
    Action                  — complete action space A (Enum)
    transition(s, a)        — raw deterministic T : S x A -> S
    constrained_transition  — T_C : S x A -> S ∪ {None (bottom)}
    admissible_actions      — A_C(s) computed EXPLICITLY by enumeration

All functions are pure: the input `State` is never mutated.  Every transition
returns a brand-new `ciir.State` via `dataclasses.replace`.

`constrained_transition` enforces the formal A_C(s) condition by checking phi
on BOTH the pre-state and the post-state of the *raw* transition.  This is the
only legal path from one State to another in the system; `executor.py` is
required to route every action through this function.
"""

from __future__ import annotations

from dataclasses import replace
from enum import Enum
from typing import FrozenSet, Optional, Tuple

from .ciir import (
    Constraint,
    State,
    all_satisfied,
)

# --------------------------------------------------------------------------
# Action space A
# --------------------------------------------------------------------------


class Action(Enum):
    """Complete action space A."""

    ALLOCATE_CPU = "allocate_cpu"  # +1 CPU unit
    RELEASE_CPU = "release_cpu"  # -1 CPU unit
    ALLOCATE_MEM = "allocate_mem"  # +1 MEM unit
    RELEASE_MEM = "release_mem"  # -1 MEM unit
    START = "start"  # status: idle -> running
    STOP = "stop"  # status: running -> idle
    NOOP = "noop"  # identity action (safe fallback)


#: The full action space, exposed as an immutable frozenset.
A_FULL: FrozenSet[Action] = frozenset(Action)


# --------------------------------------------------------------------------
# Raw transition T : S x A -> S
# --------------------------------------------------------------------------


def transition(s: State, a: Action) -> State:
    """Raw deterministic transition T(s, a) -> s'.

    PURE. Always returns a new `State`. Even illegal-looking moves (e.g.,
    ALLOCATE_CPU when already at max) are computed; legality is checked by
    `constrained_transition` against the active constraint set.
    """
    if a is Action.NOOP:
        return s
    if a is Action.ALLOCATE_CPU:
        return replace(s, cpu_used=s.cpu_used + 1)
    if a is Action.RELEASE_CPU:
        return replace(s, cpu_used=s.cpu_used - 1)
    if a is Action.ALLOCATE_MEM:
        return replace(s, mem_used=s.mem_used + 1)
    if a is Action.RELEASE_MEM:
        return replace(s, mem_used=s.mem_used - 1)
    if a is Action.START:
        return replace(s, status="running")
    if a is Action.STOP:
        return replace(s, status="idle")
    # Defensive: an exhaustive Enum match means we never reach this branch,
    # but if a new Action is added without updating T this will surface clearly.
    raise ValueError(f"unhandled action {a!r} in transition()")


# --------------------------------------------------------------------------
# Constrained transition T_C : S x A x C -> S ∪ {None}
# --------------------------------------------------------------------------


def constrained_transition(
    s: State,
    a: Action,
    constraints: FrozenSet[Constraint],
) -> Tuple[Optional[State], Tuple[str, ...]]:
    """Constrained transition T_C(s, a).

    Returns ``(s', ())`` on success or ``(None, violated_names)`` (bottom)
    when either the pre-state or post-state fails phi for any active
    constraint.

    The violated-name tuple is part of the return value so that the executor
    can populate a precise `ConstraintViolation` failure record without having
    to recompute phi.

    PURE. Performs *no* mutation and *no* logging — logging is the executor's
    responsibility.  No action may bypass this function.
    """
    # Pre-condition phi(s, c) for all c
    pre_ok, pre_violated = all_satisfied(s, constraints)
    if not pre_ok:
        return (None, pre_violated)

    # Compute candidate post-state via raw T
    s_next = transition(s, a)

    # Post-condition phi(T(s, a), c) for all c
    post_ok, post_violated = all_satisfied(s_next, constraints)
    if not post_ok:
        return (None, post_violated)

    return (s_next, ())


# --------------------------------------------------------------------------
# Admissible action set A_C(s) — explicit enumeration
# --------------------------------------------------------------------------


def admissible_actions(
    s: State,
    action_space: FrozenSet[Action],
    constraints: FrozenSet[Constraint],
) -> FrozenSet[Action]:
    """Compute A_C(s) by testing every a ∈ A.

    A_C(s) := { a ∈ A | phi(s, c) ∀ c AND phi(T(s, a), c) ∀ c }

    Implementation enumerates the full action space and asks
    `constrained_transition` whether each candidate yields a non-bottom result.
    This guarantees the formal definition is realised exactly — no shortcuts
    based on intent or heuristics.

    Note: NOOP is always admissible whenever the pre-state itself satisfies
    every constraint, because T(s, NOOP) = s.
    """
    admissible: set[Action] = set()
    for a in action_space:
        s_next, _ = constrained_transition(s, a, constraints)
        if s_next is not None:
            admissible.add(a)
    return frozenset(admissible)


__all__ = [
    "Action",
    "A_FULL",
    "transition",
    "constrained_transition",
    "admissible_actions",
]
