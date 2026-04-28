"""CRS — Computational Reality Substrate (typed state-transition system).

Formal definition:
    CRS := ⟨S, Σ, s₀, T, F⟩

where:
    S   — typed state space (shared with CIIR)
    Σ   — finite action alphabet (labels)
    s₀  — initial state
    T   — unconstrained transition function  T : S × A → S
    F   — set of terminal/accepting states

The constrained transition function is:
    T_C(s, a) := T(s, a)  if s ⊨ C_act and a ∈ A_C(s)
                 ⊥         otherwise           (represented as None)

CRS does NOT select actions — that is exclusively the role of RIC.
CRS does NOT evaluate constraints directly — it delegates to CIIR.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Callable, FrozenSet, Optional

from qagents.framework.ciir import (
    Constraint,
    ConstraintAlgebra,
    State,
)

# ---------------------------------------------------------------------------
# Action space A
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Action:
    """An atomic action a ∈ A.

    Attributes
    ----------
    name : str
        Action identifier (element of Σ).
    """

    name: str

    def __repr__(self) -> str:
        return f"Action({self.name!r})"


NOOP = Action(name="NOOP")  # Safe fallback action (always admissible by convention)


# ---------------------------------------------------------------------------
# Failure modes
# ---------------------------------------------------------------------------


class FailureMode(Enum):
    """Formal failure mode classification.

    Type I   — Constraint violation: s ⊭ C_act
    Type II  — Intent parse failure: π(i_raw) = ⊥
    Type III — Transition undefined: T_C(s, a) = ⊥
    Type IV  — Invariant breach:     T_C(s, a) ∉ Inv
    """

    TYPE_I_CONSTRAINT_VIOLATION = auto()
    TYPE_II_INTENT_PARSE_FAILURE = auto()
    TYPE_III_TRANSITION_UNDEFINED = auto()
    TYPE_IV_INVARIANT_BREACH = auto()


class CRSError(Exception):
    """Raised when CRS detects an irrecoverable failure."""

    def __init__(self, failure_mode: FailureMode, detail: str = "") -> None:
        self.failure_mode = failure_mode
        self.detail = detail
        super().__init__(f"{failure_mode.name}: {detail}")


# ---------------------------------------------------------------------------
# CRS — labeled typed state-transition system
# ---------------------------------------------------------------------------


class CRS:
    """Formal CRS := ⟨S, Σ, s₀, T, F⟩.

    Parameters
    ----------
    initial_state : State
        s₀ — the initial state.
    action_space : list[Action]
        Σ — finite action alphabet.
    transition_fn : Callable[[State, Action], State | None]
        T : S × A → S ∪ {⊥}.
        Returns None (⊥) if the transition is physically undefined
        (distinct from constraint violation).
    algebra : ConstraintAlgebra
        The CIIR constraint algebra governing this substrate.
    active_constraints : list[Constraint]
        C_act — the currently active constraint set.  May be updated between
        execution cycles but MUST NOT be mutated by RIC during a step.
    invariant : Callable[[State], bool]
        Inv ⊆ S — states satisfying the system invariant.
    terminal_states : FrozenSet[str] | None
        F — optional set of terminal-state identifiers.
    """

    def __init__(
        self,
        initial_state: State,
        action_space: list[Action],
        transition_fn: Callable[[State, Action], Optional[State]],
        algebra: ConstraintAlgebra,
        active_constraints: list[Constraint],
        invariant: Callable[[State], bool],
        terminal_states: Optional[FrozenSet[str]] = None,
    ) -> None:
        self._initial_state: State = dict(initial_state)
        self._action_space: list[Action] = list(action_space)
        self._T: Callable[[State, Action], Optional[State]] = transition_fn
        self._algebra: ConstraintAlgebra = algebra
        self._active: list[Constraint] = list(active_constraints)
        self._invariant: Callable[[State], bool] = invariant
        self._terminal: FrozenSet[str] = terminal_states or frozenset()

    # ------------------------------------------------------------------
    # Read-only accessors
    # ------------------------------------------------------------------

    @property
    def initial_state(self) -> dict[str, Any]:
        return dict(self._initial_state)

    @property
    def action_space(self) -> list[Action]:
        return list(self._action_space)

    @property
    def active_constraints(self) -> list[Constraint]:
        """C_act — read-only view.  RIC must not mutate this during a step."""
        return list(self._active)

    @property
    def algebra(self) -> ConstraintAlgebra:
        return self._algebra

    # ------------------------------------------------------------------
    # Determinism condition (Definition 4.3)
    # ------------------------------------------------------------------

    def is_deterministic(self) -> bool:
        """T is deterministic iff |T(s,a)| ≤ 1 for all (s,a).

        For a total function T : S × A → S this holds by construction.
        Returns True (structural guarantee for this implementation).
        """
        return True  # T is a total deterministic function by construction

    # ------------------------------------------------------------------
    # Invariant check
    # ------------------------------------------------------------------

    def in_invariant(self, state: State) -> bool:
        """Return True iff state ∈ Inv."""
        return bool(self._invariant(state))

    # ------------------------------------------------------------------
    # Admissible action set  A_C(s)
    # ------------------------------------------------------------------

    def admissible_actions(self, state: State) -> list[Action]:
        """Compute A_C(s) := { a ∈ A : φ(s,c) ∧ φ(T(s,a),c) ∀ c ∈ C_act }.

        Both pre-condition (current state satisfies C_act) and
        post-condition (successor state satisfies C_act) are enforced.

        Returns an empty list if the pre-condition fails (Type I failure).
        NOOP is always included if it produces a valid, constraint-satisfying
        successor.
        """
        # Pre-condition: s ⊨ C_act
        if not self._algebra.satisfies_all(state, self._active):
            return []  # Type I failure — no actions are admissible

        admissible: list[Action] = []
        for action in self._action_space:
            successor = self._T(state, action)
            if successor is None:
                # T(s, a) = ⊥ — physically undefined transition
                continue
            # Post-condition: T(s, a) ⊨ C_act AND T(s, a) ∈ Inv
            if self._algebra.satisfies_all(successor, self._active) and self.in_invariant(
                successor
            ):
                admissible.append(action)

        return admissible

    # ------------------------------------------------------------------
    # Constrained transition function T_C(s, a)
    # ------------------------------------------------------------------

    def execute(self, state: State, action: Action) -> Optional[State]:
        """T_C(s, a) — execute action only if pre/post conditions hold.

        Returns
        -------
        dict[str, Any]
            The successor state s' if the transition is valid.
        None (⊥)
            If any condition fails:
                — s ⊭ C_act            → Type I
                — T(s, a) = ⊥         → Type III
                — T(s, a) ⊭ C_act     → Type I (post-condition)
                — T(s, a) ∉ Inv       → Type IV
        """
        # Pre-condition: s ⊨ C_act
        if not self._algebra.satisfies_all(state, self._active):
            return None  # Type I

        # Physical transition
        successor = self._T(state, action)
        if successor is None:
            return None  # Type III

        # Post-condition: successor ⊨ C_act
        if not self._algebra.satisfies_all(successor, self._active):
            return None  # Type I (post-condition failure)

        # Invariant check: successor ∈ Inv
        if not self.in_invariant(successor):
            return None  # Type IV

        return dict(successor)
