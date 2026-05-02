"""RIC — deterministic controller layer.

Formal definition:
    RIC := ⟨π, ρ, A_C, Inv⟩

where:
    π  : I_raw → I ∪ {⊥}        intent parsing function
    ρ  : I × S → A               deterministic control function
    A_C : S → P(A)               admissible action correspondence (from CRS)
    Inv ⊆ S                       invariant set (from CRS)

Guarantees enforced here:
    Legality     — ρ(i, s) ∈ A_C(s) always
    Determinism  — ρ is a total, deterministic function
    Non-interference — RIC never mutates C_act during a step

The execution loop (Steps 1–9 of the formal lifecycle) is implemented in
``RIC.step()``.  Each step produces a ``TraceEntry`` record.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from qagents.framework.ciir import ObserverMap, State
from qagents.framework.crs import CRS, NOOP, Action, FailureMode

# ---------------------------------------------------------------------------
# Intent types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Intent:
    """Formal intent object i ∈ I.

    Attributes
    ----------
    goal : str
        Typed goal identifier.
    priority : int
        Priority ordering (higher = more urgent).
    preferred_action : str | None
        Optionally specifies a preferred action name.  RIC uses this only if
        the preferred action is admissible; otherwise it falls back to ρ.
    """
    goal: str
    priority: int = 0
    preferred_action: Optional[str] = None


# ---------------------------------------------------------------------------
# Trace record
# ---------------------------------------------------------------------------

@dataclass
class TraceEntry:
    """One record in the execution trace τ.

    Corresponds to the formal tuple:
        ⟨s_t, i_t, A_C(s_t), a_t, s_{t+1}, Ω(s_{t+1}), status⟩

    Attributes
    ----------
    step : int
        Step index t.
    state_before : dict
        s_t — CRS state before the step.
    raw_intent : str
        i_raw — the raw intent string received by RIC.
    parsed_intent : Intent | None
        i — the parsed formal intent (None on parse failure).
    admissible_actions : list[str]
        A_C(s_t) — names of constraint-legal actions.
    selected_action : str
        a_t — the action selected by ρ.
    state_after : dict | None
        s_{t+1} — successor state (None if transition failed).
    observation : dict | None
        Ω(s_{t+1}) — observable projection of successor state.
    failure_mode : FailureMode | None
        Failure type if any; None on success.
    status : str
        "OK" | "FAILED"
    """
    step: int
    state_before: dict
    raw_intent: str
    parsed_intent: Optional[Intent]
    admissible_actions: list[str]
    selected_action: str
    state_after: Optional[dict]
    observation: Optional[dict]
    failure_mode: Optional[FailureMode]
    status: str

    def as_dict(self) -> dict:
        return {
            "step": self.step,
            "state_before": self.state_before,
            "raw_intent": self.raw_intent,
            "parsed_intent": (
                {
                    "goal": self.parsed_intent.goal,
                    "priority": self.parsed_intent.priority,
                    "preferred_action": self.parsed_intent.preferred_action,
                }
                if self.parsed_intent is not None
                else None
            ),
            "admissible_actions": self.admissible_actions,
            "selected_action": self.selected_action,
            "state_after": self.state_after,
            "observation": self.observation,
            "failure_mode": (
                self.failure_mode.name if self.failure_mode is not None else None
            ),
            "status": self.status,
        }


# ---------------------------------------------------------------------------
# RIC controller
# ---------------------------------------------------------------------------

class RIC:
    """Formal RIC := ⟨π, ρ, A_C, Inv⟩.

    Parameters
    ----------
    crs : CRS
        The computational substrate.  RIC reads its state and action space
        but never mutates it directly.
    observer : ObserverMap
        Ω : S → O.  Applied after each transition.
    intent_parser : Callable[[str], Intent | None]
        π : I_raw → I ∪ {⊥}.  Returns None (⊥) on parse failure.
    policy : Callable[[Intent, State, list[Action]], Action] | None
        ρ : I × S × A_C(s) → A.  If None, the default lexicographic policy
        is used (prefer preferred_action if admissible, else first admissible,
        else NOOP).
    """

    def __init__(
        self,
        crs: CRS,
        observer: ObserverMap,
        intent_parser: Callable[[str], Optional[Intent]],
        policy: Optional[Callable[[Intent, State, list[Action]], Action]] = None,
    ) -> None:
        self._crs = crs
        self._observer = observer
        self._parse = intent_parser
        self._policy = policy or self._default_policy
        self._trace: list[TraceEntry] = []
        self._step_counter: int = 0

    # ------------------------------------------------------------------
    # Read-only accessors
    # ------------------------------------------------------------------

    @property
    def trace(self) -> list[TraceEntry]:
        """Full execution trace τ — read-only view."""
        return list(self._trace)

    # ------------------------------------------------------------------
    # Default lexicographic policy ρ
    # ------------------------------------------------------------------

    @staticmethod
    def _default_policy(
        intent: Intent,
        state: State,
        admissible: list[Action],
    ) -> Action:
        """Deterministic lexicographic policy (no randomness).

        Priority:
        1. Select preferred_action if admissible.
        2. Else select the first admissible action in A_C(s).
        3. Else NOOP (safe fallback — always returned even if not in A_C).

        Guarantee 5.1 (Legality): The returned action is always in A_C(s) if
        A_C(s) is non-empty; otherwise NOOP is returned as a safe fallback.
        """
        if not admissible:
            return NOOP

        # Preferred action override
        if intent.preferred_action is not None:
            for a in admissible:
                if a.name == intent.preferred_action:
                    return a

        # First admissible action (deterministic — list ordering is fixed)
        return admissible[0]

    # ------------------------------------------------------------------
    # Execution step (Steps 1–9 of the formal lifecycle)
    # ------------------------------------------------------------------

    def step(self, raw_intent: str, current_state: State) -> TraceEntry:
        """Execute one closed-loop step under CIIR–CRS governance.

        Formal pipeline (Section 9 of the whitepaper):
            Step 1: Intent Reception
            Step 2: Intent Parsing        π(i_raw) → i; handle ⊥
            Step 3: State Snapshot        verify s ∈ Inv
            Step 4: Constraint Evaluation compute A_C(s)
            Step 5: Action Selection      a = ρ(i, s, A_C(s)); verify a ∈ A_C(s)
            Step 6: Execution Dispatch    s' = T_C(s, a)
            Step 7: Post-Execution Verify s' ∈ Inv; Ω(s')
            Step 8: Trace Logging
            Step 9: State Handoff         (caller is responsible for s ← s')

        Returns
        -------
        TraceEntry
            Complete record for this step.
        """
        t = self._step_counter
        self._step_counter += 1

        # Snapshot active constraint set (non-interference: read-only during step)
        active_constraints = self._crs.active_constraints  # Step 1/4 read

        # ----------------------------------------------------------------
        # Step 2 — Intent parsing: i = π(i_raw)
        # ----------------------------------------------------------------
        parsed: Optional[Intent] = self._parse(raw_intent)

        if parsed is None:
            # Type II failure: intent parse failure
            entry = TraceEntry(
                step=t,
                state_before=dict(current_state),
                raw_intent=raw_intent,
                parsed_intent=None,
                admissible_actions=[],
                selected_action=NOOP.name,
                state_after=None,
                observation=None,
                failure_mode=FailureMode.TYPE_II_INTENT_PARSE_FAILURE,
                status="FAILED",
            )
            self._trace.append(entry)
            return entry

        # ----------------------------------------------------------------
        # Step 3 — State snapshot: verify s ∈ Inv
        # ----------------------------------------------------------------
        if not self._crs.in_invariant(current_state):
            entry = TraceEntry(
                step=t,
                state_before=dict(current_state),
                raw_intent=raw_intent,
                parsed_intent=parsed,
                admissible_actions=[],
                selected_action=NOOP.name,
                state_after=None,
                observation=None,
                failure_mode=FailureMode.TYPE_IV_INVARIANT_BREACH,
                status="FAILED",
            )
            self._trace.append(entry)
            return entry

        # ----------------------------------------------------------------
        # Step 4 — Constraint evaluation: compute A_C(s)
        # ----------------------------------------------------------------
        admissible: list[Action] = self._crs.admissible_actions(current_state)

        # Pre-condition failure: s ⊭ C_act  →  A_C(s) = ∅ (Type I)
        if not self._crs.algebra.satisfies_all(current_state, active_constraints):
            entry = TraceEntry(
                step=t,
                state_before=dict(current_state),
                raw_intent=raw_intent,
                parsed_intent=parsed,
                admissible_actions=[],
                selected_action=NOOP.name,
                state_after=None,
                observation=None,
                failure_mode=FailureMode.TYPE_I_CONSTRAINT_VIOLATION,
                status="FAILED",
            )
            self._trace.append(entry)
            return entry

        admissible_names = [a.name for a in admissible]

        # ----------------------------------------------------------------
        # Step 5 — Action selection: a = ρ(i, s, A_C(s))
        # ----------------------------------------------------------------
        # Non-interference: policy reads state/constraints but does not mutate them
        selected: Action = self._policy(parsed, current_state, admissible)

        # Legality assertion: a ∈ A_C(s)  (Guarantee 5.1)
        if selected not in admissible and selected != NOOP:
            # Controller selected an illegal action — treat as Type I failure
            entry = TraceEntry(
                step=t,
                state_before=dict(current_state),
                raw_intent=raw_intent,
                parsed_intent=parsed,
                admissible_actions=admissible_names,
                selected_action=selected.name,
                state_after=None,
                observation=None,
                failure_mode=FailureMode.TYPE_I_CONSTRAINT_VIOLATION,
                status="FAILED",
            )
            self._trace.append(entry)
            return entry

        # If A_C(s) is empty and selected is NOOP, we still record a blocked step
        if selected == NOOP and NOOP not in admissible:
            entry = TraceEntry(
                step=t,
                state_before=dict(current_state),
                raw_intent=raw_intent,
                parsed_intent=parsed,
                admissible_actions=admissible_names,
                selected_action=NOOP.name,
                state_after=None,
                observation=None,
                failure_mode=FailureMode.TYPE_I_CONSTRAINT_VIOLATION,
                status="FAILED",
            )
            self._trace.append(entry)
            return entry

        # ----------------------------------------------------------------
        # Step 6 — Execution dispatch: s' = T_C(s, a)
        # ----------------------------------------------------------------
        successor: Optional[State] = self._crs.execute(current_state, selected)

        if successor is None:
            # Type III: transition undefined under T_C
            entry = TraceEntry(
                step=t,
                state_before=dict(current_state),
                raw_intent=raw_intent,
                parsed_intent=parsed,
                admissible_actions=admissible_names,
                selected_action=selected.name,
                state_after=None,
                observation=None,
                failure_mode=FailureMode.TYPE_III_TRANSITION_UNDEFINED,
                status="FAILED",
            )
            self._trace.append(entry)
            return entry

        # ----------------------------------------------------------------
        # Step 7 — Post-execution verification: s' ∈ Inv; Ω(s')
        # ----------------------------------------------------------------
        if not self._crs.in_invariant(successor):
            # Type IV: invariant breach — T_C(s, a) ∉ Inv
            entry = TraceEntry(
                step=t,
                state_before=dict(current_state),
                raw_intent=raw_intent,
                parsed_intent=parsed,
                admissible_actions=admissible_names,
                selected_action=selected.name,
                state_after=dict(successor),
                observation=None,
                failure_mode=FailureMode.TYPE_IV_INVARIANT_BREACH,
                status="FAILED",
            )
            self._trace.append(entry)
            return entry

        observation = self._observer.observe(successor)

        # ----------------------------------------------------------------
        # Step 8 — Trace logging
        # ----------------------------------------------------------------
        entry = TraceEntry(
            step=t,
            state_before=dict(current_state),
            raw_intent=raw_intent,
            parsed_intent=parsed,
            admissible_actions=admissible_names,
            selected_action=selected.name,
            state_after=dict(successor),
            observation=observation,
            failure_mode=None,
            status="OK",
        )
        self._trace.append(entry)

        # Step 9 — State handoff: caller updates s ← s'
        return entry
