"""executor.py — the mandatory 9-step closed-loop execution.

Per execution tick the executor performs:

    1. parse_intent(raw)             -> i  or  bottom (None)
    2. assert Inv(s)                 — Type IV halt if violated
    3. compute A_C(s)
    4. select_action(i, Omega(s), A_C(s))  -> a
    5. assert a ∈ A_C(s)             — Type I if absent
    6. T_C(s, a)                     -> s' or bottom (None)
    7. assert Inv(s')                — Type IV if violated (post)
    8. log trace entry
    9. return (s', Omega(s'))

`run_step` performs exactly this sequence and *always* emits exactly one
`TraceEntry`, regardless of which step (if any) fails.  Failures are mapped to
deterministic `failures.FailureRecord`s; no exception escapes this function.

`print_trace` produces a human-inspectable rendering of any list of entries.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import FrozenSet, List, Optional, Tuple

from .ciir import C_ACTIVE, Constraint, Inv, Observation, Omega, State
from .crs import A_FULL, Action, admissible_actions, constrained_transition
from .failures import (STATUS_OK, STATUS_TYPE_I, STATUS_TYPE_II,
                       STATUS_TYPE_III, STATUS_TYPE_IV, ConstraintViolation,
                       FailureRecord, IntentParseFailure, InvalidTransition,
                       InvariantViolation)
from .ric import Intent, parse_intent, select_action

# --------------------------------------------------------------------------
# Trace schema
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class TraceEntry:
    """Mandatory trace schema (see spec §Trace Schema)."""
    step: int
    state_t: State
    intent_raw: str
    intent: Optional[Intent]
    admissible: FrozenSet[Action]
    action: Action
    state_t1: Optional[State]
    observation: Optional[Observation]
    status: str            # STATUS_OK | STATUS_TYPE_I..IV
    failure: Optional[FailureRecord] = None


@dataclass
class TraceLog:
    """Mutable container that accumulates trace entries over many ticks.

    The log itself is the *only* legitimate source of mutable state in the
    framework; states, observations and constraints remain immutable.
    """
    entries: List[TraceEntry] = field(default_factory=list)

    def append(self, entry: TraceEntry) -> None:
        self.entries.append(entry)

    def __iter__(self):  # pragma: no cover — convenience
        return iter(self.entries)

    def __len__(self) -> int:  # pragma: no cover — convenience
        return len(self.entries)


# --------------------------------------------------------------------------
# Single-tick execution
# --------------------------------------------------------------------------

def run_step(
    state: State,
    raw_intent: str,
    constraints: FrozenSet[Constraint] = C_ACTIVE,
    action_space: FrozenSet[Action] = A_FULL,
    step_index: int = 0,
) -> Tuple[State, Optional[Observation], TraceEntry]:
    """Execute one closed-loop tick.

    Returns ``(next_state, observation, entry)`` where ``next_state`` is the
    new state on success, or the *unchanged* input ``state`` on any failure
    (deterministic fallback).  ``observation`` is ``None`` if a failure
    prevents producing a new state.

    The executor never raises; every error path produces a populated
    `TraceEntry` with `failure` set.
    """
    # Step 1: parse intent ---------------------------------------------------
    intent: Optional[Intent] = parse_intent(raw_intent)
    if intent is None:
        failure = IntentParseFailure(
            f"intent parse returned bottom for raw={raw_intent!r}",
            raw=raw_intent,
        )
        entry = TraceEntry(
            step=step_index,
            state_t=state,
            intent_raw=raw_intent,
            intent=None,
            admissible=frozenset(),
            action=Action.NOOP,
            state_t1=None,
            observation=None,
            status=STATUS_TYPE_II,
            failure=failure.record(),
        )
        return state, None, entry

    # Step 2: pre-state invariant -------------------------------------------
    if not Inv(state):
        failure = InvariantViolation(
            "Inv(s_t) = False (pre-state)",
            state=repr(state),
        )
        entry = TraceEntry(
            step=step_index,
            state_t=state,
            intent_raw=raw_intent,
            intent=intent,
            admissible=frozenset(),
            action=Action.NOOP,
            state_t1=None,
            observation=None,
            status=STATUS_TYPE_IV,
            failure=failure.record(),
        )
        return state, None, entry

    # Step 3: compute A_C(s) ------------------------------------------------
    admissible: FrozenSet[Action] = admissible_actions(state, action_space, constraints)

    # Step 4: select an action (always via Omega, never raw state) ---------
    obs_pre: Observation = Omega(state)
    action: Action = select_action(intent, obs_pre, admissible)

    # Step 5: hard membership check -----------------------------------------
    if action not in admissible:
        failure = ConstraintViolation(
            f"selected action {action.name} ∉ A_C(s)",
            action=action.name,
            admissible=sorted(a.name for a in admissible),
        )
        entry = TraceEntry(
            step=step_index,
            state_t=state,
            intent_raw=raw_intent,
            intent=intent,
            admissible=admissible,
            action=action,
            state_t1=None,
            observation=None,
            status=STATUS_TYPE_I,
            failure=failure.record(),
        )
        return state, None, entry

    # Step 6: constrained transition ----------------------------------------
    s_next, violated = constrained_transition(state, action, constraints)
    if s_next is None:
        # Could be either a constraint violation that slipped past A_C (which
        # would indicate a logic bug in admissibility) or a true Type III.
        # We classify by whether any constraint names came back.
        if violated:
            failure = ConstraintViolation(
                f"T_C returned bottom: phi violated for {list(violated)}",
                action=action.name,
                violated=list(violated),
            )
            status = STATUS_TYPE_I
        else:
            failure = InvalidTransition(
                f"T_C returned bottom for action {action.name}",
                action=action.name,
            )
            status = STATUS_TYPE_III
        entry = TraceEntry(
            step=step_index,
            state_t=state,
            intent_raw=raw_intent,
            intent=intent,
            admissible=admissible,
            action=action,
            state_t1=None,
            observation=None,
            status=status,
            failure=failure.record(),
        )
        return state, None, entry

    # Step 7: post-state invariant ------------------------------------------
    if not Inv(s_next):
        failure = InvariantViolation(
            "Inv(s_{t+1}) = False (post-state)",
            action=action.name,
            state_next=repr(s_next),
        )
        entry = TraceEntry(
            step=step_index,
            state_t=state,
            intent_raw=raw_intent,
            intent=intent,
            admissible=admissible,
            action=action,
            state_t1=None,            # do NOT propagate an invariant-broken state
            observation=None,
            status=STATUS_TYPE_IV,
            failure=failure.record(),
        )
        return state, None, entry

    # Step 8 + 9: log and return --------------------------------------------
    obs_post: Observation = Omega(s_next)
    entry = TraceEntry(
        step=step_index,
        state_t=state,
        intent_raw=raw_intent,
        intent=intent,
        admissible=admissible,
        action=action,
        state_t1=s_next,
        observation=obs_post,
        status=STATUS_OK,
        failure=None,
    )
    return s_next, obs_post, entry


# --------------------------------------------------------------------------
# Multi-step driver and trace renderer
# --------------------------------------------------------------------------

def run(
    state: State,
    raw_intents: List[str],
    constraints: FrozenSet[Constraint] = C_ACTIVE,
    action_space: FrozenSet[Action] = A_FULL,
) -> Tuple[State, TraceLog]:
    """Execute a sequence of raw intents and return (final_state, trace_log)."""
    log = TraceLog()
    s = state
    for i, raw in enumerate(raw_intents):
        s, _obs, entry = run_step(s, raw, constraints, action_space, step_index=i)
        log.append(entry)
    return s, log


def print_trace(log: TraceLog) -> None:
    """Pretty-print the trace log to stdout for inspection."""
    print("=" * 78)
    print(f"TRACE — {len(log.entries)} entr{'y' if len(log.entries) == 1 else 'ies'}")
    print("=" * 78)
    for e in log.entries:
        print(f"[step {e.step}]  status={e.status}")
        print(f"  state_t     : {e.state_t}")
        print(f"  intent_raw  : {e.intent_raw!r}")
        print(f"  intent      : {e.intent.name if e.intent else 'BOTTOM'}")
        print(f"  A_C(s_t)    : {{{', '.join(sorted(a.name for a in e.admissible))}}}")
        print(f"  action      : {e.action.name}")
        print(f"  state_t+1   : {e.state_t1}")
        print(f"  observation : {e.observation}")
        if e.failure is not None:
            print(f"  failure     : {e.failure.code} {e.failure.name}: {e.failure.message}")
            if e.failure.details:
                print(f"    details   : {e.failure.details}")
        print("-" * 78)


__all__ = [
    "TraceEntry",
    "TraceLog",
    "run_step",
    "run",
    "print_trace",
]
