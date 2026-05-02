"""ric.py — Reality Interface Controller.

Implements:
    Intent           — the parsed intent space I (Enum)
    parse_intent     — pi : str -> I ∪ {None (bottom)}
    select_action    — rho : I x Observation x admissible-set -> Action

Strict module hygiene:
    * RIC NEVER touches raw `State` fields.  It receives an `Observation`
      produced by `ciir.Omega(s)` and an admissible set computed by
      `crs.admissible_actions`.  This realises the architectural rule
      "RIC must not access raw State fields directly — only through Omega".
    * `select_action` is deterministic: given the same `(intent, observation,
      admissible)` it returns the same `Action`.  No randomness, no hashing
      of mutable objects, no time-dependent branches.
    * If no valid action exists OR the intent is bottom, NOOP is returned.
"""
from __future__ import annotations

from enum import Enum
from typing import FrozenSet, Optional

# RIC imports the *types* it needs but performs no state-field access.
from .ciir import Observation
from .crs import Action


# --------------------------------------------------------------------------
# Intent space I
# --------------------------------------------------------------------------

class Intent(Enum):
    """High-level user intents.  pi maps raw strings to elements of I."""
    INCREASE_CPU = "increase_cpu"
    DECREASE_CPU = "decrease_cpu"
    INCREASE_MEM = "increase_mem"
    DECREASE_MEM = "decrease_mem"
    BEGIN = "begin"
    END = "end"
    HOLD = "hold"


# Canonical raw -> Intent mapping. Decoupled so it can be inspected/extended.
_RAW_TO_INTENT: dict[str, Intent] = {
    "increase_cpu": Intent.INCREASE_CPU,
    "more_cpu": Intent.INCREASE_CPU,
    "decrease_cpu": Intent.DECREASE_CPU,
    "less_cpu": Intent.DECREASE_CPU,
    "increase_mem": Intent.INCREASE_MEM,
    "more_mem": Intent.INCREASE_MEM,
    "decrease_mem": Intent.DECREASE_MEM,
    "less_mem": Intent.DECREASE_MEM,
    "begin": Intent.BEGIN,
    "start": Intent.BEGIN,
    "end": Intent.END,
    "stop": Intent.END,
    "hold": Intent.HOLD,
    "noop": Intent.HOLD,
}


def parse_intent(raw: str) -> Optional[Intent]:
    """pi : str -> I ∪ {None}.

    Returns None (bottom) for any input that is not a known canonical raw key,
    not a string, or empty / whitespace.  The executor maps a None return into
    a Type II failure record.
    """
    if not isinstance(raw, str):
        return None
    key = raw.strip().lower()
    if not key:
        return None
    return _RAW_TO_INTENT.get(key)


# --------------------------------------------------------------------------
# Deterministic action selector rho
# --------------------------------------------------------------------------

# Preference list per intent.  Ordering is the *deterministic* tie-break rule:
# the first action in this list that is also in the admissible set is chosen.
_PREFERENCE: dict[Intent, tuple[Action, ...]] = {
    Intent.INCREASE_CPU: (Action.ALLOCATE_CPU, Action.NOOP),
    Intent.DECREASE_CPU: (Action.RELEASE_CPU, Action.NOOP),
    Intent.INCREASE_MEM: (Action.ALLOCATE_MEM, Action.NOOP),
    Intent.DECREASE_MEM: (Action.RELEASE_MEM, Action.NOOP),
    Intent.BEGIN: (Action.START, Action.NOOP),
    Intent.END: (Action.STOP, Action.NOOP),
    Intent.HOLD: (Action.NOOP,),
}


def select_action(
    intent: Optional[Intent],
    observation: Observation,
    admissible: FrozenSet[Action],
) -> Action:
    """rho : I x Observation x 2^A -> A.

    Deterministic policy:
        1. If intent is None (parse failure), return NOOP.
        2. Walk the static `_PREFERENCE[intent]` list in order; return the
           first action also in `admissible`.
        3. If nothing in the preference list is admissible, return NOOP.
        4. NOOP itself may be inadmissible (if pre-state already violates a
           constraint); the executor must still verify membership in
           `admissible` before execution and treat absence as Type I.

    The `observation` parameter is currently used only for documentation —
    it would be consulted by more sophisticated deterministic policies (e.g.,
    refusing INCREASE_CPU when `cpu_free == 0`).  It is included in the
    signature to enforce the "RIC sees only Omega(s), never State" rule.
    """
    # Defensive read of the observation to keep the type-check tight even
    # though we don't branch on its values in this default policy.
    _ = (observation.cpu_used, observation.mem_used, observation.status)

    if intent is None:
        return Action.NOOP

    for candidate in _PREFERENCE.get(intent, (Action.NOOP,)):
        if candidate in admissible:
            return candidate
    return Action.NOOP


__all__ = [
    "Intent",
    "parse_intent",
    "select_action",
]
