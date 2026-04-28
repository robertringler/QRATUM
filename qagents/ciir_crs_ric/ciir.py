"""ciir.py — Constraint-Induced Interface Realism layer.

Implements:
    State       — typed, immutable state space S
    Constraint  — named predicate over states (element of constraint set C)
    phi(s, c)   — constraint satisfaction predicate, S x C -> Bool (PURE)
    Inv(s)      — system invariant, S -> Bool
    Omega(s)    — observer map S -> Observation (partial, only declared fields)

Domain: a small resource allocator. The state tracks integer CPU and memory
units in use, plus a status flag and a hidden "internal_notes" field that is
*not* observable through Omega — demonstrating that observability is strictly
narrower than the full state.

The `Constraint` value is parameterised by a pure predicate so that the
constraint *set* is data, not code, and `phi` does not need to switch on the
constraint name.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable, FrozenSet, Tuple


# --------------------------------------------------------------------------
# Typed state space S
# --------------------------------------------------------------------------

#: System budget constants — define the legal box of S.
MAX_CPU: int = 8           # hard upper bound on cpu_used
MAX_MEM: int = 16          # hard upper bound on mem_used
TOTAL_BUDGET: int = 20     # invariant: cpu_used + mem_used <= TOTAL_BUDGET


@dataclass(frozen=True)
class State:
    """Typed state space S.

    `frozen=True` enforces immutability — every transition must return a new
    `State`, never mutate.  Status values are restricted to a small string
    enum-like set so that the parser/controller can reason about reachability.
    """
    cpu_used: int
    mem_used: int
    status: str            # one of: 'idle', 'running', 'halted'
    internal_notes: str = ""  # NOT exposed via Omega — internal-only field

    def __post_init__(self) -> None:
        # Cheap structural typing: catch construction-time type errors early
        # so the rest of the pipeline can assume well-typed states.
        if not isinstance(self.cpu_used, int):
            raise TypeError(f"cpu_used must be int, got {type(self.cpu_used).__name__}")
        if not isinstance(self.mem_used, int):
            raise TypeError(f"mem_used must be int, got {type(self.mem_used).__name__}")
        if self.status not in {"idle", "running", "halted"}:
            raise ValueError(f"status must be idle|running|halted, got {self.status!r}")


# --------------------------------------------------------------------------
# Observer map Omega — partial projection of the state
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Observation:
    """Result of Omega(s).

    Only the fields declared here are observable.  `internal_notes` is omitted
    by design.  The numeric fields are exposed exactly; consumers of an
    Observation must NOT downcast it back into a State.
    """
    cpu_used: int
    mem_used: int
    status: str
    cpu_free: int          # derived field
    mem_free: int          # derived field


def Omega(s: State) -> Observation:
    """Observer map: project a State into a partial Observation.

    Pure function. Never exposes `internal_notes` or any future hidden fields.
    Derived quantities (`cpu_free`, `mem_free`) are computed here so that
    downstream layers do not need to know the budget constants.
    """
    return Observation(
        cpu_used=s.cpu_used,
        mem_used=s.mem_used,
        status=s.status,
        cpu_free=MAX_CPU - s.cpu_used,
        mem_free=MAX_MEM - s.mem_used,
    )


# --------------------------------------------------------------------------
# Constraint set C and satisfaction predicate phi
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Constraint:
    """A named, pure predicate over State.

    The constraint set C is then `FrozenSet[Constraint]`; `phi(s, c)` simply
    invokes the predicate.  Decoupling identity (name) from the predicate body
    makes constraints data-like and enables fast inspection and logging.
    """
    name: str
    predicate: Callable[[State], bool]
    description: str = ""

    # Constraints must be hashable by name only so two constraints with the
    # same name are equal even if their lambdas differ across imports.
    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Constraint) and self.name == other.name


def phi(s: State, c: Constraint) -> bool:
    """Constraint satisfaction predicate phi : S x C -> Bool.

    PURE. Any exception raised inside a constraint predicate is treated as
    *not satisfied* — this guarantees the satisfaction check itself can never
    crash the executor.
    """
    try:
        return bool(c.predicate(s))
    except Exception:
        return False


def all_satisfied(s: State, constraints: FrozenSet[Constraint]) -> Tuple[bool, Tuple[str, ...]]:
    """Return (all-true, names-of-violated-constraints).

    Used by `crs.constrained_transition` for both pre- and post-checks, and by
    `crs.admissible_actions` when enumerating A_C(s).
    """
    violated: list[str] = []
    for c in constraints:
        if not phi(s, c):
            violated.append(c.name)
    return (len(violated) == 0, tuple(violated))


# --------------------------------------------------------------------------
# Invariant Inv ⊆ S
# --------------------------------------------------------------------------

def Inv(s: State) -> bool:
    """System invariant.

    Inv := (status != 'halted') ∧ (cpu_used + mem_used <= TOTAL_BUDGET)
                                ∧ (cpu_used >= 0) ∧ (mem_used >= 0)

    Non-trivial: at least one execution path (Case C) attempts to exceed the
    budget and is required to be caught by Inv on the post-state.
    """
    return (
        s.status != "halted"
        and s.cpu_used >= 0
        and s.mem_used >= 0
        and (s.cpu_used + s.mem_used) <= TOTAL_BUDGET
    )


# --------------------------------------------------------------------------
# Default constraint set C_active
# --------------------------------------------------------------------------

C_ACTIVE: FrozenSet[Constraint] = frozenset({
    Constraint(
        name="cpu_in_range",
        predicate=lambda s: 0 <= s.cpu_used <= MAX_CPU,
        description="0 <= cpu_used <= MAX_CPU",
    ),
    Constraint(
        name="mem_in_range",
        predicate=lambda s: 0 <= s.mem_used <= MAX_MEM,
        description="0 <= mem_used <= MAX_MEM",
    ),
    Constraint(
        name="not_halted_for_alloc",
        predicate=lambda s: s.status != "halted",
        description="halted state forbids further allocation/scheduling",
    ),
})


__all__ = [
    "MAX_CPU",
    "MAX_MEM",
    "TOTAL_BUDGET",
    "State",
    "Observation",
    "Omega",
    "Constraint",
    "phi",
    "all_satisfied",
    "Inv",
    "C_ACTIVE",
    "replace",
]
