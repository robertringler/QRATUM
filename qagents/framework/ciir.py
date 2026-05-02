"""CIIR — Constraint-Induced Interface Realism (governing theory layer).

Formal objects defined here:
    State         — typed state space S = prod_{k} S_k
    Constraint    — atomic constraint c with satisfaction predicate φ(s, c)
    ConstraintAlgebra — (C, ⊑, ⊓, ⊤, ⊥) with consistency check
    ObserverMap   — Ω : S → O  (maps state to observable subspace)

CIIR governs:
    * which states are accessible  (S_accessible ⊆ S)
    * which observations are valid  (O_C ⊆ O)
    * which actions are pre/post-condition legal  (A_C(s))

Nothing in this module mutates state — all functions are pure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional

# ---------------------------------------------------------------------------
# State space S
# ---------------------------------------------------------------------------

# A State is a frozen mapping from typed field names to typed values.
# Immutability is enforced: no function in this module returns a mutable state.
State = Mapping[str, Any]


def state_eq(s1: State, s2: State) -> bool:
    """Structural equality for states."""
    return dict(s1) == dict(s2)


# ---------------------------------------------------------------------------
# Constraint  c ∈ C
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Constraint:
    """An atomic constraint with a named satisfaction predicate.

    Formal: φ : S × C → {0, 1}
    Here each Constraint carries its own predicate φ_c : S → bool,
    so that φ(s, c) := c.satisfies(s).

    Attributes
    ----------
    name : str
        Human-readable constraint identifier.
    predicate : Callable[[State], bool]
        Pure function implementing φ_c.  Must be deterministic and side-effect
        free.
    """

    name: str
    predicate: Callable[[State], bool] = field(compare=False, hash=False)

    def satisfies(self, state: State) -> bool:
        """Return φ(state, self).  Pure — no side effects."""
        return bool(self.predicate(state))

    def __repr__(self) -> str:
        return f"Constraint({self.name!r})"


# ---------------------------------------------------------------------------
# Constraint algebra  (C, ⊑, ⊓, ⊤, ⊥)
# ---------------------------------------------------------------------------

class ConstraintAlgebra:
    """Finite constraint algebra over a fixed state space.

    Formal structure:
        (C, ⊑, ⊓, ⊤, ⊥_C)

    where:
        ⊑  — refinement preorder: c1 ⊑ c2 iff ∀s, φ(s,c1) → φ(s,c2)
        ⊓  — conjunction:  φ(s, c1⊓c2) := φ(s,c1) ∧ φ(s,c2)
        ⊤  — vacuous constraint (always satisfied)
        ⊥_C — inconsistent constraint (never satisfied)

    Consistency: C_act is consistent iff ∃ s ∈ S s.t. φ(s,c) for all c ∈ C_act.
    Consistency checking is delegated to the caller (witness required).
    """

    def __init__(self, constraints: list[Constraint]) -> None:
        # C — the base constraint set
        self._constraints: list[Constraint] = list(constraints)

    @property
    def constraints(self) -> list[Constraint]:
        """Return the ordered constraint list (C)."""
        return list(self._constraints)

    def satisfies_all(self, state: State, active: Optional[list[Constraint]] = None) -> bool:
        """Return True iff state ⊨ c for every c in active (or all C if None).

        Formal: s ⊨ ∧C_act  ⟺  ∀ c ∈ C_act : φ(s, c) = 1
        """
        subset = active if active is not None else self._constraints
        return all(c.satisfies(state) for c in subset)

    def violated_constraints(
        self, state: State, active: Optional[list[Constraint]] = None
    ) -> list[Constraint]:
        """Return the list of constraints violated by state."""
        subset = active if active is not None else self._constraints
        return [c for c in subset if not c.satisfies(state)]

    def is_consistent(self, witness: State) -> bool:
        """Check consistency via witness: ∃ s s.t. s ⊨ ∧C.

        A full decision procedure is domain-dependent (Open Problem 2 in the
        whitepaper).  This implementation uses an explicit witness.
        """
        return self.satisfies_all(witness)

    def conjunction(self, c1: Constraint, c2: Constraint) -> Constraint:
        """Return c1 ⊓ c2 as a new Constraint."""
        return Constraint(
            name=f"({c1.name} ⊓ {c2.name})",
            predicate=lambda s, _c1=c1, _c2=c2: _c1.satisfies(s) and _c2.satisfies(s),
        )

    def top(self) -> Constraint:
        """Return ⊤ — the vacuous constraint."""
        return Constraint(name="⊤", predicate=lambda _s: True)

    def bottom(self) -> Constraint:
        """Return ⊥_C — the inconsistent constraint."""
        return Constraint(name="⊥", predicate=lambda _s: False)


# ---------------------------------------------------------------------------
# Observer map  Ω : S → O
# ---------------------------------------------------------------------------

class ObserverMap:
    """Formal observer map Ω : S → O.

    The observable subspace O_C is the image of S_accessible under Ω:

        O_C = { Ω(s) : s ∈ S_accessible(C_act) }

    Observability bound (Proposition 7.3):
        |Ω(S_accessible)| ≤ |O_C|

    This class does NOT expose full state unless explicitly configured.
    The projection function determines which fields are observable.
    """

    def __init__(self, projection: Callable[[State], dict[str, Any]]) -> None:
        """
        Parameters
        ----------
        projection : Callable[[State], dict]
            Pure function that extracts the observable subspace from a state.
            Must not mutate the state.
        """
        self._projection = projection

    def observe(self, state: State) -> dict[str, Any]:
        """Ω(s) — return the observable projection of state s."""
        return dict(self._projection(state))

    def observable_equal(self, s1: State, s2: State) -> bool:
        """Return True iff Ω(s1) = Ω(s2) (states are observationally equivalent)."""
        return self.observe(s1) == self.observe(s2)


# ---------------------------------------------------------------------------
# Accessible state predicate
# ---------------------------------------------------------------------------

def accessible_state_space(
    state: State,
    algebra: ConstraintAlgebra,
    active: Optional[list[Constraint]] = None,
) -> bool:
    """Axiom 3.1 (Interface realism): s ∈ S_accessible iff s ⊨ ∧C_act.

    Returns True iff state belongs to the constraint-accessible subspace.
    """
    return algebra.satisfies_all(state, active)
