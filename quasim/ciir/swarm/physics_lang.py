r"""Programmable Physics Language — syntax, semantics, and meta-programming.

Defines a formal "physics programming language" with:
1. Syntax: Rule definition grammar, state declarations, interaction specs
2. Semantics: Deterministic vs probabilistic, parallel vs sequential
3. Meta-programming: Rules that modify rules, constraint enforcement
4. Safety: Self-consistency condition Q(P) = P
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Any

import numpy as np
from numpy.typing import NDArray


# ================================================================
# Enumerations
# ================================================================

class UpdateMode(Enum):
    """Execution model for rule application."""
    DETERMINISTIC = auto()
    PROBABILISTIC = auto()


class ScheduleMode(Enum):
    """Rule scheduling discipline."""
    SEQUENTIAL = auto()
    PARALLEL = auto()


# ================================================================
# State declarations
# ================================================================

@dataclass
class StateDecl:
    r"""State space declaration.

    Specifies the dimensionality and constraints of the state space S.

    Parameters
    ----------
    dim : int
        Dimension of state vectors.
    constraints : list of Callable
        Each constraint maps state → bool.
    bounds : tuple of (float, float) or None
        Coordinate bounds if bounded.
    """

    dim: int
    constraints: list[Callable[[NDArray], bool]] = field(default_factory=list)
    bounds: tuple[float, float] | None = None

    def validate(self, state: NDArray) -> bool:
        """Check state satisfies all constraints."""
        if state.shape[-1] != self.dim:
            return False
        if self.bounds is not None:
            lo, hi = self.bounds
            if np.any(state < lo) or np.any(state > hi):
                return False
        return all(c(state) for c in self.constraints)


# ================================================================
# Rule declarations
# ================================================================

@dataclass
class RuleDecl:
    r"""Physics rule declaration.

    A rule R: S → S transforms states.  May be deterministic or probabilistic.

    Parameters
    ----------
    name : str
        Human-readable identifier.
    apply_fn : Callable
        (state, params, rng) → new_state
    params : dict
        Named parameters for the rule.
    mode : UpdateMode
        DETERMINISTIC or PROBABILISTIC
    priority : int
        Higher priority rules execute first in sequential mode.
    """

    name: str
    apply_fn: Callable[[NDArray, dict, np.random.Generator], NDArray]
    params: dict = field(default_factory=dict)
    mode: UpdateMode = UpdateMode.DETERMINISTIC
    priority: int = 0


# ================================================================
# Interaction specifications
# ================================================================

@dataclass
class InteractionSpec:
    r"""Interaction between two state subsystems.

    I: S_A × S_B → S_A × S_B

    Parameters
    ----------
    name : str
    subsystem_a : str
        Label of first subsystem.
    subsystem_b : str
        Label of second subsystem.
    coupling : float
        Interaction strength.
    apply_fn : Callable
        (state_a, state_b, coupling, rng) → (state_a', state_b')
    """

    name: str
    subsystem_a: str
    subsystem_b: str
    coupling: float = 1.0
    apply_fn: Callable | None = None


# ================================================================
# Meta-rules (rules that modify rules)
# ================================================================

@dataclass
class MetaRule:
    r"""Meta-programming construct: a rule that modifies other rules.

    M: R → R' — transforms the rule set itself.

    Parameters
    ----------
    name : str
    condition : Callable
        (program, history) → bool — decides whether to activate.
    transform : Callable
        (rules) → rules' — modifies the rule set.
    """

    name: str
    condition: Callable[[Any, list], bool]
    transform: Callable[[list[RuleDecl]], list[RuleDecl]]


# ================================================================
# Physics Program
# ================================================================

class PhysicsProgram:
    r"""A complete programmable physics system P = (S, R, U, Φ, C).

    Encapsulates state space, rules, update semantics, observation map,
    and consistency constraints.
    """

    def __init__(
        self,
        state_decl: StateDecl,
        schedule: ScheduleMode = ScheduleMode.SEQUENTIAL,
        seed: int = 42,
    ) -> None:
        self.state_decl = state_decl
        self.rules: list[RuleDecl] = []
        self.interactions: list[InteractionSpec] = []
        self.meta_rules: list[MetaRule] = []
        self.schedule = schedule
        self.rng = np.random.default_rng(seed)
        self._observation_map: Callable | None = None
        self._consistency_checks: list[Callable] = []

    # ---- Rule management ----------------------------------------

    def add_rule(self, rule: RuleDecl) -> None:
        """Register a physics rule."""
        self.rules.append(rule)

    def add_interaction(self, interaction: InteractionSpec) -> None:
        """Register a subsystem interaction."""
        self.interactions.append(interaction)

    def add_meta_rule(self, meta_rule: MetaRule) -> None:
        """Register a meta-rule."""
        self.meta_rules.append(meta_rule)

    def set_observation_map(self, phi: Callable[[NDArray], NDArray]) -> None:
        r"""Set the observation/interface map Φ: S → O."""
        self._observation_map = phi

    def add_consistency_check(self, check: Callable[[NDArray], bool]) -> None:
        r"""Add a consistency constraint C: S → {0, 1}."""
        self._consistency_checks.append(check)

    # ---- Execution engine (Update operator U) --------------------

    def execute_step(self, state: NDArray) -> NDArray:
        r"""Apply one update step: U(S, R) → S'.

        In SEQUENTIAL mode, rules execute in priority order.
        In PARALLEL mode, all rules execute on the same input state
        and contributions are averaged.
        """
        if self.schedule == ScheduleMode.SEQUENTIAL:
            sorted_rules = sorted(self.rules, key=lambda r: -r.priority)
            for rule in sorted_rules:
                state = rule.apply_fn(state, rule.params, self.rng)
        else:  # PARALLEL
            if not self.rules:
                return state
            contributions = []
            for rule in self.rules:
                contributions.append(rule.apply_fn(state, rule.params, self.rng))
            state = np.mean(contributions, axis=0)

        return state

    def observe(self, state: NDArray) -> NDArray:
        r"""Apply observation map Φ(S)."""
        if self._observation_map is not None:
            return self._observation_map(state)
        return state

    def check_consistency(self, state: NDArray) -> bool:
        r"""Verify C(S') = 1 for all consistency constraints."""
        if not self._consistency_checks:
            return True
        return all(c(state) for c in self._consistency_checks)

    def apply_meta_rules(self, history: list[NDArray]) -> None:
        """Apply meta-rules to potentially modify the rule set."""
        for mr in self.meta_rules:
            if mr.condition(self, history):
                self.rules = mr.transform(self.rules)

    # ---- Simulation driver --------------------------------------

    def run(self, initial_state: NDArray, steps: int) -> list[NDArray]:
        r"""Execute the full evolution for *steps* iterations.

        Returns trajectory [S_0, S_1, ..., S_n].
        """
        trajectory = [initial_state.copy()]
        state = initial_state.copy()

        for _ in range(steps):
            state = self.execute_step(state)

            if not self.check_consistency(state):
                break

            trajectory.append(state.copy())

        # Apply meta-rules after trajectory collection
        self.apply_meta_rules(trajectory)

        return trajectory

    # ---- Self-consistency check ---------------------------------

    def is_self_consistent(self, state: NDArray, tol: float = 1e-6) -> bool:
        r"""Verify safety constraint Q(P) = P (self-consistency).

        A program is self-consistent if applying it twice yields the
        same result as applying it once (idempotency in the stable limit).
        """
        s1 = self.execute_step(state)
        s2 = self.execute_step(s1)
        return bool(np.allclose(s1, s2, atol=tol))

    # ---- Properties ---------------------------------------------

    @property
    def rule_count(self) -> int:
        return len(self.rules)

    @property
    def complexity(self) -> int:
        """Kolmogorov-proxy complexity: total parameter count."""
        total = 0
        for rule in self.rules:
            total += sum(
                np.prod(np.array(v).shape) if isinstance(v, np.ndarray) else 1
                for v in rule.params.values()
            ) if rule.params else 0
        return int(total)
