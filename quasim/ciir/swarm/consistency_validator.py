r"""Agent CV — Consistency Validator.

Rejects systems violating:
- Logical consistency (axiom satisfaction)
- Closure under rules (rule application preserves state space)
- Stability under iteration (bounded trajectories)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from quasim.ciir.swarm.axiom_architect import Axiom
from quasim.ciir.swarm.physics_lang import PhysicsProgram
from quasim.ciir.swarm.simulator import SimulationResult

# ================================================================
# Validation result
# ================================================================

@dataclass
class ValidationResult:
    """Outcome of consistency checking.

    Attributes
    ----------
    passed : bool
        Overall pass/fail.
    axiom_pass : bool
    closure_pass : bool
    stability_pass : bool
    details : dict
    """

    passed: bool
    axiom_pass: bool
    closure_pass: bool
    stability_pass: bool
    details: dict


# ================================================================
# Consistency Validator agent
# ================================================================

class ConsistencyValidator:
    """Reject systems violating logical consistency, closure, or stability."""

    def __init__(
        self,
        max_norm: float = 1e6,
        stability_window: int = 20,
    ) -> None:
        self.max_norm = max_norm
        self.stability_window = stability_window

    def validate(
        self,
        program: PhysicsProgram,
        axioms: list[Axiom],
        result: SimulationResult,
    ) -> ValidationResult:
        """Full validation pipeline."""
        axiom_pass = self._check_axioms(axioms, result)
        closure_pass = self._check_closure(program, result)
        stability_pass = self._check_stability(result)

        return ValidationResult(
            passed=axiom_pass and closure_pass and stability_pass,
            axiom_pass=axiom_pass,
            closure_pass=closure_pass,
            stability_pass=stability_pass,
            details={
                "trajectory_length": len(result.trajectory),
                "converged": result.converged,
                "final_norm": float(np.linalg.norm(result.final_state)),
            },
        )

    def _check_axioms(
        self, axioms: list[Axiom], result: SimulationResult,
    ) -> bool:
        """Check all axioms hold for the final state."""
        return all(ax.predicate(result.final_state) for ax in axioms)

    def _check_closure(
        self, program: PhysicsProgram, result: SimulationResult,
    ) -> bool:
        """Check that rule application preserves the state space."""
        s = result.final_state
        s_next = program.execute_step(s)
        return program.state_decl.validate(s_next)

    def _check_stability(self, result: SimulationResult) -> bool:
        """Check that the trajectory remains bounded."""
        if not result.trajectory:
            return False

        # Check all norms are finite and bounded
        norms = [np.linalg.norm(s) for s in result.trajectory]
        if any(not np.isfinite(n) for n in norms):
            return False
        if max(norms) > self.max_norm:
            return False

        # Check late trajectory is not diverging
        if len(norms) > self.stability_window:
            late = norms[-self.stability_window:]
            if late[-1] > 2 * late[0] and late[-1] > 1.0:
                return False

        return True
