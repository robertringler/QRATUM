r"""Agent CO — Compression Optimizer.

Minimizes rule complexity while preserving behavior.

Strategies:
- Parameter pruning (remove near-zero parameters)
- Rule merging (combine equivalent rules)
- Complexity measurement (Kolmogorov proxy)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from quasim.ciir.swarm.physics_lang import (
    PhysicsProgram, RuleDecl, UpdateMode,
)


# ================================================================
# Compression result
# ================================================================

@dataclass
class CompressionResult:
    """Outcome of compression optimization.

    Attributes
    ----------
    original_complexity : int
    compressed_complexity : int
    rules_removed : int
    behavior_preserved : bool
    """

    original_complexity: int
    compressed_complexity: int
    rules_removed: int
    behavior_preserved: bool


# ================================================================
# Compression Optimizer agent
# ================================================================

class CompressionOptimizer:
    """Minimize rule complexity while preserving behavior."""

    def __init__(
        self,
        prune_threshold: float = 1e-4,
        merge_threshold: float = 0.01,
    ) -> None:
        self.prune_threshold = prune_threshold
        self.merge_threshold = merge_threshold

    def optimize(
        self,
        program: PhysicsProgram,
        reference_trajectory: list[NDArray],
        behavior_tol: float = 0.1,
    ) -> CompressionResult:
        """Compress the program's rule set.

        Removes rules that minimally affect the trajectory.
        """
        original_complexity = program.complexity
        original_rules = list(program.rules)
        original_count = len(original_rules)

        # Try removing each rule and check behavior preservation
        essential_rules: list[RuleDecl] = []
        for i, rule in enumerate(original_rules):
            test_rules = [r for j, r in enumerate(original_rules) if j != i]
            program.rules = test_rules

            if reference_trajectory:
                s0 = reference_trajectory[0]
                test_traj = program.run(s0, min(len(reference_trajectory) - 1, 50))
                preserved = self._trajectories_close(
                    reference_trajectory[: len(test_traj)],
                    test_traj,
                    behavior_tol,
                )
                if not preserved:
                    essential_rules.append(rule)
            else:
                essential_rules.append(rule)

        program.rules = essential_rules

        # Prune near-zero parameters
        for rule in program.rules:
            self._prune_params(rule)

        new_complexity = program.complexity
        rules_removed = original_count - len(essential_rules)

        # Verify behavior preserved
        if reference_trajectory:
            s0 = reference_trajectory[0]
            final_traj = program.run(s0, min(len(reference_trajectory) - 1, 50))
            preserved = self._trajectories_close(
                reference_trajectory[: len(final_traj)],
                final_traj,
                behavior_tol,
            )
        else:
            preserved = True

        return CompressionResult(
            original_complexity=original_complexity,
            compressed_complexity=new_complexity,
            rules_removed=rules_removed,
            behavior_preserved=preserved,
        )

    def _trajectories_close(
        self,
        traj_a: list[NDArray],
        traj_b: list[NDArray],
        tol: float,
    ) -> bool:
        """Compare two trajectories element-wise."""
        n = min(len(traj_a), len(traj_b))
        if n == 0:
            return True
        for i in range(n):
            if np.linalg.norm(traj_a[i] - traj_b[i]) > tol:
                return False
        return True

    def _prune_params(self, rule: RuleDecl) -> None:
        """Zero out near-zero parameters."""
        for key, val in rule.params.items():
            if isinstance(val, np.ndarray):
                val[np.abs(val) < self.prune_threshold] = 0.0

    def measure_complexity(self, program: PhysicsProgram) -> int:
        """Kolmogorov-proxy: count non-zero parameters."""
        total = 0
        for rule in program.rules:
            for val in rule.params.values():
                if isinstance(val, np.ndarray):
                    total += int(np.count_nonzero(val))
                elif isinstance(val, (int, float)):
                    total += 1 if abs(val) > 1e-30 else 0
        return total
