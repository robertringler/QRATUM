r"""Agent HM — Hypothesis Mutator.

Generates variant physics programs via:
- Rule perturbation (parameter jitter)
- Axiom mutation (relax/tighten constraints)
- Structural recombination (crossover between programs)
"""

from __future__ import annotations

import copy

import numpy as np

from quasim.ciir.swarm.physics_lang import (
    PhysicsProgram,
    StateDecl,
)

# ================================================================
# Hypothesis Mutator agent
# ================================================================


class HypothesisMutator:
    """Generate variant physics programs via mutation."""

    def __init__(self, seed: int = 42) -> None:
        self.rng = np.random.default_rng(seed)

    def perturb_rules(
        self,
        program: PhysicsProgram,
        sigma: float = 0.05,
    ) -> PhysicsProgram:
        """Return a new program with perturbed rule parameters."""
        new_prog = self._clone_program(program)
        for rule in new_prog.rules:
            for key, val in rule.params.items():
                if isinstance(val, np.ndarray):
                    rule.params[key] = val + self.rng.standard_normal(val.shape) * sigma
                elif isinstance(val, float):
                    rule.params[key] = val + self.rng.standard_normal() * sigma
        return new_prog

    def mutate_axioms(
        self,
        program: PhysicsProgram,
        mutation_rate: float = 0.1,
    ) -> PhysicsProgram:
        """Relax or tighten state space bounds."""
        new_prog = self._clone_program(program)
        if new_prog.state_decl.bounds is not None:
            lo, hi = new_prog.state_decl.bounds
            if self.rng.random() < mutation_rate:
                lo *= 1 + self.rng.standard_normal() * 0.1
            if self.rng.random() < mutation_rate:
                hi *= 1 + self.rng.standard_normal() * 0.1
            if lo < hi:
                new_prog.state_decl.bounds = (lo, hi)
        return new_prog

    def crossover(
        self,
        program_a: PhysicsProgram,
        program_b: PhysicsProgram,
    ) -> PhysicsProgram:
        """Create offspring by combining rules from two parents."""
        new_prog = self._clone_program(program_a)
        new_prog.rules = []

        rules_a = program_a.rules
        rules_b = program_b.rules

        # For each position, pick from parent A or B
        max_len = max(len(rules_a), len(rules_b))
        for i in range(max_len):
            if self.rng.random() < 0.5 and i < len(rules_a):
                new_prog.rules.append(copy.deepcopy(rules_a[i]))
            elif i < len(rules_b):
                new_prog.rules.append(copy.deepcopy(rules_b[i]))
            elif i < len(rules_a):
                new_prog.rules.append(copy.deepcopy(rules_a[i]))

        return new_prog

    def generate_variants(
        self,
        program: PhysicsProgram,
        n_variants: int = 5,
        sigma: float = 0.05,
    ) -> list[PhysicsProgram]:
        """Generate multiple variants of a program."""
        variants = []
        for _ in range(n_variants):
            strategy = self.rng.choice(["perturb", "mutate", "combined"])
            if strategy == "perturb":
                variants.append(self.perturb_rules(program, sigma))
            elif strategy == "mutate":
                variants.append(self.mutate_axioms(program))
            else:
                p = self.perturb_rules(program, sigma)
                variants.append(self.mutate_axioms(p))
        return variants

    def _clone_program(self, program: PhysicsProgram) -> PhysicsProgram:
        """Deep copy a physics program."""
        new = PhysicsProgram(
            state_decl=StateDecl(
                dim=program.state_decl.dim,
                constraints=list(program.state_decl.constraints),
                bounds=program.state_decl.bounds,
            ),
            schedule=program.schedule,
            seed=int(self.rng.integers(0, 2**31)),
        )
        new.rules = copy.deepcopy(program.rules)
        new.interactions = copy.deepcopy(program.interactions)
        new.meta_rules = copy.deepcopy(program.meta_rules)
        new._observation_map = program._observation_map
        new._consistency_checks = list(program._consistency_checks)
        return new
