r"""Agent ORC — Orchestrator.

Coordinates all swarm agents and executes the core evolution loop:

Step 1: Generate Theory   (AA → FZ → RS)
Step 2: Execute           (SIM)
Step 3: Analyze           (IM + EM)
Step 4: Validate          (CV)
Step 5: Optimize          (CO)
Step 6: Mutate            (HM)
Step 7: Select            (fitness-based)

Fitness function:
  F(P) = α·Consistency + β·EmpiricalFit + γ·Simplicity + δ·Novelty
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from quasim.ciir.swarm.axiom_architect import AxiomArchitect
from quasim.ciir.swarm.compression_optimizer import CompressionOptimizer
from quasim.ciir.swarm.consistency_validator import ConsistencyValidator
from quasim.ciir.swarm.empirical_mapper import EmpiricalMapper
from quasim.ciir.swarm.formalizer import Formalizer
from quasim.ciir.swarm.hypothesis_mutator import HypothesisMutator
from quasim.ciir.swarm.invariant_miner import InvariantMiner
from quasim.ciir.swarm.memory import KnowledgeGraph
from quasim.ciir.swarm.physics_lang import (PhysicsProgram, ScheduleMode,
                                            StateDecl)
from quasim.ciir.swarm.rule_synthesizer import RuleSynthesizer
from quasim.ciir.swarm.simulator import SimulationResult, Simulator

# ================================================================
# Fitness & evolution results
# ================================================================

@dataclass
class FitnessScores:
    """Multi-objective fitness decomposition."""
    consistency: float = 0.0
    empirical_fit: float = 0.0
    simplicity: float = 0.0
    novelty: float = 0.0
    total: float = 0.0


@dataclass
class EvolutionResult:
    """Result of one full evolution cycle.

    Attributes
    ----------
    generation : int
    best_fitness : FitnessScores
    best_program : PhysicsProgram
    sim_result : SimulationResult
    population_size : int
    memory_size : int
    """

    generation: int
    best_fitness: FitnessScores
    best_program: PhysicsProgram
    sim_result: SimulationResult
    population_size: int
    memory_size: int


# ================================================================
# Orchestrator
# ================================================================

class Orchestrator:
    r"""Coordinate all swarm agents and execute the evolution loop.

    Parameters
    ----------
    dim : int
        State space dimension.
    alpha, beta, gamma, delta : float
        Fitness weights for consistency, empirical fit, simplicity, novelty.
    seed : int
        Global random seed.
    """

    def __init__(
        self,
        dim: int = 4,
        alpha: float = 0.4,
        beta: float = 0.3,
        gamma: float = 0.2,
        delta: float = 0.1,
        seed: int = 42,
    ) -> None:
        self.dim = dim
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta

        # Shared memory
        self.memory = KnowledgeGraph()

        # Instantiate agents
        self.axiom_architect = AxiomArchitect(dim, seed)
        self.formalizer = Formalizer(dim)
        self.rule_synthesizer = RuleSynthesizer(dim, seed)
        self.simulator = Simulator(seed)
        self.invariant_miner = InvariantMiner()
        self.consistency_validator = ConsistencyValidator()
        self.empirical_mapper = EmpiricalMapper()
        self.compression_optimizer = CompressionOptimizer()
        self.hypothesis_mutator = HypothesisMutator(seed)

        self.rng = np.random.default_rng(seed)

    # ---- Core evolution loop ------------------------------------

    def run_evolution(
        self,
        generations: int = 5,
        steps_per_sim: int = 50,
        population_size: int = 5,
    ) -> list[EvolutionResult]:
        """Execute the full evolutionary loop for *generations* cycles."""
        results: list[EvolutionResult] = []

        # Step 1: Generate initial theory (AA → FZ → RS)
        axioms = self.axiom_architect.generate_minimal_axioms()
        self.axiom_architect.publish_to_memory(self.memory, axioms, generation=0)

        axiom_struct = self.formalizer.formalize_axiom_system(axioms)
        rules = self.rule_synthesizer.synthesize_default_ruleset()
        rule_struct = self.formalizer.formalize_rules(rules)
        cat_struct = self.formalizer.derive_category_structure(axiom_struct, rule_struct)
        self.formalizer.publish_to_memory(
            self.memory, [axiom_struct, rule_struct, cat_struct], generation=0,
        )
        self.rule_synthesizer.publish_to_memory(self.memory, rules, generation=0)

        # Build initial program
        program = self._build_program(rules)

        for gen in range(generations):
            # Step 2: Execute (SIM)
            s0 = self.rng.standard_normal(self.dim)
            s0 /= np.linalg.norm(s0)
            sim_result = self.simulator.run(program, s0, steps_per_sim)
            self.simulator.publish_to_memory(self.memory, sim_result, gen)

            # Step 3: Analyze (IM + EM)
            invariants = self.invariant_miner.mine_invariants(sim_result)
            self.invariant_miner.publish_to_memory(self.memory, invariants, gen)
            mappings = self.empirical_mapper.map_all(sim_result, invariants)
            self.empirical_mapper.publish_to_memory(self.memory, mappings, gen)

            # Step 4: Validate (CV)
            validation = self.consistency_validator.validate(program, axioms, sim_result)

            # Step 5: Optimize (CO)
            if sim_result.trajectory:
                self.compression_optimizer.optimize(
                    program, sim_result.trajectory, behavior_tol=0.5,
                )

            # Step 6: Mutate (HM) → population
            variants = self.hypothesis_mutator.generate_variants(
                program, n_variants=population_size,
            )

            # Step 7: Select (fitness-based)
            best_prog = program
            best_fitness = self._compute_fitness(
                program, sim_result, validation.passed, mappings, invariants,
            )

            for variant in variants:
                v_s0 = self.rng.standard_normal(self.dim)
                v_s0 /= np.linalg.norm(v_s0)
                v_result = self.simulator.run(variant, v_s0, steps_per_sim)
                v_valid = self.consistency_validator.validate(
                    variant, axioms, v_result,
                )
                v_inv = self.invariant_miner.mine_invariants(v_result)
                v_map = self.empirical_mapper.map_all(v_result, v_inv)
                v_fitness = self._compute_fitness(
                    variant, v_result, v_valid.passed, v_map, v_inv,
                )
                if v_fitness.total > best_fitness.total:
                    best_fitness = v_fitness
                    best_prog = variant
                    sim_result = v_result

            program = best_prog

            results.append(EvolutionResult(
                generation=gen,
                best_fitness=best_fitness,
                best_program=program,
                sim_result=sim_result,
                population_size=population_size,
                memory_size=self.memory.size,
            ))

        return results

    # ---- Fitness computation ------------------------------------

    def _compute_fitness(
        self,
        program: PhysicsProgram,
        sim_result: SimulationResult,
        consistent: bool,
        mappings: list,
        invariants: list,
    ) -> FitnessScores:
        """Compute F(P) = α·C + β·E + γ·S + δ·N."""
        # Consistency: binary
        c_score = 1.0 if consistent else 0.0

        # Empirical fit: average match score
        e_scores = [m.match_score for m in mappings]
        e_score = float(np.mean(e_scores)) if e_scores else 0.0

        # Simplicity: inverse of complexity
        complexity = max(program.complexity, 1)
        s_score = 1.0 / (1.0 + complexity / 100.0)

        # Novelty: number of non-trivial invariants / total
        conserved = [i for i in invariants if i.relative_variation < 0.05]
        n_score = len(conserved) / max(len(invariants), 1)

        total = (
            self.alpha * c_score
            + self.beta * e_score
            + self.gamma * s_score
            + self.delta * n_score
        )

        return FitnessScores(
            consistency=c_score,
            empirical_fit=e_score,
            simplicity=s_score,
            novelty=n_score,
            total=total,
        )

    # ---- Helpers ------------------------------------------------

    def _build_program(self, rules: list) -> PhysicsProgram:
        """Construct a PhysicsProgram from rules."""
        state_decl = StateDecl(dim=self.dim)
        program = PhysicsProgram(state_decl, ScheduleMode.SEQUENTIAL)
        for rule in rules:
            program.add_rule(rule)

        # Add consistency checks
        program.add_consistency_check(
            lambda s: bool(np.isfinite(np.linalg.norm(s)))
        )
        program.add_consistency_check(
            lambda s: float(np.linalg.norm(s)) < 1e6
        )

        return program
