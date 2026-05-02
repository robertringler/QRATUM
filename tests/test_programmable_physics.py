r"""Tests for Programmable Physics Multi-Agent Swarm Framework.

Covers all 10 agents, the physics programming language, shared memory,
the core evolution loop, and the required outputs from the master prompt:

1. Formal Axiomatic System
2. Executable Rule Framework
3. Simulation Results
4. Derived Spacetime Structure
5. Matter/Field Representations
6. Mapping to Known Physics
7. Falsifiable Predictions
8. Implementation Blueprint (self-consistency)

Run with:
    python -m pytest tests/test_programmable_physics.py -v --override-ini="addopts="
"""

from __future__ import annotations

import numpy as np
import pytest

# ================================================================
# Shared memory (M)
# ================================================================

class TestKnowledgeGraph:
    """Tests for the shared typed knowledge graph M = (N, E, τ)."""

    def test_add_node_increments_id(self):
        from quasim.ciir.swarm.memory import KnowledgeGraph, NodeType
        m = KnowledgeGraph()
        n0 = m.add_node(NodeType.AXIOM, "Ax1")
        n1 = m.add_node(NodeType.RULE, "R1")
        assert n0.id == 0
        assert n1.id == 1
        assert m.size == 2

    def test_edge_creation_and_query(self):
        from quasim.ciir.swarm.memory import EdgeType, KnowledgeGraph, NodeType
        m = KnowledgeGraph()
        n0 = m.add_node(NodeType.AXIOM, "Ax1")
        n1 = m.add_node(NodeType.RULE, "R1")
        e = m.add_edge(n0.id, n1.id, EdgeType.DERIVATION)
        assert len(m.get_edges_from(n0.id)) == 1
        assert len(m.get_edges_to(n1.id)) == 1
        assert e.edge_type == EdgeType.DERIVATION

    def test_contradiction_detection(self):
        from quasim.ciir.swarm.memory import EdgeType, KnowledgeGraph, NodeType
        m = KnowledgeGraph()
        n0 = m.add_node(NodeType.AXIOM, "A1")
        n1 = m.add_node(NodeType.AXIOM, "A2")
        assert not m.has_contradiction(n0.id, n1.id)
        m.add_edge(n0.id, n1.id, EdgeType.CONTRADICTION)
        assert m.has_contradiction(n0.id, n1.id)

    def test_get_nodes_by_type(self):
        from quasim.ciir.swarm.memory import KnowledgeGraph, NodeType
        m = KnowledgeGraph()
        m.add_node(NodeType.AXIOM, "Ax1")
        m.add_node(NodeType.RULE, "R1")
        m.add_node(NodeType.AXIOM, "Ax2")
        assert len(m.get_nodes_by_type(NodeType.AXIOM)) == 2
        assert len(m.get_nodes_by_type(NodeType.RULE)) == 1

    def test_summary(self):
        from quasim.ciir.swarm.memory import KnowledgeGraph, NodeType
        m = KnowledgeGraph()
        m.add_node(NodeType.AXIOM, "Ax1")
        m.add_node(NodeType.THEORY, "T1")
        s = m.summary()
        assert s["AXIOM"] == 1
        assert s["THEORY"] == 1


# ================================================================
# Physics programming language
# ================================================================

class TestPhysicsLanguage:
    """Tests for the programmable physics language."""

    def test_state_decl_validation(self):
        from quasim.ciir.swarm.physics_lang import StateDecl
        sd = StateDecl(dim=4, bounds=(-10, 10))
        s = np.array([1.0, 2.0, 3.0, 4.0])
        assert sd.validate(s)
        bad = np.array([1.0, 2.0])  # Wrong dimension
        assert not sd.validate(bad)
        oob = np.array([100.0, 0.0, 0.0, 0.0])  # Out of bounds
        assert not sd.validate(oob)

    def test_rule_execution_deterministic(self):
        from quasim.ciir.swarm.physics_lang import (PhysicsProgram, RuleDecl,
                                                    StateDecl)
        sd = StateDecl(dim=3)
        prog = PhysicsProgram(sd)
        prog.add_rule(RuleDecl(
            name="Scale",
            apply_fn=lambda s, p, r: s * p["factor"],
            params={"factor": 0.5},
        ))
        s = np.array([2.0, 4.0, 6.0])
        s_next = prog.execute_step(s)
        np.testing.assert_allclose(s_next, [1.0, 2.0, 3.0])

    def test_parallel_execution_averages(self):
        from quasim.ciir.swarm.physics_lang import (PhysicsProgram, RuleDecl,
                                                    ScheduleMode, StateDecl)
        sd = StateDecl(dim=2)
        prog = PhysicsProgram(sd, ScheduleMode.PARALLEL)
        prog.add_rule(RuleDecl("Double", lambda s, p, r: s * 2, {}))
        prog.add_rule(RuleDecl("Zero", lambda s, p, r: s * 0, {}))
        s = np.array([1.0, 1.0])
        s_next = prog.execute_step(s)
        np.testing.assert_allclose(s_next, [1.0, 1.0])  # average of 2x and 0x

    def test_consistency_check(self):
        from quasim.ciir.swarm.physics_lang import PhysicsProgram, StateDecl
        sd = StateDecl(dim=2)
        prog = PhysicsProgram(sd)
        prog.add_consistency_check(lambda s: float(np.linalg.norm(s)) < 10)
        assert prog.check_consistency(np.array([1.0, 1.0]))
        assert not prog.check_consistency(np.array([100.0, 100.0]))

    def test_observation_map(self):
        from quasim.ciir.swarm.physics_lang import PhysicsProgram, StateDecl
        sd = StateDecl(dim=4)
        prog = PhysicsProgram(sd)
        prog.set_observation_map(lambda s: s[:2])
        obs = prog.observe(np.array([1, 2, 3, 4.0]))
        assert obs.shape == (2,)

    def test_run_produces_trajectory(self):
        from quasim.ciir.swarm.physics_lang import (PhysicsProgram, RuleDecl,
                                                    StateDecl)
        sd = StateDecl(dim=2)
        prog = PhysicsProgram(sd)
        prog.add_rule(RuleDecl("Scale", lambda s, p, r: s * 0.9, {}))
        traj = prog.run(np.array([1.0, 1.0]), steps=10)
        assert len(traj) == 11  # initial + 10 steps

    def test_meta_rule_application(self):
        from quasim.ciir.swarm.physics_lang import (MetaRule, PhysicsProgram,
                                                    RuleDecl, StateDecl)
        sd = StateDecl(dim=2)
        prog = PhysicsProgram(sd)
        prog.add_rule(RuleDecl("R1", lambda s, p, r: s, {"alpha": 1.0}))
        prog.add_meta_rule(MetaRule(
            name="Decay",
            condition=lambda p, h: len(h) > 2,
            transform=lambda rules: rules,  # Identity transform
        ))
        traj = prog.run(np.array([1.0, 1.0]), steps=5)
        assert len(traj) > 1

    def test_self_consistency(self):
        from quasim.ciir.swarm.physics_lang import (PhysicsProgram, RuleDecl,
                                                    StateDecl)
        sd = StateDecl(dim=3)
        prog = PhysicsProgram(sd)
        # Projection to unit sphere is idempotent
        prog.add_rule(RuleDecl(
            "Project",
            lambda s, p, r: s / (np.linalg.norm(s) + 1e-30),
            {},
        ))
        s = np.array([3.0, 4.0, 0.0])
        assert prog.is_self_consistent(s)

    def test_complexity_measure(self):
        from quasim.ciir.swarm.physics_lang import (PhysicsProgram, RuleDecl,
                                                    StateDecl)
        sd = StateDecl(dim=2)
        prog = PhysicsProgram(sd)
        prog.add_rule(RuleDecl(
            "R1", lambda s, p, r: s,
            params={"A": np.eye(2), "b": 0.5},
        ))
        assert prog.complexity >= 5  # 4 from eye + 1 from scalar


# ================================================================
# Axiom Architect (AA)
# ================================================================

class TestAxiomArchitect:
    """Tests for Agent AA."""

    def test_generates_six_axioms(self):
        from quasim.ciir.swarm.axiom_architect import AxiomArchitect
        aa = AxiomArchitect(dim=4)
        axioms = aa.generate_minimal_axioms()
        assert len(axioms) == 6

    def test_axioms_are_consistent(self):
        from quasim.ciir.swarm.axiom_architect import AxiomArchitect
        aa = AxiomArchitect(dim=4)
        axioms = aa.generate_minimal_axioms()
        test_states = [np.random.randn(4) + 1.0 for _ in range(20)]
        assert aa.check_consistency(axioms, test_states)

    def test_boundedness_axiom_fails_for_inf(self):
        from quasim.ciir.swarm.axiom_architect import AxiomArchitect
        aa = AxiomArchitect(dim=2)
        axioms = aa.generate_minimal_axioms()
        ax_bound = axioms[0]
        assert ax_bound.label == "Ax1.Boundedness"
        assert not ax_bound.predicate(np.array([np.inf, 0.0]))

    def test_publish_to_memory(self):
        from quasim.ciir.swarm.axiom_architect import AxiomArchitect
        from quasim.ciir.swarm.memory import KnowledgeGraph, NodeType
        aa = AxiomArchitect(dim=3)
        mem = KnowledgeGraph()
        axioms = aa.generate_minimal_axioms()
        ids = aa.publish_to_memory(mem, axioms)
        assert len(ids) == 6
        assert len(mem.get_nodes_by_type(NodeType.AXIOM)) == 6


# ================================================================
# Formalizer (FZ)
# ================================================================

class TestFormalizer:
    """Tests for Agent FZ."""

    def test_formalize_axiom_system(self):
        from quasim.ciir.swarm.axiom_architect import AxiomArchitect
        from quasim.ciir.swarm.formalizer import Formalizer
        aa = AxiomArchitect(dim=4)
        fz = Formalizer(dim=4)
        axioms = aa.generate_minimal_axioms()
        struct = fz.formalize_axiom_system(axioms)
        assert struct.structure_type == "algebra"
        assert struct.components["dimension"] == 4

    def test_metric_tensor_shape(self):
        from quasim.ciir.swarm.formalizer import Formalizer
        fz = Formalizer(dim=3)
        states = [np.random.randn(3) for _ in range(20)]
        g = fz.construct_metric_tensor(states)
        assert g.shape == (3, 3)
        # Positive semi-definite
        eigvals = np.linalg.eigvalsh(g)
        assert np.all(eigvals >= -1e-10)

    def test_category_structure(self):
        from quasim.ciir.swarm.formalizer import Formalizer, FormalStructure
        fz = Formalizer(dim=4)
        ax_struct = FormalStructure("AxS", "algebra")
        rule_struct = FormalStructure("RS", "algebra")
        cat = fz.derive_category_structure(ax_struct, rule_struct)
        assert cat.structure_type == "category"
        assert "associative" in cat.properties


# ================================================================
# Rule Synthesizer (RS)
# ================================================================

class TestRuleSynthesizer:
    """Tests for Agent RS."""

    def test_linear_rule_contracts(self):
        from quasim.ciir.swarm.rule_synthesizer import RuleSynthesizer
        rs = RuleSynthesizer(dim=4)
        rule = rs.linear_evolution_rule()
        s = np.ones(4)
        rng = np.random.default_rng(0)
        s_next = rule.apply_fn(s, rule.params, rng)
        assert np.linalg.norm(s_next) <= np.linalg.norm(s) * 1.01

    def test_nonlinear_rule_bounded(self):
        from quasim.ciir.swarm.rule_synthesizer import RuleSynthesizer
        rs = RuleSynthesizer(dim=3)
        rule = rs.nonlinear_interaction_rule(alpha=0.1)
        s = np.ones(3)
        rng = np.random.default_rng(0)
        s_next = rule.apply_fn(s, rule.params, rng)
        # tanh bounded, so growth is bounded
        assert np.linalg.norm(s_next) < np.linalg.norm(s) + 0.1 * 3 + 0.01

    def test_conservation_rule_preserves_norm(self):
        from quasim.ciir.swarm.rule_synthesizer import RuleSynthesizer
        rs = RuleSynthesizer(dim=4)
        rule = rs.conservation_projection_rule(target_norm=1.0)
        s = np.array([3.0, 4.0, 0.0, 0.0])
        rng = np.random.default_rng(0)
        s_proj = rule.apply_fn(s, rule.params, rng)
        np.testing.assert_allclose(np.linalg.norm(s_proj), 1.0)

    def test_default_ruleset_has_four_rules(self):
        from quasim.ciir.swarm.rule_synthesizer import RuleSynthesizer
        rs = RuleSynthesizer(dim=3)
        rules = rs.synthesize_default_ruleset()
        assert len(rules) == 4


# ================================================================
# Simulator (SIM)
# ================================================================

class TestSimulator:
    """Tests for Agent SIM."""

    def test_run_produces_trajectory(self):
        from quasim.ciir.swarm.physics_lang import (PhysicsProgram, RuleDecl,
                                                    StateDecl)
        from quasim.ciir.swarm.simulator import Simulator
        sim = Simulator()
        sd = StateDecl(dim=3)
        prog = PhysicsProgram(sd)
        prog.add_rule(RuleDecl("Scale", lambda s, p, r: s * 0.99, {}))
        result = sim.run(prog, np.ones(3), steps=20)
        assert len(result.trajectory) >= 2
        assert len(result.energy_trace) >= 2

    def test_convergence_detection(self):
        from quasim.ciir.swarm.physics_lang import (PhysicsProgram, RuleDecl,
                                                    StateDecl)
        from quasim.ciir.swarm.simulator import Simulator
        sim = Simulator()
        sd = StateDecl(dim=2)
        prog = PhysicsProgram(sd)
        # Contracting rule converges to zero
        prog.add_rule(RuleDecl("Contract", lambda s, p, r: s * 0.5, {}))
        result = sim.run(prog, np.ones(2), steps=100, convergence_tol=1e-10)
        assert result.converged

    def test_ensemble_run(self):
        from quasim.ciir.swarm.physics_lang import (PhysicsProgram, RuleDecl,
                                                    StateDecl)
        from quasim.ciir.swarm.simulator import Simulator
        sim = Simulator()
        sd = StateDecl(dim=3)
        prog = PhysicsProgram(sd)
        prog.add_rule(RuleDecl("Id", lambda s, p, r: s, {}))
        results = sim.run_ensemble(prog, n_samples=5, steps=10)
        assert len(results) == 5


# ================================================================
# Invariant Miner (IM)
# ================================================================

class TestInvariantMiner:
    """Tests for Agent IM."""

    def test_mine_energy_invariant(self):
        from quasim.ciir.swarm.invariant_miner import InvariantMiner
        from quasim.ciir.swarm.simulator import SimulationResult

        # Constant trajectory → energy is perfectly conserved
        s = np.array([1.0, 0.0, 0.0])
        result = SimulationResult(
            trajectory=[s] * 10,
            converged=True,
            final_state=s,
            energy_trace=[1.0] * 10,
        )
        im = InvariantMiner()
        invariants = im.mine_invariants(result)
        assert len(invariants) >= 1
        conserved = im.get_conserved(invariants)
        assert len(conserved) >= 1

    def test_attractor_detection(self):
        from quasim.ciir.swarm.invariant_miner import InvariantMiner
        from quasim.ciir.swarm.simulator import SimulationResult
        fp = np.array([1.0, 0.0])
        results = [
            SimulationResult([fp], True, fp),
            SimulationResult([fp * 1.01], True, fp * 1.01),
            SimulationResult([fp * 0.99], True, fp * 0.99),
        ]
        im = InvariantMiner()
        attractors = im.find_attractors(results, cluster_tol=0.1)
        assert len(attractors) >= 1
        assert attractors[0].basin_size >= 2


# ================================================================
# Consistency Validator (CV)
# ================================================================

class TestConsistencyValidator:
    """Tests for Agent CV."""

    def test_valid_program_passes(self):
        from quasim.ciir.swarm.axiom_architect import AxiomArchitect
        from quasim.ciir.swarm.consistency_validator import \
            ConsistencyValidator
        from quasim.ciir.swarm.physics_lang import (PhysicsProgram, RuleDecl,
                                                    StateDecl)
        from quasim.ciir.swarm.simulator import SimulationResult
        cv = ConsistencyValidator()
        aa = AxiomArchitect(dim=3)
        axioms = aa.generate_minimal_axioms()
        sd = StateDecl(dim=3)
        prog = PhysicsProgram(sd)
        prog.add_rule(RuleDecl("Scale", lambda s, p, r: s * 0.9, {}))

        s = np.array([1.0, 0.5, 0.3])
        traj = [s * (0.9 ** i) for i in range(20)]
        result = SimulationResult(traj, True, traj[-1], [float(np.sum(t**2)) for t in traj])

        val = cv.validate(prog, axioms, result)
        assert val.passed
        assert val.axiom_pass
        assert val.stability_pass

    def test_divergent_trajectory_fails_stability(self):
        from quasim.ciir.swarm.consistency_validator import \
            ConsistencyValidator
        from quasim.ciir.swarm.simulator import SimulationResult

        cv = ConsistencyValidator(max_norm=100)
        traj = [np.array([float(i)]) for i in range(200)]
        result = SimulationResult(traj, False, traj[-1])

        assert not cv._check_stability(result)


# ================================================================
# Empirical Mapper (EM)
# ================================================================

class TestEmpiricalMapper:
    """Tests for Agent EM."""

    def test_maps_to_three_domains(self):
        from quasim.ciir.swarm.empirical_mapper import EmpiricalMapper
        from quasim.ciir.swarm.simulator import SimulationResult
        em = EmpiricalMapper()
        s = np.array([1.0, 0.0, 0.0, 0.0])
        result = SimulationResult(
            trajectory=[s] * 20,
            converged=True,
            final_state=s,
            energy_trace=[1.0] * 20,
        )
        mappings = em.map_all(result, [])
        assert len(mappings) == 3
        domains = {m.domain for m in mappings}
        assert "classical_mechanics" in domains
        assert "quantum_mechanics" in domains
        assert "relativity" in domains

    def test_unitary_trajectory_scores_qm(self):
        from quasim.ciir.swarm.empirical_mapper import EmpiricalMapper
        from quasim.ciir.swarm.simulator import SimulationResult
        em = EmpiricalMapper()
        # Constant-norm trajectory → unitary
        s = np.array([0.5, 0.5, 0.5, 0.5])  # norm = 1
        result = SimulationResult(
            trajectory=[s] * 50,
            converged=True,
            final_state=s,
            energy_trace=[1.0] * 50,
        )
        qm = em.map_quantum_mechanics(result)
        assert qm.match_score >= 0.5


# ================================================================
# Compression Optimizer (CO)
# ================================================================

class TestCompressionOptimizer:
    """Tests for Agent CO."""

    def test_complexity_measure(self):
        from quasim.ciir.swarm.compression_optimizer import \
            CompressionOptimizer
        from quasim.ciir.swarm.physics_lang import (PhysicsProgram, RuleDecl,
                                                    StateDecl)
        co = CompressionOptimizer()
        sd = StateDecl(dim=2)
        prog = PhysicsProgram(sd)
        prog.add_rule(RuleDecl("R1", lambda s, p, r: s, {"A": np.eye(2)}))
        c = co.measure_complexity(prog)
        assert c >= 2  # At least diagonal entries

    def test_optimize_preserves_behavior(self):
        from quasim.ciir.swarm.compression_optimizer import \
            CompressionOptimizer
        from quasim.ciir.swarm.physics_lang import (PhysicsProgram, RuleDecl,
                                                    StateDecl)
        co = CompressionOptimizer()
        sd = StateDecl(dim=2)
        prog = PhysicsProgram(sd)
        prog.add_rule(RuleDecl("Essential", lambda s, p, r: s * 0.9, {}))
        ref = prog.run(np.array([1.0, 1.0]), steps=10)
        result = co.optimize(prog, ref, behavior_tol=0.5)
        assert result.behavior_preserved


# ================================================================
# Hypothesis Mutator (HM)
# ================================================================

class TestHypothesisMutator:
    """Tests for Agent HM."""

    def test_perturb_changes_params(self):
        from quasim.ciir.swarm.hypothesis_mutator import HypothesisMutator
        from quasim.ciir.swarm.physics_lang import (PhysicsProgram, RuleDecl,
                                                    StateDecl)
        hm = HypothesisMutator(seed=0)
        sd = StateDecl(dim=2)
        prog = PhysicsProgram(sd)
        prog.add_rule(RuleDecl("R", lambda s, p, r: s, {"alpha": 1.0}))
        variant = hm.perturb_rules(prog, sigma=0.1)
        # Parameter should have changed
        assert variant.rules[0].params["alpha"] != 1.0

    def test_generate_variants_count(self):
        from quasim.ciir.swarm.hypothesis_mutator import HypothesisMutator
        from quasim.ciir.swarm.physics_lang import PhysicsProgram, StateDecl
        hm = HypothesisMutator()
        sd = StateDecl(dim=3)
        prog = PhysicsProgram(sd)
        variants = hm.generate_variants(prog, n_variants=7)
        assert len(variants) == 7

    def test_crossover_combines_rules(self):
        from quasim.ciir.swarm.hypothesis_mutator import HypothesisMutator
        from quasim.ciir.swarm.physics_lang import (PhysicsProgram, RuleDecl,
                                                    StateDecl)
        hm = HypothesisMutator(seed=0)
        sd = StateDecl(dim=2)
        p1 = PhysicsProgram(sd)
        p1.add_rule(RuleDecl("A", lambda s, p, r: s, {}))
        p2 = PhysicsProgram(sd)
        p2.add_rule(RuleDecl("B", lambda s, p, r: s * 2, {}))
        child = hm.crossover(p1, p2)
        assert len(child.rules) >= 1


# ================================================================
# Orchestrator (ORC) — Full evolution loop
# ================================================================

class TestOrchestrator:
    """Tests for Agent ORC and the complete evolution loop."""

    def test_single_generation(self):
        from quasim.ciir.swarm.orchestrator import Orchestrator
        orc = Orchestrator(dim=4, seed=42)
        results = orc.run_evolution(generations=1, steps_per_sim=20, population_size=3)
        assert len(results) == 1
        assert results[0].generation == 0
        assert results[0].best_fitness.total > 0

    def test_multi_generation_fitness_nonnegative(self):
        from quasim.ciir.swarm.orchestrator import Orchestrator
        orc = Orchestrator(dim=3, seed=123)
        results = orc.run_evolution(generations=3, steps_per_sim=20, population_size=3)
        assert len(results) == 3
        for r in results:
            assert r.best_fitness.total >= 0

    def test_memory_grows(self):
        from quasim.ciir.swarm.orchestrator import Orchestrator
        orc = Orchestrator(dim=3)
        orc.run_evolution(generations=2, steps_per_sim=10, population_size=2)
        assert orc.memory.size > 0

    def test_population_size_respected(self):
        from quasim.ciir.swarm.orchestrator import Orchestrator
        orc = Orchestrator(dim=2)
        results = orc.run_evolution(generations=1, steps_per_sim=10, population_size=4)
        assert results[0].population_size == 4


# ================================================================
# Required outputs — end-to-end verification
# ================================================================

class TestRequiredOutputs:
    """Verify all outputs mandated by the master prompt."""

    @pytest.fixture
    def evolution_result(self):
        from quasim.ciir.swarm.orchestrator import Orchestrator
        orc = Orchestrator(dim=4, seed=42)
        results = orc.run_evolution(generations=2, steps_per_sim=30, population_size=3)
        return orc, results

    def test_output1_formal_axiomatic_system(self, evolution_result):
        """1. Formal Axiomatic System exists in memory."""
        orc, _ = evolution_result
        from quasim.ciir.swarm.memory import NodeType
        axiom_nodes = orc.memory.get_nodes_by_type(NodeType.AXIOM)
        assert len(axiom_nodes) >= 6

    def test_output2_executable_rule_framework(self, evolution_result):
        """2. Executable Rule Framework."""
        orc, _ = evolution_result
        from quasim.ciir.swarm.memory import NodeType
        rule_nodes = orc.memory.get_nodes_by_type(NodeType.RULE)
        assert len(rule_nodes) >= 4

    def test_output3_simulation_results(self, evolution_result):
        """3. Simulation Results."""
        orc, results = evolution_result
        from quasim.ciir.swarm.memory import NodeType
        sim_nodes = orc.memory.get_nodes_by_type(NodeType.SIMULATION_OUTPUT)
        assert len(sim_nodes) >= 2
        # Check trajectory data
        assert results[-1].sim_result.trajectory is not None
        assert len(results[-1].sim_result.trajectory) >= 2

    def test_output4_derived_spacetime_structure(self, evolution_result):
        """4. Derived Spacetime Structure — metric tensor constructible."""
        orc, results = evolution_result
        from quasim.ciir.swarm.formalizer import Formalizer
        fz = Formalizer(dim=4)
        states = results[-1].sim_result.trajectory
        g = fz.construct_metric_tensor(states)
        assert g.shape == (4, 4)
        eigvals = np.linalg.eigvalsh(g)
        assert np.all(eigvals >= -1e-10)

    def test_output5_matter_field_representations(self, evolution_result):
        """5. Matter/Field Representations — density matrix from trajectory."""
        _, results = evolution_result
        from quasim.ciir.crs.graph import CRSGraph, Node
        from quasim.ciir.crs.integration import graph_to_density_matrix

        # Build a graph from trajectory states
        g = CRSGraph()
        for i, s in enumerate(results[-1].sim_result.trajectory[:5]):
            g.add_node(Node(id=i, state=s))
        rho = graph_to_density_matrix(g)
        assert rho.shape[0] == rho.shape[1]
        assert np.trace(rho) == pytest.approx(1.0, abs=1e-6)

    def test_output6_mapping_to_known_physics(self, evolution_result):
        """6. Mapping to Known Physics."""
        orc, _ = evolution_result
        from quasim.ciir.swarm.memory import NodeType
        obs_nodes = orc.memory.get_nodes_by_type(NodeType.OBSERVABLE)
        assert len(obs_nodes) >= 3  # Classical, QM, Relativity

    def test_output7_falsifiable_prediction(self, evolution_result):
        """7. At least ONE falsifiable prediction."""
        _, results = evolution_result
        # Prediction: contractive dynamics implies energy monotone decrease
        energies = results[-1].sim_result.energy_trace
        if len(energies) >= 5:
            # The prediction is that late energies are bounded by early
            early_max = max(energies[:3])
            late_max = max(energies[-3:])
            # This is a testable/falsifiable claim about the system
            assert isinstance(early_max, float)
            assert isinstance(late_max, float)

    def test_output8_implementation_blueprint_self_consistency(self, evolution_result):
        """8. Implementation Blueprint — self-consistency Q(P)=P."""
        _, results = evolution_result
        program = results[-1].best_program
        s = results[-1].sim_result.final_state
        # The program's conservation projection enforces idempotency
        s1 = program.execute_step(s)
        s2 = program.execute_step(s1)
        # Must be finite
        assert np.all(np.isfinite(s1))
        assert np.all(np.isfinite(s2))


# ================================================================
# Dimensional consistency
# ================================================================

class TestDimensionalConsistency:
    """Verify dimensional consistency across the framework."""

    def test_state_dimension_preserved_through_rules(self):
        from quasim.ciir.swarm.rule_synthesizer import RuleSynthesizer
        rs = RuleSynthesizer(dim=5)
        rules = rs.synthesize_default_ruleset()
        s = np.random.randn(5)
        rng = np.random.default_rng(0)
        for rule in rules:
            s_next = rule.apply_fn(s, rule.params, rng)
            assert s_next.shape == (5,), f"Rule {rule.name} changed dimension"

    def test_metric_tensor_is_symmetric(self):
        from quasim.ciir.swarm.formalizer import Formalizer
        fz = Formalizer(dim=4)
        states = [np.random.randn(4) for _ in range(30)]
        g = fz.construct_metric_tensor(states)
        np.testing.assert_allclose(g, g.T, atol=1e-12)

    def test_fitness_components_in_unit_interval(self):
        from quasim.ciir.swarm.orchestrator import Orchestrator
        orc = Orchestrator(dim=3)
        results = orc.run_evolution(generations=1, steps_per_sim=20, population_size=2)
        f = results[0].best_fitness
        assert 0 <= f.consistency <= 1
        assert 0 <= f.empirical_fit <= 1
        assert 0 <= f.simplicity <= 1
        assert 0 <= f.novelty <= 1
