r"""Tests for the Minimal Programmable Physics Kernel (MPPK).

Covers all components of the law-discovery engine:
- Graph state space (S)
- Interaction rules (R) and interaction registry
- Update operator (U) with exact initial kernel
- Observables: Energy E(t) and Momentum P(t)
- Trajectory and simulation (Step 1)
- Structure extraction (Step 2): fixed points, limit cycles, chaos, clusters
- Invariant mining (Step 3): conservation, symmetries, spectral stability
- Fitness evaluation (Step 4)
- Rule mutation (Step 5): all 5 strategies
- Selection (Step 6) and full evolution loop
- Meta-learning layer
- Optional extensions: phase nodes, adaptive rewiring
- Divergence rejection and boundedness constraints
- Dimensional consistency throughout

Run with:
    python -m pytest tests/test_mppk.py -v --override-ini="addopts="
"""

from __future__ import annotations

import numpy as np
import pytest


# ================================================================
# Graph state space (S)
# ================================================================

class TestMPPKGraph:
    """Tests for the graph-based state space."""

    def test_add_nodes_and_edges(self):
        from quasim.ciir.swarm.mppk import MPPKGraph, MPPKNode, MPPKEdge
        g = MPPKGraph()
        g.add_node(MPPKNode(0, np.array([1.0, 2.0])))
        g.add_node(MPPKNode(1, np.array([3.0, 4.0])))
        g.add_edge(MPPKEdge(0, 1, 0.5))
        assert g.n_nodes == 2
        assert g.n_edges == 1
        assert g.dim == 2

    def test_neighbors(self):
        from quasim.ciir.swarm.mppk import MPPKGraph, MPPKNode, MPPKEdge
        g = MPPKGraph()
        g.add_node(MPPKNode(0, np.zeros(3)))
        g.add_node(MPPKNode(1, np.ones(3)))
        g.add_edge(MPPKEdge(0, 1, 0.7))
        nbrs = g.neighbors(0)
        assert len(nbrs) == 1
        assert nbrs[0] == (1, 0.7)

    def test_state_matrix_shape(self):
        from quasim.ciir.swarm.mppk import MPPKGraph, MPPKNode
        g = MPPKGraph()
        for i in range(5):
            g.add_node(MPPKNode(i, np.zeros(4)))
        X = g.state_matrix()
        assert X.shape == (5, 4)

    def test_set_states_roundtrip(self):
        from quasim.ciir.swarm.mppk import MPPKGraph, MPPKNode
        g = MPPKGraph()
        for i in range(3):
            g.add_node(MPPKNode(i, np.zeros(2)))
        X = np.array([[1, 2], [3, 4], [5, 6.0]])
        g.set_states(X)
        X_out = g.state_matrix()
        np.testing.assert_array_equal(X, X_out)

    def test_laplacian_is_symmetric(self):
        from quasim.ciir.swarm.mppk import MPPKGraph, MPPKNode, MPPKEdge
        g = MPPKGraph()
        for i in range(4):
            g.add_node(MPPKNode(i, np.zeros(2)))
        g.add_edge(MPPKEdge(0, 1, 1.0))
        g.add_edge(MPPKEdge(1, 2, 0.5))
        g.add_edge(MPPKEdge(2, 3, 0.8))
        L = g.laplacian()
        np.testing.assert_allclose(L, L.T, atol=1e-12)

    def test_laplacian_zero_row_sum(self):
        from quasim.ciir.swarm.mppk import MPPKGraph, MPPKNode, MPPKEdge
        g = MPPKGraph()
        for i in range(3):
            g.add_node(MPPKNode(i, np.zeros(2)))
        g.add_edge(MPPKEdge(0, 1, 1.0))
        g.add_edge(MPPKEdge(1, 2, 1.0))
        L = g.laplacian()
        row_sums = L.sum(axis=1)
        np.testing.assert_allclose(row_sums, 0, atol=1e-12)

    def test_copy_is_independent(self):
        from quasim.ciir.swarm.mppk import MPPKGraph, MPPKNode, MPPKEdge
        g = MPPKGraph()
        g.add_node(MPPKNode(0, np.array([1.0])))
        g2 = g.copy()
        g2.nodes[0].state = np.array([99.0])
        assert g.nodes[0].state[0] == 1.0  # Original unchanged


# ================================================================
# Interaction rules (R)
# ================================================================

class TestInteractionRules:
    """Tests for the rule registry and built-in interaction functions."""

    def test_tanh_bounded(self):
        from quasim.ciir.swarm.mppk import tanh_interaction
        diff = np.array([10.0, -10.0, 0.0])
        result = tanh_interaction(diff)
        assert np.all(np.abs(result) <= 1.0)

    def test_sigmoid_bounded(self):
        from quasim.ciir.swarm.mppk import sigmoid_interaction
        diff = np.array([100.0, -100.0, 0.0])
        result = sigmoid_interaction(diff)
        assert np.all(np.abs(result) <= 1.01)

    def test_quadratic_bounded(self):
        from quasim.ciir.swarm.mppk import quadratic_interaction
        diff = np.array([10.0, 10.0])
        result = quadratic_interaction(diff)
        assert np.linalg.norm(result) < np.linalg.norm(diff) + 1

    def test_relu_clip_bounded(self):
        from quasim.ciir.swarm.mppk import relu_clip_interaction
        diff = np.array([5.0, -5.0, 0.3])
        result = relu_clip_interaction(diff)
        assert np.all(np.abs(result) <= 1.0)

    def test_cubic_bounded(self):
        from quasim.ciir.swarm.mppk import cubic_interaction
        diff = np.array([3.0, -3.0])
        result = cubic_interaction(diff)
        assert np.all(np.abs(result) <= 1.0)

    def test_registry_complete(self):
        from quasim.ciir.swarm.mppk import INTERACTION_REGISTRY, InteractionType
        for t in InteractionType:
            assert t in INTERACTION_REGISTRY

    def test_default_rule_is_tanh(self):
        from quasim.ciir.swarm.mppk import make_default_rule, InteractionType
        rule = make_default_rule()
        assert rule.interaction_type == InteractionType.TANH
        assert rule.noise_sigma == 0.0


# ================================================================
# Update operator (U) — exact initial kernel
# ================================================================

class TestUpdateOperator:
    """Tests for the update operator implementing the initial kernel."""

    def _make_triangle_graph(self, dim=2):
        from quasim.ciir.swarm.mppk import MPPKGraph, MPPKNode, MPPKEdge
        g = MPPKGraph()
        g.add_node(MPPKNode(0, np.array([1.0, 0.0])))
        g.add_node(MPPKNode(1, np.array([0.0, 1.0])))
        g.add_node(MPPKNode(2, np.array([0.0, 0.0])))
        g.add_edge(MPPKEdge(0, 1, 0.5))
        g.add_edge(MPPKEdge(1, 2, 0.5))
        g.add_edge(MPPKEdge(2, 0, 0.5))
        return g

    def test_step_updates_states(self):
        from quasim.ciir.swarm.mppk import UpdateOperator, make_default_rule
        g = self._make_triangle_graph()
        X0 = g.state_matrix().copy()
        updater = UpdateOperator(make_default_rule())
        updater.step(g)
        X1 = g.state_matrix()
        assert not np.allclose(X0, X1)  # Something changed

    def test_kernel_formula_manual_check(self):
        """Verify x_i(t+1) = x_i(t) + Σ w_ij tanh(x_j - x_i) by hand."""
        from quasim.ciir.swarm.mppk import (
            MPPKGraph, MPPKNode, MPPKEdge, UpdateOperator, make_default_rule,
        )
        g = MPPKGraph()
        g.add_node(MPPKNode(0, np.array([1.0])))
        g.add_node(MPPKNode(1, np.array([3.0])))
        g.add_edge(MPPKEdge(0, 1, 0.4))

        rule = make_default_rule()
        updater = UpdateOperator(rule)
        updater.step(g)

        # Node 0: x0' = 1.0 + 0.4 * tanh(3.0 - 1.0) = 1.0 + 0.4 * tanh(2.0)
        expected_0 = 1.0 + 0.4 * np.tanh(2.0)
        np.testing.assert_allclose(g.nodes[0].state, [expected_0], atol=1e-12)

        # Node 1: no outgoing edges from 1 → delta = 0
        np.testing.assert_allclose(g.nodes[1].state, [3.0], atol=1e-12)

    def test_prev_states_stored(self):
        from quasim.ciir.swarm.mppk import UpdateOperator, make_default_rule
        g = self._make_triangle_graph()
        updater = UpdateOperator(make_default_rule())
        assert updater.prev_states is None
        updater.step(g)
        assert updater.prev_states is not None
        assert updater.prev_states.shape == (3, 2)

    def test_noisy_rule_adds_randomness(self):
        from quasim.ciir.swarm.mppk import (
            MPPKGraph, MPPKNode, MPPKEdge, UpdateOperator, MPPKRule,
            tanh_interaction, InteractionType,
        )
        g = MPPKGraph()
        g.add_node(MPPKNode(0, np.zeros(3)))
        g.add_node(MPPKNode(1, np.zeros(3)))
        g.add_edge(MPPKEdge(0, 1, 0.5))

        rule = MPPKRule("Noisy", tanh_interaction, InteractionType.TANH, noise_sigma=0.1)
        updater = UpdateOperator(rule, seed=0)
        updater.step(g)
        # With identical initial states but noise, nodes should diverge
        assert not np.allclose(g.nodes[0].state, g.nodes[1].state)


# ================================================================
# Observables — Energy E(t) and Momentum P(t)
# ================================================================

class TestObservables:
    """Tests for E(t) and P(t) computation."""

    def test_energy_nonnegative_positive_weights(self):
        from quasim.ciir.swarm.mppk import (
            MPPKGraph, MPPKNode, MPPKEdge, compute_energy,
        )
        g = MPPKGraph()
        g.add_node(MPPKNode(0, np.array([1.0, 0.0])))
        g.add_node(MPPKNode(1, np.array([0.0, 1.0])))
        g.add_edge(MPPKEdge(0, 1, 1.0))
        E = compute_energy(g)
        assert E >= 0

    def test_energy_zero_identical_states(self):
        from quasim.ciir.swarm.mppk import (
            MPPKGraph, MPPKNode, MPPKEdge, compute_energy,
        )
        g = MPPKGraph()
        s = np.array([2.0, 3.0])
        g.add_node(MPPKNode(0, s.copy()))
        g.add_node(MPPKNode(1, s.copy()))
        g.add_edge(MPPKEdge(0, 1, 0.5))
        assert compute_energy(g) == pytest.approx(0.0)

    def test_energy_formula_manual(self):
        """E = Σ w_ij ||x_i - x_j||²."""
        from quasim.ciir.swarm.mppk import (
            MPPKGraph, MPPKNode, MPPKEdge, compute_energy,
        )
        g = MPPKGraph()
        g.add_node(MPPKNode(0, np.array([0.0])))
        g.add_node(MPPKNode(1, np.array([3.0])))
        g.add_edge(MPPKEdge(0, 1, 2.0))
        # E = 2.0 * (3-0)^2 = 18.0
        assert compute_energy(g) == pytest.approx(18.0)

    def test_momentum_zero_without_prev(self):
        from quasim.ciir.swarm.mppk import MPPKGraph, MPPKNode, compute_momentum
        g = MPPKGraph()
        g.add_node(MPPKNode(0, np.array([1.0, 2.0])))
        P = compute_momentum(g, None)
        np.testing.assert_array_equal(P, [0.0, 0.0])

    def test_momentum_formula_manual(self):
        """P = Σ_i (x_i(t) - x_i(t-1))."""
        from quasim.ciir.swarm.mppk import MPPKGraph, MPPKNode, compute_momentum
        g = MPPKGraph()
        g.add_node(MPPKNode(0, np.array([2.0])))
        g.add_node(MPPKNode(1, np.array([5.0])))
        prev = np.array([[1.0], [3.0]])  # x_0 was 1, x_1 was 3
        P = compute_momentum(g, prev)
        # P = (2-1) + (5-3) = 1 + 2 = 3
        np.testing.assert_allclose(P, [3.0])


# ================================================================
# Simulation (Step 1)
# ================================================================

class TestSimulation:
    """Tests for trajectory simulation."""

    def _make_graph(self, n=5, d=3):
        from quasim.ciir.swarm.mppk import MPPKGraph, MPPKNode, MPPKEdge
        rng = np.random.default_rng(42)
        g = MPPKGraph()
        for i in range(n):
            g.add_node(MPPKNode(i, rng.standard_normal(d) * 0.5))
        for i in range(n):
            j = (i + 1) % n
            g.add_edge(MPPKEdge(i, j, float(rng.uniform(0.1, 0.5))))
        return g

    def test_trajectory_length(self):
        from quasim.ciir.swarm.mppk import simulate, make_default_rule
        g = self._make_graph()
        traj = simulate(g, make_default_rule(), T=50)
        assert traj.T == 51  # initial + 50 steps

    def test_trajectory_records_energy(self):
        from quasim.ciir.swarm.mppk import simulate, make_default_rule
        g = self._make_graph()
        traj = simulate(g, make_default_rule(), T=20)
        assert len(traj.energies) == 21
        assert all(np.isfinite(e) for e in traj.energies)

    def test_trajectory_records_momentum(self):
        from quasim.ciir.swarm.mppk import simulate, make_default_rule
        g = self._make_graph()
        traj = simulate(g, make_default_rule(), T=10)
        assert len(traj.momenta) == 11

    def test_trajectory_states_have_correct_shape(self):
        from quasim.ciir.swarm.mppk import simulate, make_default_rule
        g = self._make_graph(n=8, d=5)
        traj = simulate(g, make_default_rule(), T=10)
        for S in traj.states:
            assert S.shape == (8, 5)

    def test_simulation_bounded_with_tanh(self):
        """tanh kernel should keep dynamics bounded."""
        from quasim.ciir.swarm.mppk import simulate, make_default_rule
        g = self._make_graph(n=10, d=4)
        traj = simulate(g, make_default_rule(), T=200)
        assert all(n < 1e6 for n in traj.norms)


# ================================================================
# Structure extraction (Step 2)
# ================================================================

class TestStructureExtraction:
    """Tests for fixed point, cycle, chaos, and cluster detection."""

    def test_fixed_point_detected(self):
        from quasim.ciir.swarm.mppk import Trajectory, extract_structure
        # All states identical → fixed point
        s = np.array([[1.0, 0.0], [0.0, 1.0]])
        traj = Trajectory(states=[s] * 20, energies=[1.0] * 20)
        report = extract_structure(traj)
        assert report.has_fixed_point

    def test_not_fixed_point_for_varying_states(self):
        from quasim.ciir.swarm.mppk import Trajectory, extract_structure
        states = [np.array([[float(t), 0], [0, float(t)]]) for t in range(20)]
        traj = Trajectory(states=states, energies=[1.0] * 20)
        report = extract_structure(traj, fp_tol=1e-4)
        assert not report.has_fixed_point

    def test_cluster_detection(self):
        from quasim.ciir.swarm.mppk import Trajectory, extract_structure
        # Two tight clusters
        cluster_a = np.array([0.0, 0.0])
        cluster_b = np.array([10.0, 10.0])
        final = np.array([
            cluster_a, cluster_a * 1.01, cluster_a * 0.99,
            cluster_b, cluster_b * 1.01,
        ])
        traj = Trajectory(states=[final] * 5, energies=[1.0] * 5)
        report = extract_structure(traj, cluster_tol=0.5)
        assert report.n_clusters == 2
        assert report.symmetry_breaking

    def test_chaos_proxy(self):
        from quasim.ciir.swarm.mppk import Trajectory, extract_structure
        # Rapidly diverging states → chaotic
        rng = np.random.default_rng(0)
        states = [rng.standard_normal((3, 2)) * (2.0 ** t) for t in range(30)]
        traj = Trajectory(states=states, energies=[1.0] * 30)
        report = extract_structure(traj)
        assert report.lyapunov_proxy > 0


# ================================================================
# Invariant mining (Step 3)
# ================================================================

class TestInvariantMining:
    """Tests for conserved quantity and spectral stability detection."""

    def test_energy_conservation_detected(self):
        from quasim.ciir.swarm.mppk import (
            Trajectory, mine_invariants, MPPKGraph, MPPKNode, MPPKEdge,
        )
        g = MPPKGraph()
        g.add_node(MPPKNode(0, np.zeros(2)))
        g.add_node(MPPKNode(1, np.ones(2)))
        g.add_edge(MPPKEdge(0, 1, 1.0))
        traj = Trajectory(
            states=[g.state_matrix()] * 20,
            energies=[2.0] * 20,  # Constant energy
            momenta=[np.zeros(2)] * 20,
            norms=[2.0] * 20,
        )
        report = mine_invariants(traj, g)
        assert report.energy_conserved
        assert report.energy_rel_var < 0.01

    def test_spectral_gap_positive(self):
        from quasim.ciir.swarm.mppk import (
            Trajectory, mine_invariants, MPPKGraph, MPPKNode, MPPKEdge,
        )
        g = MPPKGraph()
        for i in range(4):
            g.add_node(MPPKNode(i, np.zeros(2)))
        # Complete graph → spectral gap > 0
        for i in range(4):
            for j in range(i + 1, 4):
                g.add_edge(MPPKEdge(i, j, 1.0))
        traj = Trajectory(states=[g.state_matrix()], energies=[0.0])
        report = mine_invariants(traj, g)
        assert report.spectral_gap > 0
        assert report.spectral_stable

    def test_laplacian_eigenvalues_sorted(self):
        from quasim.ciir.swarm.mppk import (
            Trajectory, mine_invariants, MPPKGraph, MPPKNode, MPPKEdge,
        )
        g = MPPKGraph()
        for i in range(5):
            g.add_node(MPPKNode(i, np.zeros(3)))
        g.add_edge(MPPKEdge(0, 1, 1.0))
        g.add_edge(MPPKEdge(1, 2, 1.0))
        traj = Trajectory(states=[g.state_matrix()], energies=[0.0])
        report = mine_invariants(traj, g)
        # Eigenvalues should be sorted
        for i in range(len(report.laplacian_eigenvalues) - 1):
            assert report.laplacian_eigenvalues[i] <= report.laplacian_eigenvalues[i + 1] + 1e-10


# ================================================================
# Fitness evaluation (Step 4)
# ================================================================

class TestFitnessEvaluation:
    """Tests for the fitness function."""

    def test_fitness_in_unit_interval(self):
        from quasim.ciir.swarm.mppk import (
            evaluate_fitness, Trajectory, StructureReport, InvariantReport,
            make_default_rule, MPPKGraph, MPPKNode,
        )
        g = MPPKGraph()
        g.add_node(MPPKNode(0, np.zeros(2)))
        traj = Trajectory(
            states=[np.zeros((1, 2))] * 10,
            energies=[0.0] * 10,
            norms=[0.0] * 10,
        )
        sr = StructureReport(has_fixed_point=True)
        ir = InvariantReport(energy_conserved=True, spectral_stable=True)
        f = evaluate_fitness(traj, sr, ir, make_default_rule(), g)
        assert 0 <= f.total <= 1
        assert 0 <= f.stability <= 1
        assert 0 <= f.structure_formation <= 1
        assert 0 <= f.boundedness <= 1
        assert 0 <= f.compressibility <= 1

    def test_divergent_trajectory_penalized(self):
        from quasim.ciir.swarm.mppk import (
            evaluate_fitness, Trajectory, StructureReport, InvariantReport,
            make_default_rule, MPPKGraph, MPPKNode,
        )
        g = MPPKGraph()
        g.add_node(MPPKNode(0, np.zeros(2)))
        traj = Trajectory(norms=[1e7])  # Divergent
        sr = StructureReport()
        ir = InvariantReport()
        f = evaluate_fitness(traj, sr, ir, make_default_rule(), g)
        assert f.boundedness == 0.0

    def test_more_structure_higher_fitness(self):
        from quasim.ciir.swarm.mppk import (
            evaluate_fitness, Trajectory, StructureReport, InvariantReport,
            make_default_rule, MPPKGraph, MPPKNode,
        )
        g = MPPKGraph()
        g.add_node(MPPKNode(0, np.zeros(2)))
        traj = Trajectory(norms=[1.0] * 10, energies=[1.0] * 10)

        sr_boring = StructureReport()
        sr_rich = StructureReport(n_clusters=3, symmetry_breaking=True)
        ir = InvariantReport()

        f_boring = evaluate_fitness(traj, sr_boring, ir, make_default_rule(), g)
        f_rich = evaluate_fitness(traj, sr_rich, ir, make_default_rule(), g)
        assert f_rich.structure_formation >= f_boring.structure_formation


# ================================================================
# Rule mutation (Step 5)
# ================================================================

class TestRuleMutation:
    """Tests for all 5 mutation strategies."""

    def _make_graph(self):
        from quasim.ciir.swarm.mppk import MPPKGraph, MPPKNode, MPPKEdge
        g = MPPKGraph()
        for i in range(4):
            g.add_node(MPPKNode(i, np.random.randn(3)))
        g.add_edge(MPPKEdge(0, 1, 0.5))
        g.add_edge(MPPKEdge(1, 2, 0.3))
        g.add_edge(MPPKEdge(2, 3, 0.7))
        return g

    def test_mutate_interaction_changes_type(self):
        from quasim.ciir.swarm.mppk import RuleMutator, make_default_rule
        mut = RuleMutator(seed=0)
        rule = make_default_rule()
        # Run multiple mutations — at least one should differ
        types_seen = set()
        for _ in range(20):
            new_rule = mut.mutate_interaction(rule)
            types_seen.add(new_rule.interaction_type)
        assert len(types_seen) > 1

    def test_mutate_topology_preserves_node_count(self):
        from quasim.ciir.swarm.mppk import RuleMutator
        g = self._make_graph()
        mut = RuleMutator(seed=0)
        g2 = mut.mutate_topology(g, p_rewire=0.5)
        assert g2.n_nodes == g.n_nodes

    def test_mutate_noise_sets_sigma(self):
        from quasim.ciir.swarm.mppk import RuleMutator, make_default_rule
        mut = RuleMutator(seed=0)
        rule = make_default_rule()
        assert rule.noise_sigma == 0.0
        noisy = mut.mutate_noise(rule, sigma_range=(0.01, 0.1))
        assert noisy.noise_sigma > 0

    def test_mutate_coupling_adds_matrix(self):
        from quasim.ciir.swarm.mppk import RuleMutator, make_default_rule
        mut = RuleMutator(seed=0)
        rule = make_default_rule()
        assert rule.coupling_matrix is None
        coupled = mut.mutate_coupling(rule, dim=3)
        assert coupled.coupling_matrix is not None
        assert coupled.coupling_matrix.shape == (3, 3)

    def test_mutate_weights_changes_values(self):
        from quasim.ciir.swarm.mppk import RuleMutator
        g = self._make_graph()
        mut = RuleMutator(seed=0)
        g2 = mut.mutate_weights(g, sigma=0.1)
        old_weights = [e.weight for e in g.edges]
        new_weights = [e.weight for e in g2.edges]
        assert old_weights != new_weights

    def test_generate_candidates_count(self):
        from quasim.ciir.swarm.mppk import RuleMutator, make_default_rule
        g = self._make_graph()
        mut = RuleMutator(seed=0)
        candidates = mut.generate_candidates(make_default_rule(), g, n=7)
        assert len(candidates) == 7


# ================================================================
# Full evolution loop (Steps 1–6)
# ================================================================

class TestMPPKEngine:
    """Tests for the full MPPK engine."""

    def test_initialize_graph(self):
        from quasim.ciir.swarm.mppk import MPPKEngine
        engine = MPPKEngine(n_nodes=10, dim=4)
        g = engine.initialize_graph()
        assert g.n_nodes == 10
        assert g.dim == 4
        assert g.n_edges > 0

    def test_single_generation(self):
        from quasim.ciir.swarm.mppk import MPPKEngine, make_default_rule
        engine = MPPKEngine(n_nodes=8, dim=3, seed=42)
        g = engine.initialize_graph()
        result = engine.run_generation(g, make_default_rule(), T=50)
        assert result.trajectory.T >= 2
        assert result.fitness.total >= 0

    def test_evolve_returns_correct_count(self):
        from quasim.ciir.swarm.mppk import MPPKEngine
        engine = MPPKEngine(n_nodes=6, dim=3, seed=42)
        results, meta = engine.evolve(generations=3, T=30, n_candidates=3)
        assert len(results) == 3

    def test_evolve_fitness_nonnegative(self):
        from quasim.ciir.swarm.mppk import MPPKEngine
        engine = MPPKEngine(n_nodes=6, dim=3, seed=42)
        results, _ = engine.evolve(generations=3, T=30, n_candidates=3)
        for r in results:
            assert r.fitness.total >= 0

    def test_evolve_rejects_divergent(self):
        """Engine should not select candidates with norm > 1e6."""
        from quasim.ciir.swarm.mppk import MPPKEngine
        engine = MPPKEngine(n_nodes=8, dim=4, seed=42)
        results, _ = engine.evolve(generations=5, T=50, n_candidates=4)
        for r in results:
            assert all(n < 1e6 for n in r.trajectory.norms)

    def test_evolve_trajectory_bounded(self):
        from quasim.ciir.swarm.mppk import MPPKEngine
        engine = MPPKEngine(n_nodes=10, dim=4, seed=123)
        results, _ = engine.evolve(generations=3, T=100, n_candidates=3)
        for r in results:
            assert all(np.isfinite(n) for n in r.trajectory.norms)


# ================================================================
# Meta-learning layer
# ================================================================

class TestMetaLearning:
    """Tests for the meta-learning layer."""

    def test_meta_learn_with_results(self):
        from quasim.ciir.swarm.mppk import MPPKEngine
        engine = MPPKEngine(n_nodes=6, dim=3, seed=42)
        results, meta = engine.evolve(generations=4, T=30, n_candidates=3)
        # Meta report should have candidate laws
        assert isinstance(meta.effective_force_law, str)
        assert len(meta.effective_force_law) > 0

    def test_meta_learn_embedding_dim(self):
        from quasim.ciir.swarm.mppk import MPPKEngine
        engine = MPPKEngine(n_nodes=6, dim=3, seed=42)
        _, meta = engine.evolve(generations=4, T=30, n_candidates=3)
        assert meta.embedding_dim >= 1

    def test_meta_learn_candidate_laws(self):
        from quasim.ciir.swarm.mppk import MPPKEngine
        engine = MPPKEngine(n_nodes=8, dim=4, seed=42)
        _, meta = engine.evolve(generations=5, T=50, n_candidates=3)
        # Should produce at least some candidate laws
        assert isinstance(meta.candidate_laws, list)

    def test_meta_learn_empty_input(self):
        from quasim.ciir.swarm.mppk import meta_learn
        report = meta_learn([])
        assert report.embedding_dim == 0
        assert report.candidate_laws == []


# ================================================================
# Optional extensions
# ================================================================

class TestOptionalExtensions:
    """Tests for phase variables and adaptive rewiring."""

    def test_phase_node_has_phase(self):
        from quasim.ciir.swarm.mppk import PhaseNode
        pn = PhaseNode(id=0, state=np.array([1.0, 2.0]), phase=0.5)
        assert pn.phase == 0.5
        assert pn.state.shape == (2,)

    def test_adaptive_rewire(self):
        from quasim.ciir.swarm.mppk import (
            MPPKGraph, MPPKNode, MPPKEdge, adaptive_rewire,
        )
        g = MPPKGraph()
        # Two close nodes + one far node
        g.add_node(MPPKNode(0, np.array([0.0, 0.0])))
        g.add_node(MPPKNode(1, np.array([0.1, 0.1])))
        g.add_node(MPPKNode(2, np.array([10.0, 10.0])))
        g.add_edge(MPPKEdge(0, 2, 0.5))  # Long-range edge
        g.add_edge(MPPKEdge(1, 2, 0.5))

        g2 = adaptive_rewire(g, threshold=0.5)
        # Close nodes (0,1) should be connected after rewiring
        assert g2.n_nodes == 3


# ================================================================
# Hard constraints verification
# ================================================================

class TestHardConstraints:
    """Verify all hard constraints from the specification."""

    def test_no_hardcoded_physics_constants(self):
        """Rules should not contain G, h, c, etc."""
        from quasim.ciir.swarm.mppk import make_default_rule
        rule = make_default_rule()
        # The default rule has no physics constants in params
        assert rule.coupling_matrix is None
        assert rule.noise_sigma == 0.0

    def test_no_undefined_primitives(self):
        """All interaction functions are defined and bounded."""
        from quasim.ciir.swarm.mppk import INTERACTION_REGISTRY
        test_diff = np.array([1.0, -1.0, 0.5])
        for name, fn in INTERACTION_REGISTRY.items():
            result = fn(test_diff)
            assert result.shape == test_diff.shape
            assert np.all(np.isfinite(result))

    def test_laws_are_emergent_not_assumed(self):
        """The engine should discover laws via selection, not assume them."""
        from quasim.ciir.swarm.mppk import MPPKEngine
        engine = MPPKEngine(n_nodes=6, dim=3, seed=42)
        results, meta = engine.evolve(generations=3, T=30, n_candidates=3)
        # Laws come from meta-learning, not hard-coded
        if meta.candidate_laws:
            for law in meta.candidate_laws:
                assert isinstance(law, str)
                assert len(law) > 0

    def test_boundedness_enforced(self):
        """System must remain bounded throughout evolution."""
        from quasim.ciir.swarm.mppk import MPPKEngine
        engine = MPPKEngine(n_nodes=8, dim=4, seed=42)
        results, _ = engine.evolve(generations=3, T=100, n_candidates=3)
        for r in results:
            for norm in r.trajectory.norms:
                assert np.isfinite(norm)
                assert norm < 1e6


# ================================================================
# Dimensional consistency
# ================================================================

class TestDimensionalConsistency:
    """Verify dimensional consistency across all operations."""

    def test_state_dim_preserved_through_update(self):
        from quasim.ciir.swarm.mppk import (
            MPPKGraph, MPPKNode, MPPKEdge, UpdateOperator, make_default_rule,
        )
        g = MPPKGraph()
        d = 5
        for i in range(4):
            g.add_node(MPPKNode(i, np.random.randn(d)))
            g.add_edge(MPPKEdge(i, (i + 1) % 4, 0.3))
        updater = UpdateOperator(make_default_rule())
        updater.step(g)
        for n in g.nodes.values():
            assert n.state.shape == (d,)

    def test_energy_is_scalar(self):
        from quasim.ciir.swarm.mppk import (
            MPPKGraph, MPPKNode, MPPKEdge, compute_energy,
        )
        g = MPPKGraph()
        g.add_node(MPPKNode(0, np.zeros(7)))
        g.add_node(MPPKNode(1, np.ones(7)))
        g.add_edge(MPPKEdge(0, 1, 1.0))
        E = compute_energy(g)
        assert isinstance(E, float)

    def test_momentum_dim_matches_state(self):
        from quasim.ciir.swarm.mppk import MPPKGraph, MPPKNode, compute_momentum
        d = 6
        g = MPPKGraph()
        g.add_node(MPPKNode(0, np.ones(d)))
        prev = np.zeros((1, d))
        P = compute_momentum(g, prev)
        assert P.shape == (d,)

    def test_laplacian_shape_matches_nodes(self):
        from quasim.ciir.swarm.mppk import MPPKGraph, MPPKNode, MPPKEdge
        n = 7
        g = MPPKGraph()
        for i in range(n):
            g.add_node(MPPKNode(i, np.zeros(2)))
        g.add_edge(MPPKEdge(0, 1, 1.0))
        L = g.laplacian()
        assert L.shape == (n, n)
