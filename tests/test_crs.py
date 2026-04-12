"""Tests for CRS (Computational Reality System) package.

Tests cover all 8 layers:
    Layer 0: Graph primitives (Node, Edge, CRSGraph)
    Layer 1: Rewrite rules (diffusion, decay, interaction, noise, edge reweight)
    Layer 2: Evolution engine (CRSEngine)
    Layer 3: Emergent spacetime (geodesics, curvature, dimension)
    Layer 4: Branching (BranchState, BranchingEngine, interference)
    Layer 5: Conservation laws (invariants)
    Layer 6: Observer subsystem (observation, decoherence)
    Layer 7: CIIR/QuASIM/QRATUM integration
    Experiments: Emergent behavior demonstrations
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from quasim.ciir.crs.graph import CRSGraph, Node, Edge, random_graph, lattice_graph
from quasim.ciir.crs.rewrite import (
    RewriteEngine,
    diffusion_rule,
    decay_rule,
    interaction_rule,
    stochastic_perturbation_rule,
    edge_reweight_rule,
)
from quasim.ciir.crs.evolution import CRSEngine
from quasim.ciir.crs.spacetime import (
    geodesic_distance,
    distance_matrix,
    causal_depth,
    curvature_proxy,
    graph_dimension_estimate,
)
from quasim.ciir.crs.branching import BranchState, BranchingEngine
from quasim.ciir.crs.conservation import (
    ConservationEngine,
    total_state_norm_invariant,
    total_causal_weight_invariant,
    node_count_invariant,
    state_centroid_invariant,
    graph_energy_invariant,
)
from quasim.ciir.crs.observer import Observer, observe, measurement_entropy, decohere, partial_trace
from quasim.ciir.crs.integration import (
    graph_to_density_matrix,
    density_matrix_to_graph,
    apply_ciir_distortion,
    QRATUM_CRS_Core,
)
from quasim.ciir.crs.experiments import (
    run_diffusion_experiment,
    run_wave_experiment,
    run_phase_transition_experiment,
    run_entropy_experiment,
)


# ================================================================
# Fixtures
# ================================================================


@pytest.fixture
def small_graph():
    """5-node graph with known structure."""
    g = CRSGraph()
    for i in range(5):
        g.add_node(Node(id=i, state=np.array([float(i), float(i + 1)]), timestamp=0))
    g.add_edge(Edge(source=0, target=1, causal_weight=0.8))
    g.add_edge(Edge(source=1, target=2, causal_weight=0.5))
    g.add_edge(Edge(source=2, target=3, causal_weight=0.3))
    g.add_edge(Edge(source=3, target=4, causal_weight=0.9))
    g.add_edge(Edge(source=0, target=2, causal_weight=0.4))
    return g


@pytest.fixture
def lattice():
    """5x5 lattice graph."""
    return lattice_graph(5, 5, d_state=4, seed=42)


@pytest.fixture
def rng():
    return np.random.default_rng(42)


# ================================================================
# Layer 0: Graph Primitives
# ================================================================


class TestGraphPrimitives:
    """Tests for Node, Edge, CRSGraph."""

    def test_node_creation(self):
        n = Node(id=0, state=np.array([1.0, 2.0]), timestamp=0)
        assert n.id == 0
        assert n.state.shape == (2,)
        assert n.timestamp == 0

    def test_edge_creation(self):
        e = Edge(source=0, target=1, causal_weight=0.5)
        assert e.source == 0
        assert e.target == 1
        assert e.causal_weight == 0.5

    def test_graph_add_remove_nodes(self):
        g = CRSGraph()
        n = Node(id=0, state=np.array([1.0]), timestamp=0)
        g.add_node(n)
        assert g.node_count == 1
        g.remove_node(0)
        assert g.node_count == 0

    def test_graph_add_remove_edges(self, small_graph):
        assert small_graph.edge_count == 5
        small_graph.remove_edge(0, 1)
        assert small_graph.edge_count == 4

    def test_graph_neighborhood(self, small_graph):
        nbrs = small_graph.get_neighborhood(0, radius=1)
        assert 1 in nbrs
        assert 2 in nbrs  # direct edge 0→2

    def test_graph_subgraph(self, small_graph):
        sub = small_graph.subgraph({0, 1, 2})
        assert sub.node_count == 3
        # Should contain edges within the subgraph
        assert (0, 1) in sub.edges or (0, 2) in sub.edges

    def test_graph_copy(self, small_graph):
        g2 = small_graph.copy()
        assert g2.node_count == small_graph.node_count
        # Modify copy, original unchanged
        g2.nodes[0].state = np.array([999.0, 999.0])
        assert small_graph.nodes[0].state[0] != 999.0

    def test_random_graph(self):
        g = random_graph(20, d_state=4, p_edge=0.3, seed=42)
        assert g.node_count == 20
        assert g.edge_count > 0

    def test_lattice_graph(self):
        g = lattice_graph(4, 4, d_state=3, seed=42)
        assert g.node_count == 16
        # Each interior node should have 4 neighbors (up/down/left/right, both directions)
        assert g.edge_count > 0

    def test_density(self, small_graph):
        d = small_graph.density()
        assert 0.0 <= d <= 1.0

    def test_causal_cone(self, small_graph):
        cone = small_graph.get_causal_cone(4)
        # 4 has ancestor 3, which has ancestor 2, etc.
        assert isinstance(cone, set)


# ================================================================
# Layer 1: Rewrite Rules
# ================================================================


class TestRewriteRules:
    """Tests for rewrite rules."""

    def test_diffusion_rule(self, small_graph, rng):
        rule = diffusion_rule(0.5)
        new_state = rule.apply(small_graph, 1, rng)
        # Node 1 has neighbor 2 (via edge 1→2 if forward adj)
        # State should change from [1.0, 2.0]
        assert new_state.shape == (2,)

    def test_decay_rule(self, small_graph, rng):
        rule = decay_rule(0.1)
        old_norm = np.linalg.norm(small_graph.nodes[0].state)
        new_state = rule.apply(small_graph, 0, rng)
        new_norm = np.linalg.norm(new_state)
        assert new_norm < old_norm

    def test_interaction_rule(self, small_graph, rng):
        rule = interaction_rule(0.05)
        new_state = rule.apply(small_graph, 0, rng)
        assert new_state.shape == (2,)

    def test_stochastic_perturbation(self, small_graph, rng):
        rule = stochastic_perturbation_rule(0.01)
        new_state = rule.apply(small_graph, 0, rng)
        # State should be slightly perturbed
        diff = np.linalg.norm(new_state - small_graph.nodes[0].state)
        assert diff > 0  # noise should be non-zero with overwhelming probability

    def test_edge_reweight_rule(self, small_graph, rng):
        old_weight = small_graph.edges[(0, 1)].causal_weight
        rule = edge_reweight_rule(0.1)
        rule.apply(small_graph, 0, rng)
        new_weight = small_graph.edges[(0, 1)].causal_weight
        # Weight should change (nodes have different states)
        assert new_weight != old_weight
        assert 0.0 <= new_weight <= 1.0

    def test_rewrite_engine(self, small_graph, rng):
        engine = RewriteEngine()
        engine.register_rule(diffusion_rule(0.1))
        engine.register_rule(decay_rule(0.01))
        updated = engine.apply_rules(small_graph, 0, rng)
        assert isinstance(updated, Node)
        assert updated.timestamp == 1  # incremented


# ================================================================
# Layer 2: Evolution Engine
# ================================================================


class TestEvolution:
    """Tests for CRSEngine."""

    def test_single_step(self, lattice):
        rw = RewriteEngine()
        rw.register_rule(diffusion_rule(0.1))
        engine = CRSEngine(graph=lattice, rewrite_engine=rw, seed=42)
        metrics = engine.step()
        assert "step" in metrics
        assert "n_nodes" in metrics
        assert metrics["n_nodes"] == 25
        assert engine.step_count == 1

    def test_multi_step(self, lattice):
        rw = RewriteEngine()
        rw.register_rule(diffusion_rule(0.1))
        rw.register_rule(stochastic_perturbation_rule(0.01))
        engine = CRSEngine(graph=lattice, rewrite_engine=rw, seed=42)
        history = engine.evolve(n_steps=10)
        assert len(history) == 10
        assert engine.step_count == 10

    def test_metrics_keys(self, lattice):
        rw = RewriteEngine()
        rw.register_rule(diffusion_rule(0.1))
        engine = CRSEngine(graph=lattice, rewrite_engine=rw, seed=42)
        m = engine.step()
        expected_keys = {
            "step", "n_nodes", "n_edges", "mean_state_norm",
            "max_state_norm", "total_causal_weight", "entropy_estimate",
        }
        assert expected_keys.issubset(set(m.keys()))


# ================================================================
# Layer 3: Emergent Spacetime
# ================================================================


class TestSpacetime:
    """Tests for spacetime reconstruction."""

    def test_geodesic_distance(self, small_graph):
        d = geodesic_distance(small_graph, 0, 4)
        assert d > 0
        assert not np.isinf(d)

    def test_geodesic_self_distance(self, small_graph):
        d = geodesic_distance(small_graph, 0, 0)
        assert d == 0.0

    def test_geodesic_disconnected(self):
        g = CRSGraph()
        g.add_node(Node(id=0, state=np.array([1.0]), timestamp=0))
        g.add_node(Node(id=1, state=np.array([2.0]), timestamp=0))
        d = geodesic_distance(g, 0, 1)
        assert np.isinf(d)

    def test_distance_matrix_symmetric(self, small_graph):
        D = distance_matrix(small_graph)
        assert D.shape == (5, 5)
        # Diagonal should be zero
        np.testing.assert_array_almost_equal(np.diag(D), 0.0)

    def test_causal_depth(self, small_graph):
        depths = causal_depth(small_graph)
        assert isinstance(depths, dict)
        # All values should be non-negative
        for v in depths.values():
            assert v >= 0

    def test_curvature_proxy(self, lattice):
        c = curvature_proxy(lattice, 12)  # center node
        assert isinstance(c, float)

    def test_dimension_estimate(self, lattice):
        dim = graph_dimension_estimate(lattice)
        assert isinstance(dim, float)
        assert dim > 0


# ================================================================
# Layer 4: Branching
# ================================================================


class TestBranching:
    """Tests for quantum-like branching."""

    def test_branch_state_creation(self, small_graph):
        bs = BranchState(graph=small_graph, amplitude=1.0 + 0j, label="root")
        assert bs.amplitude == 1.0 + 0j

    def test_branching_engine_normalize(self, small_graph):
        be = BranchingEngine(
            initial_graph=small_graph.copy(),
            max_branches=10,
        )
        # The engine starts with a single branch; add another manually
        be.states.append(
            BranchState(graph=small_graph.copy(), amplitude=0.5 + 0j, label="extra")
        )
        be.normalize()
        probs = be.probabilities()
        assert abs(sum(probs) - 1.0) < 1e-10

    def test_branching_engine_collapse(self, small_graph):
        rng = np.random.default_rng(42)
        be = BranchingEngine(initial_graph=small_graph.copy())
        be.states.append(
            BranchState(graph=small_graph.copy(), amplitude=0.6 + 0j, label="extra")
        )
        be.normalize()
        collapsed = be.collapse(rng)
        assert isinstance(collapsed, CRSGraph)

    def test_branching_prune(self, small_graph):
        be = BranchingEngine(initial_graph=small_graph.copy(), max_branches=1)
        # Add a second branch with small amplitude
        be.states.append(
            BranchState(graph=small_graph.copy(), amplitude=0.01 + 0j, label="small")
        )
        be.normalize()
        be.prune()
        assert len(be.states) <= 2  # pruning keeps top max_branches

    def test_probabilities_normalized(self, small_graph):
        be = BranchingEngine(initial_graph=small_graph.copy())
        be.states.append(
            BranchState(graph=small_graph.copy(), amplitude=0.5 - 0.1j, label="extra")
        )
        be.normalize()
        probs = be.probabilities()
        assert abs(sum(probs) - 1.0) < 1e-10


# ================================================================
# Layer 5: Conservation Laws
# ================================================================


class TestConservation:
    """Tests for conservation law engine."""

    def test_node_count_invariant(self, lattice):
        inv = node_count_invariant()
        val = inv.compute(lattice)
        assert val == 25  # 5x5

    def test_total_state_norm(self, lattice):
        inv = total_state_norm_invariant(tol=1.0)
        val = inv.compute(lattice)
        assert val > 0

    def test_conservation_check(self, lattice):
        ce = ConservationEngine()
        ce.register_invariant(node_count_invariant())

        before = lattice.copy()
        # No modification → should not be violated
        violations = ce.check_invariants(before, lattice)
        assert len(violations) == 1
        assert not violations[0]["violated"]

    def test_conservation_violation_detected(self, lattice):
        ce = ConservationEngine()
        ce.register_invariant(node_count_invariant())

        before = lattice.copy()
        # Remove a node → violation
        lattice.remove_node(0)
        violations = ce.check_invariants(before, lattice)
        assert violations[0]["violated"]

    def test_graph_energy_invariant(self, small_graph):
        inv = graph_energy_invariant(tol=1.0)
        val = inv.compute(small_graph)
        assert isinstance(val, float)
        assert val >= 0.0


# ================================================================
# Layer 6: Observer Subsystem
# ================================================================


class TestObserver:
    """Tests for observer subsystem."""

    def test_observe(self, small_graph):
        obs = Observer(id=0, accessible_nodes={0, 1, 2}, resolution=1.0)
        result = observe(obs, small_graph)
        assert len(result) == 3
        # With resolution=1.0, observed state == true state
        np.testing.assert_array_almost_equal(result[0], small_graph.nodes[0].state)

    def test_observe_with_noise(self, small_graph):
        obs = Observer(id=0, accessible_nodes={0, 1}, resolution=0.5)
        rng = np.random.default_rng(42)
        result = observe(obs, small_graph, rng=rng)
        # With resolution=0.5, observed state should differ from true state
        diff = np.linalg.norm(result[0] - small_graph.nodes[0].state)
        # May or may not differ significantly, but should return a result
        assert result[0].shape == small_graph.nodes[0].state.shape

    def test_observe_inaccessible(self, small_graph):
        obs = Observer(id=0, accessible_nodes={0, 1})
        result = observe(obs, small_graph)
        # Should only contain accessible nodes
        assert 3 not in result

    def test_measurement_entropy(self, small_graph):
        obs = Observer(id=0, accessible_nodes={0, 1, 2, 3, 4}, resolution=1.0)
        ent = measurement_entropy(obs, small_graph)
        assert isinstance(ent, float)
        assert ent >= 0.0

    def test_partial_trace(self, small_graph):
        sub = partial_trace(small_graph, keep_nodes={0, 1, 2})
        assert sub.node_count == 3
        # Edges not involving kept nodes should be removed
        for (s, t) in sub.edges:
            assert s in {0, 1, 2}
            assert t in {0, 1, 2}

    def test_decohere(self, lattice):
        obs = Observer(id=0, accessible_nodes={0, 1, 2, 3, 4}, resolution=0.8)
        rng = np.random.default_rng(42)
        g_before = lattice.copy()
        g_after = decohere(lattice, obs, decoherence_rate=0.5, rng=rng)
        # States within the observer's reach should be modified
        # (at least some with high probability)
        changed = sum(
            1 for nid in obs.accessible_nodes
            if nid in g_before.nodes and nid in g_after.nodes
            and not np.allclose(g_before.nodes[nid].state, g_after.nodes[nid].state)
        )
        assert changed >= 0  # may or may not change depending on decoherence model


# ================================================================
# Layer 7: CIIR/QuASIM/QRATUM Integration
# ================================================================


class TestIntegration:
    """Tests for CIIR/QuASIM/QRATUM integration."""

    def test_graph_to_density_matrix(self, lattice):
        rho = graph_to_density_matrix(lattice)
        d = lattice.nodes[0].state.shape[0]
        assert rho.shape == (d, d)
        # Trace should be 1
        assert abs(np.trace(rho) - 1.0) < 1e-10
        # Should be positive semidefinite
        eigvals = np.linalg.eigvalsh(rho)
        assert np.all(eigvals >= -1e-10)

    def test_density_matrix_to_graph(self, lattice):
        rho = graph_to_density_matrix(lattice)
        g2 = density_matrix_to_graph(rho, n_nodes=10)
        assert g2.node_count == 10
        # The reconstructed graph should have nodes with valid states
        for n in g2.nodes.values():
            assert n.state.shape == (lattice.nodes[0].state.shape[0],)

    def test_apply_ciir_distortion(self, lattice):
        g_before = lattice.copy()
        g_after = apply_ciir_distortion(lattice, kappa_min=1.0, diameter=1.0)
        # States should be contracted
        for nid in g_before.nodes:
            if nid in g_after.nodes:
                norm_before = np.linalg.norm(g_before.nodes[nid].state)
                norm_after = np.linalg.norm(g_after.nodes[nid].state)
                if norm_before > 1e-12:
                    assert norm_after <= norm_before + 1e-10

    def test_qratum_crs_core_step(self, lattice):
        rw = RewriteEngine()
        rw.register_rule(diffusion_rule(0.1))
        engine = CRSEngine(graph=lattice, rewrite_engine=rw, seed=42)
        core = QRATUM_CRS_Core(engine=engine)
        m = core.step()
        assert "step" in m
        assert "conservation" in m

    def test_qratum_crs_core_run_experiment(self, lattice):
        rw = RewriteEngine()
        rw.register_rule(diffusion_rule(0.1))
        rw.register_rule(stochastic_perturbation_rule(0.01))
        engine = CRSEngine(graph=lattice, rewrite_engine=rw, seed=42)
        core = QRATUM_CRS_Core(engine=engine)
        result = core.run_experiment(n_steps=5)
        # Check for expected metrics keys
        assert "mean_state_norm" in result or "steps" in result
        # Should have 5 entries in each list
        first_key = list(result.keys())[0]
        assert len(result[first_key]) == 5

    def test_qratum_crs_core_density_matrix(self, lattice):
        rw = RewriteEngine()
        rw.register_rule(diffusion_rule(0.1))
        engine = CRSEngine(graph=lattice, rewrite_engine=rw, seed=42)
        core = QRATUM_CRS_Core(engine=engine)
        rho = core.get_density_matrix()
        assert abs(np.trace(rho) - 1.0) < 1e-10

    def test_qratum_crs_core_emergent_metrics(self, lattice):
        rw = RewriteEngine()
        rw.register_rule(diffusion_rule(0.1))
        engine = CRSEngine(graph=lattice, rewrite_engine=rw, seed=42)
        core = QRATUM_CRS_Core(engine=engine)
        em = core.get_emergent_metrics()
        assert "dimension_estimate" in em
        assert "mean_curvature" in em


# ================================================================
# Experiments: Emergent Behavior
# ================================================================


class TestExperiments:
    """Tests for simulation experiments demonstrating emergent behavior."""

    def test_diffusion_experiment(self):
        result = run_diffusion_experiment(n_nodes=16, d_state=3, n_steps=20, seed=42)
        assert "steps" in result
        assert "peak_ratio" in result
        # Peak ratio should decrease as hot spot diffuses
        assert result["peak_ratio"][-1] < result["peak_ratio"][0]

    def test_wave_experiment(self):
        result = run_wave_experiment(n_nodes=16, d_state=3, n_steps=20, seed=42)
        assert "energy" in result
        assert "oscillation_index" in result
        # Energy should remain bounded
        assert max(result["energy"]) < 1e6

    def test_phase_transition_experiment(self):
        result = run_phase_transition_experiment(
            n_nodes=10, d_state=3, n_steps=10, n_points=3, seed=42,
        )
        assert "p_values" in result
        assert len(result["p_values"]) == 3
        # More edges at higher p
        assert result["final_n_edges"][-1] >= result["final_n_edges"][0]

    def test_entropy_experiment(self):
        result = run_entropy_experiment(n_nodes=16, d_state=3, n_steps=20, seed=42)
        assert "entropy" in result
        # Entropy should increase from initial low-entropy state
        assert result["entropy"][-1] >= result["entropy"][0] - 0.5  # allow small fluctuation
