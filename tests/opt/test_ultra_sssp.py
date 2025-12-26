"""Tests for UltraSSSP shortest path algorithm."""

from quasim.opt.graph import QGraph
from quasim.opt.ultra_sssp import (
    FrontierBatch,
    SSSPSimulationConfig,
    UltraSSSP,
    dijkstra_baseline,
    run_sssp_simulation,
    validate_sssp_results,
)


class TestDijkstraBaseline:
    """Test classical Dijkstra implementation."""

    def test_dijkstra_simple_path(self):
        """Test Dijkstra on simple linear path."""
        graph = QGraph(num_nodes=3)
        graph.add_edge(0, 1, 1.0)
        graph.add_edge(1, 2, 2.0)

        distances, metrics = dijkstra_baseline(graph, source=0)

        assert distances[0] == 0.0
        assert distances[1] == 1.0
        assert distances[2] == 3.0
        assert metrics.nodes_visited == 3

    def test_dijkstra_unreachable_nodes(self):
        """Test Dijkstra with unreachable nodes."""
        graph = QGraph(num_nodes=4)
        graph.add_edge(0, 1, 1.0)
        graph.add_edge(2, 3, 1.0)

        distances, metrics = dijkstra_baseline(graph, source=0)

        assert distances[0] == 0.0
        assert distances[1] == 1.0
        assert distances[2] == float("inf")
        assert distances[3] == float("inf")

    def test_dijkstra_triangle(self):
        """Test Dijkstra on triangle graph."""
        graph = QGraph(num_nodes=3)
        graph.add_edge(0, 1, 5.0)
        graph.add_edge(0, 2, 2.0)
        graph.add_edge(2, 1, 1.0)  # Shorter path via node 2

        distances, metrics = dijkstra_baseline(graph, source=0)

        assert distances[0] == 0.0
        assert distances[1] == 3.0  # Via node 2
        assert distances[2] == 2.0

    def test_dijkstra_single_node(self):
        """Test Dijkstra on single node graph."""
        graph = QGraph(num_nodes=1)
        distances, metrics = dijkstra_baseline(graph, source=0)

        assert distances[0] == 0.0
        assert metrics.nodes_visited == 1


class TestFrontierBatch:
    """Test frontier batch management."""

    def test_init_empty_batch(self):
        """Test empty batch initialization."""
        batch = FrontierBatch()
        assert batch.size() == 0
        assert batch.min_distance == float("inf")
        assert batch.max_distance == 0.0

    def test_add_nodes_to_batch(self):
        """Test adding nodes to batch."""
        batch = FrontierBatch()
        batch.add_node(1, 2.0)
        batch.add_node(2, 5.0)
        batch.add_node(3, 1.0)

        assert batch.size() == 3
        assert batch.min_distance == 1.0
        assert batch.max_distance == 5.0
        assert 1 in batch.nodes
        assert 2 in batch.nodes
        assert 3 in batch.nodes


class TestUltraSSSP:
    """Test UltraSSSP algorithm."""

    def test_ultra_sssp_simple_path(self):
        """Test UltraSSSP on simple linear path."""
        graph = QGraph(num_nodes=3)
        graph.add_edge(0, 1, 1.0)
        graph.add_edge(1, 2, 2.0)

        sssp = UltraSSSP(graph, batch_size=10)
        distances, metrics = sssp.solve(source=0)

        assert distances[0] == 0.0
        assert distances[1] == 1.0
        assert distances[2] == 3.0
        assert metrics.total_time > 0

    def test_ultra_sssp_triangle(self):
        """Test UltraSSSP on triangle graph."""
        graph = QGraph(num_nodes=3)
        graph.add_edge(0, 1, 5.0)
        graph.add_edge(0, 2, 2.0)
        graph.add_edge(2, 1, 1.0)  # Shorter path

        sssp = UltraSSSP(graph, batch_size=10)
        distances, metrics = sssp.solve(source=0)

        assert distances[0] == 0.0
        assert distances[1] == 3.0  # Via node 2
        assert distances[2] == 2.0

    def test_ultra_sssp_with_hierarchy(self):
        """Test UltraSSSP with hierarchical contraction."""
        graph = QGraph.random_graph(num_nodes=50, edge_probability=0.1, seed=42)

        sssp = UltraSSSP(graph, batch_size=10, use_hierarchy=True, hierarchy_levels=2)
        distances, metrics = sssp.solve(source=0)

        # Should complete without errors
        assert len(distances) == 50
        assert distances[0] == 0.0
        assert metrics.total_time > 0

    def test_ultra_sssp_matches_dijkstra(self):
        """Test that UltraSSSP matches Dijkstra results."""
        graph = QGraph.random_graph(num_nodes=30, edge_probability=0.15, seed=123)

        # Run both algorithms
        dijkstra_distances, _ = dijkstra_baseline(graph, source=0)

        sssp = UltraSSSP(graph, batch_size=10)
        ultra_distances, _ = sssp.solve(source=0)

        # Results should match
        assert validate_sssp_results(ultra_distances, dijkstra_distances)

    def test_quantum_pivot_fallback(self):
        """Test quantum pivot selection fallback."""
        graph = QGraph(num_nodes=3)
        graph.add_edge(0, 1, 1.0)
        graph.add_edge(0, 2, 2.0)

        sssp = UltraSSSP(graph, batch_size=10)

        # Test fallback selects minimum distance
        candidates = [1, 2]
        distances = [0.0, 1.0, 2.0]
        pivot = sssp.quantum_pivot_select(candidates, distances)

        assert pivot == 1  # Node 1 has smaller distance

    def test_quantum_pivot_custom_function(self):
        """Test custom quantum pivot function."""
        graph = QGraph(num_nodes=3)
        graph.add_edge(0, 1, 1.0)
        graph.add_edge(0, 2, 2.0)

        # Custom function always picks last candidate
        def custom_pivot(candidates, distances):
            return candidates[-1]

        sssp = UltraSSSP(graph, batch_size=10, quantum_pivot_fn=custom_pivot)

        candidates = [1, 2]
        distances = [0.0, 1.0, 2.0]
        pivot = sssp.quantum_pivot_select(candidates, distances)

        assert pivot == 2  # Custom function picks last


class TestValidation:
    """Test validation utilities."""

    def test_validate_identical_distances(self):
        """Test validation with identical distances."""
        distances1 = [0.0, 1.0, 2.0, float("inf")]
        distances2 = [0.0, 1.0, 2.0, float("inf")]

        assert validate_sssp_results(distances1, distances2)

    def test_validate_different_lengths(self):
        """Test validation rejects different length arrays."""
        distances1 = [0.0, 1.0, 2.0]
        distances2 = [0.0, 1.0]

        assert not validate_sssp_results(distances1, distances2)

    def test_validate_different_values(self):
        """Test validation detects different values."""
        distances1 = [0.0, 1.0, 2.0]
        distances2 = [0.0, 1.0, 3.0]

        assert not validate_sssp_results(distances1, distances2)

    def test_validate_with_tolerance(self):
        """Test validation with numerical tolerance."""
        distances1 = [0.0, 1.0, 2.0]
        distances2 = [0.0, 1.0000001, 2.0]

        assert validate_sssp_results(distances1, distances2, tolerance=1e-5)
        assert not validate_sssp_results(distances1, distances2, tolerance=1e-8)


class TestSimulationConfig:
    """Test simulation configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = SSSPSimulationConfig()

        assert config.num_nodes == 1000
        assert config.edge_probability == 0.01
        assert config.source_node == 0
        assert config.batch_size == 100
        assert config.seed == 42
        assert config.validate_against_dijkstra is True

    def test_custom_config(self):
        """Test custom configuration values."""
        config = SSSPSimulationConfig(
            num_nodes=500, edge_probability=0.05, batch_size=50, use_hierarchy=True, seed=123
        )

        assert config.num_nodes == 500
        assert config.edge_probability == 0.05
        assert config.batch_size == 50
        assert config.use_hierarchy is True
        assert config.seed == 123


class TestSimulation:
    """Test complete simulation pipeline."""

    def test_run_small_simulation(self):
        """Test running small simulation."""
        config = SSSPSimulationConfig(
            num_nodes=20,
            edge_probability=0.2,
            batch_size=5,
            seed=42,
            validate_against_dijkstra=True,
        )

        results = run_sssp_simulation(config)

        # Check results structure
        assert "distances" in results
        assert "ultra_sssp_metrics" in results
        assert "dijkstra_metrics" in results
        assert "correctness" in results
        assert "speedup" in results
        assert "graph_info" in results

        # Check correctness
        assert results["correctness"] is True

        # Check distances
        assert len(results["distances"]) == 20
        assert results["distances"][0] == 0.0  # Source distance

    def test_run_simulation_without_validation(self):
        """Test simulation without validation."""
        config = SSSPSimulationConfig(
            num_nodes=20, edge_probability=0.2, validate_against_dijkstra=False
        )

        results = run_sssp_simulation(config)

        assert "distances" in results
        assert "ultra_sssp_metrics" in results
        assert "dijkstra_metrics" not in results
        assert "correctness" not in results

    def test_run_simulation_with_hierarchy(self):
        """Test simulation with hierarchical contraction."""
        config = SSSPSimulationConfig(
            num_nodes=50,
            edge_probability=0.1,
            use_hierarchy=True,
            hierarchy_levels=2,
            validate_against_dijkstra=True,
        )

        results = run_sssp_simulation(config)

        assert results["correctness"] is True
        assert len(results["distances"]) == 50
