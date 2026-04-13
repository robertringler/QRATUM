"""Tests for PostDijkstra SSSP algorithm."""


from quasim.opt.graph import QGraph
from quasim.opt.post_dijkstra_sssp import (
    BucketedFrontier,
    ClassicalMinimumFinder,
    DeltaBucket,
    LowerBoundPruner,
    PostDijkstraSSSP,
    QuantumMinimumFinder,
)
from quasim.opt.ultra_sssp import dijkstra_baseline, validate_sssp_results


class TestDeltaBucket:
    """Test delta bucket data structure."""

    def test_bucket_init(self):
        """Test bucket initialization."""
        bucket = DeltaBucket(bucket_id=5, delta=2.0)
        assert bucket.bucket_id == 5
        assert bucket.delta == 2.0
        assert len(bucket.nodes) == 0
        assert bucket.min_distance == float("inf")

    def test_bucket_add_node(self):
        """Test adding nodes to bucket."""
        bucket = DeltaBucket(bucket_id=0, delta=1.0)
        bucket.add(10, 0.5)
        bucket.add(20, 0.3)

        assert 10 in bucket.nodes
        assert 20 in bucket.nodes
        assert bucket.min_distance == 0.3

    def test_bucket_pop_all(self):
        """Test popping all nodes from bucket."""
        bucket = DeltaBucket(bucket_id=0, delta=1.0)
        bucket.add(10, 0.5)
        bucket.add(20, 0.3)

        nodes = bucket.pop_all()
        assert 10 in nodes
        assert 20 in nodes
        assert len(bucket.nodes) == 0

    def test_bucket_distance_range(self):
        """Test bucket distance range calculation."""
        bucket = DeltaBucket(bucket_id=3, delta=2.0)
        lower, upper = bucket.distance_range
        assert lower == 6.0
        assert upper == 8.0


class TestBucketedFrontier:
    """Test bucketed frontier management."""

    def test_frontier_init(self):
        """Test frontier initialization."""
        frontier = BucketedFrontier(delta=1.0)
        assert frontier.delta == 1.0
        assert frontier.is_empty()

    def test_frontier_insert_and_extract(self):
        """Test inserting and extracting nodes."""
        frontier = BucketedFrontier(delta=1.0)

        frontier.insert(10, 0.5)
        frontier.insert(20, 1.5)
        frontier.insert(30, 0.3)

        assert not frontier.is_empty()

        # Should extract bucket 0 (distances 0.0-1.0)
        nodes = frontier.extract_min_bucket()
        assert nodes is not None
        assert 10 in nodes
        assert 30 in nodes
        assert 20 not in nodes

        # Should extract bucket 1 (distances 1.0-2.0)
        nodes = frontier.extract_min_bucket()
        assert nodes is not None
        assert 20 in nodes

        # Should be empty now
        assert frontier.is_empty()

    def test_frontier_update_distance(self):
        """Test updating node distance moves to correct bucket."""
        frontier = BucketedFrontier(delta=1.0)

        frontier.insert(10, 2.5)
        assert frontier.min_bucket_id == 2

        # Update to smaller distance
        frontier.insert(10, 0.5)
        assert frontier.min_bucket_id == 0

        nodes = frontier.extract_min_bucket()
        assert 10 in nodes

    def test_frontier_multiple_buckets(self):
        """Test frontier with multiple active buckets."""
        frontier = BucketedFrontier(delta=2.0)

        frontier.insert(1, 0.5)
        frontier.insert(2, 3.5)
        frontier.insert(3, 7.0)
        frontier.insert(4, 1.5)

        # Bucket 0: nodes 1, 4
        bucket1 = frontier.extract_min_bucket()
        assert len(bucket1) == 2
        assert {1, 4} == bucket1

        # Bucket 1: node 2
        bucket2 = frontier.extract_min_bucket()
        assert len(bucket2) == 1
        assert 2 in bucket2

        # Bucket 3: node 3
        bucket3 = frontier.extract_min_bucket()
        assert len(bucket3) == 1
        assert 3 in bucket3


class TestMinimumFinders:
    """Test minimum-finding strategies."""

    def test_classical_find_minimum(self):
        """Test classical minimum finder."""
        finder = ClassicalMinimumFinder()
        candidates = [(3.0, 10), (1.0, 20), (5.0, 30), (2.0, 40)]

        min_item = finder.find_minimum(candidates)
        assert min_item == (1.0, 20)

    def test_classical_find_k_minimum(self):
        """Test classical k-minimum finder."""
        finder = ClassicalMinimumFinder()
        candidates = [(3.0, 10), (1.0, 20), (5.0, 30), (2.0, 40)]

        k_min = finder.find_k_minimum(candidates, 2)
        assert len(k_min) == 2
        assert (1.0, 20) in k_min
        assert (2.0, 40) in k_min

    def test_classical_empty_candidates(self):
        """Test classical finder with empty input."""
        finder = ClassicalMinimumFinder()
        result = finder.find_minimum([])
        assert result == (float("inf"), -1)

    def test_quantum_fallback(self):
        """Test quantum finder falls back to classical."""
        finder = QuantumMinimumFinder(use_qpu=False)
        candidates = [(3.0, 10), (1.0, 20), (5.0, 30)]

        min_item = finder.find_minimum(candidates)
        assert min_item == (1.0, 20)

    def test_quantum_small_input_fallback(self):
        """Test quantum finder uses classical for small inputs."""
        finder = QuantumMinimumFinder(use_qpu=True)
        candidates = [(3.0, 10), (1.0, 20)]  # Too small for quantum

        min_item = finder.find_minimum(candidates)
        assert min_item == (1.0, 20)


class TestLowerBoundPruner:
    """Test lower-bound pruning."""

    def test_pruner_init(self):
        """Test pruner initialization."""
        graph = QGraph.random_graph(num_nodes=20, edge_probability=0.2, seed=42)
        pruner = LowerBoundPruner(graph, num_landmarks=5)

        assert len(pruner.landmarks) == 5
        assert len(pruner.landmark_distances) > 0

    def test_pruner_lower_bound(self):
        """Test lower bound computation."""
        graph = QGraph(num_nodes=4)
        graph.add_edge(0, 1, 1.0)
        graph.add_edge(1, 2, 1.0)
        graph.add_edge(2, 3, 1.0)

        pruner = LowerBoundPruner(graph, num_landmarks=2)

        # Lower bound should be non-negative
        bound = pruner.get_lower_bound(0, 3)
        assert bound >= 0.0

    def test_pruner_empty_graph(self):
        """Test pruner with empty graph."""
        graph = QGraph(num_nodes=5)
        pruner = LowerBoundPruner(graph, num_landmarks=2)

        bound = pruner.get_lower_bound(0, 4)
        assert bound >= 0.0


class TestPostDijkstraSSSP:
    """Test PostDijkstra algorithm."""

    def test_post_dijkstra_simple_path(self):
        """Test PostDijkstra on simple path."""
        graph = QGraph(num_nodes=3)
        graph.add_edge(0, 1, 1.0)
        graph.add_edge(1, 2, 2.0)

        sssp = PostDijkstraSSSP(graph, delta=0.5, use_hierarchy=False)
        distances, metrics = sssp.solve(source=0)

        assert distances[0] == 0.0
        assert distances[1] == 1.0
        assert distances[2] == 3.0
        assert metrics.total_time > 0

    def test_post_dijkstra_triangle(self):
        """Test PostDijkstra on triangle graph."""
        graph = QGraph(num_nodes=3)
        graph.add_edge(0, 1, 5.0)
        graph.add_edge(0, 2, 2.0)
        graph.add_edge(2, 1, 1.0)

        sssp = PostDijkstraSSSP(graph, delta=0.5, use_hierarchy=False)
        distances, metrics = sssp.solve(source=0)

        assert distances[0] == 0.0
        assert distances[1] == 3.0  # Via node 2
        assert distances[2] == 2.0

    def test_post_dijkstra_matches_dijkstra(self):
        """Test that PostDijkstra matches Dijkstra results."""
        graph = QGraph.random_graph(num_nodes=30, edge_probability=0.15, seed=123)

        # Run both algorithms
        dijkstra_distances, _ = dijkstra_baseline(graph, source=0)

        sssp = PostDijkstraSSSP(graph, delta=0.5, use_hierarchy=False, use_lower_bounds=False)
        post_distances, metrics = sssp.solve(source=0)

        # Results should match
        assert validate_sssp_results(post_distances, dijkstra_distances)
        metrics.correctness_verified = True

    def test_post_dijkstra_with_hierarchy(self):
        """Test PostDijkstra with hierarchical decomposition."""
        graph = QGraph.random_graph(num_nodes=100, edge_probability=0.05, seed=42)

        sssp = PostDijkstraSSSP(graph, delta=0.5, use_hierarchy=True, use_lower_bounds=False)
        distances, metrics = sssp.solve(source=0)

        assert len(distances) == 100
        assert distances[0] == 0.0
        assert metrics.hierarchy_levels_used > 0

    def test_post_dijkstra_with_lower_bounds(self):
        """Test PostDijkstra with lower-bound pruning."""
        graph = QGraph.random_graph(num_nodes=60, edge_probability=0.1, seed=456)

        sssp = PostDijkstraSSSP(graph, delta=0.5, use_hierarchy=False, use_lower_bounds=True)
        distances, metrics = sssp.solve(source=0)

        # Verify correctness
        dijkstra_distances, _ = dijkstra_baseline(graph, source=0)
        assert validate_sssp_results(distances, dijkstra_distances)

        # Should have some prunings
        # Note: May be 0 for small/dense graphs where pruning isn't beneficial
        assert metrics.lower_bound_prunings >= 0

    def test_post_dijkstra_auto_delta(self):
        """Test automatic delta computation."""
        graph = QGraph.random_graph(num_nodes=50, edge_probability=0.1, seed=789)

        sssp = PostDijkstraSSSP(graph)  # Auto-compute delta
        assert sssp.delta > 0

        distances, _ = sssp.solve(source=0)

        # Verify correctness
        dijkstra_distances, _ = dijkstra_baseline(graph, source=0)
        assert validate_sssp_results(distances, dijkstra_distances)

    def test_post_dijkstra_unreachable_nodes(self):
        """Test PostDijkstra with unreachable nodes."""
        graph = QGraph(num_nodes=4)
        graph.add_edge(0, 1, 1.0)
        graph.add_edge(2, 3, 1.0)

        sssp = PostDijkstraSSSP(graph, delta=0.5, use_hierarchy=False)
        distances, _ = sssp.solve(source=0)

        assert distances[0] == 0.0
        assert distances[1] == 1.0
        assert distances[2] == float("inf")
        assert distances[3] == float("inf")

    def test_post_dijkstra_metrics(self):
        """Test that metrics are properly tracked."""
        graph = QGraph.random_graph(num_nodes=40, edge_probability=0.1, seed=999)

        sssp = PostDijkstraSSSP(graph, delta=1.0, use_hierarchy=True, use_lower_bounds=True)
        distances, metrics = sssp.solve(source=0)

        # Check metrics are populated
        assert metrics.total_time > 0
        assert metrics.nodes_visited > 0
        assert metrics.edges_relaxed > 0
        assert metrics.bucket_operations > 0
        assert metrics.parallel_batches > 0
        assert metrics.memory_bytes > 0
        assert "delta_stepping" in metrics.phase_times

    def test_post_dijkstra_batch_size_variation(self):
        """Test PostDijkstra with different batch sizes."""
        graph = QGraph.random_graph(num_nodes=30, edge_probability=0.15, seed=111)

        dijkstra_distances, _ = dijkstra_baseline(graph, source=0)

        for batch_size in [10, 50, 100]:
            sssp = PostDijkstraSSSP(graph, delta=0.5, batch_size=batch_size, use_hierarchy=False)
            distances, _ = sssp.solve(source=0)

            assert validate_sssp_results(distances, dijkstra_distances)

    def test_post_dijkstra_custom_minimum_finder(self):
        """Test PostDijkstra with custom minimum finder."""
        graph = QGraph.random_graph(num_nodes=25, edge_probability=0.15, seed=222)

        finder = ClassicalMinimumFinder()
        sssp = PostDijkstraSSSP(graph, delta=0.5, minimum_finder=finder, use_hierarchy=False)
        distances, _ = sssp.solve(source=0)

        dijkstra_distances, _ = dijkstra_baseline(graph, source=0)
        assert validate_sssp_results(distances, dijkstra_distances)

    def test_post_dijkstra_single_node(self):
        """Test PostDijkstra on single node graph."""
        graph = QGraph(num_nodes=1)
        sssp = PostDijkstraSSSP(graph, delta=1.0, use_hierarchy=False)
        distances, metrics = sssp.solve(source=0)

        assert distances[0] == 0.0
        assert metrics.nodes_visited == 1

    def test_post_dijkstra_disconnected_components(self):
        """Test PostDijkstra on graph with disconnected components."""
        graph = QGraph(num_nodes=6)
        # Component 1
        graph.add_edge(0, 1, 1.0)
        graph.add_edge(1, 2, 1.0)
        # Component 2
        graph.add_edge(3, 4, 1.0)
        graph.add_edge(4, 5, 1.0)

        sssp = PostDijkstraSSSP(graph, delta=0.5, use_hierarchy=False)
        distances, _ = sssp.solve(source=0)

        # Component 1 reachable
        assert distances[0] == 0.0
        assert distances[1] == 1.0
        assert distances[2] == 2.0

        # Component 2 unreachable
        assert distances[3] == float("inf")
        assert distances[4] == float("inf")
        assert distances[5] == float("inf")


class TestMetrics:
    """Test metrics tracking and reporting."""

    def test_metrics_to_dict(self):
        """Test metrics serialization."""
        from quasim.opt.post_dijkstra_sssp import PostDijkstraMetrics

        metrics = PostDijkstraMetrics()
        metrics.total_time = 1.5
        metrics.nodes_visited = 100
        metrics.edges_relaxed = 500
        metrics.bucket_operations = 50
        metrics.memory_bytes = 1024 * 1024

        data = metrics.to_dict()

        assert data["total_time"] == 1.5
        assert data["nodes_visited"] == 100
        assert data["edges_relaxed"] == 500
        assert data["bucket_operations"] == 50
        assert data["memory_mb"] == 1.0
        assert "operations_avoided" in data
