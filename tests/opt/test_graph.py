"""Tests for QRATUM-native graph structures."""

import pytest

from quasim.opt.graph import HierarchicalGraph, QGraph


class TestQGraph:
    """Test QGraph data structure."""

    def test_init_empty_graph(self):
        """Test initialization of empty graph."""
        graph = QGraph(num_nodes=5)
        assert graph.num_nodes == 5
        assert graph.directed is True
        assert graph.edge_count() == 0
        for i in range(5):
            assert graph.degree(i) == 0

    def test_add_directed_edge(self):
        """Test adding directed edges."""
        graph = QGraph(num_nodes=3, directed=True)
        graph.add_edge(0, 1, 2.5)

        assert graph.degree(0) == 1
        assert graph.degree(1) == 0
        assert graph.edge_count() == 1

        neighbors = list(graph.neighbors(0))
        assert len(neighbors) == 1
        assert neighbors[0] == (1, 2.5)

    def test_add_undirected_edge(self):
        """Test adding undirected edges."""
        graph = QGraph(num_nodes=3, directed=False)
        graph.add_edge(0, 1, 3.0)

        assert graph.degree(0) == 1
        assert graph.degree(1) == 1
        assert graph.edge_count() == 2  # Both directions

        neighbors_0 = list(graph.neighbors(0))
        neighbors_1 = list(graph.neighbors(1))
        assert (1, 3.0) in neighbors_0
        assert (0, 3.0) in neighbors_1

    def test_negative_weight_raises_error(self):
        """Test that negative weights are rejected."""
        graph = QGraph(num_nodes=2)
        with pytest.raises(ValueError, match="non-negative"):
            graph.add_edge(0, 1, -1.0)

    def test_out_of_range_nodes_raise_error(self):
        """Test that out-of-range node ids raise errors."""
        graph = QGraph(num_nodes=3)
        with pytest.raises(ValueError, match="range"):
            graph.add_edge(0, 5, 1.0)
        with pytest.raises(ValueError, match="range"):
            graph.add_edge(-1, 1, 1.0)

    def test_from_edge_list(self):
        """Test creating graph from edge list."""
        edge_list = [
            (0, 1, 1.0),
            (1, 2, 2.0),
            (2, 0, 3.0),
        ]
        graph = QGraph.from_edge_list(num_nodes=3, edge_list=edge_list, directed=True)

        assert graph.num_nodes == 3
        assert graph.edge_count() == 3
        assert graph.degree(0) == 1
        assert graph.degree(1) == 1
        assert graph.degree(2) == 1

    def test_random_graph(self):
        """Test random graph generation."""
        graph = QGraph.random_graph(
            num_nodes=10,
            edge_probability=0.3,
            seed=42,
            directed=True,
            max_weight=5.0
        )

        assert graph.num_nodes == 10
        assert graph.edge_count() > 0  # Should have some edges

        # Check edge weights are in valid range
        for node in range(graph.num_nodes):
            for neighbor, weight in graph.neighbors(node):
                assert 0 <= neighbor < graph.num_nodes
                assert 1.0 <= weight <= 5.0

    def test_neighbors_iterator(self):
        """Test neighbors iterator."""
        graph = QGraph(num_nodes=3)
        graph.add_edge(0, 1, 1.0)
        graph.add_edge(0, 2, 2.0)

        neighbors = list(graph.neighbors(0))
        assert len(neighbors) == 2
        assert (1, 1.0) in neighbors
        assert (2, 2.0) in neighbors

        # Node with no neighbors
        neighbors = list(graph.neighbors(1))
        assert len(neighbors) == 0


class TestHierarchicalGraph:
    """Test hierarchical graph structures."""

    def test_init_empty_hierarchy(self):
        """Test initialization of empty hierarchy."""
        hierarchy = HierarchicalGraph()
        assert hierarchy.num_levels == 0

    def test_add_levels(self):
        """Test adding levels to hierarchy."""
        hierarchy = HierarchicalGraph()

        base = QGraph(num_nodes=4)
        hierarchy.add_level(base, mapping=None)
        assert hierarchy.num_levels == 1

        level1 = QGraph(num_nodes=2)
        mapping = {0: 0, 1: 0, 2: 1, 3: 1}
        hierarchy.add_level(level1, mapping=mapping)
        assert hierarchy.num_levels == 2

    def test_get_supernode(self):
        """Test supernode lookup."""
        hierarchy = HierarchicalGraph()

        base = QGraph(num_nodes=4)
        hierarchy.add_level(base, mapping=None)

        level1 = QGraph(num_nodes=2)
        mapping = {0: 0, 1: 0, 2: 1, 3: 1}
        hierarchy.add_level(level1, mapping=mapping)

        # Nodes 0 and 1 map to supernode 0
        assert hierarchy.get_supernode(0, 0) == 0
        assert hierarchy.get_supernode(0, 1) == 0

        # Nodes 2 and 3 map to supernode 1
        assert hierarchy.get_supernode(0, 2) == 1
        assert hierarchy.get_supernode(0, 3) == 1

    def test_from_contraction(self):
        """Test hierarchical graph creation by contraction."""
        base = QGraph.random_graph(
            num_nodes=20,
            edge_probability=0.2,
            seed=42,
            directed=True
        )

        hierarchy = HierarchicalGraph.from_contraction(
            base_graph=base,
            num_levels=3,
            contraction_factor=0.5
        )

        assert hierarchy.num_levels == 3
        assert hierarchy.levels[0].num_nodes == 20
        assert hierarchy.levels[1].num_nodes == 10  # 50% of 20
        assert hierarchy.levels[2].num_nodes == 5   # 50% of 10

        # Check that mappings exist
        assert len(hierarchy.node_mappings) == 2

    def test_contraction_preserves_connectivity(self):
        """Test that contraction maintains graph structure."""
        # Create simple triangle graph
        base = QGraph(num_nodes=3)
        base.add_edge(0, 1, 1.0)
        base.add_edge(1, 2, 1.0)
        base.add_edge(2, 0, 1.0)

        hierarchy = HierarchicalGraph.from_contraction(
            base_graph=base,
            num_levels=2,
            contraction_factor=0.5
        )

        # Should have 2 levels
        assert hierarchy.num_levels == 2

        # Contracted graph should have fewer or equal nodes
        assert hierarchy.levels[1].num_nodes <= base.num_nodes
