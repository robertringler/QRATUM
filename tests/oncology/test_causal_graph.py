"""Tests for the CausalOncologyGraph module."""

import pytest

from qratum.oncology.causal_graph import (
    CausalOncologyGraph,
    OncogenicNode,
    CausalEdge,
    MutationState,
    EpigeneticState,
    ImmuneEvasionMechanism,
    TumorMicroenvironment,
    CancerHallmark,
    NodeType,
    EdgeType,
    create_example_egfr_graph,
)


class TestMutationState:
    """Tests for MutationState dataclass."""

    def test_creation(self):
        """Test MutationState creation."""
        mutation = MutationState(
            gene="TP53",
            mutation_type="missense",
            variant="R248W",
            allele_frequency=0.45,
            driver_probability=0.95,
        )
        assert mutation.gene == "TP53"
        assert mutation.variant == "R248W"
        assert mutation.driver_probability == 0.95

    def test_to_dict(self):
        """Test serialization."""
        mutation = MutationState(gene="KRAS", mutation_type="missense", variant="G12D")
        d = mutation.to_dict()
        assert d["gene"] == "KRAS"
        assert d["variant"] == "G12D"


class TestEpigeneticState:
    """Tests for EpigeneticState dataclass."""

    def test_creation(self):
        """Test EpigeneticState creation."""
        epigenetic = EpigeneticState(
            region="BRCA1_promoter",
            modification_type="methylation",
            level=0.8,
            silenced_genes=["BRCA1"],
        )
        assert epigenetic.level == 0.8
        assert "BRCA1" in epigenetic.silenced_genes

    def test_to_dict(self):
        """Test serialization."""
        epigenetic = EpigeneticState(region="test", modification_type="acetylation")
        d = epigenetic.to_dict()
        assert d["modification_type"] == "acetylation"


class TestOncogenicNode:
    """Tests for OncogenicNode dataclass."""

    def test_creation(self):
        """Test OncogenicNode creation."""
        node = OncogenicNode(
            node_id="EGFR_mut",
            name="EGFR Activating Mutation",
            node_type=NodeType.MUTATION,
            hallmarks=[CancerHallmark.SUSTAINING_PROLIFERATIVE_SIGNALING],
            druggability=0.85,
        )
        assert node.node_id == "EGFR_mut"
        assert node.druggability == 0.85
        assert CancerHallmark.SUSTAINING_PROLIFERATIVE_SIGNALING in node.hallmarks

    def test_to_dict(self):
        """Test serialization."""
        node = OncogenicNode(
            node_id="test",
            name="Test Node",
            node_type=NodeType.PATHWAY,
        )
        d = node.to_dict()
        assert d["node_id"] == "test"
        assert d["node_type"] == "pathway"


class TestCausalEdge:
    """Tests for CausalEdge dataclass."""

    def test_creation(self):
        """Test CausalEdge creation."""
        edge = CausalEdge(
            source_id="A",
            target_id="B",
            edge_type=EdgeType.ACTIVATES,
            strength=0.9,
            confidence=0.95,
        )
        assert edge.source_id == "A"
        assert edge.strength == 0.9

    def test_to_dict(self):
        """Test serialization."""
        edge = CausalEdge(
            source_id="X",
            target_id="Y",
            edge_type=EdgeType.INHIBITS,
        )
        d = edge.to_dict()
        assert d["edge_type"] == "inhibits"


class TestCausalOncologyGraph:
    """Tests for CausalOncologyGraph class."""

    def test_initialization(self):
        """Test graph initialization."""
        graph = CausalOncologyGraph(
            name="Test_Graph",
            cancer_type="NSCLC",
            seed=42,
        )
        assert graph.name == "Test_Graph"
        assert graph.cancer_type == "NSCLC"

    def test_add_node(self):
        """Test adding nodes."""
        graph = CausalOncologyGraph(name="Test")
        node = OncogenicNode(
            node_id="test_node",
            name="Test Node",
            node_type=NodeType.MUTATION,
        )
        graph.add_node(node)
        assert graph.get_node("test_node") is not None

    def test_add_edge(self):
        """Test adding edges."""
        graph = CausalOncologyGraph(name="Test")

        node1 = OncogenicNode(
            node_id="A", name="Node A", node_type=NodeType.MUTATION
        )
        node2 = OncogenicNode(
            node_id="B", name="Node B", node_type=NodeType.PATHWAY
        )
        graph.add_node(node1)
        graph.add_node(node2)

        edge = CausalEdge(
            source_id="A",
            target_id="B",
            edge_type=EdgeType.ACTIVATES,
        )
        graph.add_edge(edge)

        assert graph.get_edge("A", "B") is not None
        assert "B" in graph.get_downstream_nodes("A")
        assert "A" in graph.get_upstream_nodes("B")

    def test_add_edge_invalid_source(self):
        """Test that adding edge with invalid source raises error."""
        graph = CausalOncologyGraph(name="Test")
        node = OncogenicNode(
            node_id="B", name="Node B", node_type=NodeType.PATHWAY
        )
        graph.add_node(node)

        edge = CausalEdge(
            source_id="A",  # Doesn't exist
            target_id="B",
            edge_type=EdgeType.ACTIVATES,
        )
        with pytest.raises(ValueError, match="Source node"):
            graph.add_edge(edge)

    def test_find_feedback_loops(self):
        """Test feedback loop detection."""
        graph = CausalOncologyGraph(name="Test")

        # Create a simple cycle: A -> B -> C -> A
        for name in ["A", "B", "C"]:
            graph.add_node(
                OncogenicNode(node_id=name, name=f"Node {name}", node_type=NodeType.PATHWAY)
            )

        graph.add_edge(CausalEdge(source_id="A", target_id="B", edge_type=EdgeType.ACTIVATES))
        graph.add_edge(CausalEdge(source_id="B", target_id="C", edge_type=EdgeType.ACTIVATES))
        graph.add_edge(CausalEdge(source_id="C", target_id="A", edge_type=EdgeType.ACTIVATES))

        loops = graph.find_feedback_loops()
        assert len(loops) > 0

    def test_compute_stability_manifold(self):
        """Test stability manifold computation."""
        graph = CausalOncologyGraph(name="Test", seed=42)

        for name in ["A", "B"]:
            graph.add_node(
                OncogenicNode(node_id=name, name=f"Node {name}", node_type=NodeType.PATHWAY)
            )
        graph.add_edge(CausalEdge(source_id="A", target_id="B", edge_type=EdgeType.ACTIVATES, strength=0.5))

        stability = graph.compute_stability_manifold(n_simulations=10)
        assert "A" in stability
        assert "B" in stability
        assert 0.0 <= stability["A"] <= 1.0

    def test_identify_attractor_states(self):
        """Test attractor state identification."""
        graph = CausalOncologyGraph(name="Test", seed=42)

        graph.add_node(
            OncogenicNode(
                node_id="high_activity",
                name="High Activity Node",
                node_type=NodeType.PHENOTYPE,
                activity_level=0.9,
            )
        )
        graph.add_node(
            OncogenicNode(
                node_id="low_activity",
                name="Low Activity Node",
                node_type=NodeType.PHENOTYPE,
                activity_level=0.1,
            )
        )

        attractors = graph.identify_attractor_states()
        assert "high_activity" in attractors
        assert "low_activity" in attractors

    def test_find_intervention_targets(self):
        """Test intervention target identification."""
        graph = CausalOncologyGraph(name="Test")

        graph.add_node(
            OncogenicNode(
                node_id="druggable",
                name="Druggable Target",
                node_type=NodeType.PROTEIN,
                druggability=0.9,
            )
        )
        graph.add_node(
            OncogenicNode(
                node_id="not_druggable",
                name="Not Druggable",
                node_type=NodeType.PATHWAY,
                druggability=0.1,
            )
        )

        targets = graph.find_intervention_targets(min_druggability=0.5)
        assert len(targets) == 1
        assert targets[0].node_id == "druggable"

    def test_get_causal_paths(self):
        """Test causal path finding."""
        graph = CausalOncologyGraph(name="Test")

        for name in ["A", "B", "C"]:
            graph.add_node(
                OncogenicNode(node_id=name, name=f"Node {name}", node_type=NodeType.PATHWAY)
            )

        graph.add_edge(CausalEdge(source_id="A", target_id="B", edge_type=EdgeType.ACTIVATES))
        graph.add_edge(CausalEdge(source_id="B", target_id="C", edge_type=EdgeType.ACTIVATES))

        paths = graph.get_causal_paths("A", "C")
        assert len(paths) > 0
        assert paths[0] == ["A", "B", "C"]

    def test_compute_graph_hash(self):
        """Test graph hash computation is deterministic."""
        graph = CausalOncologyGraph(name="Test")
        graph.add_node(OncogenicNode(node_id="A", name="A", node_type=NodeType.MUTATION))

        hash1 = graph.compute_graph_hash()
        hash2 = graph.compute_graph_hash()
        assert hash1 == hash2

    def test_serialization_roundtrip(self):
        """Test to_dict and from_dict roundtrip."""
        graph = CausalOncologyGraph(name="Test", cancer_type="NSCLC")
        graph.add_node(OncogenicNode(node_id="A", name="A", node_type=NodeType.MUTATION))

        d = graph.to_dict()
        restored = CausalOncologyGraph.from_dict(d)

        assert restored.name == graph.name
        assert restored.cancer_type == graph.cancer_type
        assert restored.get_node("A") is not None


class TestExampleEGFRGraph:
    """Tests for the example EGFR graph."""

    def test_create_example(self):
        """Test example graph creation."""
        graph = create_example_egfr_graph()
        assert graph.name == "EGFR_Model"
        assert graph.cancer_type == "NSCLC"

    def test_example_has_nodes(self):
        """Test example graph has expected nodes."""
        graph = create_example_egfr_graph()
        assert graph.get_node("EGFR_mut") is not None
        assert graph.get_node("RAS_pathway") is not None
        assert graph.get_node("proliferation") is not None

    def test_example_has_edges(self):
        """Test example graph has expected edges."""
        graph = create_example_egfr_graph()
        assert graph.get_edge("EGFR_mut", "RAS_pathway") is not None
        assert graph.get_edge("EGFR_mut", "PI3K_pathway") is not None

    def test_example_causal_paths(self):
        """Test causal paths in example graph."""
        graph = create_example_egfr_graph()
        paths = graph.get_causal_paths("EGFR_mut", "proliferation")
        assert len(paths) >= 1
