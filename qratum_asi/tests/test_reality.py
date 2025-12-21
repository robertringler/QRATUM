"""Tests for Q-REALITY component."""

import pytest
from qratum_asi.components.reality import QReality, KnowledgeNode, CausalLink
from qratum_asi.core.contracts import ASIContract
from qratum_asi.core.types import ASISafetyLevel, AuthorizationType


class TestQReality:
    """Test Q-REALITY world model."""

    def test_initialization(self):
        """Test Q-REALITY initialization."""
        reality = QReality()
        assert len(reality.knowledge_nodes) == 0
        assert len(reality.causal_links) == 0
        assert len(reality.verticals) == 14

    def test_add_knowledge_node(self):
        """Test adding knowledge node."""
        reality = QReality()
        contract = ASIContract(
            contract_id="test_001",
            operation_type="add_knowledge",
            safety_level=ASISafetyLevel.ROUTINE,
            authorization_type=AuthorizationType.NONE,
            payload={},
        )

        node = reality.add_knowledge_node(
            node_id="node_001",
            content={"statement": "Test knowledge"},
            source_vertical="JURIS",
            confidence=0.9,
            provenance=["test_source"],
            contract=contract,
        )

        assert node.node_id == "node_001"
        assert node.confidence == 0.9
        assert "node_001" in reality.knowledge_nodes
        assert reality.merkle_chain.get_chain_length() == 1

    def test_add_knowledge_node_invalid_contract(self):
        """Test adding knowledge node with invalid contract."""
        reality = QReality()
        contract = ASIContract(
            contract_id="test_002",
            operation_type="add_knowledge",
            safety_level=ASISafetyLevel.SENSITIVE,
            authorization_type=AuthorizationType.SINGLE_HUMAN,
            payload={},
            authorized=False,  # Not authorized
        )

        with pytest.raises(ValueError):
            reality.add_knowledge_node(
                node_id="node_002",
                content={"statement": "Test"},
                source_vertical="JURIS",
                confidence=0.9,
                provenance=["test"],
                contract=contract,
            )

    def test_add_causal_link(self):
        """Test adding causal link."""
        reality = QReality()
        contract = ASIContract(
            contract_id="test_003",
            operation_type="add_knowledge",
            safety_level=ASISafetyLevel.ROUTINE,
            authorization_type=AuthorizationType.NONE,
            payload={},
        )

        # Add nodes first
        reality.add_knowledge_node(
            node_id="node_001",
            content={"statement": "Cause"},
            source_vertical="JURIS",
            confidence=0.9,
            provenance=["test"],
            contract=contract,
        )

        reality.add_knowledge_node(
            node_id="node_002",
            content={"statement": "Effect"},
            source_vertical="JURIS",
            confidence=0.9,
            provenance=["test"],
            contract=contract,
        )

        # Add link
        link = reality.add_causal_link(
            link_id="link_001",
            source_node_id="node_001",
            target_node_id="node_002",
            relationship_type="causes",
            confidence=0.85,
            evidence=["observation"],
            contract=contract,
        )

        assert link.link_id == "link_001"
        assert "link_001" in reality.causal_links

    def test_query_knowledge(self):
        """Test querying knowledge."""
        reality = QReality()
        contract = ASIContract(
            contract_id="test_004",
            operation_type="add_knowledge",
            safety_level=ASISafetyLevel.ROUTINE,
            authorization_type=AuthorizationType.NONE,
            payload={},
        )

        reality.add_knowledge_node(
            node_id="node_001",
            content={"statement": "Test"},
            source_vertical="JURIS",
            confidence=0.9,
            provenance=["test"],
            contract=contract,
        )

        results = reality.query_knowledge(
            query={"source_vertical": "JURIS"},
            min_confidence=0.5,
        )

        assert len(results) == 1
        assert results[0].node_id == "node_001"

    def test_get_causal_path(self):
        """Test finding causal path."""
        reality = QReality()
        contract = ASIContract(
            contract_id="test_005",
            operation_type="add_knowledge",
            safety_level=ASISafetyLevel.ROUTINE,
            authorization_type=AuthorizationType.NONE,
            payload={},
        )

        # Create chain: node_001 -> node_002 -> node_003
        for i in range(1, 4):
            reality.add_knowledge_node(
                node_id=f"node_00{i}",
                content={"statement": f"Node {i}"},
                source_vertical="JURIS",
                confidence=0.9,
                provenance=["test"],
                contract=contract,
            )

        reality.add_causal_link(
            link_id="link_001",
            source_node_id="node_001",
            target_node_id="node_002",
            relationship_type="causes",
            confidence=0.85,
            evidence=["test"],
            contract=contract,
        )

        reality.add_causal_link(
            link_id="link_002",
            source_node_id="node_002",
            target_node_id="node_003",
            relationship_type="causes",
            confidence=0.85,
            evidence=["test"],
            contract=contract,
        )

        path = reality.get_causal_path("node_001", "node_003")
        assert path == ["node_001", "node_002", "node_003"]
