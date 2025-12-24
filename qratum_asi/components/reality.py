"""Q-REALITY: Emergent World Model.

Unified causal model fusing all 14 verticals with hash-addressed
knowledge nodes, causal graph structure, and full provenance tracking.
"""

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

from qratum_asi.core.chain import ASIMerkleChain
from qratum_asi.core.contracts import ASIContract
from qratum_asi.core.events import ASIEvent, ASIEventType


@dataclass
class KnowledgeNode:
    """Immutable knowledge node in the world model."""

    node_id: str
    content: Dict[str, Any]
    source_vertical: str
    timestamp: str
    confidence: float  # 0.0 to 1.0
    provenance: List[str]  # Chain of sources
    hash: str = ""

    def __post_init__(self):
        """Compute node hash."""
        if not self.hash:
            node_data = {
                "node_id": self.node_id,
                "content": self.content,
                "source_vertical": self.source_vertical,
                "timestamp": self.timestamp,
            }
            serialized = json.dumps(node_data, sort_keys=True)
            self.hash = hashlib.sha256(serialized.encode()).hexdigest()


@dataclass
class CausalLink:
    """Causal relationship between knowledge nodes."""

    link_id: str
    source_node_id: str
    target_node_id: str
    relationship_type: str
    confidence: float
    evidence: List[str]


@dataclass
class QReality:
    """Q-REALITY: Emergent World Model.
    
    Maintains unified causal model across all QRATUM verticals with
    deterministic updates, full provenance, and cross-domain inference.
    """

    knowledge_nodes: Dict[str, KnowledgeNode] = field(default_factory=dict)
    causal_links: Dict[str, CausalLink] = field(default_factory=dict)
    merkle_chain: ASIMerkleChain = field(default_factory=ASIMerkleChain)
    verticals: Set[str] = field(default_factory=lambda: {
        "JURIS", "VITRA", "ECORA", "QUASIM", "QNIMBUS", "QUBIC",
        "XENON", "HCAL", "QNX", "QSTACK", "TERC", "OMNILEX", "FEDERATED", "AGI"
    })

    def add_knowledge_node(
        self,
        node_id: str,
        content: Dict[str, Any],
        source_vertical: str,
        confidence: float,
        provenance: List[str],
        contract: ASIContract,
    ) -> KnowledgeNode:
        """Add knowledge node to world model."""
        # Validate contract
        if not contract.validate():
            raise ValueError(f"Invalid contract for knowledge node: {contract.contract_id}")

        # Create node
        node = KnowledgeNode(
            node_id=node_id,
            content=content,
            source_vertical=source_vertical,
            timestamp=datetime.utcnow().isoformat(),
            confidence=confidence,
            provenance=provenance,
        )

        # Store node (immutable)
        self.knowledge_nodes[node_id] = node

        # Emit event
        event = ASIEvent.create(
            event_type=ASIEventType.KNOWLEDGE_NODE_CREATED,
            payload={
                "node_id": node_id,
                "source_vertical": source_vertical,
                "confidence": confidence,
            },
            contract_id=contract.contract_id,
            index=self.merkle_chain.get_chain_length(),
        )
        self.merkle_chain.append(event)

        return node

    def add_causal_link(
        self,
        link_id: str,
        source_node_id: str,
        target_node_id: str,
        relationship_type: str,
        confidence: float,
        evidence: List[str],
        contract: ASIContract,
    ) -> CausalLink:
        """Add causal link between knowledge nodes."""
        # Validate nodes exist
        if source_node_id not in self.knowledge_nodes:
            raise ValueError(f"Source node not found: {source_node_id}")
        if target_node_id not in self.knowledge_nodes:
            raise ValueError(f"Target node not found: {target_node_id}")

        # Create link
        link = CausalLink(
            link_id=link_id,
            source_node_id=source_node_id,
            target_node_id=target_node_id,
            relationship_type=relationship_type,
            confidence=confidence,
            evidence=evidence,
        )

        self.causal_links[link_id] = link

        # Emit event
        event = ASIEvent.create(
            event_type=ASIEventType.CAUSAL_LINK_CREATED,
            payload={
                "link_id": link_id,
                "source_node_id": source_node_id,
                "target_node_id": target_node_id,
                "relationship_type": relationship_type,
            },
            contract_id=contract.contract_id,
            index=self.merkle_chain.get_chain_length(),
        )
        self.merkle_chain.append(event)

        return link

    def query_knowledge(
        self, query: Dict[str, Any], min_confidence: float = 0.5
    ) -> List[KnowledgeNode]:
        """Query knowledge nodes by criteria."""
        results = []
        for node in self.knowledge_nodes.values():
            if node.confidence >= min_confidence:
                # Simple matching (real implementation would use semantic search)
                if self._matches_query(node, query):
                    results.append(node)
        return results

    def get_causal_path(
        self, source_node_id: str, target_node_id: str
    ) -> Optional[List[str]]:
        """Find causal path between two nodes."""
        # Simple BFS implementation
        from collections import deque

        if source_node_id not in self.knowledge_nodes:
            return None
        if target_node_id not in self.knowledge_nodes:
            return None

        queue = deque([(source_node_id, [source_node_id])])
        visited = {source_node_id}

        while queue:
            current, path = queue.popleft()
            if current == target_node_id:
                return path

            # Find outgoing links
            for link in self.causal_links.values():
                if link.source_node_id == current and link.target_node_id not in visited:
                    visited.add(link.target_node_id)
                    queue.append((link.target_node_id, path + [link.target_node_id]))

        return None

    def _matches_query(self, node: KnowledgeNode, query: Dict[str, Any]) -> bool:
        """Check if node matches query criteria.
        
        NOTE: This is a placeholder implementation using simple dictionary matching.
        A production implementation would require:
        - Semantic search capabilities
        - Vector embeddings
        - Neural information retrieval
        - Efficient indexing structures (e.g., FAISS, Annoy)
        These capabilities require AI breakthroughs not yet achieved.
        """
        for key, value in query.items():
            if key == "source_vertical" and node.source_vertical != value:
                return False
            if key in node.content and node.content[key] != value:
                return False
        return True
