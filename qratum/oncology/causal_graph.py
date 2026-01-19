"""
Causal Oncology Graph - VITRA Integration

This module implements symbolic causal graphs of oncogenesis, representing
cancer as a multi-scale dynamical system with:
- Genetic mutations
- Epigenetic dysregulation
- Tumor microenvironment feedback
- Immune evasion loops

The causal graph enables:
1. Control-theoretic modeling of tumor growth
2. Stability vs instability manifold analysis
3. Attractor state identification (remission vs progression)

RESEARCH USE ONLY - Not for clinical diagnosis or treatment.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


class CancerHallmark(Enum):
    """The Hallmarks of Cancer (Hanahan & Weinberg, 2011)."""

    SUSTAINING_PROLIFERATIVE_SIGNALING = "sustaining_proliferative_signaling"
    EVADING_GROWTH_SUPPRESSORS = "evading_growth_suppressors"
    RESISTING_CELL_DEATH = "resisting_cell_death"
    ENABLING_REPLICATIVE_IMMORTALITY = "enabling_replicative_immortality"
    INDUCING_ANGIOGENESIS = "inducing_angiogenesis"
    ACTIVATING_INVASION_METASTASIS = "activating_invasion_metastasis"
    DEREGULATING_CELLULAR_ENERGETICS = "deregulating_cellular_energetics"
    AVOIDING_IMMUNE_DESTRUCTION = "avoiding_immune_destruction"
    GENOME_INSTABILITY = "genome_instability"
    TUMOR_PROMOTING_INFLAMMATION = "tumor_promoting_inflammation"


class NodeType(Enum):
    """Types of nodes in the causal oncology graph."""

    MUTATION = "mutation"
    EPIGENETIC = "epigenetic"
    PATHWAY = "pathway"
    PROTEIN = "protein"
    METABOLITE = "metabolite"
    IMMUNE_CELL = "immune_cell"
    MICROENVIRONMENT = "microenvironment"
    PHENOTYPE = "phenotype"
    TREATMENT_TARGET = "treatment_target"


class EdgeType(Enum):
    """Types of causal relationships."""

    ACTIVATES = "activates"
    INHIBITS = "inhibits"
    UPREGULATES = "upregulates"
    DOWNREGULATES = "downregulates"
    INDUCES = "induces"
    SUPPRESSES = "suppresses"
    FEEDBACK_POSITIVE = "feedback_positive"
    FEEDBACK_NEGATIVE = "feedback_negative"
    EPIGENETIC_SILENCING = "epigenetic_silencing"
    EPIGENETIC_ACTIVATION = "epigenetic_activation"


@dataclass
class MutationState:
    """Represents a genetic mutation state.

    Attributes:
        gene: Gene symbol (e.g., 'TP53', 'KRAS', 'EGFR')
        mutation_type: Type of mutation (missense, nonsense, frameshift, etc.)
        variant: Specific variant notation (e.g., 'G12D', 'R248W')
        allele_frequency: Variant allele frequency (0.0 to 1.0)
        functional_impact: Predicted functional impact score
        driver_probability: Probability that this is a driver mutation
    """

    gene: str
    mutation_type: str
    variant: Optional[str] = None
    allele_frequency: float = 0.5
    functional_impact: float = 0.0
    driver_probability: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "gene": self.gene,
            "mutation_type": self.mutation_type,
            "variant": self.variant,
            "allele_frequency": self.allele_frequency,
            "functional_impact": self.functional_impact,
            "driver_probability": self.driver_probability,
        }


@dataclass
class EpigeneticState:
    """Represents epigenetic dysregulation state.

    Attributes:
        region: Genomic region or gene name
        modification_type: Type of modification (methylation, histone, etc.)
        level: Level of modification (0.0 to 1.0)
        silenced_genes: List of genes silenced by this modification
        activated_genes: List of genes activated by this modification
    """

    region: str
    modification_type: str
    level: float = 0.5
    silenced_genes: list[str] = field(default_factory=list)
    activated_genes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "region": self.region,
            "modification_type": self.modification_type,
            "level": self.level,
            "silenced_genes": self.silenced_genes,
            "activated_genes": self.activated_genes,
        }


@dataclass
class ImmuneEvasionMechanism:
    """Represents an immune evasion mechanism.

    Attributes:
        mechanism_name: Name of the evasion mechanism
        target_immune_cells: Immune cell types affected
        checkpoint_molecules: Checkpoint molecules involved
        efficacy: Efficacy of evasion (0.0 to 1.0)
        reversibility: Probability of reversing this mechanism
    """

    mechanism_name: str
    target_immune_cells: list[str] = field(default_factory=list)
    checkpoint_molecules: list[str] = field(default_factory=list)
    efficacy: float = 0.5
    reversibility: float = 0.5

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "mechanism_name": self.mechanism_name,
            "target_immune_cells": self.target_immune_cells,
            "checkpoint_molecules": self.checkpoint_molecules,
            "efficacy": self.efficacy,
            "reversibility": self.reversibility,
        }


@dataclass
class TumorMicroenvironment:
    """Represents tumor microenvironment state.

    Attributes:
        hypoxia_level: Level of hypoxia (0.0 to 1.0)
        acidosis_level: Level of acidosis (0.0 to 1.0)
        immune_infiltration: Dictionary of immune cell infiltration levels
        stromal_activation: Level of stromal cell activation
        angiogenesis_score: Level of angiogenesis
        metabolic_reprogramming: Dictionary of metabolic alterations
    """

    hypoxia_level: float = 0.0
    acidosis_level: float = 0.0
    immune_infiltration: dict[str, float] = field(default_factory=dict)
    stromal_activation: float = 0.0
    angiogenesis_score: float = 0.0
    metabolic_reprogramming: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "hypoxia_level": self.hypoxia_level,
            "acidosis_level": self.acidosis_level,
            "immune_infiltration": self.immune_infiltration,
            "stromal_activation": self.stromal_activation,
            "angiogenesis_score": self.angiogenesis_score,
            "metabolic_reprogramming": self.metabolic_reprogramming,
        }


@dataclass
class OncogenicNode:
    """A node in the causal oncology graph.

    Represents a biological entity involved in oncogenesis.

    Attributes:
        node_id: Unique identifier
        name: Human-readable name
        node_type: Type of node (mutation, pathway, etc.)
        hallmarks: Associated cancer hallmarks
        state_data: State-specific data (mutation, epigenetic, etc.)
        activity_level: Current activity level (0.0 to 1.0)
        stability: Stability score (higher = more stable state)
        druggability: Druggability score (0.0 to 1.0)
    """

    node_id: str
    name: str
    node_type: NodeType
    hallmarks: list[CancerHallmark] = field(default_factory=list)
    state_data: Optional[Any] = None
    activity_level: float = 0.5
    stability: float = 0.5
    druggability: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        state_data_dict = None
        if self.state_data is not None and hasattr(self.state_data, "to_dict"):
            state_data_dict = self.state_data.to_dict()

        return {
            "node_id": self.node_id,
            "name": self.name,
            "node_type": self.node_type.value,
            "hallmarks": [h.value for h in self.hallmarks],
            "state_data": state_data_dict,
            "activity_level": self.activity_level,
            "stability": self.stability,
            "druggability": self.druggability,
            "metadata": self.metadata,
        }


@dataclass
class CausalEdge:
    """A causal edge in the oncology graph.

    Represents a causal relationship between two nodes.

    Attributes:
        source_id: Source node ID
        target_id: Target node ID
        edge_type: Type of causal relationship
        strength: Strength of the causal effect (-1.0 to 1.0)
        confidence: Confidence in this causal relationship (0.0 to 1.0)
        evidence_sources: Literature sources supporting this edge
        time_delay: Estimated time delay for the effect (hours)
        is_therapeutic_target: Whether this edge is a potential therapeutic target
    """

    source_id: str
    target_id: str
    edge_type: EdgeType
    strength: float = 0.5
    confidence: float = 0.5
    evidence_sources: list[str] = field(default_factory=list)
    time_delay: float = 0.0
    is_therapeutic_target: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "edge_type": self.edge_type.value,
            "strength": self.strength,
            "confidence": self.confidence,
            "evidence_sources": self.evidence_sources,
            "time_delay": self.time_delay,
            "is_therapeutic_target": self.is_therapeutic_target,
        }


class CausalOncologyGraph:
    """
    VITRA-integrated causal graph for oncogenesis modeling.

    This class represents cancer as a multi-scale dynamical system,
    enabling control-theoretic analysis of tumor growth and treatment response.

    Key capabilities:
    1. Build symbolic causal graphs from multi-omic data
    2. Identify feedback loops and attractor states
    3. Compute stability manifolds for treatment response
    4. Generate intervention targets via causal inference

    RESEARCH USE ONLY - Not for clinical diagnosis or treatment.
    """

    # Research disclaimer
    RESEARCH_DISCLAIMER = """
    This causal model is for RESEARCH PURPOSES ONLY.
    Outputs require experimental validation and clinical oversight.
    Not approved for clinical decision-making.
    """

    def __init__(
        self,
        name: str,
        cancer_type: Optional[str] = None,
        seed: int = 42,
    ) -> None:
        """Initialize the causal oncology graph.

        Args:
            name: Graph identifier
            cancer_type: Type of cancer being modeled (e.g., 'NSCLC', 'CRC')
            seed: Random seed for reproducibility
        """
        self.name = name
        self.cancer_type = cancer_type
        self.seed = seed
        self.rng = np.random.RandomState(seed)

        self._nodes: dict[str, OncogenicNode] = {}
        self._edges: dict[str, CausalEdge] = {}
        self._adjacency: dict[str, list[str]] = {}
        self._reverse_adjacency: dict[str, list[str]] = {}

        self.created_at = datetime.now(timezone.utc).isoformat()
        self.provenance: list[str] = []

        logger.info(
            f"Initialized CausalOncologyGraph: {name} " f"(cancer_type={cancer_type}, seed={seed})"
        )

    def add_node(self, node: OncogenicNode) -> None:
        """Add a node to the graph.

        Args:
            node: OncogenicNode to add
        """
        if node.node_id in self._nodes:
            logger.warning(f"Node {node.node_id} already exists, updating")

        self._nodes[node.node_id] = node
        if node.node_id not in self._adjacency:
            self._adjacency[node.node_id] = []
        if node.node_id not in self._reverse_adjacency:
            self._reverse_adjacency[node.node_id] = []

    def add_edge(self, edge: CausalEdge) -> None:
        """Add a causal edge to the graph.

        Args:
            edge: CausalEdge to add

        Raises:
            ValueError: If source or target node doesn't exist
        """
        if edge.source_id not in self._nodes:
            raise ValueError(f"Source node '{edge.source_id}' not found")
        if edge.target_id not in self._nodes:
            raise ValueError(f"Target node '{edge.target_id}' not found")

        edge_key = f"{edge.source_id}->{edge.target_id}"
        self._edges[edge_key] = edge
        self._adjacency[edge.source_id].append(edge.target_id)
        self._reverse_adjacency[edge.target_id].append(edge.source_id)

    def get_node(self, node_id: str) -> Optional[OncogenicNode]:
        """Get a node by ID."""
        return self._nodes.get(node_id)

    def get_edge(self, source_id: str, target_id: str) -> Optional[CausalEdge]:
        """Get an edge by source and target IDs."""
        edge_key = f"{source_id}->{target_id}"
        return self._edges.get(edge_key)

    def get_downstream_nodes(self, node_id: str) -> list[str]:
        """Get all nodes downstream of the given node."""
        return self._adjacency.get(node_id, [])

    def get_upstream_nodes(self, node_id: str) -> list[str]:
        """Get all nodes upstream of the given node."""
        return self._reverse_adjacency.get(node_id, [])

    def find_feedback_loops(self) -> list[list[str]]:
        """Identify feedback loops in the causal graph.

        Returns:
            List of feedback loops (each loop is a list of node IDs)
        """
        loops = []
        visited = set()
        path = []
        path_set = set()

        def dfs(node_id: str) -> None:
            if node_id in path_set:
                # Found a cycle
                cycle_start = path.index(node_id)
                cycle = path[cycle_start:] + [node_id]
                loops.append(cycle)
                return

            if node_id in visited:
                return

            visited.add(node_id)
            path.append(node_id)
            path_set.add(node_id)

            for neighbor in self._adjacency.get(node_id, []):
                dfs(neighbor)

            path.pop()
            path_set.remove(node_id)

        for node_id in self._nodes:
            dfs(node_id)

        return loops

    def compute_stability_manifold(
        self,
        perturbation_magnitude: float = 0.1,
        n_simulations: int = 100,
    ) -> dict[str, float]:
        """Compute stability scores for each node.

        Uses Monte Carlo simulation to estimate how perturbations
        propagate through the causal graph.

        Args:
            perturbation_magnitude: Size of random perturbations
            n_simulations: Number of simulation runs

        Returns:
            Dictionary mapping node IDs to stability scores
        """
        stability_scores = {}

        for node_id in self._nodes:
            total_perturbation = 0.0

            for _ in range(n_simulations):
                # Apply random perturbation to this node
                perturbation = self.rng.uniform(-perturbation_magnitude, perturbation_magnitude)

                # Propagate through downstream nodes
                propagated = self._propagate_perturbation(node_id, perturbation)
                total_perturbation += abs(propagated)

            # Stability is inverse of average perturbation propagation
            avg_propagation = total_perturbation / n_simulations
            stability_scores[node_id] = 1.0 / (1.0 + avg_propagation)

        return stability_scores

    def _propagate_perturbation(
        self, source_id: str, initial_perturbation: float, max_depth: int = 10
    ) -> float:
        """Propagate a perturbation through the causal graph.

        Args:
            source_id: Starting node for perturbation
            initial_perturbation: Initial perturbation magnitude
            max_depth: Maximum propagation depth

        Returns:
            Total propagated perturbation across all downstream nodes
        """
        total = abs(initial_perturbation)
        visited = {source_id}
        current = [(source_id, initial_perturbation)]
        depth = 0

        while current and depth < max_depth:
            next_level = []

            for node_id, perturbation in current:
                for neighbor in self._adjacency.get(node_id, []):
                    if neighbor not in visited:
                        edge = self.get_edge(node_id, neighbor)
                        if edge:
                            # Scale perturbation by edge strength
                            propagated = perturbation * edge.strength
                            total += abs(propagated)
                            visited.add(neighbor)
                            next_level.append((neighbor, propagated))

            current = next_level
            depth += 1

        return total

    def identify_attractor_states(
        self,
        activity_threshold: float = 0.5,
    ) -> dict[str, str]:
        """Identify attractor states in the system.

        Attractor states represent stable configurations that the
        tumor system tends toward (e.g., progression vs remission).

        Args:
            activity_threshold: Threshold for node activity classification

        Returns:
            Dictionary mapping node IDs to attractor classification
            ('progression', 'remission', 'unstable')
        """
        attractor_states = {}
        stability = self.compute_stability_manifold()

        for node_id, node in self._nodes.items():
            node_stability = stability.get(node_id, 0.5)

            if node_stability > 0.7:
                # Stable state
                if node.activity_level > activity_threshold:
                    # High activity stable state = progression
                    attractor_states[node_id] = "progression"
                else:
                    # Low activity stable state = remission
                    attractor_states[node_id] = "remission"
            else:
                # Unstable state
                attractor_states[node_id] = "unstable"

        return attractor_states

    def find_intervention_targets(
        self,
        min_druggability: float = 0.3,
        max_targets: int = 10,
    ) -> list[OncogenicNode]:
        """Identify potential intervention targets.

        Ranks nodes by their potential as therapeutic targets based on:
        - Druggability score
        - Downstream influence (number of affected nodes)
        - Involvement in feedback loops

        Args:
            min_druggability: Minimum druggability score
            max_targets: Maximum number of targets to return

        Returns:
            List of OncogenicNodes ranked by therapeutic potential
        """
        candidates = []
        feedback_loops = self.find_feedback_loops()

        # Count feedback loop participation
        loop_participation = {}
        for loop in feedback_loops:
            for node_id in loop:
                loop_participation[node_id] = loop_participation.get(node_id, 0) + 1

        for node_id, node in self._nodes.items():
            if node.druggability < min_druggability:
                continue

            # Calculate downstream influence
            downstream = self._get_all_downstream(node_id)
            downstream_influence = len(downstream)

            # Calculate therapeutic score
            score = (
                node.druggability * 0.4
                + (downstream_influence / max(len(self._nodes), 1)) * 0.3
                + (loop_participation.get(node_id, 0) / max(len(feedback_loops), 1)) * 0.3
            )

            candidates.append((node, score))

        # Sort by score and return top targets
        candidates.sort(key=lambda x: x[1], reverse=True)
        return [node for node, _ in candidates[:max_targets]]

    def _get_all_downstream(self, node_id: str) -> set[str]:
        """Get all nodes downstream of the given node."""
        visited = set()
        to_visit = [node_id]

        while to_visit:
            current = to_visit.pop()
            if current in visited:
                continue
            visited.add(current)
            to_visit.extend(self._adjacency.get(current, []))

        visited.discard(node_id)  # Remove the starting node
        return visited

    def get_causal_paths(
        self,
        source_id: str,
        target_id: str,
        max_paths: int = 10,
    ) -> list[list[str]]:
        """Find all causal paths between two nodes.

        Args:
            source_id: Starting node
            target_id: Ending node
            max_paths: Maximum number of paths to return

        Returns:
            List of paths (each path is a list of node IDs)
        """
        if source_id not in self._nodes or target_id not in self._nodes:
            return []

        paths = []
        visited = set()

        def dfs(current: str, path: list[str]) -> None:
            if len(paths) >= max_paths:
                return

            if current == target_id:
                paths.append(path.copy())
                return

            if current in visited:
                return

            visited.add(current)
            for neighbor in self._adjacency.get(current, []):
                path.append(neighbor)
                dfs(neighbor, path)
                path.pop()
            visited.remove(current)

        dfs(source_id, [source_id])
        return paths

    def compute_graph_hash(self) -> str:
        """Compute unique hash of graph topology and parameters."""
        graph_dict = {
            "nodes": sorted(
                [n.to_dict() for n in self._nodes.values()], key=lambda x: x["node_id"]
            ),
            "edges": sorted(
                [e.to_dict() for e in self._edges.values()],
                key=lambda x: (x["source_id"], x["target_id"]),
            ),
        }
        graph_json = json.dumps(graph_dict, sort_keys=True)
        return hashlib.sha256(graph_json.encode()).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        """Serialize graph to dictionary."""
        return {
            "name": self.name,
            "cancer_type": self.cancer_type,
            "seed": self.seed,
            "created_at": self.created_at,
            "provenance": self.provenance,
            "nodes": [n.to_dict() for n in self._nodes.values()],
            "edges": [e.to_dict() for e in self._edges.values()],
            "hash": self.compute_graph_hash(),
            "disclaimer": self.RESEARCH_DISCLAIMER,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CausalOncologyGraph:
        """Deserialize graph from dictionary."""
        graph = cls(
            name=data["name"],
            cancer_type=data.get("cancer_type"),
            seed=data.get("seed", 42),
        )
        graph.created_at = data.get("created_at", graph.created_at)
        graph.provenance = data.get("provenance", [])

        # Reconstruct nodes
        for node_data in data.get("nodes", []):
            node = OncogenicNode(
                node_id=node_data["node_id"],
                name=node_data["name"],
                node_type=NodeType(node_data["node_type"]),
                hallmarks=[CancerHallmark(h) for h in node_data.get("hallmarks", [])],
                activity_level=node_data.get("activity_level", 0.5),
                stability=node_data.get("stability", 0.5),
                druggability=node_data.get("druggability", 0.0),
                metadata=node_data.get("metadata", {}),
            )
            graph.add_node(node)

        # Reconstruct edges
        for edge_data in data.get("edges", []):
            edge = CausalEdge(
                source_id=edge_data["source_id"],
                target_id=edge_data["target_id"],
                edge_type=EdgeType(edge_data["edge_type"]),
                strength=edge_data.get("strength", 0.5),
                confidence=edge_data.get("confidence", 0.5),
                evidence_sources=edge_data.get("evidence_sources", []),
                time_delay=edge_data.get("time_delay", 0.0),
                is_therapeutic_target=edge_data.get("is_therapeutic_target", False),
            )
            graph.add_edge(edge)

        return graph

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"CausalOncologyGraph(name='{self.name}', "
            f"cancer_type='{self.cancer_type}', "
            f"nodes={len(self._nodes)}, "
            f"edges={len(self._edges)})"
        )


def create_example_egfr_graph() -> CausalOncologyGraph:
    """Create an example EGFR-driven cancer causal graph.

    This is a simplified model for demonstration purposes.
    Real models require extensive literature and data curation.

    Returns:
        CausalOncologyGraph modeling EGFR-driven oncogenesis
    """
    graph = CausalOncologyGraph(name="EGFR_Model", cancer_type="NSCLC", seed=42)

    # Add mutation node
    egfr_mutation = OncogenicNode(
        node_id="EGFR_mut",
        name="EGFR Activating Mutation",
        node_type=NodeType.MUTATION,
        hallmarks=[CancerHallmark.SUSTAINING_PROLIFERATIVE_SIGNALING],
        state_data=MutationState(
            gene="EGFR",
            mutation_type="missense",
            variant="L858R",
            driver_probability=0.95,
        ),
        activity_level=0.9,
        druggability=0.85,
    )
    graph.add_node(egfr_mutation)

    # Add downstream signaling
    ras_pathway = OncogenicNode(
        node_id="RAS_pathway",
        name="RAS/MAPK Pathway Activation",
        node_type=NodeType.PATHWAY,
        hallmarks=[CancerHallmark.SUSTAINING_PROLIFERATIVE_SIGNALING],
        activity_level=0.8,
        druggability=0.6,
    )
    graph.add_node(ras_pathway)

    pi3k_pathway = OncogenicNode(
        node_id="PI3K_pathway",
        name="PI3K/AKT/mTOR Pathway",
        node_type=NodeType.PATHWAY,
        hallmarks=[
            CancerHallmark.SUSTAINING_PROLIFERATIVE_SIGNALING,
            CancerHallmark.RESISTING_CELL_DEATH,
        ],
        activity_level=0.75,
        druggability=0.7,
    )
    graph.add_node(pi3k_pathway)

    # Add proliferation phenotype
    proliferation = OncogenicNode(
        node_id="proliferation",
        name="Uncontrolled Proliferation",
        node_type=NodeType.PHENOTYPE,
        hallmarks=[CancerHallmark.SUSTAINING_PROLIFERATIVE_SIGNALING],
        activity_level=0.85,
        druggability=0.0,
    )
    graph.add_node(proliferation)

    # Add edges
    graph.add_edge(
        CausalEdge(
            source_id="EGFR_mut",
            target_id="RAS_pathway",
            edge_type=EdgeType.ACTIVATES,
            strength=0.9,
            confidence=0.95,
            is_therapeutic_target=True,
        )
    )
    graph.add_edge(
        CausalEdge(
            source_id="EGFR_mut",
            target_id="PI3K_pathway",
            edge_type=EdgeType.ACTIVATES,
            strength=0.85,
            confidence=0.9,
            is_therapeutic_target=True,
        )
    )
    graph.add_edge(
        CausalEdge(
            source_id="RAS_pathway",
            target_id="proliferation",
            edge_type=EdgeType.INDUCES,
            strength=0.8,
            confidence=0.85,
        )
    )
    graph.add_edge(
        CausalEdge(
            source_id="PI3K_pathway",
            target_id="proliferation",
            edge_type=EdgeType.INDUCES,
            strength=0.75,
            confidence=0.85,
        )
    )

    graph.provenance.append("Created as example EGFR-driven NSCLC model")
    return graph
