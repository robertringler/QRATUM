"""Extended Domain Registry for SI Transition.

Manages the extended set of cognitive domains beyond the original 14
QRATUM verticals, including domain definitions, interconnections, and
capability mappings for cross-domain reasoning.

Maintains backward compatibility with existing verticals while enabling
arbitrary domain expansion under safety constraints.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from qratum_asi.generalization.types import (
    CognitiveDomain,
    DomainCapability,
    DomainCapabilityProfile,
    SynthesisSafetyLevel,
)


@dataclass
class DomainDefinition:
    """Complete definition of a cognitive domain.

    Attributes:
        domain: The cognitive domain identifier
        name: Human-readable name
        description: Detailed description
        capabilities: Available capabilities in this domain
        abstraction_level: How abstract the domain is (0-1)
        formalization_level: How formalized/axiomatic (0-1)
        parent_domains: Domains this domain derives from
        child_domains: Specialized sub-domains
        safety_constraints: Domain-specific safety constraints
        requires_expertise: Whether human expert review is required
        is_core_vertical: Whether this is an original QRATUM vertical
    """

    domain: CognitiveDomain
    name: str
    description: str
    capabilities: list[DomainCapability]
    abstraction_level: float
    formalization_level: float
    parent_domains: list[CognitiveDomain] = field(default_factory=list)
    child_domains: list[CognitiveDomain] = field(default_factory=list)
    safety_constraints: list[str] = field(default_factory=list)
    requires_expertise: bool = True
    is_core_vertical: bool = False

    def to_capability_profile(self) -> DomainCapabilityProfile:
        """Convert to immutable capability profile."""
        interconnected = list(set(self.parent_domains + self.child_domains))
        return DomainCapabilityProfile(
            domain=self.domain,
            primary_capabilities=tuple(self.capabilities),
            interconnected_domains=tuple(interconnected),
            abstraction_level=self.abstraction_level,
            formalization_level=self.formalization_level,
        )


@dataclass
class DomainInterconnection:
    """Represents a connection between two cognitive domains.

    Models how knowledge and methods can transfer between domains
    for cross-domain synthesis.

    Attributes:
        source_domain: Origin domain
        target_domain: Destination domain
        connection_strength: How strong the connection is (0-1)
        transfer_mechanisms: How knowledge transfers
        bidirectional: Whether transfer works both ways
        requires_translation: Whether semantic translation is needed
        safety_level: Safety classification for this connection
    """

    source_domain: CognitiveDomain
    target_domain: CognitiveDomain
    connection_strength: float
    transfer_mechanisms: list[str]
    bidirectional: bool = True
    requires_translation: bool = False
    safety_level: SynthesisSafetyLevel = SynthesisSafetyLevel.ROUTINE


class ExtendedDomainRegistry:
    """Registry managing all cognitive domains and their interconnections.

    Provides a comprehensive catalog of domains, their capabilities,
    and relationships for enabling general reasoning across arbitrary
    cognitive territory.

    Enforces:
    - Backward compatibility with original 14 verticals
    - Safety constraints on domain interactions
    - Human oversight for sensitive domain combinations
    - Provenance tracking for domain additions
    """

    # Original QRATUM verticals for backward compatibility
    CORE_VERTICALS = frozenset(
        [
            CognitiveDomain.VITRA,
            CognitiveDomain.CAPRA,
            CognitiveDomain.STRATA,
            CognitiveDomain.ECORA,
            CognitiveDomain.NEURA,
            CognitiveDomain.FLUXA,
            CognitiveDomain.CHRONA,
            CognitiveDomain.COHORA,
            CognitiveDomain.FUSIA,
            CognitiveDomain.GEONA,
            CognitiveDomain.JURIS,
            CognitiveDomain.ORBIA,
            CognitiveDomain.SENTRA,
            CognitiveDomain.VEXOR,
        ]
    )

    def __init__(self):
        """Initialize the domain registry with all domains."""
        self.domains: dict[CognitiveDomain, DomainDefinition] = {}
        self.interconnections: dict[
            tuple[CognitiveDomain, CognitiveDomain], DomainInterconnection
        ] = {}
        self.capability_index: dict[DomainCapability, list[CognitiveDomain]] = {}
        self._registry_version = "1.0.0"
        self._last_updated = datetime.utcnow().isoformat()

        # Initialize all domains
        self._initialize_core_domains()
        self._initialize_extended_domains()
        self._build_interconnections()
        self._build_capability_index()

    def _initialize_core_domains(self) -> None:
        """Initialize the original 14 QRATUM verticals."""
        core_definitions = [
            DomainDefinition(
                domain=CognitiveDomain.VITRA,
                name="VITRA - Life Sciences",
                description="Drug discovery, genomics, and biological systems",
                capabilities=[
                    DomainCapability.ANALYSIS,
                    DomainCapability.PREDICTION,
                    DomainCapability.OPTIMIZATION,
                    DomainCapability.GENERATION,
                ],
                abstraction_level=0.5,
                formalization_level=0.7,
                is_core_vertical=True,
                safety_constraints=["medical_safety", "regulatory_compliance"],
            ),
            DomainDefinition(
                domain=CognitiveDomain.CAPRA,
                name="CAPRA - Financial Modeling",
                description="Economic and financial systems analysis",
                capabilities=[
                    DomainCapability.ANALYSIS,
                    DomainCapability.PREDICTION,
                    DomainCapability.OPTIMIZATION,
                ],
                abstraction_level=0.6,
                formalization_level=0.8,
                is_core_vertical=True,
                safety_constraints=["market_integrity", "regulatory_compliance"],
            ),
            DomainDefinition(
                domain=CognitiveDomain.STRATA,
                name="STRATA - Strategic Planning",
                description="Long-term strategic analysis and planning",
                capabilities=[
                    DomainCapability.ANALYSIS,
                    DomainCapability.SYNTHESIS,
                    DomainCapability.PREDICTION,
                ],
                abstraction_level=0.7,
                formalization_level=0.5,
                is_core_vertical=True,
            ),
            DomainDefinition(
                domain=CognitiveDomain.ECORA,
                name="ECORA - Environmental Systems",
                description="Climate and environmental modeling",
                capabilities=[
                    DomainCapability.ANALYSIS,
                    DomainCapability.PREDICTION,
                    DomainCapability.SYNTHESIS,
                ],
                abstraction_level=0.5,
                formalization_level=0.6,
                is_core_vertical=True,
            ),
            DomainDefinition(
                domain=CognitiveDomain.NEURA,
                name="NEURA - Neuroscience",
                description="Neural systems and cognitive science",
                capabilities=[
                    DomainCapability.ANALYSIS,
                    DomainCapability.EXPLANATION,
                    DomainCapability.PREDICTION,
                ],
                abstraction_level=0.6,
                formalization_level=0.6,
                is_core_vertical=True,
            ),
            DomainDefinition(
                domain=CognitiveDomain.FLUXA,
                name="FLUXA - Complex Systems",
                description="Dynamical systems and complexity science",
                capabilities=[
                    DomainCapability.ANALYSIS,
                    DomainCapability.SYNTHESIS,
                    DomainCapability.ABSTRACTION,
                ],
                abstraction_level=0.8,
                formalization_level=0.7,
                is_core_vertical=True,
            ),
            DomainDefinition(
                domain=CognitiveDomain.CHRONA,
                name="CHRONA - Temporal Analysis",
                description="Time series and temporal patterns",
                capabilities=[
                    DomainCapability.ANALYSIS,
                    DomainCapability.PREDICTION,
                ],
                abstraction_level=0.6,
                formalization_level=0.8,
                is_core_vertical=True,
            ),
            DomainDefinition(
                domain=CognitiveDomain.COHORA,
                name="COHORA - Social Systems",
                description="Social dynamics and collective behavior",
                capabilities=[
                    DomainCapability.ANALYSIS,
                    DomainCapability.PREDICTION,
                    DomainCapability.EXPLANATION,
                ],
                abstraction_level=0.6,
                formalization_level=0.4,
                is_core_vertical=True,
            ),
            DomainDefinition(
                domain=CognitiveDomain.FUSIA,
                name="FUSIA - Integration",
                description="Cross-domain synthesis and integration",
                capabilities=[
                    DomainCapability.SYNTHESIS,
                    DomainCapability.ANALOGICAL,
                    DomainCapability.ABSTRACTION,
                ],
                abstraction_level=0.9,
                formalization_level=0.5,
                is_core_vertical=True,
            ),
            DomainDefinition(
                domain=CognitiveDomain.GEONA,
                name="GEONA - Geospatial",
                description="Geographic and spatial analysis",
                capabilities=[
                    DomainCapability.ANALYSIS,
                    DomainCapability.SYNTHESIS,
                ],
                abstraction_level=0.4,
                formalization_level=0.7,
                is_core_vertical=True,
            ),
            DomainDefinition(
                domain=CognitiveDomain.JURIS,
                name="JURIS - Legal",
                description="Legal analysis and regulatory compliance",
                capabilities=[
                    DomainCapability.ANALYSIS,
                    DomainCapability.VERIFICATION,
                    DomainCapability.EXPLANATION,
                ],
                abstraction_level=0.5,
                formalization_level=0.8,
                is_core_vertical=True,
                safety_constraints=["legal_accuracy", "jurisdiction_awareness"],
            ),
            DomainDefinition(
                domain=CognitiveDomain.ORBIA,
                name="ORBIA - Global Systems",
                description="Global interconnected systems",
                capabilities=[
                    DomainCapability.ANALYSIS,
                    DomainCapability.SYNTHESIS,
                    DomainCapability.PREDICTION,
                ],
                abstraction_level=0.7,
                formalization_level=0.5,
                is_core_vertical=True,
            ),
            DomainDefinition(
                domain=CognitiveDomain.SENTRA,
                name="SENTRA - Sensing",
                description="Perception and sensor systems",
                capabilities=[
                    DomainCapability.ANALYSIS,
                    DomainCapability.SYNTHESIS,
                ],
                abstraction_level=0.4,
                formalization_level=0.7,
                is_core_vertical=True,
            ),
            DomainDefinition(
                domain=CognitiveDomain.VEXOR,
                name="VEXOR - Problem Solving",
                description="Complex problem solving and optimization",
                capabilities=[
                    DomainCapability.ANALYSIS,
                    DomainCapability.OPTIMIZATION,
                    DomainCapability.GENERATION,
                ],
                abstraction_level=0.8,
                formalization_level=0.6,
                is_core_vertical=True,
            ),
        ]

        for definition in core_definitions:
            self.domains[definition.domain] = definition

    def _initialize_extended_domains(self) -> None:
        """Initialize extended domains for SI capability."""
        extended_definitions = [
            DomainDefinition(
                domain=CognitiveDomain.MATHEMATICS,
                name="Mathematics",
                description="Pure and applied mathematics including algebra, analysis, topology, number theory",
                capabilities=[
                    DomainCapability.ANALYSIS,
                    DomainCapability.VERIFICATION,
                    DomainCapability.ABSTRACTION,
                    DomainCapability.GENERATION,
                ],
                abstraction_level=0.95,
                formalization_level=0.99,
                child_domains=[CognitiveDomain.LOGIC],
            ),
            DomainDefinition(
                domain=CognitiveDomain.PHYSICS,
                name="Physics",
                description="Fundamental physics including mechanics, quantum theory, relativity, thermodynamics",
                capabilities=[
                    DomainCapability.ANALYSIS,
                    DomainCapability.PREDICTION,
                    DomainCapability.EXPLANATION,
                    DomainCapability.VERIFICATION,
                ],
                abstraction_level=0.8,
                formalization_level=0.9,
                parent_domains=[CognitiveDomain.MATHEMATICS],
            ),
            DomainDefinition(
                domain=CognitiveDomain.ENGINEERING,
                name="Engineering",
                description="All engineering disciplines: mechanical, electrical, civil, software, chemical",
                capabilities=[
                    DomainCapability.ANALYSIS,
                    DomainCapability.OPTIMIZATION,
                    DomainCapability.GENERATION,
                    DomainCapability.VERIFICATION,
                ],
                abstraction_level=0.5,
                formalization_level=0.7,
                parent_domains=[CognitiveDomain.PHYSICS, CognitiveDomain.MATHEMATICS],
            ),
            DomainDefinition(
                domain=CognitiveDomain.PHILOSOPHY,
                name="Philosophy",
                description="Epistemology, ethics, metaphysics, logic, philosophy of mind",
                capabilities=[
                    DomainCapability.ANALYSIS,
                    DomainCapability.ABSTRACTION,
                    DomainCapability.EXPLANATION,
                    DomainCapability.ANALOGICAL,
                ],
                abstraction_level=0.95,
                formalization_level=0.6,
                child_domains=[CognitiveDomain.LOGIC],
            ),
            DomainDefinition(
                domain=CognitiveDomain.GEOPOLITICS,
                name="Geopolitics",
                description="International relations, strategic studies, political analysis",
                capabilities=[
                    DomainCapability.ANALYSIS,
                    DomainCapability.PREDICTION,
                    DomainCapability.SYNTHESIS,
                ],
                abstraction_level=0.6,
                formalization_level=0.4,
                safety_constraints=["political_neutrality", "no_regime_preference"],
            ),
            DomainDefinition(
                domain=CognitiveDomain.CREATIVE_ARTS,
                name="Creative Arts",
                description="Music, visual arts, literature, creative expression",
                capabilities=[
                    DomainCapability.GENERATION,
                    DomainCapability.CREATIVE,
                    DomainCapability.ANALOGICAL,
                ],
                abstraction_level=0.7,
                formalization_level=0.2,
                safety_constraints=["respect_copyright", "no_harmful_content"],
            ),
            DomainDefinition(
                domain=CognitiveDomain.SOCIAL_DYNAMICS,
                name="Social Dynamics",
                description="Sociology, psychology, anthropology, behavioral science",
                capabilities=[
                    DomainCapability.ANALYSIS,
                    DomainCapability.PREDICTION,
                    DomainCapability.EXPLANATION,
                ],
                abstraction_level=0.6,
                formalization_level=0.5,
                parent_domains=[CognitiveDomain.COHORA],
                safety_constraints=["no_manipulation", "respect_privacy"],
            ),
            DomainDefinition(
                domain=CognitiveDomain.COMPUTER_SCIENCE,
                name="Computer Science",
                description="Algorithms, systems, AI, programming languages, complexity theory",
                capabilities=[
                    DomainCapability.ANALYSIS,
                    DomainCapability.GENERATION,
                    DomainCapability.OPTIMIZATION,
                    DomainCapability.VERIFICATION,
                ],
                abstraction_level=0.8,
                formalization_level=0.9,
                parent_domains=[CognitiveDomain.MATHEMATICS, CognitiveDomain.LOGIC],
            ),
            DomainDefinition(
                domain=CognitiveDomain.CHEMISTRY,
                name="Chemistry",
                description="Molecular science, materials, biochemistry",
                capabilities=[
                    DomainCapability.ANALYSIS,
                    DomainCapability.PREDICTION,
                    DomainCapability.GENERATION,
                ],
                abstraction_level=0.6,
                formalization_level=0.8,
                parent_domains=[CognitiveDomain.PHYSICS],
                child_domains=[CognitiveDomain.VITRA],
            ),
            DomainDefinition(
                domain=CognitiveDomain.BIOLOGY,
                name="Biology",
                description="Fundamental biology, evolution, ecology, genetics",
                capabilities=[
                    DomainCapability.ANALYSIS,
                    DomainCapability.PREDICTION,
                    DomainCapability.EXPLANATION,
                ],
                abstraction_level=0.5,
                formalization_level=0.6,
                parent_domains=[CognitiveDomain.CHEMISTRY],
                child_domains=[CognitiveDomain.VITRA, CognitiveDomain.ECORA],
            ),
            DomainDefinition(
                domain=CognitiveDomain.LINGUISTICS,
                name="Linguistics",
                description="Language structure, semantics, pragmatics, communication theory",
                capabilities=[
                    DomainCapability.ANALYSIS,
                    DomainCapability.ABSTRACTION,
                    DomainCapability.GENERATION,
                ],
                abstraction_level=0.7,
                formalization_level=0.7,
            ),
            DomainDefinition(
                domain=CognitiveDomain.HISTORY,
                name="History",
                description="Historical analysis, pattern recognition across time",
                capabilities=[
                    DomainCapability.ANALYSIS,
                    DomainCapability.EXPLANATION,
                    DomainCapability.ANALOGICAL,
                ],
                abstraction_level=0.5,
                formalization_level=0.3,
                parent_domains=[CognitiveDomain.CHRONA],
            ),
            DomainDefinition(
                domain=CognitiveDomain.ASTRONOMY,
                name="Astronomy",
                description="Cosmology, astrophysics, planetary science",
                capabilities=[
                    DomainCapability.ANALYSIS,
                    DomainCapability.PREDICTION,
                    DomainCapability.EXPLANATION,
                ],
                abstraction_level=0.7,
                formalization_level=0.8,
                parent_domains=[CognitiveDomain.PHYSICS],
            ),
            DomainDefinition(
                domain=CognitiveDomain.LOGIC,
                name="Formal Logic",
                description="Propositional logic, predicate logic, proof theory, model theory",
                capabilities=[
                    DomainCapability.ANALYSIS,
                    DomainCapability.VERIFICATION,
                    DomainCapability.ABSTRACTION,
                ],
                abstraction_level=0.99,
                formalization_level=1.0,
                parent_domains=[CognitiveDomain.MATHEMATICS, CognitiveDomain.PHILOSOPHY],
            ),
            DomainDefinition(
                domain=CognitiveDomain.ECONOMICS,
                name="Economics",
                description="Economic theory, mechanism design, market dynamics",
                capabilities=[
                    DomainCapability.ANALYSIS,
                    DomainCapability.PREDICTION,
                    DomainCapability.OPTIMIZATION,
                ],
                abstraction_level=0.7,
                formalization_level=0.7,
                parent_domains=[CognitiveDomain.MATHEMATICS],
                child_domains=[CognitiveDomain.CAPRA],
            ),
            DomainDefinition(
                domain=CognitiveDomain.GAME_THEORY,
                name="Game Theory",
                description="Strategic interaction, mechanism design, equilibrium analysis",
                capabilities=[
                    DomainCapability.ANALYSIS,
                    DomainCapability.OPTIMIZATION,
                    DomainCapability.PREDICTION,
                ],
                abstraction_level=0.85,
                formalization_level=0.9,
                parent_domains=[CognitiveDomain.MATHEMATICS, CognitiveDomain.ECONOMICS],
            ),
        ]

        for definition in extended_definitions:
            self.domains[definition.domain] = definition

    def _build_interconnections(self) -> None:
        """Build domain interconnection graph."""
        # Define key interconnections with transfer mechanisms
        connections = [
            # Mathematics → Other Sciences
            DomainInterconnection(
                source_domain=CognitiveDomain.MATHEMATICS,
                target_domain=CognitiveDomain.PHYSICS,
                connection_strength=0.95,
                transfer_mechanisms=["formal_modeling", "proof_techniques", "abstraction"],
            ),
            DomainInterconnection(
                source_domain=CognitiveDomain.MATHEMATICS,
                target_domain=CognitiveDomain.COMPUTER_SCIENCE,
                connection_strength=0.9,
                transfer_mechanisms=["algorithms", "complexity_analysis", "proof"],
            ),
            DomainInterconnection(
                source_domain=CognitiveDomain.MATHEMATICS,
                target_domain=CognitiveDomain.ECONOMICS,
                connection_strength=0.8,
                transfer_mechanisms=["optimization", "game_theory", "statistics"],
            ),
            # Physics → Engineering
            DomainInterconnection(
                source_domain=CognitiveDomain.PHYSICS,
                target_domain=CognitiveDomain.ENGINEERING,
                connection_strength=0.9,
                transfer_mechanisms=["physical_laws", "modeling", "simulation"],
            ),
            DomainInterconnection(
                source_domain=CognitiveDomain.PHYSICS,
                target_domain=CognitiveDomain.CHEMISTRY,
                connection_strength=0.85,
                transfer_mechanisms=["quantum_mechanics", "thermodynamics"],
            ),
            # Philosophy → Logic → Computer Science
            DomainInterconnection(
                source_domain=CognitiveDomain.PHILOSOPHY,
                target_domain=CognitiveDomain.LOGIC,
                connection_strength=0.9,
                transfer_mechanisms=["formal_reasoning", "epistemology", "semantics"],
            ),
            DomainInterconnection(
                source_domain=CognitiveDomain.LOGIC,
                target_domain=CognitiveDomain.COMPUTER_SCIENCE,
                connection_strength=0.85,
                transfer_mechanisms=["proof_systems", "type_theory", "verification"],
            ),
            # Biology → Chemistry → VITRA
            DomainInterconnection(
                source_domain=CognitiveDomain.CHEMISTRY,
                target_domain=CognitiveDomain.BIOLOGY,
                connection_strength=0.85,
                transfer_mechanisms=["molecular_biology", "biochemistry"],
            ),
            DomainInterconnection(
                source_domain=CognitiveDomain.BIOLOGY,
                target_domain=CognitiveDomain.VITRA,
                connection_strength=0.9,
                transfer_mechanisms=["drug_targets", "genomics", "pathways"],
            ),
            # Cross-domain paradigm connections
            DomainInterconnection(
                source_domain=CognitiveDomain.FLUXA,
                target_domain=CognitiveDomain.ECONOMICS,
                connection_strength=0.7,
                transfer_mechanisms=["complex_systems", "emergence", "dynamics"],
            ),
            DomainInterconnection(
                source_domain=CognitiveDomain.GAME_THEORY,
                target_domain=CognitiveDomain.GEOPOLITICS,
                connection_strength=0.75,
                transfer_mechanisms=["strategic_analysis", "equilibrium"],
                safety_level=SynthesisSafetyLevel.SENSITIVE,
            ),
        ]

        for conn in connections:
            key = (conn.source_domain, conn.target_domain)
            self.interconnections[key] = conn
            if conn.bidirectional:
                reverse_key = (conn.target_domain, conn.source_domain)
                reverse_conn = DomainInterconnection(
                    source_domain=conn.target_domain,
                    target_domain=conn.source_domain,
                    connection_strength=conn.connection_strength * 0.9,  # Slightly weaker reverse
                    transfer_mechanisms=conn.transfer_mechanisms,
                    bidirectional=False,
                    requires_translation=conn.requires_translation,
                    safety_level=conn.safety_level,
                )
                self.interconnections[reverse_key] = reverse_conn

    def _build_capability_index(self) -> None:
        """Build index from capabilities to domains."""
        for domain, definition in self.domains.items():
            for capability in definition.capabilities:
                if capability not in self.capability_index:
                    self.capability_index[capability] = []
                self.capability_index[capability].append(domain)

    def get_domain(self, domain: CognitiveDomain) -> DomainDefinition | None:
        """Get domain definition."""
        return self.domains.get(domain)

    def get_all_domains(self) -> list[CognitiveDomain]:
        """Get list of all registered domains."""
        return list(self.domains.keys())

    def get_core_verticals(self) -> list[CognitiveDomain]:
        """Get original 14 QRATUM verticals."""
        return [d for d in self.domains.keys() if d in self.CORE_VERTICALS]

    def get_extended_domains(self) -> list[CognitiveDomain]:
        """Get extended domains beyond core verticals."""
        return [d for d in self.domains.keys() if d not in self.CORE_VERTICALS]

    def get_domains_with_capability(self, capability: DomainCapability) -> list[CognitiveDomain]:
        """Get all domains that have a specific capability."""
        return self.capability_index.get(capability, [])

    def get_interconnection(
        self, source: CognitiveDomain, target: CognitiveDomain
    ) -> DomainInterconnection | None:
        """Get interconnection between two domains."""
        return self.interconnections.get((source, target))

    def get_connected_domains(self, domain: CognitiveDomain) -> list[tuple[CognitiveDomain, float]]:
        """Get all domains connected to a given domain with connection strength."""
        connected = []
        for (source, target), conn in self.interconnections.items():
            if source == domain:
                connected.append((target, conn.connection_strength))
        return sorted(connected, key=lambda x: x[1], reverse=True)

    def find_synthesis_path(
        self,
        source: CognitiveDomain,
        target: CognitiveDomain,
        max_hops: int = 3,
    ) -> list[list[CognitiveDomain]] | None:
        """Find paths for knowledge synthesis between domains.

        Uses BFS to find paths up to max_hops length.

        Args:
            source: Starting domain
            target: Target domain
            max_hops: Maximum path length

        Returns:
            List of possible paths (each path is a list of domains)
        """
        if source == target:
            return [[source]]

        paths: list[list[CognitiveDomain]] = []
        queue: list[tuple[CognitiveDomain, list[CognitiveDomain]]] = [(source, [source])]

        while queue:
            current, path = queue.pop(0)

            if len(path) > max_hops:
                continue

            for neighbor, _ in self.get_connected_domains(current):
                if neighbor in path:
                    continue

                new_path = path + [neighbor]

                if neighbor == target:
                    paths.append(new_path)
                else:
                    queue.append((neighbor, new_path))

        return paths if paths else None

    def get_registry_stats(self) -> dict[str, Any]:
        """Get registry statistics."""
        return {
            "version": self._registry_version,
            "last_updated": self._last_updated,
            "total_domains": len(self.domains),
            "core_verticals": len(self.CORE_VERTICALS),
            "extended_domains": len(self.domains) - len(self.CORE_VERTICALS),
            "total_interconnections": len(self.interconnections),
            "capabilities_indexed": len(self.capability_index),
        }
