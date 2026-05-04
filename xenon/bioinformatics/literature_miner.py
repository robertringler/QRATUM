"""Literature mining utilities for XENON.

Provides functionality for:
- PubMed API integration
- Article metadata extraction
- Protein-protein interaction mining
- Mechanism evidence extraction
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class Publication:
    """Scientific publication metadata.

    Attributes:
        pmid: PubMed ID
        title: Article title
        authors: List of author names
        journal: Journal name
        year: Publication year
        abstract: Article abstract
        keywords: List of keywords
        doi: Digital Object Identifier
    """

    pmid: str
    title: str
    authors: list[str]
    journal: str
    year: int
    abstract: str = ""
    keywords: list[str] = field(default_factory=list)
    doi: Optional[str] = None

    def mentions_protein(self, protein_name: str) -> bool:
        """Check if publication mentions a protein.

        Args:
            protein_name: Protein name to search for

        Returns:
            True if protein is mentioned in title or abstract
        """

        search_text = (self.title + " " + self.abstract).lower()
        return protein_name.lower() in search_text

    def extract_interactions(self) -> list[tuple[str, str]]:
        """Extract potential protein-protein interactions from text.

        Returns:
            List of (protein1, protein2) tuples
        """

        # Simple pattern matching for interaction keywords
        interaction_patterns = [
            r"(\w+)\s+(?:interacts with|binds to|phosphorylates)\s+(\w+)",
            r"(\w+)-(\w+)\s+(?:interaction|complex|binding)",
        ]

        interactions = []
        text = self.title + " " + self.abstract

        for pattern in interaction_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            interactions.extend(matches)

        return interactions


@dataclass
class InteractionEvidence:
    """Evidence for protein-protein interaction.

    Attributes:
        protein_a: First protein
        protein_b: Second protein
        interaction_type: Type of interaction (binding, phosphorylation, etc.)
        publications: Supporting publications
        confidence: Confidence score (0-1)
    """

    protein_a: str
    protein_b: str
    interaction_type: str
    publications: list[str] = field(default_factory=list)
    confidence: float = 0.0

    def add_publication(self, pmid: str) -> None:
        """Add supporting publication."""

        if pmid not in self.publications:
            self.publications.append(pmid)
            # Increase confidence with more publications
            self.confidence = min(1.0, 0.3 + len(self.publications) * 0.1)


class LiteratureMiner:
    """Literature mining and evidence extraction.

    Integrates with PubMed and other databases to extract
    biological mechanism evidence from scientific literature.
    """

    def __init__(self):
        """Initialize literature miner."""

        self._publications: dict[str, Publication] = {}
        self._interactions: dict[tuple[str, str], InteractionEvidence] = {}
        self._protein_citations: dict[str, set[str]] = {}

    def parse_pubmed_xml(self, xml_content: str) -> list[Publication]:
        """Parse PubMed XML format (simplified).

        Note: Full implementation would use xml.etree.ElementTree.
        This is a placeholder for Phase 1.

        Args:
            xml_content: PubMed XML string

        Returns:
            List of Publication objects
        """

        # Placeholder: In Phase 2+, implement full XML parsing
        return []

    def query_pubmed(
        self,
        query: str,
        max_results: int = 100,
    ) -> list[Publication]:
        """Query PubMed database.

        Note: Requires internet access and NCBI API key for production use.
        This is a mock implementation for Phase 1.

        Args:
            query: Search query
            max_results: Maximum number of results

        Returns:
            List of publications
        """

        # Phase 1: Mock implementation
        # Phase 2+: Implement actual PubMed E-utilities API calls
        return []

    def add_publication(self, publication: Publication) -> None:
        """Add a publication to the database.

        Args:
            publication: Publication object
        """

        self._publications[publication.pmid] = publication

        # Extract and index interactions
        interactions = publication.extract_interactions()
        for prot_a, prot_b in interactions:
            key = tuple(sorted([prot_a.upper(), prot_b.upper()]))

            if key not in self._interactions:
                self._interactions[key] = InteractionEvidence(
                    protein_a=key[0],
                    protein_b=key[1],
                    interaction_type="unknown",
                )

            self._interactions[key].add_publication(publication.pmid)

    def get_protein_citations(self, protein_name: str) -> list[Publication]:
        """Get all publications mentioning a protein.

        Args:
            protein_name: Protein name to search for

        Returns:
            List of publications
        """

        publications = []
        for pub in self._publications.values():
            if pub.mentions_protein(protein_name):
                publications.append(pub)

        return publications

    def get_citation_count(self, protein_name: str) -> int:
        """Get number of publications mentioning a protein.

        Args:
            protein_name: Protein name

        Returns:
            Number of citations
        """

        return len(self.get_protein_citations(protein_name))

    def get_interactions(
        self,
        protein: Optional[str] = None,
        min_confidence: float = 0.0,
    ) -> list[InteractionEvidence]:
        """Get protein-protein interactions.

        Args:
            protein: Filter by protein (optional)
            min_confidence: Minimum confidence threshold

        Returns:
            List of interaction evidence
        """

        interactions = []

        for evidence in self._interactions.values():
            if evidence.confidence < min_confidence:
                continue

            if protein is not None:
                protein_upper = protein.upper()
                if protein_upper not in (evidence.protein_a, evidence.protein_b):
                    continue

            interactions.append(evidence)

        return interactions

    def compute_literature_prior(self, protein: str) -> float:
        """Compute literature-based prior probability.

        Args:
            protein: Protein name

        Returns:
            Prior probability (0-1) based on citation count
        """

        citation_count = self.get_citation_count(protein)

        # Log-scale normalization
        # Typical proteins: 10-1000 citations
        # Well-studied proteins: >10,000 citations
        if citation_count == 0:
            return 0.1  # Minimum prior

        log_citations = np.log10(citation_count + 1)
        # Normalize to 0-1 range (assuming max ~10^5 citations)
        return min(1.0, log_citations / 5.0)

    def extract_mechanism_keywords(self, publication: Publication) -> list[str]:
        """Extract mechanism-related keywords from publication.

        Args:
            publication: Publication object

        Returns:
            List of mechanism keywords
        """

        # Keywords related to biological mechanisms
        mechanism_keywords = [
            "phosphorylation",
            "activation",
            "inhibition",
            "binding",
            "catalysis",
            "signaling",
            "pathway",
            "regulation",
            "conformational change",
            "allosteric",
            "enzyme",
            "substrate",
            "product",
            "kinase",
            "phosphatase",
        ]

        found_keywords = []
        text = (publication.title + " " + publication.abstract).lower()

        for keyword in mechanism_keywords:
            if keyword in text:
                found_keywords.append(keyword)

        return found_keywords

    def rank_publications_by_relevance(
        self,
        protein: str,
        mechanism_type: Optional[str] = None,
    ) -> list[tuple[Publication, float]]:
        """Rank publications by relevance to protein/mechanism.

        Args:
            protein: Target protein name
            mechanism_type: Mechanism type (e.g., 'phosphorylation')

        Returns:
            List of (publication, relevance_score) tuples
        """

        ranked = []
        protein_lower = protein.lower()

        for pub in self._publications.values():
            score = 0.0
            text = (pub.title + " " + pub.abstract).lower()

            # Count protein mentions
            protein_count = text.count(protein_lower)
            score += protein_count * 0.1

            # Boost if in title
            if protein_lower in pub.title.lower():
                score += 0.5

            # Mechanism type relevance
            if mechanism_type is not None:
                if mechanism_type.lower() in text:
                    score += 0.3

            # Recent publications get boost
            current_year = datetime.now().year
            years_old = current_year - pub.year
            recency_boost = max(0, 1.0 - years_old / 20.0)
            score += recency_boost * 0.2

            if score > 0:
                ranked.append((pub, score))

        # Sort by score descending
        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked

    def generate_literature_summary(
        self,
        protein: str,
        max_publications: int = 10,
    ) -> dict[str, any]:
        """Generate summary of literature for a protein.

        Args:
            protein: Protein name
            max_publications: Maximum publications to include

        Returns:
            Dictionary with summary statistics
        """

        publications = self.get_protein_citations(protein)
        ranked = self.rank_publications_by_relevance(protein)[:max_publications]
        interactions = self.get_interactions(protein=protein, min_confidence=0.3)

        # Extract common keywords
        all_keywords: dict[str, int] = {}
        for pub in publications:
            keywords = self.extract_mechanism_keywords(pub)
            for kw in keywords:
                all_keywords[kw] = all_keywords.get(kw, 0) + 1

        top_keywords = sorted(all_keywords.items(), key=lambda x: x[1], reverse=True)[:10]

        return {
            "protein": protein,
            "total_publications": len(publications),
            "top_publications": [(pub.pmid, pub.title, score) for pub, score in ranked],
            "interactions": [(ev.protein_a, ev.protein_b, ev.confidence) for ev in interactions],
            "common_keywords": top_keywords,
            "literature_prior": self.compute_literature_prior(protein),
        }


# Import numpy for numerical operations
import numpy as np
