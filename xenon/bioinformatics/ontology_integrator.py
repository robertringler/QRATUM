"""Ontology integration for XENON.

Provides functionality for:
- Gene Ontology (GO) integration
- ChEBI chemical ontology
- UniProt protein database
- Pathway databases (KEGG, Reactome)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple


@dataclass
class GOTerm:
    """Gene Ontology term.
    
    Attributes:
        go_id: GO identifier (e.g., GO:0006915)
        name: Term name
        namespace: GO namespace (biological_process, molecular_function, cellular_component)
        definition: Term definition
        is_a: Parent terms
        part_of: Part-of relationships
    """
    
    go_id: str
    name: str
    namespace: str
    definition: str = ""
    is_a: List[str] = field(default_factory=list)
    part_of: List[str] = field(default_factory=list)


@dataclass
class ProteinAnnotation:
    """Protein annotation from UniProt.
    
    Attributes:
        uniprot_id: UniProt accession
        protein_name: Full protein name
        gene_name: Gene symbol
        organism: Organism name
        go_terms: Associated GO terms
        function: Functional description
        pathways: Associated pathways
        domains: Protein domains
    """
    
    uniprot_id: str
    protein_name: str
    gene_name: str
    organism: str
    go_terms: List[str] = field(default_factory=list)
    function: str = ""
    pathways: List[str] = field(default_factory=list)
    domains: List[str] = field(default_factory=list)


@dataclass
class ChemicalCompound:
    """Chemical compound from ChEBI.
    
    Attributes:
        chebi_id: ChEBI identifier
        name: Compound name
        formula: Molecular formula
        mass: Molecular mass
        charge: Formal charge
        inchi: InChI identifier
        smiles: SMILES string
    """
    
    chebi_id: str
    name: str
    formula: str = ""
    mass: float = 0.0
    charge: int = 0
    inchi: str = ""
    smiles: str = ""


@dataclass
class Pathway:
    """Biological pathway.
    
    Attributes:
        pathway_id: Pathway identifier (KEGG, Reactome, etc.)
        name: Pathway name
        proteins: List of protein IDs in pathway
        compounds: List of chemical compounds
        reactions: List of reactions
        database: Source database
    """
    
    pathway_id: str
    name: str
    proteins: List[str] = field(default_factory=list)
    compounds: List[str] = field(default_factory=list)
    reactions: List[str] = field(default_factory=list)
    database: str = "unknown"


class OntologyIntegrator:
    """Ontology and database integration for biological knowledge.
    
    Integrates multiple biological databases and ontologies to provide
    structured knowledge for mechanism discovery.
    """
    
    def __init__(self):
        """Initialize ontology integrator."""
        self._go_terms: Dict[str, GOTerm] = {}
        self._proteins: Dict[str, ProteinAnnotation] = {}
        self._compounds: Dict[str, ChemicalCompound] = {}
        self._pathways: Dict[str, Pathway] = {}
        self._protein_go_index: Dict[str, Set[str]] = {}
    
    def add_go_term(self, term: GOTerm) -> None:
        """Add a GO term to the ontology.
        
        Args:
            term: GO term object
        """
        self._go_terms[term.go_id] = term
    
    def add_protein_annotation(self, annotation: ProteinAnnotation) -> None:
        """Add protein annotation.
        
        Args:
            annotation: Protein annotation object
        """
        self._proteins[annotation.uniprot_id] = annotation
        
        # Index GO terms
        for go_id in annotation.go_terms:
            if go_id not in self._protein_go_index:
                self._protein_go_index[go_id] = set()
            self._protein_go_index[go_id].add(annotation.uniprot_id)
    
    def add_compound(self, compound: ChemicalCompound) -> None:
        """Add chemical compound.
        
        Args:
            compound: Chemical compound object
        """
        self._compounds[compound.chebi_id] = compound
    
    def add_pathway(self, pathway: Pathway) -> None:
        """Add biological pathway.
        
        Args:
            pathway: Pathway object
        """
        self._pathways[pathway.pathway_id] = pathway
    
    def get_protein_function(self, uniprot_id: str) -> Optional[str]:
        """Get protein function description.
        
        Args:
            uniprot_id: UniProt accession
        
        Returns:
            Function description if found
        """
        annotation = self._proteins.get(uniprot_id)
        return annotation.function if annotation else None
    
    def get_protein_go_terms(self, uniprot_id: str) -> List[GOTerm]:
        """Get GO terms for a protein.
        
        Args:
            uniprot_id: UniProt accession
        
        Returns:
            List of GO terms
        """
        annotation = self._proteins.get(uniprot_id)
        if not annotation:
            return []
        
        terms = []
        for go_id in annotation.go_terms:
            term = self._go_terms.get(go_id)
            if term:
                terms.append(term)
        
        return terms
    
    def get_proteins_by_go_term(self, go_id: str) -> List[str]:
        """Get proteins annotated with a GO term.
        
        Args:
            go_id: GO identifier
        
        Returns:
            List of UniProt IDs
        """
        return list(self._protein_go_index.get(go_id, set()))
    
    def get_related_proteins(
        self,
        uniprot_id: str,
        relationship: str = "go_overlap",
    ) -> List[Tuple[str, float]]:
        """Find related proteins based on annotations.
        
        Args:
            uniprot_id: UniProt accession
            relationship: Type of relationship ('go_overlap', 'pathway', 'domain')
        
        Returns:
            List of (uniprot_id, similarity_score) tuples
        """
        annotation = self._proteins.get(uniprot_id)
        if not annotation:
            return []
        
        if relationship == "go_overlap":
            return self._find_go_similar_proteins(uniprot_id)
        elif relationship == "pathway":
            return self._find_pathway_proteins(uniprot_id)
        else:
            return []
    
    def _find_go_similar_proteins(self, uniprot_id: str) -> List[Tuple[str, float]]:
        """Find proteins with similar GO annotations.
        
        Args:
            uniprot_id: UniProt accession
        
        Returns:
            List of (uniprot_id, similarity) tuples
        """
        annotation = self._proteins.get(uniprot_id)
        if not annotation:
            return []
        
        query_go = set(annotation.go_terms)
        if not query_go:
            return []
        
        similar = []
        for other_id, other_annotation in self._proteins.items():
            if other_id == uniprot_id:
                continue
            
            other_go = set(other_annotation.go_terms)
            if not other_go:
                continue
            
            # Jaccard similarity
            intersection = len(query_go & other_go)
            union = len(query_go | other_go)
            similarity = intersection / union if union > 0 else 0.0
            
            if similarity > 0:
                similar.append((other_id, similarity))
        
        similar.sort(key=lambda x: x[1], reverse=True)
        return similar
    
    def _find_pathway_proteins(self, uniprot_id: str) -> List[Tuple[str, float]]:
        """Find proteins in same pathways.
        
        Args:
            uniprot_id: UniProt accession
        
        Returns:
            List of (uniprot_id, pathway_count) tuples
        """
        # Find pathways containing this protein
        protein_pathways = []
        for pathway in self._pathways.values():
            if uniprot_id in pathway.proteins:
                protein_pathways.append(pathway.pathway_id)
        
        if not protein_pathways:
            return []
        
        # Find other proteins in these pathways
        related: Dict[str, int] = {}
        for pathway_id in protein_pathways:
            pathway = self._pathways[pathway_id]
            for other_id in pathway.proteins:
                if other_id != uniprot_id:
                    related[other_id] = related.get(other_id, 0) + 1
        
        result = [(pid, count) for pid, count in related.items()]
        result.sort(key=lambda x: x[1], reverse=True)
        return result
    
    def compute_conservation_prior(self, protein_name: str) -> float:
        """Compute conservation-based prior probability.
        
        Args:
            protein_name: Protein name or gene symbol
        
        Returns:
            Prior probability (0-1)
        """
        # Find protein annotation
        annotation = None
        for ann in self._proteins.values():
            if ann.gene_name == protein_name or ann.protein_name == protein_name:
                annotation = ann
                break
        
        if not annotation:
            return 0.5  # Neutral prior
        
        # Proteins with GO annotations are likely more conserved/studied
        go_count = len(annotation.go_terms)
        pathway_count = len(annotation.pathways)
        domain_count = len(annotation.domains)
        
        # Normalize to 0-1
        # Typical well-studied protein: 10+ GO terms, 3+ pathways, 2+ domains
        score = (
            min(1.0, go_count / 10.0) * 0.5 +
            min(1.0, pathway_count / 3.0) * 0.3 +
            min(1.0, domain_count / 2.0) * 0.2
        )
        
        return score
    
    def find_mechanistic_links(
        self,
        protein_a: str,
        protein_b: str,
    ) -> List[Dict[str, any]]:
        """Find mechanistic links between two proteins.
        
        Args:
            protein_a: First protein ID
            protein_b: Second protein ID
        
        Returns:
            List of evidence dictionaries
        """
        evidence = []
        
        # Check for shared GO terms
        go_a = set(self._proteins.get(protein_a, ProteinAnnotation("", "", "", "")).go_terms)
        go_b = set(self._proteins.get(protein_b, ProteinAnnotation("", "", "", "")).go_terms)
        shared_go = go_a & go_b
        
        if shared_go:
            evidence.append({
                "type": "shared_go_terms",
                "terms": list(shared_go),
                "confidence": len(shared_go) / max(len(go_a), len(go_b)) if go_a or go_b else 0,
            })
        
        # Check for shared pathways
        for pathway in self._pathways.values():
            if protein_a in pathway.proteins and protein_b in pathway.proteins:
                evidence.append({
                    "type": "shared_pathway",
                    "pathway_id": pathway.pathway_id,
                    "pathway_name": pathway.name,
                    "confidence": 0.7,
                })
        
        return evidence
    
    def query_by_function(self, function_keywords: List[str]) -> List[str]:
        """Find proteins by functional keywords.
        
        Args:
            function_keywords: Keywords to search for
        
        Returns:
            List of UniProt IDs
        """
        matching = []
        
        for uniprot_id, annotation in self._proteins.items():
            function_text = annotation.function.lower()
            
            for keyword in function_keywords:
                if keyword.lower() in function_text:
                    matching.append(uniprot_id)
                    break
        
        return matching
    
    def export_protein_network(
        self,
        proteins: List[str],
        relationship_type: str = "go_overlap",
        min_similarity: float = 0.3,
    ) -> Dict[str, any]:
        """Export protein network as graph data.
        
        Args:
            proteins: List of protein IDs
            relationship_type: Type of relationships to include
            min_similarity: Minimum similarity threshold
        
        Returns:
            Dictionary with nodes and edges
        """
        nodes = []
        edges = []
        
        # Add nodes
        for protein_id in proteins:
            annotation = self._proteins.get(protein_id)
            if annotation:
                nodes.append({
                    "id": protein_id,
                    "name": annotation.protein_name,
                    "gene": annotation.gene_name,
                })
        
        # Add edges
        for protein_id in proteins:
            related = self.get_related_proteins(protein_id, relationship_type)
            
            for other_id, similarity in related:
                if other_id in proteins and similarity >= min_similarity:
                    edges.append({
                        "source": protein_id,
                        "target": other_id,
                        "weight": similarity,
                        "type": relationship_type,
                    })
        
        return {
            "nodes": nodes,
            "edges": edges,
        }
