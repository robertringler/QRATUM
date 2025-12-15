"""Integration between bioinformatics modules and XENON learning system.

Enhances XENON runtime with:
- Literature-based prior computation
- Sequence conservation priors
- Pathway context for mechanism ranking
- Structure-guided hypothesis generation
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from xenon.core.mechanism import BioMechanism
from xenon.bioinformatics.sequence_analyzer import SequenceAnalyzer
from xenon.bioinformatics.literature_miner import LiteratureMiner
from xenon.bioinformatics.ontology_integrator import OntologyIntegrator
from xenon.bioinformatics.pathway_analyzer import PathwayAnalyzer


class BioinformaticsEnhancedPrior:
    """Enhanced prior computation using bioinformatics databases.
    
    Combines multiple information sources:
    1. Literature citations (PubMed)
    2. Sequence conservation (UniProt, homology)
    3. Ontology annotations (GO, ChEBI)
    4. Pathway membership (KEGG, Reactome)
    5. Structure information (PDB)
    """
    
    def __init__(
        self,
        literature_weight: float = 0.3,
        conservation_weight: float = 0.3,
        ontology_weight: float = 0.2,
        pathway_weight: float = 0.2,
    ):
        """Initialize enhanced prior.
        
        Args:
            literature_weight: Weight for literature evidence
            conservation_weight: Weight for sequence conservation
            ontology_weight: Weight for ontology annotations
            pathway_weight: Weight for pathway membership
        """
        self.literature_weight = literature_weight
        self.conservation_weight = conservation_weight
        self.ontology_weight = ontology_weight
        self.pathway_weight = pathway_weight
        
        # Initialize bioinformatics modules
        self.literature_miner = LiteratureMiner()
        self.sequence_analyzer = SequenceAnalyzer()
        self.ontology = OntologyIntegrator()
        self.pathway_analyzer = PathwayAnalyzer()
    
    def compute_enhanced_prior(
        self,
        mechanism: BioMechanism,
        protein_name: str,
    ) -> float:
        """Compute enhanced prior probability.
        
        Args:
            mechanism: Candidate mechanism
            protein_name: Target protein name
        
        Returns:
            Enhanced prior probability (0-1)
        """
        # Compute individual priors
        lit_prior = self._compute_literature_prior(protein_name)
        cons_prior = self._compute_conservation_prior(protein_name)
        onto_prior = self._compute_ontology_prior(protein_name)
        pathway_prior = self._compute_pathway_prior(protein_name)
        
        # Weighted combination
        prior = (
            self.literature_weight * lit_prior +
            self.conservation_weight * cons_prior +
            self.ontology_weight * onto_prior +
            self.pathway_weight * pathway_prior
        )
        
        return prior
    
    def _compute_literature_prior(self, protein_name: str) -> float:
        """Compute literature-based prior.
        
        Args:
            protein_name: Protein name
        
        Returns:
            Prior probability (0-1)
        """
        citation_count = self.literature_miner.get_citation_count(protein_name)
        
        # Log-scale normalization
        if citation_count == 0:
            return 0.1
        
        log_citations = np.log10(citation_count + 1)
        return float(min(1.0, log_citations / 5.0))
    
    def _compute_conservation_prior(self, protein_name: str) -> float:
        """Compute conservation-based prior.
        
        Args:
            protein_name: Protein name
        
        Returns:
            Prior probability (0-1)
        """
        return self.ontology.compute_conservation_prior(protein_name)
    
    def _compute_ontology_prior(self, protein_name: str) -> float:
        """Compute ontology-based prior.
        
        Args:
            protein_name: Protein name
        
        Returns:
            Prior probability (0-1)
        """
        # Find protein by name
        protein_id = None
        for pid, annotation in self.ontology._proteins.items():
            if annotation.gene_name == protein_name or annotation.protein_name == protein_name:
                protein_id = pid
                break
        
        if not protein_id:
            return 0.5
        
        # Use number of GO annotations as proxy for knowledge depth
        go_terms = self.ontology.get_protein_go_terms(protein_id)
        return float(min(1.0, len(go_terms) / 10.0))
    
    def _compute_pathway_prior(self, protein_name: str) -> float:
        """Compute pathway-based prior.
        
        Args:
            protein_name: Protein name
        
        Returns:
            Prior probability (0-1)
        """
        # Count pathways containing this protein
        pathway_count = 0
        for pathway in self.pathway_analyzer._pathways.values():
            for reaction in pathway.reactions:
                if protein_name in reaction.enzymes:
                    pathway_count += 1
                    break
        
        return float(min(1.0, pathway_count / 5.0))
    
    def rank_mechanisms_by_evidence(
        self,
        mechanisms: List[BioMechanism],
        protein_name: str,
    ) -> List[Tuple[BioMechanism, float]]:
        """Rank mechanisms by combined evidence.
        
        Args:
            mechanisms: List of candidate mechanisms
            protein_name: Target protein name
        
        Returns:
            List of (mechanism, evidence_score) tuples
        """
        ranked = []
        
        for mechanism in mechanisms:
            evidence_score = self.compute_enhanced_prior(mechanism, protein_name)
            ranked.append((mechanism, evidence_score))
        
        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked
    
    def identify_related_proteins(
        self,
        protein_name: str,
        min_similarity: float = 0.5,
    ) -> List[Tuple[str, float, str]]:
        """Identify proteins related to target.
        
        Args:
            protein_name: Target protein name
            min_similarity: Minimum similarity threshold
        
        Returns:
            List of (protein_id, similarity, relationship_type) tuples
        """
        related = []
        
        # Find protein ID
        protein_id = None
        for pid, annotation in self.ontology._proteins.items():
            if annotation.gene_name == protein_name or annotation.protein_name == protein_name:
                protein_id = pid
                break
        
        if not protein_id:
            return []
        
        # GO overlap
        go_related = self.ontology.get_related_proteins(protein_id, "go_overlap")
        for other_id, similarity in go_related:
            if similarity >= min_similarity:
                related.append((other_id, similarity, "go_overlap"))
        
        # Pathway membership
        pathway_related = self.ontology.get_related_proteins(protein_id, "pathway")
        for other_id, count in pathway_related:
            if count > 0:
                similarity = min(1.0, count / 3.0)
                if similarity >= min_similarity:
                    related.append((other_id, similarity, "pathway"))
        
        return related
    
    def generate_hypothesis_from_homology(
        self,
        protein_name: str,
        sequence: str,
    ) -> List[Dict[str, any]]:
        """Generate mechanism hypotheses from homologous proteins.
        
        Args:
            protein_name: Target protein name
            sequence: Protein sequence
        
        Returns:
            List of hypothesis dictionaries
        """
        hypotheses = []
        
        # Find similar sequences
        for seq_id, protein_seq in self.sequence_analyzer._sequences.items():
            similarity = self.sequence_analyzer.compute_similarity(
                sequence, protein_seq.sequence
            )
            
            if similarity >= 50.0:  # At least 50% similar
                # Look up known interactions for homolog
                interactions = self.literature_miner.get_interactions(protein=seq_id)
                
                for interaction in interactions:
                    hypothesis = {
                        "source_protein": protein_name,
                        "homolog": seq_id,
                        "similarity": similarity,
                        "predicted_interaction": interaction.protein_b,
                        "confidence": interaction.confidence * (similarity / 100.0),
                    }
                    hypotheses.append(hypothesis)
        
        return hypotheses
    
    def validate_mechanism_with_structure(
        self,
        mechanism: BioMechanism,
        protein_structures: Dict[str, str],
    ) -> Tuple[bool, List[str]]:
        """Validate mechanism using structural information.
        
        Args:
            mechanism: Candidate mechanism
            protein_structures: Dictionary mapping protein names to PDB IDs
        
        Returns:
            Tuple of (is_valid, validation_messages)
        """
        from xenon.bioinformatics.structure_analyzer import StructureAnalyzer
        
        structure_analyzer = StructureAnalyzer()
        validation_messages = []
        is_valid = True
        
        # Check if proteins in mechanism have known structures
        for state in mechanism._states.values():
            protein_name = state.molecule
            
            if protein_name not in protein_structures:
                validation_messages.append(
                    f"No structure available for {protein_name}"
                )
                continue
            
            pdb_id = protein_structures[protein_name]
            structure = structure_analyzer.get_structure(pdb_id)
            
            if structure:
                # Assess structure quality
                quality = structure_analyzer.assess_structure_quality(structure)
                
                if quality["resolution"] > 3.0:
                    validation_messages.append(
                        f"Low resolution structure for {protein_name}: "
                        f"{quality['resolution']:.2f} Å"
                    )
                
                if quality["completeness"] < 0.8:
                    validation_messages.append(
                        f"Incomplete structure for {protein_name}: "
                        f"{quality['completeness']*100:.1f}% complete"
                    )
                    is_valid = False
        
        return is_valid, validation_messages
    
    def export_enriched_mechanism(
        self,
        mechanism: BioMechanism,
        protein_name: str,
    ) -> Dict[str, any]:
        """Export mechanism with enriched bioinformatics annotations.
        
        Args:
            mechanism: Mechanism to export
            protein_name: Target protein name
        
        Returns:
            Dictionary with mechanism and annotations
        """
        # Get basic mechanism data
        mech_dict = mechanism.to_dict()
        
        # Add bioinformatics annotations
        mech_dict["literature_citations"] = len(
            self.literature_miner.get_protein_citations(protein_name)
        )
        
        mech_dict["conservation_score"] = self._compute_conservation_prior(protein_name)
        
        # Add pathway context
        pathways = []
        for pathway_id, pathway in self.pathway_analyzer._pathways.items():
            for reaction in pathway.reactions:
                if protein_name in reaction.enzymes:
                    pathways.append({
                        "pathway_id": pathway_id,
                        "pathway_name": pathway.name,
                    })
        mech_dict["pathways"] = pathways
        
        # Add related proteins
        related = self.identify_related_proteins(protein_name, min_similarity=0.5)
        mech_dict["related_proteins"] = [
            {"protein_id": pid, "similarity": sim, "relationship": rel}
            for pid, sim, rel in related[:10]  # Top 10
        ]
        
        return mech_dict
