#!/usr/bin/env python3
"""Demonstration of XENON bioinformatics capabilities.

This script showcases the key features of the bioinformatics modules:
- Sequence analysis
- Literature mining
- Drug-target scoring
- Multi-omics integration
"""

from __future__ import annotations

import numpy as np

from xenon.bioinformatics.drug_target_scoring import (DrugCandidate,
                                                      DrugTargetScorer)
from xenon.bioinformatics.literature_miner import LiteratureMiner, Publication
from xenon.bioinformatics.multiomics_integrator import (MultiOmicsIntegrator,
                                                        OmicsData)
from xenon.bioinformatics.sequence_analyzer import SequenceAnalyzer


def demo_sequence_analysis():
    """Demonstrate sequence analysis capabilities."""

    print("=" * 60)
    print("XENON Sequence Analysis Demo")
    print("=" * 60)

    analyzer = SequenceAnalyzer()

    # Example FASTA content
    fasta_content = """>sp|P04626|ERBB2_HUMAN Receptor tyrosine-protein kinase erbB-2
MELAALCRWGLLLALLPPGAASTQVCTGTDMKLRLPASPETHLDMLRHLYQGCQVVQGNLELTYLPTNASL
SFLQDIQEVQGYVLIAHNQVRQVPLQRLRIVRGTQLFEDNYALAVLDNGDPLNNTTPVTGASPGGLRELQL
RSLTEILKGGVLIQRNPQLCYQDTILWKDIFHKNNQLALTLIDTNRSRACHPCSPMCKGSRCWGESSEDCQ
>sp|P00533|EGFR_HUMAN Epidermal growth factor receptor
MRPSGTAGAALLALLAALCPASRALEEKKVCQGTSNKLTQLGTFEDHFLSLQRMFNNCEVVLGNLEITYVQ
RNYDLSFLKTIQEVAGYVLIALNTVERIPLENLQIIRGNMYYENSYALAVLSNYDANKTGLKELPMRNLQE
ILHGAVRFSNNPALCNVESIQWRDIVSSDFLSNMSMDFQNHLGSCQKCDPSCPNGSCWGAGEENCQKLTKII"""

    sequences = analyzer.parse_fasta(fasta_content)
    print(f"\nParsed {len(sequences)} sequences")

    for seq in sequences:
        print(f"\n{seq.id}: {seq.name}")
        print(f"  Length: {seq.length()} amino acids")
        print(f"  Molecular weight: {analyzer.compute_molecular_weight(seq.sequence):.1f} Da")
        print(f"  Hydrophobicity: {analyzer.compute_hydrophobicity(seq.sequence):.2f}")
        print(f"  Isoelectric point: {analyzer.compute_isoelectric_point(seq.sequence):.2f}")

    # Sequence alignment
    if len(sequences) >= 2:
        print(f"\nAligning {sequences[0].id} and {sequences[1].id}:")
        aligned1, aligned2, score = analyzer.align_sequences(
            sequences[0].sequence[:50], sequences[1].sequence[:50]
        )
        print(f"  Alignment score: {score:.1f}")
        print(f"  Aligned seq1: {aligned1[:60]}...")
        print(f"  Aligned seq2: {aligned2[:60]}...")


def demo_literature_mining():
    """Demonstrate literature mining capabilities."""

    print("\n" + "=" * 60)
    print("XENON Literature Mining Demo")
    print("=" * 60)

    miner = LiteratureMiner()

    # Add sample publications
    pubs = [
        Publication(
            pmid="12345678",
            title="EGFR signaling in cancer progression",
            authors=["Smith J", "Jones A"],
            journal="Nature",
            year=2020,
            abstract="EGFR interacts with RAS to activate downstream signaling...",
            keywords=["EGFR", "cancer", "signaling"],
        ),
        Publication(
            pmid="23456789",
            title="ErbB2 phosphorylation mechanisms",
            authors=["Brown B", "Wilson C"],
            journal="Cell",
            year=2021,
            abstract="ErbB2 binds to EGFR forming heterodimers that enhance signaling...",
            keywords=["ErbB2", "EGFR", "phosphorylation"],
        ),
    ]

    for pub in pubs:
        miner.add_publication(pub)

    # Query for EGFR
    protein = "EGFR"
    citations = miner.get_protein_citations(protein)
    print(f"\nFound {len(citations)} publications mentioning {protein}")

    for pub in citations:
        print(f"\n  PMID: {pub.pmid}")
        print(f"  Title: {pub.title}")
        print(f"  Year: {pub.year}")
        print(f"  Keywords: {', '.join(pub.keywords)}")

    # Find interactions
    interactions = miner.get_interactions(protein=protein, min_confidence=0.0)
    print(f"\n{len(interactions)} interaction(s) found:")
    for interaction in interactions:
        print(
            f"  {interaction.protein_a} <-> {interaction.protein_b} (confidence: {interaction.confidence:.2f})"
        )


def demo_drug_target_scoring():
    """Demonstrate drug-target scoring capabilities."""

    print("\n" + "=" * 60)
    print("XENON Drug-Target Scoring Demo")
    print("=" * 60)

    scorer = DrugTargetScorer()

    # Add sample drug candidates
    drugs = [
        DrugCandidate(
            compound_id="DRUG001",
            name="Gefitinib",
            smiles="COC1=C(C=C2C(=C1)N=CN=C2NC3=CC(=C(C=C3)F)Cl)OCCCN4CCOCC4",
            molecular_weight=446.9,
            logp=3.5,
            hbd=1,
            hba=6,
            tpsa=68.7,
            rotatable_bonds=9,
        ),
        DrugCandidate(
            compound_id="DRUG002",
            name="Erlotinib",
            smiles="COC1=C(C=C2C(=C1)N=CN=C2NC3=CC=CC(=C3)C#C)OCCCN4CCOCC4",
            molecular_weight=393.4,
            logp=2.8,
            hbd=1,
            hba=6,
            tpsa=74.7,
            rotatable_bonds=8,
        ),
    ]

    for drug in drugs:
        scorer.add_drug(drug)

    target = "EGFR"

    print(f"\nDrug candidates for {target}:")
    for drug in drugs:
        print(f"\n  {drug.name} ({drug.compound_id})")
        print(f"    MW: {drug.molecular_weight:.1f} Da")
        print(f"    LogP: {drug.logp:.2f}")
        print(f"    Drug-like: {drug.is_drug_like()}")

        drug_likeness = scorer.compute_drug_likeness(drug.compound_id)
        print(f"    Drug-likeness score: {drug_likeness['overall_score']:.3f}")

        binding_score = scorer.compute_binding_affinity_score(drug.compound_id, target)
        print(f"    Binding affinity score: {binding_score:.3f}")

        admet = scorer.predict_admet(drug.compound_id)
        print(f"    Absorption: {admet.absorption:.3f}")
        print(f"    Toxicity: {admet.toxicity:.3f}")

    # Rank candidates
    print(f"\nRanked drug candidates for {target}:")
    ranked = scorer.rank_drug_candidates(target)
    for i, (drug_id, score) in enumerate(ranked, 1):
        drug = scorer._drugs[drug_id]
        print(f"  {i}. {drug.name}: {score:.3f}")


def demo_multiomics_integration():
    """Demonstrate multi-omics integration capabilities."""

    print("\n" + "=" * 60)
    print("XENON Multi-Omics Integration Demo")
    print("=" * 60)

    integrator = MultiOmicsIntegrator()

    # Create sample omics data
    np.random.seed(42)

    # Healthy samples (group 1)
    for i in range(5):
        sample = OmicsData(
            sample_id=f"HEALTHY_{i + 1}",
            transcriptomics={
                "EGFR": np.random.normal(100, 10),
                "KRAS": np.random.normal(150, 15),
                "TP53": np.random.normal(200, 20),
            },
            proteomics={
                "EGFR": np.random.normal(50, 5),
                "KRAS": np.random.normal(75, 7),
                "TP53": np.random.normal(100, 10),
            },
            metabolomics={
                "Glucose": np.random.normal(5.0, 0.5),
                "Lactate": np.random.normal(1.0, 0.1),
            },
            metadata={"condition": "healthy"},
        )
        integrator.add_sample(sample)

    # Disease samples (group 2)
    for i in range(5):
        sample = OmicsData(
            sample_id=f"DISEASE_{i + 1}",
            transcriptomics={
                "EGFR": np.random.normal(200, 20),  # Upregulated
                "KRAS": np.random.normal(140, 14),
                "TP53": np.random.normal(80, 8),  # Downregulated
            },
            proteomics={
                "EGFR": np.random.normal(100, 10),  # Upregulated
                "KRAS": np.random.normal(70, 7),
                "TP53": np.random.normal(40, 4),  # Downregulated
            },
            metabolomics={
                "Glucose": np.random.normal(3.0, 0.3),  # Decreased
                "Lactate": np.random.normal(2.0, 0.2),  # Increased
            },
            metadata={"condition": "disease"},
        )
        integrator.add_sample(sample)

    print(f"\nAnalyzing {len(integrator._samples)} samples")

    # Identify biomarkers
    healthy_ids = [f"HEALTHY_{i + 1}" for i in range(5)]
    disease_ids = [f"DISEASE_{i + 1}" for i in range(5)]

    biomarkers = integrator.identify_biomarkers(disease_ids, healthy_ids, effect_size_threshold=1.3)

    print(f"\nIdentified {len(biomarkers)} potential biomarkers:")
    for biomarker in sorted(biomarkers, key=lambda x: x.fdr)[:5]:
        print(f"  {biomarker.feature_id} ({biomarker.omics_type})")
        print(f"    Effect size: {biomarker.effect_size:.2f}")
        print(f"    P-value: {biomarker.p_value:.4f}")
        print(f"    FDR: {biomarker.fdr:.4f}")
        print(f"    Significant: {biomarker.is_significant()}")

    # Cross-omics correlation
    print("\nCross-omics correlations:")
    corr, pval = integrator.compute_cross_omics_correlation(
        "transcriptomics", "EGFR", "proteomics", "EGFR"
    )
    print(f"  mRNA-Protein EGFR: r={corr:.3f}, p={pval:.4f}")

    corr, pval = integrator.compute_cross_omics_correlation(
        "transcriptomics", "EGFR", "metabolomics", "Glucose"
    )
    print(f"  EGFR-Glucose: r={corr:.3f}, p={pval:.4f}")


def main():
    """Run all demonstrations."""

    print("\n" + "=" * 60)
    print("XENON Bioinformatics Capabilities Demonstration")
    print("=" * 60)

    demo_sequence_analysis()
    demo_literature_mining()
    demo_drug_target_scoring()
    demo_multiomics_integration()

    print("\n" + "=" * 60)
    print("Demo Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
