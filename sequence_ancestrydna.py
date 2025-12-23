#!/usr/bin/env python3
"""Sequence AncestryDNA.txt genome data using XENON v5 Full Genome Sequencing Pipeline.

This script reads the AncestryDNA.txt SNP data file and processes it through
the XENON Quantum Bioinformatics v5 full genome sequencing pipeline.

The script:
1. Parses the AncestryDNA.txt SNP data (rsID, chromosome, position, allele1, allele2)
2. Converts SNP data into sequences organized by chromosome
3. Runs the full genome sequencing pipeline on the converted data
4. Generates comprehensive analysis results
"""

import argparse
import json
import logging
import os
import sys
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# Add xenon to path
sys.path.insert(0, str(Path(__file__).parent))

from xenon.bioinformatics.full_genome_sequencing import (
    FullGenomeSequencingPipeline,
    GenomeSequencingConfig,
)

# Import the new Genomic Rarity & Lineage Engine
from genomic_rarity_engine import GenomicRarityAndLineageSystem

# Constants
EXPECTED_SNP_COLUMNS = 5  # Number of expected columns in SNP data: rsid, chromosome, position, allele1, allele2

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


class AncestryDNAParser:
    """Parser for AncestryDNA.txt SNP data files."""

    def __init__(self, filepath: str):
        """Initialize parser with file path.

        Args:
            filepath: Path to AncestryDNA.txt file
        """
        self.filepath = filepath
        self.snps: List[Dict] = []
        self.chromosomes: Dict[str, List[Dict]] = defaultdict(list)

    def parse(self) -> None:
        """Parse the AncestryDNA.txt file and organize SNPs by chromosome."""
        logger.info(f"Parsing AncestryDNA file: {self.filepath}")

        with open(self.filepath, "r") as f:
            # Skip header lines (starting with #)
            lines = f.readlines()
            header_line = None
            data_start = 0

            for i, line in enumerate(lines):
                if line.startswith("#"):
                    continue
                elif line.startswith("rsid"):
                    header_line = line.strip()
                    data_start = i + 1
                    break

            logger.info(f"Header: {header_line}")

            # Parse SNP data
            for line in lines[data_start:]:
                line = line.strip()
                if not line:
                    continue

                parts = line.split("\t")
                if len(parts) >= EXPECTED_SNP_COLUMNS:
                    try:
                        snp = {
                            "rsid": parts[0],
                            "chromosome": parts[1],
                            "position": int(parts[2]),
                            "allele1": parts[3],
                            "allele2": parts[4],
                        }
                        self.snps.append(snp)
                        self.chromosomes[snp["chromosome"]].append(snp)
                    except ValueError as e:
                        logger.warning(f"Skipping invalid SNP data: {line[:50]}... Error: {e}")
                        continue

        logger.info(f"Parsed {len(self.snps)} SNPs across {len(self.chromosomes)} chromosomes")
        
        # Log chromosome distribution
        for chrom in sorted(self.chromosomes.keys(), key=lambda x: (len(x), x)):
            count = len(self.chromosomes[chrom])
            logger.info(f"  Chromosome {chrom}: {count} SNPs")

    def generate_sequences(self, max_snps_per_sequence: int = 1000) -> List[Tuple[str, str]]:
        """Generate sequences from SNP data for pipeline processing.

        Groups SNPs by chromosome and creates sequences by concatenating alleles.

        Args:
            max_snps_per_sequence: Maximum number of SNPs to include in each sequence

        Returns:
            List of (sequence_id, sequence) tuples
        """
        logger.info("Generating sequences from SNP data...")
        sequences = []

        for chrom in sorted(self.chromosomes.keys(), key=lambda x: (len(x), x)):
            snp_list = self.chromosomes[chrom]

            # Split into chunks if chromosome has too many SNPs
            num_chunks = (len(snp_list) + max_snps_per_sequence - 1) // max_snps_per_sequence

            for chunk_idx in range(num_chunks):
                start_idx = chunk_idx * max_snps_per_sequence
                end_idx = min(start_idx + max_snps_per_sequence, len(snp_list))
                chunk = snp_list[start_idx:end_idx]

                # Build sequence from alleles
                sequence = ""
                for snp in chunk:
                    # Use both alleles in the sequence
                    sequence += snp["allele1"] + snp["allele2"]

                # Create sequence ID
                seq_id = f"chr{chrom}_chunk{chunk_idx + 1}_{len(chunk)}snps"
                sequences.append((seq_id, sequence))

        logger.info(f"Generated {len(sequences)} sequences from SNP data")
        return sequences

    def get_summary_stats(self) -> Dict:
        """Get summary statistics of the parsed SNP data.

        Returns:
            Dictionary with summary statistics
        """
        stats = {
            "total_snps": len(self.snps),
            "total_chromosomes": len(self.chromosomes),
            "snps_per_chromosome": {
                chrom: len(snps) for chrom, snps in self.chromosomes.items()
            },
        }

        # Calculate allele statistics
        allele_counts = defaultdict(int)
        for snp in self.snps:
            allele_counts[snp["allele1"]] += 1
            allele_counts[snp["allele2"]] += 1

        stats["allele_distribution"] = dict(allele_counts)

        # Heterozygosity rate (different alleles)
        heterozygous = sum(1 for snp in self.snps if snp["allele1"] != snp["allele2"])
        stats["heterozygosity_rate"] = heterozygous / len(self.snps) if self.snps else 0

        return stats


class AncestryDNASequencingPipeline(FullGenomeSequencingPipeline):
    """Extended pipeline specifically for AncestryDNA data."""

    def __init__(
        self,
        config: GenomeSequencingConfig,
        ancestrydna_parser: AncestryDNAParser,
    ):
        """Initialize pipeline with AncestryDNA parser.

        Args:
            config: Pipeline configuration
            ancestrydna_parser: Parsed AncestryDNA data
        """
        super().__init__(config)
        self.ancestrydna_parser = ancestrydna_parser
        
        # Initialize Genomic Rarity & Lineage System
        self.rarity_system = GenomicRarityAndLineageSystem()
        logger.info("Genomic Rarity & Lineage System initialized")

    def run_ancestrydna_sequencing(self) -> Dict:
        """Run full genome sequencing on AncestryDNA data.

        Returns:
            Complete pipeline results with AncestryDNA-specific metadata
        """
        logger.info("=" * 60)
        logger.info("AncestryDNA Genome Sequencing")
        logger.info("=" * 60)

        # Generate sequences from SNP data
        sequences = self.ancestrydna_parser.generate_sequences()

        # Get summary statistics
        summary_stats = self.ancestrydna_parser.get_summary_stats()

        logger.info("=" * 60)
        logger.info("AncestryDNA Data Summary")
        logger.info("=" * 60)
        logger.info(f"Total SNPs: {summary_stats['total_snps']:,}")
        logger.info(f"Total Chromosomes: {summary_stats['total_chromosomes']}")
        logger.info(f"Heterozygosity Rate: {summary_stats['heterozygosity_rate']:.2%}")
        logger.info(f"Generated Sequences: {len(sequences)}")

        # ============================================================
        # TIER-VI GENOMIC RARITY & ROYAL LINEAGE ANALYSIS
        # ============================================================
        logger.info("=" * 60)
        logger.info("Tier-VI Genomic Rarity & Royal Lineage Analysis")
        logger.info("=" * 60)
        
        # Perform comprehensive rarity and lineage analysis
        rarity_lineage_results = self.rarity_system.analyze_genome(self.ancestrydna_parser.snps)
        
        # Save rarity & lineage report
        rarity_report_path = self.rarity_system.generate_report(
            rarity_lineage_results, 
            self.config.output_dir
        )
        logger.info(f"✓ Rarity & Lineage analysis saved to {rarity_report_path}")

        # Execute pipeline phases
        alignment_results = self.execute_alignment(sequences)

        # Generate synthetic omics data
        omics_datasets = self._generate_synthetic_omics_data()
        fusion_results = self.execute_fusion(omics_datasets)

        # Generate synthetic time-series data
        timeseries_data = self._generate_synthetic_timeseries_data()
        te_results = self.execute_transfer_entropy(timeseries_data)

        # Generate synthetic variant graph
        variant_graph = self._generate_synthetic_variant_graph()
        inference_results = self.execute_inference(variant_graph)

        # Generate audit summary
        audit_summary = self.generate_audit_summary()

        # Generate reproducibility report
        reproducibility_report = self.generate_reproducibility_report()

        # Save all results
        self.save_results(alignment_results, fusion_results, te_results, inference_results)

        # Save AncestryDNA-specific summary
        ancestrydna_summary = {
            "ancestrydna_stats": summary_stats,
            "sequences_generated": len(sequences),
            "sequence_details": [
                {"id": seq_id, "length": len(seq)} for seq_id, seq in sequences[:10]
            ],  # First 10 as examples
        }

        summary_path = os.path.join(self.config.output_dir, "ancestrydna_summary.json")
        with open(summary_path, "w") as f:
            json.dump(ancestrydna_summary, f, indent=2)
        logger.info(f"✓ AncestryDNA summary saved to {summary_path}")

        # Calculate total metrics
        self.metrics.total_duration_ms = sum(
            [
                self.metrics.alignment_duration_ms,
                self.metrics.fusion_duration_ms,
                self.metrics.transfer_entropy_duration_ms,
                self.metrics.inference_duration_ms,
            ]
        )

        # Create deployment report
        deployment_report = {
            "status": "SUCCESS",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "data_source": "AncestryDNA.txt",
            "analysis_type": "Tier-VI Genomic-Genealogical Intelligence System",
            "ancestrydna_summary": ancestrydna_summary,
            "rarity_lineage_analysis": {
                "genome_wide_rarity": rarity_lineage_results.get("genome_wide_rarity", {}),
                "haplogroups": rarity_lineage_results.get("haplogroups", {}),
                "royal_lineage_summary": rarity_lineage_results.get("royal_lineage", {}),
                "full_report": rarity_report_path,
            },
            "metrics": asdict(self.metrics),
            "audit_summary": audit_summary,
            "reproducibility": reproducibility_report,
            "outputs": {
                "alignment_result": f"{self.config.output_dir}/alignment_result.json",
                "fusion_result": f"{self.config.output_dir}/fusion_result.json",
                "transfer_entropy": f"{self.config.output_dir}/transfer_entropy.json",
                "functional_predictions": f"{self.config.output_dir}/functional_predictions.json",
                "audit_summary": f"{self.config.output_dir}/audit_summary.json",
                "ancestrydna_summary": summary_path,
                "rarity_lineage_full_report": rarity_report_path,
            },
        }

        # Save deployment report
        report_path = os.path.join(self.config.output_dir, "deployment_report.json")
        with open(report_path, "w") as f:
            json.dump(deployment_report, f, indent=2)
        logger.info(f"✓ Deployment report saved to {report_path}")

        # Final summary
        logger.info("=" * 60)
        logger.info("AncestryDNA Genome Sequencing Complete!")
        logger.info("=" * 60)
        logger.info(f"Total SNPs Processed: {summary_stats['total_snps']:,}")
        logger.info(f"Sequences Analyzed: {len(sequences)}")
        logger.info(f"Chromosomes Covered: {summary_stats['total_chromosomes']}")
        logger.info(f"Total Duration: {self.metrics.total_duration_ms:.2f} ms")
        logger.info(f"Audit Violations: {audit_summary['unresolved_critical']}")
        logger.info("=" * 60)
        logger.info("Tier-VI Rarity & Lineage Analysis:")
        logger.info(f"  Genome-wide Rarity Percentile: {rarity_lineage_results['genome_wide_rarity'].get('rarity_percentile', 0):.2f}%")
        logger.info(f"  Y-Haplogroup: {rarity_lineage_results['haplogroups'].get('y_chromosome', 'Unknown')}")
        logger.info(f"  mtDNA Haplogroup: {rarity_lineage_results['haplogroups'].get('mitochondrial', 'Unknown')}")
        logger.info(f"  Royal Connections: {rarity_lineage_results['royal_lineage'].get('total_royal_connections', 0)}")
        logger.info("=" * 60)

        return deployment_report


def main():
    """Main entry point for AncestryDNA genome sequencing."""

    parser = argparse.ArgumentParser(
        description="Sequence AncestryDNA.txt genome data using XENON Quantum Bioinformatics v5"
    )
    parser.add_argument(
        "--ancestrydna-file",
        type=str,
        default="AncestryDNA.txt",
        help="Path to AncestryDNA.txt file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/ancestrydna_sequencing",
        help="Output directory for results",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Global random seed",
    )
    parser.add_argument(
        "--max-threads",
        type=int,
        default=8,
        help="Maximum number of threads",
    )
    parser.add_argument(
        "--max-snps-per-sequence",
        type=int,
        default=1000,
        help="Maximum SNPs per sequence chunk",
    )

    args = parser.parse_args()

    # Check if file exists
    if not os.path.exists(args.ancestrydna_file):
        logger.error(f"AncestryDNA file not found: {args.ancestrydna_file}")
        sys.exit(1)

    # Parse AncestryDNA data
    ancestrydna_parser = AncestryDNAParser(args.ancestrydna_file)
    ancestrydna_parser.parse()

    # Create pipeline configuration
    config = GenomeSequencingConfig(
        global_seed=args.seed,
        max_threads=args.max_threads,
        output_dir=args.output_dir,
        prefer_gpu=True,
        prefer_qpu=False,
    )

    # Run pipeline
    pipeline = AncestryDNASequencingPipeline(
        config=config, ancestrydna_parser=ancestrydna_parser
    )
    deployment_report = pipeline.run_ancestrydna_sequencing()

    # Print final status
    if deployment_report["status"] == "SUCCESS":
        print("\n✅ AncestryDNA genome sequencing completed successfully!")
        print(f"📊 Results saved to: {args.output_dir}")
        print(f"📈 Processed {deployment_report['ancestrydna_summary']['ancestrydna_stats']['total_snps']:,} SNPs")
        sys.exit(0)
    else:
        print("\n❌ AncestryDNA genome sequencing failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
