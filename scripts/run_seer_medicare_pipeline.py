#!/usr/bin/env python3
"""
SEER-Medicare Pipeline Runner

Runs the complete SEER-Medicare → QRATUM oncology analysis pipeline.

RESEARCH USE ONLY - Not for clinical diagnosis or treatment decisions.

Usage:
    python scripts/run_seer_medicare_pipeline.py \
        --seer_dir /data/seer/ \
        --claims_dir /data/medicare/ \
        --cancer_site "lung" \
        --histology "adenocarcinoma" \
        --stage "IV" \
        --diagnosis_year_min 2010 \
        --diagnosis_year_max 2018 \
        --lookback_days 365 \
        --followup_days 1095 \
        --seed 42 \
        --max_depth 6 \
        --n_sequences 10 \
        --output_dir artifacts/seer_medicare_run_...

All outputs are DUA-compliant:
- No patient-level identifiers in outputs
- Cell sizes < 11 are suppressed
- Deterministic runs with fixed seed
- All artifacts include hashes for reproducibility
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Add repository root to path if needed
repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from qratum.oncology.registry.seer_medicare.cohort import CohortBuilder, CohortDefinition
from qratum.oncology.registry.seer_medicare.features import FeatureEngineer
from qratum.oncology.registry.seer_medicare.io import (
    RunArtifacts,
    create_dataset_manifest,
    create_output_directory,
)
from qratum.oncology.registry.seer_medicare.linkages import PatientLinker
from qratum.oncology.registry.seer_medicare.medicare_claims import parse_claims_directory
from qratum.oncology.registry.seer_medicare.privacy import (
    DUAComplianceChecker,
    PrivacyConfig,
    SafeLogger,
)
from qratum.oncology.registry.seer_medicare.seer_registry import SEERRegistryParser
from qratum.oncology.registry.seer_medicare.timelines import (
    CodeMappingLibrary,
    TreatmentTimelineBuilder,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


RESEARCH_DISCLAIMER = """
================================================================================
RESEARCH USE ONLY - NOT FOR CLINICAL DIAGNOSIS OR TREATMENT

This pipeline produces research hypotheses and retrospective analyses.
All outputs require clinical validation before any interpretation.
Results should be reviewed by domain experts and statisticians.

DUA Compliance:
- All patient identifiers are hashed
- Cell sizes < 11 are suppressed
- No data is uploaded to external systems
- All artifacts include reproducibility hashes
================================================================================
"""


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run SEER-Medicare → QRATUM oncology analysis pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=RESEARCH_DISCLAIMER,
    )

    # Required arguments
    parser.add_argument(
        "--seer_dir",
        type=str,
        required=True,
        help="Directory containing SEER registry files (CSV format)",
    )
    parser.add_argument(
        "--claims_dir",
        type=str,
        required=True,
        help="Directory containing Medicare claims files (CSV format)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: artifacts/seer_medicare_run_<timestamp>)",
    )

    # Cohort definition
    parser.add_argument(
        "--cancer_site",
        type=str,
        default="lung",
        help="Cancer site filter (e.g., 'lung', 'breast', 'colon')",
    )
    parser.add_argument(
        "--histology",
        type=str,
        default=None,
        help="Histology filter (e.g., 'adenocarcinoma')",
    )
    parser.add_argument(
        "--stage",
        type=str,
        default=None,
        help="Stage filter (e.g., 'IV', 'III')",
    )
    parser.add_argument(
        "--diagnosis_year_min",
        type=int,
        default=2010,
        help="Minimum diagnosis year (inclusive)",
    )
    parser.add_argument(
        "--diagnosis_year_max",
        type=int,
        default=2018,
        help="Maximum diagnosis year (inclusive)",
    )
    parser.add_argument(
        "--age_min",
        type=int,
        default=65,
        help="Minimum age at diagnosis",
    )
    parser.add_argument(
        "--age_max",
        type=int,
        default=None,
        help="Maximum age at diagnosis",
    )

    # Timeline parameters
    parser.add_argument(
        "--lookback_days",
        type=int,
        default=365,
        help="Days before index date for lookback window",
    )
    parser.add_argument(
        "--followup_days",
        type=int,
        default=1095,
        help="Days after index date for follow-up window",
    )
    parser.add_argument(
        "--gap_days",
        type=int,
        default=45,
        help="Gap days to define new line of therapy",
    )

    # QRATUM parameters
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--max_depth",
        type=int,
        default=6,
        help="Maximum depth for XENON search tree",
    )
    parser.add_argument(
        "--n_sequences",
        type=int,
        default=10,
        help="Number of top sequences to return",
    )

    # Privacy settings
    parser.add_argument(
        "--min_cell_size",
        type=int,
        default=11,
        help="Minimum cell size for suppression",
    )
    parser.add_argument(
        "--salt",
        type=str,
        default="",
        help="Salt for patient key hashing",
    )

    # Optional
    parser.add_argument(
        "--enrollment_file",
        type=str,
        default=None,
        help="Path to enrollment file for continuous enrollment check",
    )
    parser.add_argument(
        "--code_mapping_file",
        type=str,
        default=None,
        help="Path to custom code mapping YAML file",
    )
    parser.add_argument(
        "--skip_qratum",
        action="store_true",
        help="Skip QRATUM analysis (cohort and timeline only)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    return parser.parse_args()


def run_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    """Run the complete pipeline.

    Args:
        args: Parsed command line arguments

    Returns:
        Dictionary of results
    """
    print(RESEARCH_DISCLAIMER)

    # Setup privacy config
    privacy_config = PrivacyConfig(
        min_cell_size=args.min_cell_size,
        salt=args.salt,
    )
    safe_logger = SafeLogger("pipeline", privacy_config)

    # Create output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = create_output_directory()

    safe_logger.info(f"Output directory: {output_dir}")

    # Build run config
    run_config = {
        "seer_dir": str(args.seer_dir),
        "claims_dir": str(args.claims_dir),
        "cancer_site": args.cancer_site,
        "histology": args.histology,
        "stage": args.stage,
        "diagnosis_year_min": args.diagnosis_year_min,
        "diagnosis_year_max": args.diagnosis_year_max,
        "age_min": args.age_min,
        "age_max": args.age_max,
        "lookback_days": args.lookback_days,
        "followup_days": args.followup_days,
        "gap_days": args.gap_days,
        "seed": args.seed,
        "max_depth": args.max_depth,
        "n_sequences": args.n_sequences,
        "min_cell_size": args.min_cell_size,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "research_disclaimer": "RESEARCH USE ONLY - NOT FOR CLINICAL USE",
    }

    # Initialize artifacts
    artifacts = RunArtifacts(output_dir=output_dir, run_config=run_config)

    # Step 1: Create dataset manifest
    safe_logger.info("Step 1: Creating dataset manifest...")
    manifest = create_dataset_manifest(
        seer_dir=Path(args.seer_dir),
        claims_dir=Path(args.claims_dir),
        compute_hashes=True,
    )
    artifacts.dataset_manifest = manifest
    safe_logger.info_aggregate("Files discovered", manifest.file_count)

    # Step 2: Parse SEER registry
    safe_logger.info("Step 2: Parsing SEER registry...")
    seer_parser = SEERRegistryParser(privacy_config=privacy_config)
    linker = PatientLinker(privacy_config=privacy_config)

    seer_dir = Path(args.seer_dir)
    if seer_dir.exists():
        cases = seer_parser.parse_directory(seer_dir, file_pattern="*.csv")
        linker.add_registry_cases(cases)
    else:
        safe_logger.warning(f"SEER directory not found: {seer_dir}")

    # Step 3: Load enrollment if provided
    if args.enrollment_file:
        safe_logger.info("Step 3: Loading enrollment data...")
        enrollment_path = Path(args.enrollment_file)
        if enrollment_path.exists():
            linker.load_enrollment_file(enrollment_path)
        else:
            safe_logger.warning(f"Enrollment file not found: {enrollment_path}")
    else:
        safe_logger.info("Step 3: Skipping enrollment check (no file provided)")

    # Build linked patients
    linked_patients = linker.build_linked_patients()

    # Step 4: Build cohort
    safe_logger.info("Step 4: Building cohort...")
    cohort_def = CohortDefinition(
        name=f"{args.cancer_site}_{args.diagnosis_year_min}_{args.diagnosis_year_max}",
        cancer_site=args.cancer_site,
        histology=args.histology,
        stage=args.stage,
        diagnosis_year_min=args.diagnosis_year_min,
        diagnosis_year_max=args.diagnosis_year_max,
        age_min=args.age_min,
        age_max=args.age_max,
        require_continuous_enrollment=args.enrollment_file is not None,
    )

    cohort_builder = CohortBuilder(cohort_def, privacy_config=privacy_config)
    cohort = cohort_builder.build_cohort(iter(linked_patients.values()))
    cohort_stats = cohort_builder.get_stats_dict()
    artifacts.cohort_counts = cohort_stats

    safe_logger.info_aggregate("Cohort size", len(cohort))

    # Step 5: Parse claims and build timelines
    safe_logger.info("Step 5: Parsing claims and building timelines...")

    # Load code mappings
    code_library = CodeMappingLibrary.create_default_library()
    if args.code_mapping_file:
        mapping_path = Path(args.code_mapping_file)
        if mapping_path.exists():
            if mapping_path.suffix == ".yaml":
                code_library.load_from_yaml(mapping_path)
            else:
                code_library.load_from_json(mapping_path)

    # Build timeline builder
    timeline_builder = TreatmentTimelineBuilder(
        code_library=code_library,
        lookback_days=args.lookback_days,
        followup_days=args.followup_days,
        gap_days_for_new_line=args.gap_days,
        privacy_config=privacy_config,
    )

    # Parse claims
    claims_dir = Path(args.claims_dir)
    timelines = []

    if claims_dir.exists() and cohort:
        # Create patient key lookup
        cohort_keys = {p.patient_key for p in cohort}

        # Collect claims by patient
        patient_claims: dict[str, list] = {key: [] for key in cohort_keys}

        for claim in parse_claims_directory(claims_dir, privacy_config):
            if claim.patient_key in patient_claims:
                patient_claims[claim.patient_key].append(claim)

        # Build timelines
        for patient in cohort:
            claims = patient_claims.get(patient.patient_key, [])
            timeline = timeline_builder.build_timeline(patient, claims)
            timelines.append(timeline)

    timeline_summary = timeline_builder.get_summary_dict()
    artifacts.timeline_summary = timeline_summary

    # Step 6: QRATUM Analysis (if not skipped)
    qratum_results: dict[str, Any] = {}
    if not args.skip_qratum and timelines:
        safe_logger.info("Step 6: Running QRATUM analysis...")
        qratum_results = run_qratum_analysis(
            timelines=timelines,
            seed=args.seed,
            max_depth=args.max_depth,
            n_sequences=args.n_sequences,
            privacy_config=privacy_config,
        )
    else:
        safe_logger.info("Step 6: QRATUM analysis skipped")
        qratum_results = {"status": "skipped", "reason": "No timelines or --skip_qratum"}

    artifacts.qratum_sequences = qratum_results

    # Save all artifacts
    safe_logger.info("Saving artifacts...")
    artifacts.save_all()

    # Generate DUA checklist
    dua_checker = DUAComplianceChecker(privacy_config)
    dua_checklist = dua_checker.generate_checklist()
    with open(output_dir / "dua_checklist.json", "w") as f:
        json.dump(dua_checklist, f, indent=2)

    safe_logger.info(f"Pipeline complete. Outputs in: {output_dir}")

    return {
        "output_dir": str(output_dir),
        "cohort_size": len(cohort),
        "timelines_built": len(timelines),
        "qratum_status": "complete" if qratum_results else "skipped",
    }


def run_qratum_analysis(
    timelines: list,
    seed: int,
    max_depth: int,
    n_sequences: int,
    privacy_config: PrivacyConfig,
) -> dict[str, Any]:
    """Run QRATUM causal graph and intervention search.

    Args:
        timelines: List of PatientTimeline objects
        seed: Random seed
        max_depth: Maximum search depth
        n_sequences: Number of sequences to return
        privacy_config: Privacy configuration

    Returns:
        Dictionary of QRATUM results
    """
    try:
        from qratum.oncology.causal_graph import CausalOncologyGraph, NodeType, OncogenicNode
        from qratum.oncology.intervention_search import (
            XENONInterventionSearch,
            create_example_drug_library,
        )
    except ImportError:
        return {
            "status": "error",
            "message": "QRATUM oncology modules not available",
        }

    feature_engineer = FeatureEngineer(privacy_config=privacy_config)

    # Aggregate state features across cohort
    state_vectors = []
    for timeline in timelines:
        if timeline.index_date:
            baseline = feature_engineer.extract_baseline_features(timeline)
            state = feature_engineer.extract_state_features(timeline, timeline.index_date)
            state_dict = feature_engineer.create_qratum_state_dict(baseline, state)
            state_vectors.append(state_dict)

    # Compute average state for cohort
    avg_state: dict[str, float] = {}
    if state_vectors:
        for key in state_vectors[0].keys():
            values = [s.get(key, 0.5) for s in state_vectors]
            avg_state[key] = sum(values) / len(values)
    else:
        # Default state when no timelines available
        # These are PLACEHOLDER values for testing/demo only
        # Rationale:
        # - tumor_burden=0.8: Assumes advanced disease (stage IV typical)
        # - immune_engagement=0.2: Low baseline without immunotherapy
        # - toxicity_level=0.2: Moderate baseline toxicity
        # - proliferation_rate=0.6: Active disease state
        # These should be derived from cohort data when available
        avg_state = {
            "tumor_burden": 0.8,  # High burden for advanced cancer cohort
            "immune_engagement": 0.2,  # Low baseline immune engagement
            "toxicity_level": 0.2,  # Moderate baseline toxicity
            "proliferation_rate": 0.6,  # Active disease proliferation
        }

    # Build causal graph
    graph = CausalOncologyGraph(
        name="SEER_Medicare_Cohort",
        cancer_type="claims_derived",
        seed=seed,
    )

    # Add basic nodes
    for node_name in ["tumor_burden", "immune_engagement", "toxicity", "treatment_response"]:
        node = OncogenicNode(
            node_id=node_name,
            name=node_name.replace("_", " ").title(),
            node_type=NodeType.PHENOTYPE,
            activity_level=avg_state.get(node_name, 0.5),
            druggability=0.5 if "treatment" in node_name else 0.0,
        )
        graph.add_node(node)

    # Run XENON search
    search = XENONInterventionSearch(
        seed=seed,
        max_depth=max_depth,
    )

    # Add drugs from library
    drug_library = create_example_drug_library()
    for drug in drug_library.values():
        search.add_drug(drug)

    # Search for sequences
    sequences = search.search_best_sequences(
        initial_tumor_state=avg_state,
        n_sequences=n_sequences,
    )

    # Format results
    results = {
        "status": "complete",
        "seed": seed,
        "max_depth": max_depth,
        "cohort_avg_state": avg_state,
        "graph_hash": graph.compute_graph_hash(),
        "sequences": [],
        "research_disclaimer": (
            "These sequences are RESEARCH HYPOTHESES only. "
            "They require experimental validation and clinical trials. "
            "NOT for clinical treatment decisions."
        ),
    }

    for seq in sequences:
        seq_dict = {
            "sequence_id": seq.sequence_id,
            "total_efficacy": seq.total_efficacy,
            "total_toxicity": seq.total_toxicity,
            "resistance_suppression_score": seq.resistance_suppression_score,
            "immune_engagement_score": seq.immune_engagement_score,
            "rationale": seq.rationale,
            "interventions": [
                {
                    "drug_name": drug.name,
                    "dosage": dosage,
                    "timing_week": week,
                }
                for drug, dosage, week in seq.interventions
            ],
            "risk_flags": [
                "Retrospective analysis - requires prospective validation",
                "Medicare population bias (age 65+)",
                "Claims-based proxies, not clinical measurements",
                "Treatment mappings require expert validation",
            ],
        }
        results["sequences"].append(seq_dict)

    return results


def main() -> None:
    """Main entry point."""
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        results = run_pipeline(args)
        print("\nPipeline completed successfully.")
        print(f"Output directory: {results['output_dir']}")
        print(f"Cohort size: {results['cohort_size']}")
        print(f"Timelines built: {results['timelines_built']}")
        print(f"QRATUM status: {results['qratum_status']}")
    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
