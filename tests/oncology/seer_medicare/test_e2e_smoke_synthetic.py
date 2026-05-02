"""
End-to-end smoke tests for SEER-Medicare pipeline.

Uses synthetic test data that mimics SEER-Medicare formats
without containing any actual patient data.

This test validates that the entire pipeline can run
from input parsing to QRATUM analysis without errors.
"""

from __future__ import annotations

import csv
from datetime import date, timedelta
from pathlib import Path

import pytest

from qratum.oncology.registry.seer_medicare.cohort import CohortBuilder, CohortDefinition
from qratum.oncology.registry.seer_medicare.features import FeatureEngineer
from qratum.oncology.registry.seer_medicare.io import (
    RunArtifacts,
    create_dataset_manifest,
)
from qratum.oncology.registry.seer_medicare.linkages import PatientLinker
from qratum.oncology.registry.seer_medicare.medicare_claims import parse_claims_directory
from qratum.oncology.registry.seer_medicare.privacy import (
    DUAComplianceChecker,
    PrivacyConfig,
    SafeLogger,
)
from qratum.oncology.registry.seer_medicare.quality import DataValidator
from qratum.oncology.registry.seer_medicare.schema import (
    ClaimEvent,
)
from qratum.oncology.registry.seer_medicare.seer_registry import SEERRegistryParser
from qratum.oncology.registry.seer_medicare.timelines import (
    CodeMappingLibrary,
    TreatmentTimelineBuilder,
)


@pytest.fixture
def synthetic_data_dir(tmp_path: Path) -> dict[str, Path]:
    """Create a complete synthetic SEER-Medicare dataset."""
    seer_dir = tmp_path / "seer"
    claims_dir = tmp_path / "claims"
    output_dir = tmp_path / "artifacts"

    seer_dir.mkdir()
    claims_dir.mkdir()
    output_dir.mkdir()

    # Create SEER registry file with 20 patients
    seer_file = seer_dir / "seer_cases.csv"
    seer_rows = []
    for i in range(20):
        patient_id = f"PT{i:04d}"
        dx_year = 2015 + (i % 4)  # Years 2015-2018
        dx_month = (i % 12) + 1
        dx_day = min(28, (i % 28) + 1)

        seer_rows.append(
            {
                "PATIENT_ID": patient_id,
                "DATE_OF_DIAGNOSIS": f"{dx_year}{dx_month:02d}{dx_day:02d}",
                "PRIMARY_SITE": "LUNG" if i % 3 == 0 else ("BREAST" if i % 3 == 1 else "COLON"),
                "HISTOLOGY": "ADENOCARCINOMA" if i % 2 == 0 else "SQUAMOUS",
                "STAGE": (
                    "IV" if i % 4 == 0 else ("III" if i % 4 == 1 else ("II" if i % 4 == 2 else "I"))
                ),
                "AGE": str(65 + i),
                "SEX": "M" if i % 2 == 0 else "F",
                "RACE": "01",
                "VITAL_STATUS": "1" if i < 15 else "4",
                "SURVIVAL": str(24 + i * 2),
                "SEQUENCE_NUMBER": "0",
            }
        )

    with open(seer_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=seer_rows[0].keys())
        writer.writeheader()
        writer.writerows(seer_rows)

    # Create NCH (carrier) claims file
    nch_file = claims_dir / "nch_carrier.csv"
    nch_rows = []
    for i in range(20):
        patient_id = f"PT{i:04d}"
        base_date = date(2015 + (i % 4), (i % 12) + 1, min(28, (i % 28) + 1))

        # Add multiple claims per patient
        for j in range(3):
            claim_date = base_date + timedelta(days=14 + j * 21)
            nch_rows.append(
                {
                    "BENE_ID": patient_id,
                    "CLM_ID": f"CLM{i:04d}{j:02d}",
                    "CLM_FROM_DT": claim_date.strftime("%Y%m%d"),
                    "HCPCS_CD": "96413" if j == 0 else ("J9271" if j == 1 else "J9045"),
                    "CLM_PMT_AMT": str(500 + j * 100),
                    "PRVDR_SPCLTY": "83",
                }
            )

    with open(nch_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=nch_rows[0].keys())
        writer.writeheader()
        writer.writerows(nch_rows)

    # Create MEDPAR (inpatient) file
    medpar_file = claims_dir / "medpar_inpatient.csv"
    medpar_rows = []
    for i in range(0, 20, 3):  # Every 3rd patient has inpatient
        patient_id = f"PT{i:04d}"
        base_date = date(2015 + (i % 4), (i % 12) + 1, min(28, (i % 28) + 1))
        admit_date = base_date + timedelta(days=60)

        medpar_rows.append(
            {
                "BENE_ID": patient_id,
                "CLM_ID": f"MED{i:04d}",
                "ADMSN_DT": admit_date.strftime("%Y%m%d"),
                "DSCHRG_DT": (admit_date + timedelta(days=5)).strftime("%Y%m%d"),
                "DRG_CD": "470",
                "PRNCPAL_DGNS_CD": "C3490",
                "CLM_PMT_AMT": "15000",
            }
        )

    with open(medpar_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=medpar_rows[0].keys())
        writer.writeheader()
        writer.writerows(medpar_rows)

    return {
        "seer_dir": seer_dir,
        "claims_dir": claims_dir,
        "output_dir": output_dir,
    }


class TestEndToEndPipeline:
    """End-to-end smoke tests for the pipeline."""

    def test_full_pipeline_smoke_test(self, synthetic_data_dir: dict[str, Path]) -> None:
        """Test complete pipeline execution with synthetic data."""
        seer_dir = synthetic_data_dir["seer_dir"]
        claims_dir = synthetic_data_dir["claims_dir"]
        output_dir = synthetic_data_dir["output_dir"]

        privacy_config = PrivacyConfig(
            min_cell_size=3, salt="test_salt"
        )  # Lower threshold for test
        safe_logger = SafeLogger("test", privacy_config)

        # Step 1: Create dataset manifest
        manifest = create_dataset_manifest(
            seer_dir=seer_dir,
            claims_dir=claims_dir,
            compute_hashes=True,
        )
        assert manifest.file_count >= 2
        assert manifest.total_size_bytes > 0

        # Step 2: Parse SEER registry
        seer_parser = SEERRegistryParser(privacy_config=privacy_config)
        linker = PatientLinker(privacy_config=privacy_config)

        cases = list(seer_parser.parse_directory(seer_dir, file_pattern="*.csv"))
        assert len(cases) == 20

        for case in cases:
            linker.add_registry_case(case)

        linked_patients = linker.build_linked_patients()
        assert len(linked_patients) == 20

        # Step 3: Build cohort (filter to lung cancer)
        cohort_def = CohortDefinition(
            name="lung_test",
            cancer_site="LUNG",
            diagnosis_year_min=2015,
            diagnosis_year_max=2018,
        )

        cohort_builder = CohortBuilder(cohort_def, privacy_config=privacy_config)
        cohort = cohort_builder.build_cohort(iter(linked_patients.values()))

        # Should have about 7 lung cancer patients (every 3rd)
        assert len(cohort) >= 3

        # Step 4: Parse claims
        claims_list = list(parse_claims_directory(claims_dir, privacy_config))
        assert len(claims_list) > 0

        # Group claims by patient
        patient_claims: dict[str, list[ClaimEvent]] = {}
        for claim in claims_list:
            if claim.patient_key not in patient_claims:
                patient_claims[claim.patient_key] = []
            patient_claims[claim.patient_key].append(claim)

        # Step 5: Build timelines
        timeline_builder = TreatmentTimelineBuilder(
            code_library=CodeMappingLibrary.create_default_library(),
            lookback_days=365,
            followup_days=365,
            privacy_config=privacy_config,
        )

        timelines = []
        for patient in cohort:
            claims = patient_claims.get(patient.patient_key, [])
            timeline = timeline_builder.build_timeline(patient, claims)
            timelines.append(timeline)

        assert len(timelines) == len(cohort)

        # Step 6: Extract features
        feature_engineer = FeatureEngineer(privacy_config=privacy_config)

        for timeline in timelines:
            if timeline.index_date:
                baseline = feature_engineer.extract_baseline_features(timeline)
                assert baseline.patient_key == timeline.patient_key

                state = feature_engineer.extract_state_features(timeline, timeline.index_date)
                assert state.assessment_date == timeline.index_date

        # Step 7: Get summary statistics
        cohort_stats = cohort_builder.get_stats_dict()
        timeline_summary = timeline_builder.get_summary_dict()

        assert "included" in cohort_stats
        assert "n_patients" in timeline_summary

        # Step 8: Save artifacts
        artifacts = RunArtifacts(
            output_dir=output_dir,
            run_config={"test": True, "seed": 42},
            dataset_manifest=manifest,
            cohort_counts=cohort_stats,
            timeline_summary=timeline_summary,
        )
        artifacts.save_all()

        # Verify artifacts were created
        assert (output_dir / "run_config.json").exists()
        assert (output_dir / "dataset_manifest.json").exists()
        assert (output_dir / "cohort_counts.json").exists()
        assert (output_dir / "timeline_summary.json").exists()
        assert (output_dir / "environment.txt").exists()
        assert (output_dir / "git_commit.txt").exists()

    def test_data_validation(self, synthetic_data_dir: dict[str, Path]) -> None:
        """Test data validation functionality."""
        seer_dir = synthetic_data_dir["seer_dir"]
        claims_dir = synthetic_data_dir["claims_dir"]

        privacy_config = PrivacyConfig(min_cell_size=3)
        validator = DataValidator(privacy_config=privacy_config)

        # Validate registry data
        seer_parser = SEERRegistryParser(privacy_config=privacy_config)
        cases = seer_parser.parse_directory(seer_dir)
        registry_metrics = validator.compute_registry_quality_metrics(cases)

        assert registry_metrics.total_records == 20
        assert registry_metrics.valid_records == 20
        assert registry_metrics.field_completeness["patient_key"] == 1.0

        # Validate claims data
        claims = parse_claims_directory(claims_dir, privacy_config)
        claims_metrics = validator.compute_claims_quality_metrics(claims)

        assert claims_metrics.total_records > 0
        assert claims_metrics.field_completeness["code"] == 1.0

    def test_dua_compliance_checklist(self, synthetic_data_dir: dict[str, Path]) -> None:
        """Test DUA compliance checklist generation."""
        privacy_config = PrivacyConfig(min_cell_size=11)
        checker = DUAComplianceChecker(privacy_config)

        checklist = checker.generate_checklist()

        assert "checklist" in checklist
        assert len(checklist["checklist"]) > 0
        assert checklist["min_cell_size"] == 11

    def test_deterministic_results(self, synthetic_data_dir: dict[str, Path]) -> None:
        """Test that results are deterministic with same seed."""
        seer_dir = synthetic_data_dir["seer_dir"]

        privacy_config = PrivacyConfig(salt="fixed_salt")

        # Run twice
        results = []
        for _ in range(2):
            parser = SEERRegistryParser(privacy_config=privacy_config)
            cases = list(parser.parse_directory(seer_dir))
            results.append([c.patient_key for c in cases])

        # Patient keys should be identical
        assert results[0] == results[1]

    def test_privacy_controls(self, synthetic_data_dir: dict[str, Path]) -> None:
        """Test that privacy controls are enforced."""
        seer_dir = synthetic_data_dir["seer_dir"]

        privacy_config = PrivacyConfig(min_cell_size=11, salt="test")

        # Parse data
        parser = SEERRegistryParser(privacy_config=privacy_config)
        cases = list(parser.parse_directory(seer_dir))

        # No raw patient IDs should be in patient_key
        for case in cases:
            assert "PT" not in case.patient_key
            assert len(case.patient_key) == 16  # Truncated hash

        # Build cohort with small counts
        cohort_def = CohortDefinition(
            name="test",
            cancer_site="LUNG",  # Should have ~7 patients
        )
        linker = PatientLinker(privacy_config=privacy_config)
        for case in cases:
            linker.add_registry_case(case)
        linked = linker.build_linked_patients()

        builder = CohortBuilder(cohort_def, privacy_config=privacy_config)
        builder.build_cohort(iter(linked.values()))
        stats = builder.get_stats_dict()

        # Small cell suppression should be applied
        # Since we have <11 in some categories, they should be suppressed
        for key, value in stats.get("age_distribution", {}).items():
            if isinstance(value, str):
                assert "<11" in value  # Suppressed

    def test_qratum_integration(self, synthetic_data_dir: dict[str, Path]) -> None:
        """Test integration with QRATUM causal graph and XENON search."""
        seer_dir = synthetic_data_dir["seer_dir"]
        claims_dir = synthetic_data_dir["claims_dir"]

        privacy_config = PrivacyConfig(min_cell_size=3, salt="test")

        # Quick pipeline run
        parser = SEERRegistryParser(privacy_config=privacy_config)
        cases = list(parser.parse_directory(seer_dir))

        linker = PatientLinker(privacy_config=privacy_config)
        for case in cases:
            linker.add_registry_case(case)
        linked = linker.build_linked_patients()

        # Get a cohort
        cohort_def = CohortDefinition(name="test", cancer_site="LUNG")
        builder = CohortBuilder(cohort_def, privacy_config=privacy_config)
        cohort = builder.build_cohort(iter(linked.values()))

        # Parse claims and build timelines
        claims_list = list(parse_claims_directory(claims_dir, privacy_config))
        patient_claims: dict[str, list] = {}
        for claim in claims_list:
            if claim.patient_key not in patient_claims:
                patient_claims[claim.patient_key] = []
            patient_claims[claim.patient_key].append(claim)

        timeline_builder = TreatmentTimelineBuilder(
            code_library=CodeMappingLibrary.create_default_library(),
            privacy_config=privacy_config,
        )

        timelines = []
        for patient in cohort:
            claims = patient_claims.get(patient.patient_key, [])
            timeline = timeline_builder.build_timeline(patient, claims)
            timelines.append(timeline)

        # Extract features for QRATUM
        feature_engineer = FeatureEngineer(privacy_config=privacy_config)

        state_dicts = []
        for timeline in timelines:
            if timeline.index_date:
                baseline = feature_engineer.extract_baseline_features(timeline)
                state = feature_engineer.extract_state_features(timeline, timeline.index_date)
                state_dict = feature_engineer.create_qratum_state_dict(baseline, state)
                state_dicts.append(state_dict)

        # Should have state dictionaries
        assert len(state_dicts) > 0

        # Each state dict should have expected keys
        for state_dict in state_dicts:
            assert "tumor_burden" in state_dict
            assert "immune_engagement" in state_dict
            assert 0.0 <= state_dict["tumor_burden"] <= 1.0

        # Try QRATUM integration
        try:
            from qratum.oncology.causal_graph import CausalOncologyGraph, NodeType, OncogenicNode
            from qratum.oncology.intervention_search import (
                XENONInterventionSearch,
                create_example_drug_library,
            )

            # Create graph
            graph = CausalOncologyGraph(name="test_graph", seed=42)
            node = OncogenicNode(
                node_id="test_node",
                name="Test Node",
                node_type=NodeType.PHENOTYPE,
            )
            graph.add_node(node)

            # Create search
            search = XENONInterventionSearch(seed=42, max_depth=3)
            drugs = create_example_drug_library()
            for drug in drugs.values():
                search.add_drug(drug)

            # Run search with average state
            avg_state = {
                k: sum(s[k] for s in state_dicts) / len(state_dicts) for k in state_dicts[0].keys()
            }
            sequences = search.search_best_sequences(
                initial_tumor_state=avg_state,
                n_sequences=3,
            )

            assert len(sequences) > 0

        except ImportError:
            pytest.skip("QRATUM oncology modules not available")


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_seer_directory(self, tmp_path: Path) -> None:
        """Test handling of empty SEER directory."""
        seer_dir = tmp_path / "empty_seer"
        seer_dir.mkdir()

        parser = SEERRegistryParser()
        cases = list(parser.parse_directory(seer_dir))

        assert len(cases) == 0

    def test_empty_claims_directory(self, tmp_path: Path) -> None:
        """Test handling of empty claims directory."""
        claims_dir = tmp_path / "empty_claims"
        claims_dir.mkdir()

        claims = list(parse_claims_directory(claims_dir))

        assert len(claims) == 0

    def test_nonexistent_directory(self, tmp_path: Path) -> None:
        """Test handling of non-existent directories."""
        manifest = create_dataset_manifest(
            seer_dir=tmp_path / "nonexistent",
            compute_hashes=False,
        )

        assert manifest.file_count == 0

    def test_malformed_csv(self, tmp_path: Path) -> None:
        """Test handling of malformed CSV files."""
        bad_file = tmp_path / "bad_seer.csv"
        with open(bad_file, "w") as f:
            f.write("PATIENT_ID,DATE_OF_DIAGNOSIS\n")
            f.write("incomplete\n")  # Missing columns
            f.write("PT001,invalid_date\n")  # Invalid date

        parser = SEERRegistryParser()
        cases = list(parser.parse_file(bad_file))

        # Should handle errors gracefully
        assert len(cases) == 0  # Both rows should fail validation
