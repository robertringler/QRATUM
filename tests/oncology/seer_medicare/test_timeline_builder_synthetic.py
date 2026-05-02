"""
Tests for Treatment Timeline Builder functionality.

Uses synthetic test data that mimics SEER-Medicare formats
without containing any actual patient data.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from qratum.oncology.registry.seer_medicare.linkages import LinkedPatient
from qratum.oncology.registry.seer_medicare.schema import (
    ClaimEvent,
    ClaimSetting,
    CodeSystem,
    RegistryCase,
    VitalStatus,
)
from qratum.oncology.registry.seer_medicare.timelines import (
    CodeMapping,
    CodeMappingLibrary,
    TreatmentTimelineBuilder,
)


class TestCodeMappingLibrary:
    """Tests for code mapping library."""

    def test_add_and_get_mapping(self) -> None:
        """Test adding and retrieving mappings."""
        library = CodeMappingLibrary()

        mapping = CodeMapping(
            code="96413",
            code_system=CodeSystem.CPT,
            treatment_type="chemo_admin",
            description="Chemo IV infusion",
        )
        library.add_mapping(mapping)

        result = library.get_mapping("96413", CodeSystem.CPT)
        assert result is not None
        assert result.treatment_type == "chemo_admin"

    def test_lookup_code(self) -> None:
        """Test code lookup."""
        library = CodeMappingLibrary()

        mapping = CodeMapping(
            code="J9271",
            code_system=CodeSystem.HCPCS,
            treatment_type="immunotherapy",
            drug_name="pembrolizumab",
            confidence=0.9,
        )
        library.add_mapping(mapping)

        result = library.lookup_code("J9271", CodeSystem.HCPCS)
        assert result is not None
        treatment_type, drug_name, confidence = result
        assert treatment_type == "immunotherapy"
        assert drug_name == "pembrolizumab"
        assert confidence == 0.9

    def test_lookup_nonexistent_code(self) -> None:
        """Test lookup of code that doesn't exist."""
        library = CodeMappingLibrary()
        result = library.lookup_code("XXXXX", CodeSystem.CPT)
        assert result is None

    def test_create_default_library(self) -> None:
        """Test default library creation."""
        library = CodeMappingLibrary.create_default_library()

        # Should have some mappings
        assert len(library._mappings) > 0

        # Should have chemo admin codes
        result = library.lookup_code("96413", CodeSystem.CPT)
        assert result is not None
        assert result[0] == "chemo_admin"

        # Should have immunotherapy codes
        result = library.lookup_code("J9271", CodeSystem.HCPCS)
        assert result is not None

    def test_get_treatment_types(self) -> None:
        """Test getting all treatment types."""
        library = CodeMappingLibrary.create_default_library()
        types = library.get_treatment_types()

        assert "chemo_admin" in types
        assert "radiation" in types
        assert "surgery" in types


class TestTreatmentTimelineBuilder:
    """Tests for treatment timeline builder."""

    @pytest.fixture
    def code_library(self) -> CodeMappingLibrary:
        """Create a test code library."""
        library = CodeMappingLibrary()

        # Add some test mappings
        mappings = [
            CodeMapping(
                code="96413",
                code_system=CodeSystem.CPT,
                treatment_type="chemo_admin",
            ),
            CodeMapping(
                code="J9271",
                code_system=CodeSystem.HCPCS,
                treatment_type="immunotherapy",
                drug_name="pembrolizumab",
            ),
            CodeMapping(
                code="J9045",
                code_system=CodeSystem.HCPCS,
                treatment_type="chemo_drug",
                drug_name="carboplatin",
            ),
            CodeMapping(
                code="77385",
                code_system=CodeSystem.CPT,
                treatment_type="radiation",
            ),
            CodeMapping(
                code="32663",
                code_system=CodeSystem.CPT,
                treatment_type="surgery",
            ),
        ]

        for m in mappings:
            library.add_mapping(m)

        return library

    @pytest.fixture
    def sample_patient(self) -> LinkedPatient:
        """Create a sample linked patient."""
        case = RegistryCase(
            patient_key="test_patient_key",
            dx_date=date(2015, 3, 15),
            cancer_site="LUNG",
            histology="ADENOCARCINOMA",
            stage="IV",
            age_at_dx=72,
            sex="M",
            vital_status=VitalStatus.ALIVE,
        )

        return LinkedPatient(
            patient_key="test_patient_key",
            registry_case=case,
        )

    @pytest.fixture
    def sample_claims(self) -> list[ClaimEvent]:
        """Create sample claims for a patient."""
        patient_key = "test_patient_key"
        index_date = date(2015, 3, 15)

        claims = [
            # Pre-index: prior outpatient visit
            ClaimEvent(
                patient_key=patient_key,
                event_date=index_date - timedelta(days=30),
                setting=ClaimSetting.OUTPATIENT,
                code_system=CodeSystem.CPT,
                code="99213",
            ),
            # Week 1: Chemo administration
            ClaimEvent(
                patient_key=patient_key,
                event_date=index_date + timedelta(days=14),
                setting=ClaimSetting.OUTPATIENT,
                code_system=CodeSystem.CPT,
                code="96413",
            ),
            # Week 1: Carboplatin
            ClaimEvent(
                patient_key=patient_key,
                event_date=index_date + timedelta(days=14),
                setting=ClaimSetting.OUTPATIENT,
                code_system=CodeSystem.HCPCS,
                code="J9045",
            ),
            # Week 4: Chemo administration
            ClaimEvent(
                patient_key=patient_key,
                event_date=index_date + timedelta(days=35),
                setting=ClaimSetting.OUTPATIENT,
                code_system=CodeSystem.CPT,
                code="96413",
            ),
            # Week 8: Immunotherapy
            ClaimEvent(
                patient_key=patient_key,
                event_date=index_date + timedelta(days=56),
                setting=ClaimSetting.OUTPATIENT,
                code_system=CodeSystem.HCPCS,
                code="J9271",
            ),
            # Week 20: Radiation
            ClaimEvent(
                patient_key=patient_key,
                event_date=index_date + timedelta(days=140),
                setting=ClaimSetting.OUTPATIENT,
                code_system=CodeSystem.CPT,
                code="77385",
            ),
        ]

        return claims

    def test_build_timeline(
        self,
        code_library: CodeMappingLibrary,
        sample_patient: LinkedPatient,
        sample_claims: list[ClaimEvent],
    ) -> None:
        """Test building a treatment timeline."""
        builder = TreatmentTimelineBuilder(
            code_library=code_library,
            lookback_days=365,
            followup_days=365,
        )

        timeline = builder.build_timeline(sample_patient, sample_claims)

        assert timeline.patient_key == "test_patient_key"
        assert timeline.index_date == date(2015, 3, 15)
        assert len(timeline.treatment_events) > 0

    def test_timeline_treatment_types(
        self,
        code_library: CodeMappingLibrary,
        sample_patient: LinkedPatient,
        sample_claims: list[ClaimEvent],
    ) -> None:
        """Test that treatment types are correctly identified."""
        builder = TreatmentTimelineBuilder(
            code_library=code_library,
            lookback_days=365,
            followup_days=365,
        )

        timeline = builder.build_timeline(sample_patient, sample_claims)
        treatment_types = {e.treatment_type for e in timeline.treatment_events}

        assert "chemo_admin" in treatment_types
        assert "chemo_drug" in treatment_types
        assert "immunotherapy" in treatment_types
        assert "radiation" in treatment_types

    def test_timeline_time_to_first_treatment(
        self,
        code_library: CodeMappingLibrary,
        sample_patient: LinkedPatient,
        sample_claims: list[ClaimEvent],
    ) -> None:
        """Test time to first treatment calculation."""
        builder = TreatmentTimelineBuilder(
            code_library=code_library,
            lookback_days=365,
            followup_days=365,
        )

        timeline = builder.build_timeline(sample_patient, sample_claims)

        # First treatment is at day 14
        assert timeline.time_to_first_treatment == 14

    def test_lines_of_therapy_assignment(
        self,
        code_library: CodeMappingLibrary,
        sample_patient: LinkedPatient,
    ) -> None:
        """Test lines of therapy assignment based on gaps."""
        builder = TreatmentTimelineBuilder(
            code_library=code_library,
            lookback_days=365,
            followup_days=365,
            gap_days_for_new_line=45,  # New line if gap > 45 days
        )

        index_date = date(2015, 3, 15)

        # Create claims with a clear gap
        claims = [
            # Line 1: Day 14
            ClaimEvent(
                patient_key="test_patient_key",
                event_date=index_date + timedelta(days=14),
                setting=ClaimSetting.OUTPATIENT,
                code_system=CodeSystem.HCPCS,
                code="J9045",  # Carboplatin
            ),
            # Line 1: Day 35 (21 day gap - same line)
            ClaimEvent(
                patient_key="test_patient_key",
                event_date=index_date + timedelta(days=35),
                setting=ClaimSetting.OUTPATIENT,
                code_system=CodeSystem.HCPCS,
                code="J9045",
            ),
            # Line 2: Day 100 (65 day gap - new line)
            ClaimEvent(
                patient_key="test_patient_key",
                event_date=index_date + timedelta(days=100),
                setting=ClaimSetting.OUTPATIENT,
                code_system=CodeSystem.HCPCS,
                code="J9271",  # Pembrolizumab
            ),
        ]

        timeline = builder.build_timeline(sample_patient, claims)

        # Check lines assigned
        lines = [e.line_of_therapy for e in timeline.treatment_events]
        assert max(lines) == 2  # Should have 2 lines

    def test_lookback_window(
        self,
        code_library: CodeMappingLibrary,
        sample_patient: LinkedPatient,
    ) -> None:
        """Test that lookback window is respected."""
        builder = TreatmentTimelineBuilder(
            code_library=code_library,
            lookback_days=30,  # Only 30 days lookback
            followup_days=365,
        )

        index_date = date(2015, 3, 15)

        claims = [
            # Outside lookback (should be excluded)
            ClaimEvent(
                patient_key="test_patient_key",
                event_date=index_date - timedelta(days=60),
                setting=ClaimSetting.OUTPATIENT,
                code_system=CodeSystem.CPT,
                code="96413",
            ),
            # Within lookback
            ClaimEvent(
                patient_key="test_patient_key",
                event_date=index_date - timedelta(days=15),
                setting=ClaimSetting.OUTPATIENT,
                code_system=CodeSystem.CPT,
                code="96413",
            ),
            # Post-index
            ClaimEvent(
                patient_key="test_patient_key",
                event_date=index_date + timedelta(days=14),
                setting=ClaimSetting.OUTPATIENT,
                code_system=CodeSystem.CPT,
                code="96413",
            ),
        ]

        timeline = builder.build_timeline(sample_patient, claims)

        # Should have 2 claims in window
        assert len(timeline.claim_events) == 2

    def test_followup_window(
        self,
        code_library: CodeMappingLibrary,
        sample_patient: LinkedPatient,
    ) -> None:
        """Test that followup window is respected."""
        builder = TreatmentTimelineBuilder(
            code_library=code_library,
            lookback_days=365,
            followup_days=30,  # Only 30 days followup
        )

        index_date = date(2015, 3, 15)

        claims = [
            # Within followup
            ClaimEvent(
                patient_key="test_patient_key",
                event_date=index_date + timedelta(days=14),
                setting=ClaimSetting.OUTPATIENT,
                code_system=CodeSystem.CPT,
                code="96413",
            ),
            # Outside followup (should be excluded)
            ClaimEvent(
                patient_key="test_patient_key",
                event_date=index_date + timedelta(days=60),
                setting=ClaimSetting.OUTPATIENT,
                code_system=CodeSystem.CPT,
                code="96413",
            ),
        ]

        timeline = builder.build_timeline(sample_patient, claims)

        # Should have 1 claim in window
        assert len(timeline.claim_events) == 1

    def test_summary_statistics(
        self,
        code_library: CodeMappingLibrary,
        sample_patient: LinkedPatient,
        sample_claims: list[ClaimEvent],
    ) -> None:
        """Test summary statistics generation."""
        builder = TreatmentTimelineBuilder(
            code_library=code_library,
            lookback_days=365,
            followup_days=365,
        )

        # Build timeline
        builder.build_timeline(sample_patient, sample_claims)

        # Get summary
        summary = builder.get_summary()

        assert summary.n_patients == 1
        assert summary.n_with_any_treatment == 1
        assert len(summary.treatment_type_counts) > 0


class TestTimelineEdgeCases:
    """Tests for edge cases in timeline building."""

    def test_empty_claims(self) -> None:
        """Test with no claims."""
        builder = TreatmentTimelineBuilder()

        case = RegistryCase(
            patient_key="test",
            dx_date=date(2015, 1, 1),
            cancer_site="LUNG",
        )
        patient = LinkedPatient(patient_key="test", registry_case=case)

        timeline = builder.build_timeline(patient, [])

        assert len(timeline.treatment_events) == 0
        assert timeline.time_to_first_treatment is None

    def test_patient_without_registry_case(self) -> None:
        """Test with patient missing registry case."""
        builder = TreatmentTimelineBuilder()

        patient = LinkedPatient(patient_key="test", registry_case=None)

        timeline = builder.build_timeline(patient, [])

        assert timeline.patient_key == "test"
        assert timeline.index_date is None

    def test_claims_with_no_matching_codes(self) -> None:
        """Test with claims that don't match any codes."""
        builder = TreatmentTimelineBuilder()

        case = RegistryCase(
            patient_key="test",
            dx_date=date(2015, 1, 1),
            cancer_site="LUNG",
        )
        patient = LinkedPatient(patient_key="test", registry_case=case)

        # Create claims with non-treatment codes
        claims = [
            ClaimEvent(
                patient_key="test",
                event_date=date(2015, 2, 1),
                setting=ClaimSetting.OUTPATIENT,
                code_system=CodeSystem.CPT,
                code="99213",  # Office visit - not a treatment
            ),
        ]

        timeline = builder.build_timeline(patient, claims)

        assert len(timeline.treatment_events) == 0
        assert len(timeline.claim_events) == 1  # Raw claim is still there

    def test_deterministic_ordering(self) -> None:
        """Test that timeline events are deterministically ordered."""
        builder = TreatmentTimelineBuilder()

        case = RegistryCase(
            patient_key="test",
            dx_date=date(2015, 1, 1),
            cancer_site="LUNG",
        )
        patient = LinkedPatient(patient_key="test", registry_case=case)

        # Create claims in random order
        claims = [
            ClaimEvent(
                patient_key="test",
                event_date=date(2015, 3, 1),
                setting=ClaimSetting.OUTPATIENT,
                code_system=CodeSystem.CPT,
                code="96413",
            ),
            ClaimEvent(
                patient_key="test",
                event_date=date(2015, 2, 1),
                setting=ClaimSetting.OUTPATIENT,
                code_system=CodeSystem.CPT,
                code="96413",
            ),
        ]

        timeline = builder.build_timeline(patient, claims)

        # Events should be sorted by date
        dates = [e.event_date for e in timeline.treatment_events]
        assert dates == sorted(dates)
