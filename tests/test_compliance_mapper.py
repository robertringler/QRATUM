"""Tests for Compliance Mapper.

Tests compliance mapping and artifact generation.
"""

import pytest

from qratum_asi.discovery_acceleration.compliance_mapper import ComplianceMapper
from qratum_asi.discovery_acceleration.types import DiscoveryType


class TestComplianceMapper:
    """Tests for ComplianceMapper."""

    def test_mapper_initialization(self):
        """Test mapper initializes correctly."""
        mapper = ComplianceMapper()

        assert len(mapper.artifacts) == 0
        assert mapper._artifact_counter == 0
        assert mapper.merkle_chain.verify_integrity()

    def test_get_compliance_mapping_all_types(self):
        """Test compliance mapping for all discovery types."""
        mapper = ComplianceMapper()

        for dt in DiscoveryType:
            mapping = mapper.get_compliance_mapping(dt)

            assert mapping.discovery_type == dt
            assert len(mapping.frameworks) >= 3  # At least GDPR, HIPAA, ISO 27001
            assert "gdpr" in mapping.frameworks
            assert "hipaa" in mapping.frameworks
            assert "iso_27001" in mapping.frameworks
            assert mapping.status in ["compliant", "non_compliant"]
            assert len(mapping.controls) > 0
            assert len(mapping.audit_requirements) > 0

    def test_common_frameworks_present(self):
        """Test common frameworks present in all mappings."""
        mapper = ComplianceMapper()

        for dt in DiscoveryType:
            mapping = mapper.get_compliance_mapping(dt)

            # Common frameworks
            assert "gdpr" in mapping.frameworks
            assert mapping.frameworks["gdpr"]["applicable"] is True
            assert mapping.frameworks["gdpr"]["status"] == "compliant"

            assert "hipaa" in mapping.frameworks
            assert "iso_27001" in mapping.frameworks

    def test_discovery_specific_frameworks(self):
        """Test discovery-specific frameworks."""
        mapper = ComplianceMapper()

        # COMPLEX_DISEASE_GENETICS should have GINA and Common Rule
        mapping = mapper.get_compliance_mapping(DiscoveryType.COMPLEX_DISEASE_GENETICS)
        assert "gina" in mapping.frameworks
        assert "common_rule" in mapping.frameworks

        # PERSONALIZED_DRUG_DESIGN should have FDA 21 CFR Part 11
        mapping = mapper.get_compliance_mapping(DiscoveryType.PERSONALIZED_DRUG_DESIGN)
        assert "fda_21_cfr_part_11" in mapping.frameworks

        # NATURAL_DRUG_DISCOVERY should have Nagoya Protocol
        mapping = mapper.get_compliance_mapping(DiscoveryType.NATURAL_DRUG_DISCOVERY)
        assert "nagoya_protocol" in mapping.frameworks

        # ANTI_AGING_PATHWAYS should have ethics review
        mapping = mapper.get_compliance_mapping(DiscoveryType.ANTI_AGING_PATHWAYS)
        assert "ethics_review" in mapping.frameworks

    def test_generate_compliance_artifact(self):
        """Test generating compliance artifact."""
        mapper = ComplianceMapper()

        contract_id = "dc_test_001"
        dt = DiscoveryType.PERSONALIZED_DRUG_DESIGN

        artifact = mapper.generate_compliance_artifact(contract_id, dt)

        assert artifact.artifact_id.startswith("ca_personalized_drug_design_")
        assert artifact.contract_id == contract_id
        assert artifact.framework == "multi_framework"
        assert "contract_id" in artifact.evidence
        assert "discovery_type" in artifact.evidence
        assert artifact.evidence["discovery_type"] == dt.value
        assert len(artifact.merkle_root) == 64  # SHA256 hex

        # Should be stored
        assert artifact.artifact_id in mapper.artifacts

    def test_generate_multiple_artifacts(self):
        """Test generating multiple artifacts increments counter."""
        mapper = ComplianceMapper()

        artifact1 = mapper.generate_compliance_artifact(
            "dc_001", DiscoveryType.COMPLEX_DISEASE_GENETICS
        )
        artifact2 = mapper.generate_compliance_artifact(
            "dc_002", DiscoveryType.PERSONALIZED_DRUG_DESIGN
        )

        assert artifact1.artifact_id != artifact2.artifact_id
        assert len(mapper.artifacts) == 2
        assert mapper._artifact_counter == 2

    def test_validate_compliance_compliant(self):
        """Test compliance validation for compliant workflow."""
        mapper = ComplianceMapper()

        workflow_id = "wf_compliant_001"
        dt = DiscoveryType.COMPLEX_DISEASE_GENETICS

        result = mapper.validate_compliance(workflow_id, dt)

        assert result.workflow_id == workflow_id
        assert result.is_compliant is True  # All frameworks compliant by default
        assert len(result.validated_frameworks) > 0
        assert len(result.violations) == 0
        assert len(result.recommendations) > 0

    def test_validate_compliance_recommendations(self):
        """Test compliance validation includes recommendations."""
        mapper = ComplianceMapper()

        result = mapper.validate_compliance("wf_test_001", DiscoveryType.NATURAL_DRUG_DISCOVERY)

        # Should have recommendations even if compliant
        assert len(result.recommendations) > 0
        # Should mention continuous monitoring
        assert any(
            "monitoring" in rec.lower() or "review" in rec.lower() for rec in result.recommendations
        )

    def test_get_framework_details_common(self):
        """Test getting details for common frameworks."""
        mapper = ComplianceMapper()

        gdpr = mapper.get_framework_details("gdpr")
        assert gdpr is not None
        assert gdpr["name"] == "General Data Protection Regulation"
        assert gdpr["applicable"] is True
        assert "controls" in gdpr
        assert "audit_requirements" in gdpr

        hipaa = mapper.get_framework_details("hipaa")
        assert hipaa is not None
        assert "phi_encryption" in hipaa["controls"]

    def test_get_framework_details_specific(self):
        """Test getting details for discovery-specific frameworks."""
        mapper = ComplianceMapper()

        nagoya = mapper.get_framework_details("nagoya_protocol")
        assert nagoya is not None
        assert "Nagoya Protocol" in nagoya["name"]
        assert "benefit_sharing" in nagoya["controls"]

        fda = mapper.get_framework_details("fda_21_cfr_part_11")
        assert fda is not None
        assert "FDA 21 CFR Part 11" in fda["name"]

    def test_get_framework_details_nonexistent(self):
        """Test getting details for nonexistent framework returns None."""
        mapper = ComplianceMapper()

        result = mapper.get_framework_details("nonexistent_framework")
        assert result is None

    def test_get_all_frameworks(self):
        """Test getting all available frameworks."""
        mapper = ComplianceMapper()

        all_frameworks = mapper.get_all_frameworks()

        # Should include common frameworks
        assert "gdpr" in all_frameworks
        assert "hipaa" in all_frameworks
        assert "iso_27001" in all_frameworks

        # Should include discovery-specific frameworks
        assert "nagoya_protocol" in all_frameworks
        assert "fda_21_cfr_part_11" in all_frameworks
        assert "gina" in all_frameworks
        assert "common_rule" in all_frameworks

    def test_export_compliance_report(self):
        """Test exporting compliance report."""
        mapper = ComplianceMapper()

        dt = DiscoveryType.CLIMATE_GENE_CONNECTIONS

        report = mapper.export_compliance_report(dt)

        assert report["report_id"] == f"compliance_report_{dt.value}"
        assert report["discovery_type"] == dt.value
        assert report["overall_status"] == "compliant"
        assert "frameworks" in report
        assert "controls" in report
        assert "audit_requirements" in report
        assert len(report["merkle_root"]) == 64
        assert report["provenance_valid"] is True

    def test_compliance_mapping_serialization(self):
        """Test compliance mapping can be serialized."""
        mapper = ComplianceMapper()

        mapping = mapper.get_compliance_mapping(DiscoveryType.ECONOMIC_BIOLOGICAL_MODEL)
        mapping_dict = mapping.to_dict()

        assert mapping_dict["discovery_type"] == "economic_biological_model"
        assert "frameworks" in mapping_dict
        assert "controls" in mapping_dict
        assert "audit_requirements" in mapping_dict
        assert mapping_dict["status"] == mapping.status

    def test_compliance_artifact_serialization(self):
        """Test compliance artifact can be serialized."""
        mapper = ComplianceMapper()

        artifact = mapper.generate_compliance_artifact(
            "dc_test_serialize", DiscoveryType.PERSONALIZED_DRUG_DESIGN
        )
        artifact_dict = artifact.to_dict()

        assert artifact_dict["artifact_id"] == artifact.artifact_id
        assert artifact_dict["contract_id"] == "dc_test_serialize"
        assert artifact_dict["framework"] == "multi_framework"
        assert "evidence" in artifact_dict
        assert "timestamp" in artifact_dict
        assert "merkle_root" in artifact_dict

    def test_compliance_validation_result_serialization(self):
        """Test compliance validation result can be serialized."""
        mapper = ComplianceMapper()

        result = mapper.validate_compliance("wf_test_serialize", DiscoveryType.ANTI_AGING_PATHWAYS)
        result_dict = result.to_dict()

        assert result_dict["workflow_id"] == "wf_test_serialize"
        assert result_dict["is_compliant"] == result.is_compliant
        assert "validated_frameworks" in result_dict
        assert "violations" in result_dict
        assert "recommendations" in result_dict

    def test_merkle_chain_integrity(self):
        """Test merkle chain maintains integrity."""
        mapper = ComplianceMapper()

        # Perform multiple operations
        mapper.generate_compliance_artifact("dc_001", DiscoveryType.COMPLEX_DISEASE_GENETICS)
        mapper.validate_compliance("wf_001", DiscoveryType.PERSONALIZED_DRUG_DESIGN)
        mapper.generate_compliance_artifact("dc_002", DiscoveryType.NATURAL_DRUG_DISCOVERY)

        # Chain should still be valid
        assert mapper.merkle_chain.verify_integrity()
        assert len(mapper.merkle_chain.chain) > 1

    def test_framework_controls_deduplication(self):
        """Test that controls are deduplicated in mapping."""
        mapper = ComplianceMapper()

        mapping = mapper.get_compliance_mapping(DiscoveryType.COMPLEX_DISEASE_GENETICS)

        # Controls should be deduplicated (no duplicates)
        assert len(mapping.controls) == len(set(mapping.controls))

    def test_framework_audit_requirements_deduplication(self):
        """Test that audit requirements are deduplicated in mapping."""
        mapper = ComplianceMapper()

        mapping = mapper.get_compliance_mapping(DiscoveryType.PERSONALIZED_DRUG_DESIGN)

        # Audit requirements should be deduplicated
        assert len(mapping.audit_requirements) == len(set(mapping.audit_requirements))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
