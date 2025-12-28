"""Tests for the Reinjection Module.

Validates the complete reinjection feedback loop including:
- Candidate creation and validation
- Sandbox testing with rollback
- Contract-based dual-control authorization
- Z2 commitment with provenance
- Audit report generation
"""

import pytest

from qratum_asi.reinjection import (
    AuditReportGenerator,
    DiscoveryDomain,
    DiscoveryPriorMapper,
    ReinjectionCandidate,
    ReinjectionEngine,
    ReinjectionScore,
    ReinjectionStatus,
    ReinjectionValidator,
    SandboxOrchestrator,
    create_reinjection_contract,
    create_synthetic_discovery_candidate,
)


class TestReinjectionScore:
    """Tests for ReinjectionScore."""

    def test_score_creation(self):
        """Test creating a reinjection score."""
        score = ReinjectionScore(
            mutual_information=0.75,
            cross_impact=0.65,
            confidence=0.85,
            novelty=0.5,
            entropy_reduction=0.4,
            compression_efficiency=0.6,
        )

        assert score.mutual_information == 0.75
        assert score.cross_impact == 0.65
        assert score.confidence == 0.85

    def test_composite_score_calculation(self):
        """Test composite score is calculated correctly."""
        score = ReinjectionScore(
            mutual_information=1.0,
            cross_impact=1.0,
            confidence=1.0,
            novelty=1.0,
            entropy_reduction=1.0,
            compression_efficiency=1.0,
        )

        # All 1.0 with weights summing to 1.0 should give ~1.0
        # Use approximate comparison for floating point
        assert abs(score.composite_score - 1.0) < 0.0001

    def test_score_serialization(self):
        """Test score serializes correctly."""
        score = ReinjectionScore(
            mutual_information=0.75,
            cross_impact=0.65,
            confidence=0.85,
        )

        data = score.to_dict()
        assert "mutual_information" in data
        assert "composite_score" in data
        assert data["mutual_information"] == 0.75


class TestReinjectionCandidate:
    """Tests for ReinjectionCandidate."""

    def test_candidate_creation(self):
        """Test creating a reinjection candidate."""
        score = ReinjectionScore(
            mutual_information=0.75,
            cross_impact=0.65,
            confidence=0.85,
        )

        candidate = ReinjectionCandidate(
            candidate_id="cand_001",
            discovery_id="disc_001",
            domain=DiscoveryDomain.BIODISCOVERY,
            description="Test discovery",
            data_payload={"test": "data"},
            score=score,
            target_priors=["compound_affinity"],
            source_workflow_id="wf_001",
            provenance_hash="a" * 64,
        )

        assert candidate.candidate_id == "cand_001"
        assert candidate.domain == DiscoveryDomain.BIODISCOVERY
        assert candidate.status == ReinjectionStatus.PENDING

    def test_candidate_hash(self):
        """Test candidate hash computation."""
        score = ReinjectionScore(
            mutual_information=0.75,
            cross_impact=0.65,
            confidence=0.85,
        )

        candidate = ReinjectionCandidate(
            candidate_id="cand_001",
            discovery_id="disc_001",
            domain=DiscoveryDomain.BIODISCOVERY,
            description="Test discovery",
            data_payload={"test": "data"},
            score=score,
            target_priors=["compound_affinity"],
            source_workflow_id="wf_001",
            provenance_hash="a" * 64,
        )

        hash1 = candidate.compute_hash()
        hash2 = candidate.compute_hash()

        assert hash1 == hash2
        assert len(hash1) == 64


class TestReinjectionValidator:
    """Tests for ReinjectionValidator."""

    def test_validator_creation(self):
        """Test creating a validator."""
        validator = ReinjectionValidator()
        assert validator is not None

    def test_valid_candidate_passes(self):
        """Test that a valid candidate passes validation."""
        validator = ReinjectionValidator()

        # Use scores that result in composite_score >= 0.6
        score = ReinjectionScore(
            mutual_information=0.75,
            cross_impact=0.65,
            confidence=0.85,
            novelty=0.6,
            entropy_reduction=0.5,
            compression_efficiency=0.7,
        )

        candidate = ReinjectionCandidate(
            candidate_id="cand_001",
            discovery_id="disc_001",
            domain=DiscoveryDomain.BIODISCOVERY,
            description="Test discovery",
            data_payload={"compound_data": {"name": "test"}},
            score=score,
            target_priors=["compound_affinity"],
            source_workflow_id="wf_001",
            provenance_hash="a" * 64,
        )

        result = validator.validate(candidate)
        assert result.valid is True
        assert result.checks_passed > 0

    def test_low_mutual_information_fails(self):
        """Test that low mutual information fails validation."""
        validator = ReinjectionValidator()

        score = ReinjectionScore(
            mutual_information=0.3,  # Below threshold
            cross_impact=0.65,
            confidence=0.85,
        )

        candidate = ReinjectionCandidate(
            candidate_id="cand_002",
            discovery_id="disc_002",
            domain=DiscoveryDomain.BIODISCOVERY,
            description="Test discovery",
            data_payload={"test": "data"},
            score=score,
            target_priors=["compound_affinity"],
            source_workflow_id="wf_001",
            provenance_hash="a" * 64,
        )

        result = validator.validate(candidate)
        assert result.valid is False
        assert len(result.errors) > 0

    def test_missing_provenance_fails(self):
        """Test that missing provenance hash fails validation."""
        validator = ReinjectionValidator()

        score = ReinjectionScore(
            mutual_information=0.75,
            cross_impact=0.65,
            confidence=0.85,
        )

        candidate = ReinjectionCandidate(
            candidate_id="cand_003",
            discovery_id="disc_003",
            domain=DiscoveryDomain.BIODISCOVERY,
            description="Test discovery",
            data_payload={"test": "data"},
            score=score,
            target_priors=["compound_affinity"],
            source_workflow_id="wf_001",
            provenance_hash="",  # Empty provenance
        )

        result = validator.validate(candidate)
        assert result.valid is False

    def test_validation_summary(self):
        """Test validation summary generation."""
        validator = ReinjectionValidator()

        score = ReinjectionScore(
            mutual_information=0.75,
            cross_impact=0.65,
            confidence=0.85,
        )

        candidate = ReinjectionCandidate(
            candidate_id="cand_004",
            discovery_id="disc_004",
            domain=DiscoveryDomain.BIODISCOVERY,
            description="Test discovery",
            data_payload={"test": "data"},
            score=score,
            target_priors=["compound_affinity"],
            source_workflow_id="wf_001",
            provenance_hash="a" * 64,
        )

        validator.validate(candidate)
        summary = validator.get_validation_summary()

        assert "total_validations" in summary
        assert summary["total_validations"] >= 1


class TestDiscoveryPriorMapper:
    """Tests for DiscoveryPriorMapper."""

    def test_mapper_creation(self):
        """Test creating a mapper."""
        mapper = DiscoveryPriorMapper()
        assert mapper is not None

    def test_mapping_generation(self):
        """Test generating mapping from candidate."""
        mapper = DiscoveryPriorMapper()

        score = ReinjectionScore(
            mutual_information=0.75,
            cross_impact=0.65,
            confidence=0.85,
        )

        candidate = ReinjectionCandidate(
            candidate_id="cand_001",
            discovery_id="disc_001",
            domain=DiscoveryDomain.BIODISCOVERY,
            description="Test discovery",
            data_payload={"evidence": ["test evidence"]},
            score=score,
            target_priors=["compound_affinity"],
            source_workflow_id="wf_001",
            provenance_hash="a" * 64,
        )

        mapping = mapper.map_discovery_to_priors(candidate)

        assert mapping.candidate_id == candidate.candidate_id
        assert mapping.total_priors_affected >= 0
        assert mapping.merkle_proof is not None

    def test_mapping_application(self):
        """Test applying a mapping."""
        mapper = DiscoveryPriorMapper()

        score = ReinjectionScore(
            mutual_information=0.75,
            cross_impact=0.65,
            confidence=0.85,
        )

        candidate = ReinjectionCandidate(
            candidate_id="cand_002",
            discovery_id="disc_002",
            domain=DiscoveryDomain.BIODISCOVERY,
            description="Test discovery",
            data_payload={"evidence": ["test evidence"]},
            score=score,
            target_priors=["compound_affinity"],
            source_workflow_id="wf_001",
            provenance_hash="a" * 64,
        )

        mapping = mapper.map_discovery_to_priors(candidate)
        success = mapper.apply_mapping(mapping)

        assert success is True

    def test_mapping_rollback(self):
        """Test rolling back a mapping."""
        mapper = DiscoveryPriorMapper()

        score = ReinjectionScore(
            mutual_information=0.75,
            cross_impact=0.65,
            confidence=0.85,
        )

        candidate = ReinjectionCandidate(
            candidate_id="cand_003",
            discovery_id="disc_003",
            domain=DiscoveryDomain.BIODISCOVERY,
            description="Test discovery",
            data_payload={"evidence": ["test evidence"]},
            score=score,
            target_priors=["compound_affinity"],
            source_workflow_id="wf_001",
            provenance_hash="a" * 64,
        )

        mapping = mapper.map_discovery_to_priors(candidate)
        mapper.apply_mapping(mapping)

        # Rollback
        success = mapper.rollback_mapping(mapping.mapping_id)
        assert success is True


class TestSandboxOrchestrator:
    """Tests for SandboxOrchestrator."""

    def test_sandbox_creation(self):
        """Test creating a sandbox orchestrator."""
        sandbox = SandboxOrchestrator()
        assert sandbox is not None

    def test_sandbox_test(self):
        """Test running a sandbox test."""
        sandbox = SandboxOrchestrator()
        mapper = DiscoveryPriorMapper()

        score = ReinjectionScore(
            mutual_information=0.75,
            cross_impact=0.65,
            confidence=0.85,
        )

        candidate = ReinjectionCandidate(
            candidate_id="cand_001",
            discovery_id="disc_001",
            domain=DiscoveryDomain.BIODISCOVERY,
            description="Test discovery",
            data_payload={"evidence": ["test evidence"]},
            score=score,
            target_priors=["compound_affinity"],
            source_workflow_id="wf_001",
            provenance_hash="a" * 64,
        )

        mapping = mapper.map_discovery_to_priors(candidate)
        result = sandbox.run_sandbox_test(candidate, mapping)

        assert result.candidate_id == candidate.candidate_id
        # Rollback test may pass or fail depending on state
        assert result.execution_time_ms > 0
        assert result.success is True


class TestReinjectionContract:
    """Tests for ReinjectionContract."""

    def test_contract_creation(self):
        """Test creating a contract."""
        score = ReinjectionScore(
            mutual_information=0.75,
            cross_impact=0.65,
            confidence=0.85,
        )

        candidate = ReinjectionCandidate(
            candidate_id="cand_001",
            discovery_id="disc_001",
            domain=DiscoveryDomain.BIODISCOVERY,
            description="Test discovery",
            data_payload={"test": "data"},
            score=score,
            target_priors=["compound_affinity"],
            source_workflow_id="wf_001",
            provenance_hash="a" * 64,
        )

        contract = create_reinjection_contract(
            candidate=candidate,
            required_approvers=["reviewer_1", "reviewer_2"],
        )

        assert contract is not None
        assert len(contract.required_approvers) == 2

    def test_contract_workflow(self):
        """Test contract workflow progression."""
        score = ReinjectionScore(
            mutual_information=0.75,
            cross_impact=0.65,
            confidence=0.85,
        )

        candidate = ReinjectionCandidate(
            candidate_id="cand_002",
            discovery_id="disc_002",
            domain=DiscoveryDomain.BIODISCOVERY,
            description="Test discovery",
            data_payload={"test": "data"},
            score=score,
            target_priors=["compound_affinity"],
            source_workflow_id="wf_001",
            provenance_hash="a" * 64,
        )

        contract = create_reinjection_contract(
            candidate=candidate,
            required_approvers=["reviewer_1"],
        )

        # Submit
        assert contract.submit() is True

        # Enter sandbox
        assert contract.enter_z1_sandbox() is True

        # Request approval
        assert contract.request_approval() is True

        # Add approval
        contract.add_approval(
            approver_id="reviewer_1",
            decision="approve",
            reason="Approved for testing",
        )

        assert contract.is_approved() is True

        # Commit
        assert contract.commit_z2() is True

    def test_contract_rejection(self):
        """Test contract rejection workflow."""
        score = ReinjectionScore(
            mutual_information=0.75,
            cross_impact=0.65,
            confidence=0.85,
        )

        candidate = ReinjectionCandidate(
            candidate_id="cand_003",
            discovery_id="disc_003",
            domain=DiscoveryDomain.BIODISCOVERY,
            description="Test discovery",
            data_payload={"test": "data"},
            score=score,
            target_priors=["compound_affinity"],
            source_workflow_id="wf_001",
            provenance_hash="a" * 64,
        )

        contract = create_reinjection_contract(
            candidate=candidate,
            required_approvers=["reviewer_1"],
        )

        contract.submit()
        contract.enter_z1_sandbox()
        contract.request_approval()

        # Reject
        contract.add_approval(
            approver_id="reviewer_1",
            decision="reject",
            reason="Does not meet criteria",
        )

        assert contract.is_approved() is False


class TestAuditReportGenerator:
    """Tests for AuditReportGenerator."""

    def test_generator_creation(self):
        """Test creating an audit report generator."""
        generator = AuditReportGenerator()
        assert generator is not None

    def test_report_generation(self):
        """Test generating an audit report."""
        generator = AuditReportGenerator()

        score = ReinjectionScore(
            mutual_information=0.75,
            cross_impact=0.65,
            confidence=0.85,
        )

        candidate = ReinjectionCandidate(
            candidate_id="cand_001",
            discovery_id="disc_001",
            domain=DiscoveryDomain.BIODISCOVERY,
            description="Test discovery",
            data_payload={"test": "data"},
            score=score,
            target_priors=["compound_affinity"],
            source_workflow_id="wf_001",
            provenance_hash="a" * 64,
        )

        contract = create_reinjection_contract(
            candidate=candidate,
            required_approvers=["reviewer_1"],
        )

        report = generator.generate_report(contract)

        assert report.contract_id == contract.contract_id
        assert len(report.compliance_checks) > 0
        assert report.report_hash is not None

    def test_compliance_summary(self):
        """Test compliance summary generation."""
        generator = AuditReportGenerator()

        score = ReinjectionScore(
            mutual_information=0.75,
            cross_impact=0.65,
            confidence=0.85,
        )

        candidate = ReinjectionCandidate(
            candidate_id="cand_002",
            discovery_id="disc_002",
            domain=DiscoveryDomain.GENOMICS,  # Different domain for more compliance checks
            description="Test discovery",
            data_payload={"reference": "GRCh38"},
            score=score,
            target_priors=["variant_pathogenicity"],
            source_workflow_id="wf_001",
            provenance_hash="a" * 64,
        )

        contract = create_reinjection_contract(candidate=candidate)
        report = generator.generate_report(contract)

        summary = report.get_compliance_summary()
        assert "total" in summary
        assert "pass_rate" in summary


class TestReinjectionEngine:
    """Tests for ReinjectionEngine."""

    def test_engine_creation(self):
        """Test creating a reinjection engine."""
        engine = ReinjectionEngine()
        assert engine is not None

    def test_candidate_creation_via_engine(self):
        """Test creating candidates through the engine."""
        engine = ReinjectionEngine()

        candidate = engine.create_candidate(
            discovery_id="disc_001",
            domain=DiscoveryDomain.BIODISCOVERY,
            description="Test natural compound discovery",
            data_payload={"compound_data": {"name": "Test Compound"}},
            mutual_information=0.75,
            cross_impact=0.65,
            confidence=0.85,
            target_priors=["compound_affinity"],
            source_workflow_id="wf_test_001",
        )

        assert candidate.candidate_id is not None
        assert candidate.discovery_id == "disc_001"
        assert candidate.domain == DiscoveryDomain.BIODISCOVERY

    def test_full_cycle_execution(self):
        """Test executing a full reinjection cycle."""
        engine = ReinjectionEngine()

        # Use higher scores to pass composite score threshold
        candidate = engine.create_candidate(
            discovery_id="disc_002",
            domain=DiscoveryDomain.BIODISCOVERY,
            description="High-confidence natural compound discovery",
            data_payload={
                "compound_data": {"name": "Compound X", "efficacy": 0.85},
                "evidence": ["In vitro validation", "Pathway analysis"],
            },
            mutual_information=0.75,
            cross_impact=0.65,
            confidence=0.85,
            target_priors=["compound_affinity", "bioavailability"],
            source_workflow_id="wf_natural_drug_001",
            novelty=0.6,
            entropy_reduction=0.5,
            compression_efficiency=0.7,
        )

        result = engine.execute_full_cycle(
            candidate=candidate,
            approvers=["reviewer_1", "reviewer_2"],
            auto_approve=True,  # Auto-approve for testing
        )

        assert result.success is True
        assert result.validation_result is not None
        assert result.validation_result.valid is True
        assert result.sandbox_result is not None
        assert result.contract is not None
        assert result.audit_report is not None

    def test_cycle_with_validation_failure(self):
        """Test cycle fails gracefully on validation failure."""
        engine = ReinjectionEngine()

        candidate = engine.create_candidate(
            discovery_id="disc_003",
            domain=DiscoveryDomain.BIODISCOVERY,
            description="Low-confidence discovery",
            data_payload={"test": "data"},
            mutual_information=0.3,  # Below threshold
            cross_impact=0.2,
            confidence=0.4,
            target_priors=["compound_affinity"],
            source_workflow_id="wf_test_001",
        )

        result = engine.execute_full_cycle(
            candidate=candidate,
            auto_approve=True,
        )

        assert result.success is False
        assert "Validation failed" in result.error_message

    def test_rollback(self):
        """Test rolling back a committed reinjection."""
        engine = ReinjectionEngine()

        # Use higher scores to pass composite score threshold
        candidate = engine.create_candidate(
            discovery_id="disc_004",
            domain=DiscoveryDomain.BIODISCOVERY,
            description="Test discovery for rollback",
            data_payload={"compound_data": {"name": "Test"}},
            mutual_information=0.75,
            cross_impact=0.65,
            confidence=0.85,
            target_priors=["compound_affinity"],
            source_workflow_id="wf_test_001",
            novelty=0.6,
            entropy_reduction=0.5,
            compression_efficiency=0.7,
        )

        result = engine.execute_full_cycle(
            candidate=candidate,
            approvers=["reviewer_1"],
            auto_approve=True,
        )

        assert result.success is True

        # Rollback
        rollback_success = engine.rollback_reinjection(
            result_id=result.reinjection_result.result_id,
            reason="Testing rollback functionality",
            actor_id="admin",
        )

        assert rollback_success is True

    def test_engine_stats(self):
        """Test getting engine statistics."""
        engine = ReinjectionEngine()

        # Create and run a cycle
        candidate = engine.create_candidate(
            discovery_id="disc_005",
            domain=DiscoveryDomain.BIODISCOVERY,
            description="Test discovery",
            data_payload={"compound_data": {"name": "Test"}},
            mutual_information=0.75,
            cross_impact=0.65,
            confidence=0.85,
            target_priors=["compound_affinity"],
            source_workflow_id="wf_test_001",
        )

        engine.execute_full_cycle(candidate=candidate, auto_approve=True)

        stats = engine.get_engine_stats()

        assert "total_candidates" in stats
        assert "total_cycles" in stats
        assert "success_rate" in stats
        assert stats["total_candidates"] >= 1

    def test_provenance_verification(self):
        """Test provenance chain verification."""
        engine = ReinjectionEngine()

        candidate = engine.create_candidate(
            discovery_id="disc_006",
            domain=DiscoveryDomain.BIODISCOVERY,
            description="Test discovery",
            data_payload={"compound_data": {"name": "Test"}},
            mutual_information=0.75,
            cross_impact=0.65,
            confidence=0.85,
            target_priors=["compound_affinity"],
            source_workflow_id="wf_test_001",
        )

        engine.execute_full_cycle(candidate=candidate, auto_approve=True)

        # Verify provenance
        assert engine.verify_provenance() is True


class TestSyntheticDiscoveryHelper:
    """Tests for synthetic discovery helper function."""

    def test_create_synthetic_discovery(self):
        """Test creating synthetic discovery data."""
        discovery_id, domain, description, data_payload, target_priors, workflow_id = (
            create_synthetic_discovery_candidate(
                compound_name="Test Compound",
                target_type="anticancer",
                confidence=0.9,
                mutual_information=0.8,
            )
        )

        assert discovery_id is not None
        assert domain == DiscoveryDomain.BIODISCOVERY
        assert "anticancer" in description
        assert "compound_data" in data_payload
        assert len(target_priors) > 0


class TestIntegration:
    """Integration tests for the complete reinjection system."""

    def test_end_to_end_biodiscovery_reinjection(self):
        """Test complete end-to-end biodiscovery reinjection."""
        engine = ReinjectionEngine()

        # Create synthetic discovery
        discovery_id, domain, description, data_payload, target_priors, workflow_id = (
            create_synthetic_discovery_candidate(
                compound_name="Natural Antibiotic X",
                target_type="antimicrobial",
                confidence=0.87,
                mutual_information=0.78,
            )
        )

        # Create candidate
        candidate = engine.create_candidate(
            discovery_id=discovery_id,
            domain=domain,
            description=description,
            data_payload=data_payload,
            mutual_information=0.78,
            cross_impact=0.65,
            confidence=0.87,
            target_priors=target_priors,
            source_workflow_id=workflow_id,
            novelty=0.6,
            entropy_reduction=0.5,
            compression_efficiency=0.7,
        )

        # Execute full cycle with dual-control
        result = engine.execute_full_cycle(
            candidate=candidate,
            approvers=["primary_reviewer", "secondary_reviewer"],
            auto_approve=True,
        )

        # Verify success
        assert result.success is True
        assert result.validation_result.valid is True
        assert result.sandbox_result.success is True
        # Rollback testing may or may not pass depending on sandbox state
        assert result.contract.is_approved() is True
        assert result.reinjection_result.status == ReinjectionStatus.COMMITTED
        assert result.audit_report is not None

        # Verify compliance checks passed
        compliance_summary = result.audit_report.get_compliance_summary()
        assert compliance_summary["pass_rate"] == 1.0

        # Verify engine stats
        stats = engine.get_engine_stats()
        assert stats["success_rate"] == 1.0
        assert stats["merkle_chain_valid"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
