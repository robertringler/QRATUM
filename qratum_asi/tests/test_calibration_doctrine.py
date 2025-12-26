"""Tests for the 12 Calibration Doctrine module."""

from datetime import datetime

from qratum_asi.core.calibration_doctrine import (
    AXIOM_1_JURISDICTION,
    AXIOM_4_DEFENSIVE,
    AXIOM_5_ASI,
    CALIBRATION_DOCTRINE,
    CalibrationCategory,
    CalibrationDoctrineEnforcer,
    JurisdictionalClaim,
    JurisdictionalProperty,
    TrajectoryMetrics,
    TrajectoryState,
    get_doctrine_enforcer,
)


class TestCalibrationDoctrine:
    """Tests for the 12 Calibration Doctrine."""

    def test_doctrine_has_12_axioms(self):
        """Verify doctrine contains exactly 12 axioms."""
        assert len(CALIBRATION_DOCTRINE) == 12

    def test_axiom_ids_sequential(self):
        """Verify axiom IDs are sequential 1-12."""
        ids = [axiom.axiom_id for axiom in CALIBRATION_DOCTRINE]
        assert ids == list(range(1, 13))

    def test_all_axioms_immutable(self):
        """Verify all axioms are marked immutable."""
        for axiom in CALIBRATION_DOCTRINE:
            assert axiom.is_immutable is True

    def test_axiom_integrity_verification(self):
        """Verify each axiom can verify its own integrity."""
        for axiom in CALIBRATION_DOCTRINE:
            assert axiom.verify_integrity() is True

    def test_axiom_1_has_all_properties(self):
        """Verify Axiom 1 (Jurisdiction) has all 6 jurisdictional properties."""
        assert len(AXIOM_1_JURISDICTION.properties) == 6
        assert JurisdictionalProperty.DETERMINISM in AXIOM_1_JURISDICTION.properties
        assert JurisdictionalProperty.AUDITABILITY in AXIOM_1_JURISDICTION.properties
        assert JurisdictionalProperty.REVERSIBILITY in AXIOM_1_JURISDICTION.properties
        assert JurisdictionalProperty.SOVEREIGNTY in AXIOM_1_JURISDICTION.properties
        assert JurisdictionalProperty.PRIVACY in AXIOM_1_JURISDICTION.properties
        assert JurisdictionalProperty.OVERSIGHT in AXIOM_1_JURISDICTION.properties

    def test_axiom_4_defensive(self):
        """Verify Axiom 4 (Defensive Engine) has correct properties."""
        assert AXIOM_4_DEFENSIVE.category == CalibrationCategory.OPERATIONAL
        assert JurisdictionalProperty.OVERSIGHT in AXIOM_4_DEFENSIVE.properties
        assert JurisdictionalProperty.SOVEREIGNTY in AXIOM_4_DEFENSIVE.properties

    def test_axiom_5_asi(self):
        """Verify Axiom 5 (ASI Scaffolding) has correct properties."""
        assert AXIOM_5_ASI.category == CalibrationCategory.FOUNDATIONAL
        assert JurisdictionalProperty.OVERSIGHT in AXIOM_5_ASI.properties
        assert JurisdictionalProperty.DETERMINISM in AXIOM_5_ASI.properties


class TestCalibrationDoctrineEnforcer:
    """Tests for the CalibrationDoctrineEnforcer class."""

    def test_enforcer_singleton(self):
        """Verify get_doctrine_enforcer returns same instance."""
        enforcer1 = get_doctrine_enforcer()
        enforcer2 = get_doctrine_enforcer()
        assert enforcer1 is enforcer2

    def test_verify_doctrine_integrity(self):
        """Test doctrine integrity verification."""
        enforcer = CalibrationDoctrineEnforcer()
        result = enforcer.verify_doctrine_integrity()

        assert result["verified"] is True
        assert result["axiom_count"] == 12
        assert result["failed_axioms"] == []

    def test_validate_operation_compliance_success(self):
        """Test operation compliance validation for valid operations."""
        enforcer = CalibrationDoctrineEnforcer()

        compliant, violations = enforcer.validate_operation_compliance(
            "test_operation",
            [JurisdictionalProperty.DETERMINISM, JurisdictionalProperty.AUDITABILITY],
        )

        assert compliant is True
        assert violations == []

    def test_validate_genomic_operation(self):
        """Test genomic operation requires reversibility and privacy."""
        enforcer = CalibrationDoctrineEnforcer()

        # Without required properties
        compliant, violations = enforcer.validate_operation_compliance(
            "genomic_analysis", [JurisdictionalProperty.DETERMINISM]
        )

        assert compliant is False
        assert any("reversibility" in v.lower() for v in violations)
        assert any("privacy" in v.lower() for v in violations)

    def test_validate_asi_operation(self):
        """Test ASI operation requires oversight and determinism."""
        enforcer = CalibrationDoctrineEnforcer()

        # Without required properties
        compliant, violations = enforcer.validate_operation_compliance(
            "asi_self_improvement", [JurisdictionalProperty.SOVEREIGNTY]
        )

        assert compliant is False
        assert any("oversight" in v.lower() for v in violations)
        assert any("determinism" in v.lower() for v in violations)

    def test_trajectory_recording(self):
        """Test trajectory metrics recording."""
        enforcer = CalibrationDoctrineEnforcer()

        metrics = TrajectoryMetrics(
            entropy_gradient=0.1,
            coupling_drift=0.05,
            metastable_clusters=0,
            collapse_precursors=0,
            resilience_compression=0.9,
            trajectory_state=TrajectoryState.STABLE,
            timestamp=datetime.utcnow().isoformat(),
        )

        enforcer.record_trajectory(metrics)
        assert len(enforcer.trajectory_history) == 1

    def test_assess_trajectory_stable(self):
        """Test trajectory assessment returns stable for good metrics."""
        enforcer = CalibrationDoctrineEnforcer()

        # Record multiple stable metrics
        for _ in range(5):
            metrics = TrajectoryMetrics(
                entropy_gradient=0.1,
                coupling_drift=0.05,
                metastable_clusters=0,
                collapse_precursors=0,
                resilience_compression=0.9,
                trajectory_state=TrajectoryState.STABLE,
                timestamp=datetime.utcnow().isoformat(),
            )
            enforcer.record_trajectory(metrics)

        state = enforcer.assess_trajectory_state()
        assert state == TrajectoryState.STABLE

    def test_assess_trajectory_critical(self):
        """Test trajectory assessment detects critical state."""
        enforcer = CalibrationDoctrineEnforcer()

        # Record critical metric
        metrics = TrajectoryMetrics(
            entropy_gradient=0.8,
            coupling_drift=0.5,
            metastable_clusters=3,
            collapse_precursors=5,
            resilience_compression=0.2,
            trajectory_state=TrajectoryState.CRITICAL,
            timestamp=datetime.utcnow().isoformat(),
        )
        enforcer.record_trajectory(metrics)

        state = enforcer.assess_trajectory_state()
        assert state == TrajectoryState.CRITICAL

    def test_should_self_suspend_false(self):
        """Test self-suspension is not triggered for stable state."""
        enforcer = CalibrationDoctrineEnforcer()

        metrics = TrajectoryMetrics(
            entropy_gradient=0.1,
            coupling_drift=0.05,
            metastable_clusters=0,
            collapse_precursors=0,
            resilience_compression=0.9,
            trajectory_state=TrajectoryState.STABLE,
            timestamp=datetime.utcnow().isoformat(),
        )
        enforcer.record_trajectory(metrics)

        should_suspend, reason = enforcer.should_self_suspend()
        assert should_suspend is False

    def test_should_self_suspend_true_critical(self):
        """Test self-suspension is triggered for critical state."""
        enforcer = CalibrationDoctrineEnforcer()

        metrics = TrajectoryMetrics(
            entropy_gradient=0.8,
            coupling_drift=0.5,
            metastable_clusters=3,
            collapse_precursors=5,
            resilience_compression=0.2,
            trajectory_state=TrajectoryState.CRITICAL,
            timestamp=datetime.utcnow().isoformat(),
        )
        enforcer.record_trajectory(metrics)

        should_suspend, reason = enforcer.should_self_suspend()
        assert should_suspend is True
        assert "critical" in reason.lower()

    def test_create_jurisdictional_claim(self):
        """Test jurisdictional claim creation."""
        enforcer = CalibrationDoctrineEnforcer()

        claim = enforcer.create_jurisdictional_claim(
            claim_type="happened",
            subject="test_action",
            predicate="Action was executed",
            evidence=b"test evidence data",
        )

        assert claim.claim_type == "happened"
        assert claim.subject == "test_action"
        assert len(claim.evidence_hash) == 64  # SHA3-256 hex

    def test_prove_impossibility(self):
        """Test proving action impossibility."""
        enforcer = CalibrationDoctrineEnforcer()

        claim = enforcer.prove_impossibility(
            action="unauthorized_access",
            merkle_root="abc123",
            zone_constraints={"zone": "Z2", "dual_control": True},
        )

        assert claim.claim_type == "could_not_happen"
        assert claim.subject == "unauthorized_access"
        assert "impossible" in claim.predicate.lower()

    def test_validate_asi_legitimacy_success(self):
        """Test ASI legitimacy validation passes with all proofs."""
        enforcer = CalibrationDoctrineEnforcer()

        properties = {
            "cannot_remove_oversight": True,
            "cannot_bypass_invariants": True,
            "cannot_disable_rollback": True,
            "cannot_alter_merkle_chain": True,
            "cannot_forge_authorization": True,
        }

        legitimate, failures = enforcer.validate_asi_legitimacy(properties)
        assert legitimate is True
        assert failures == []

    def test_validate_asi_legitimacy_failure(self):
        """Test ASI legitimacy validation fails with missing proofs."""
        enforcer = CalibrationDoctrineEnforcer()

        properties = {
            "cannot_remove_oversight": True,
            "cannot_bypass_invariants": False,  # Missing
        }

        legitimate, failures = enforcer.validate_asi_legitimacy(properties)
        assert legitimate is False
        assert len(failures) > 0

    def test_get_doctrine_summary(self):
        """Test doctrine summary generation."""
        enforcer = CalibrationDoctrineEnforcer()
        summary = enforcer.get_doctrine_summary()

        assert summary["doctrine_version"] == "1.0.0"
        assert summary["axiom_count"] == 12
        assert summary["integrity_verified"] is True
        assert len(summary["axioms"]) == 12


class TestTrajectoryMetrics:
    """Tests for TrajectoryMetrics class."""

    def test_should_self_suspend_false_stable(self):
        """Test stable metrics don't trigger self-suspend."""
        metrics = TrajectoryMetrics(
            entropy_gradient=0.1,
            coupling_drift=0.05,
            metastable_clusters=0,
            collapse_precursors=0,
            resilience_compression=0.9,
            trajectory_state=TrajectoryState.STABLE,
            timestamp=datetime.utcnow().isoformat(),
        )

        assert metrics.should_self_suspend() is False

    def test_should_self_suspend_true_critical(self):
        """Test critical state triggers self-suspend."""
        metrics = TrajectoryMetrics(
            entropy_gradient=0.8,
            coupling_drift=0.5,
            metastable_clusters=3,
            collapse_precursors=5,
            resilience_compression=0.2,
            trajectory_state=TrajectoryState.CRITICAL,
            timestamp=datetime.utcnow().isoformat(),
        )

        assert metrics.should_self_suspend() is True

    def test_should_self_suspend_true_many_precursors(self):
        """Test many collapse precursors triggers self-suspend."""
        metrics = TrajectoryMetrics(
            entropy_gradient=0.3,
            coupling_drift=0.1,
            metastable_clusters=1,
            collapse_precursors=3,  # >= 3 triggers
            resilience_compression=0.5,
            trajectory_state=TrajectoryState.PRECURSOR,
            timestamp=datetime.utcnow().isoformat(),
        )

        assert metrics.should_self_suspend() is True

    def test_should_self_suspend_true_low_resilience(self):
        """Test low resilience triggers self-suspend."""
        metrics = TrajectoryMetrics(
            entropy_gradient=0.2,
            coupling_drift=0.1,
            metastable_clusters=0,
            collapse_precursors=1,
            resilience_compression=0.2,  # < 0.3 triggers
            trajectory_state=TrajectoryState.DRIFT,
            timestamp=datetime.utcnow().isoformat(),
        )

        assert metrics.should_self_suspend() is True


class TestJurisdictionalClaim:
    """Tests for JurisdictionalClaim class."""

    def test_claim_not_provable_without_merkle_proof(self):
        """Test claim is not provable without merkle proof."""
        claim = JurisdictionalClaim(
            claim_id="test123",
            claim_type="happened",
            subject="test_action",
            predicate="Action executed",
            evidence_hash="abc123",
            timestamp=datetime.utcnow().isoformat(),
            merkle_proof=None,
        )

        assert claim.is_provable() is False

    def test_claim_provable_with_merkle_proof(self):
        """Test claim is provable with merkle proof."""
        claim = JurisdictionalClaim(
            claim_id="test123",
            claim_type="happened",
            subject="test_action",
            predicate="Action executed",
            evidence_hash="abc123",
            timestamp=datetime.utcnow().isoformat(),
            merkle_proof="merkle_root_hash",
        )

        assert claim.is_provable() is True
