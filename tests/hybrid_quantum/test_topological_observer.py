"""Tests for the Enhanced Topological Observer."""

import pytest
import numpy as np

try:
    from quasim.hybrid_quantum.topological_observer import (
        EnhancedTopologicalObserver,
        TopologicalObservation,
        CollapseMetrics,
        FidelityMetrics,
        DiagnosticFinding,
        DiagnosticSeverity,
        DiagnosticCategory,
    )
    OBSERVER_AVAILABLE = True
except ImportError:
    OBSERVER_AVAILABLE = False


@pytest.mark.skipif(not OBSERVER_AVAILABLE, reason="Topological observer module not available")
class TestCollapseMetrics:
    """Test CollapseMetrics class."""

    def test_collapse_metrics_creation(self):
        """Test creating collapse metrics."""
        metrics = CollapseMetrics(
            collapse_index=0.5,
            concentration_coefficient=0.3,
            effective_support=4,
        )
        assert metrics.collapse_index == 0.5
        assert metrics.concentration_coefficient == 0.3
        assert metrics.effective_support == 4

    def test_collapse_target_check(self):
        """Test P1 collapse reduction target (≥30%)."""
        # Collapse index 0.7 means 30% reduction from 1.0
        metrics = CollapseMetrics(collapse_index=0.7, target_reduction=0.3)
        assert metrics.meets_target is True
        
        # Collapse index 0.8 is above threshold (1 - 0.3 = 0.7)
        metrics = CollapseMetrics(collapse_index=0.8, target_reduction=0.3)
        assert metrics.meets_target is False


@pytest.mark.skipif(not OBSERVER_AVAILABLE, reason="Topological observer module not available")
class TestFidelityMetrics:
    """Test FidelityMetrics class."""

    def test_fidelity_metrics_creation(self):
        """Test creating fidelity metrics."""
        metrics = FidelityMetrics(
            simulation_fidelity=0.995,
            state_fidelity=0.998,
            error_rate=0.002,
        )
        assert metrics.simulation_fidelity == 0.995
        assert metrics.error_rate == 0.002

    def test_p1_fidelity_target(self):
        """Test P1 fidelity target (≥0.999)."""
        metrics = FidelityMetrics(simulation_fidelity=0.9995)
        assert metrics.meets_p1_target is True
        
        metrics = FidelityMetrics(simulation_fidelity=0.998)
        assert metrics.meets_p1_target is False


@pytest.mark.skipif(not OBSERVER_AVAILABLE, reason="Topological observer module not available")
class TestDiagnosticFinding:
    """Test DiagnosticFinding class."""

    def test_finding_creation(self):
        """Test creating diagnostic finding."""
        finding = DiagnosticFinding(
            finding_id="test-123",
            category=DiagnosticCategory.FIDELITY,
            severity=DiagnosticSeverity.WARNING,
            description="Low fidelity detected",
            recommended_action="Increase shots",
        )
        assert finding.finding_id == "test-123"
        assert finding.category == DiagnosticCategory.FIDELITY
        assert finding.severity == DiagnosticSeverity.WARNING

    def test_finding_to_proposal(self):
        """Test converting finding to proposal-only format."""
        finding = DiagnosticFinding(
            finding_id="test-123",
            category=DiagnosticCategory.NOISE,
            severity=DiagnosticSeverity.CRITICAL,
            description="High noise detected",
            recommended_action="Enable error mitigation",
            observation_id="obs-456",
        )
        
        proposal = finding.to_proposal()
        
        assert "proposal_id" in proposal
        assert proposal["finding_id"] == "test-123"
        assert proposal["requires_approval"] is True
        assert proposal["status"] == "pending"


@pytest.mark.skipif(not OBSERVER_AVAILABLE, reason="Topological observer module not available")
class TestEnhancedTopologicalObserver:
    """Test EnhancedTopologicalObserver class."""

    def test_observer_creation(self):
        """Test creating observer with defaults."""
        observer = EnhancedTopologicalObserver()
        assert observer.fidelity_threshold == 0.999
        assert observer.collapse_reduction_target == 0.3
        assert len(observer.get_observations()) == 0

    def test_observer_custom_thresholds(self):
        """Test creating observer with custom thresholds."""
        observer = EnhancedTopologicalObserver(
            fidelity_threshold=0.995,
            collapse_reduction_target=0.4,
            noise_threshold=0.2,
        )
        assert observer.fidelity_threshold == 0.995
        assert observer.collapse_reduction_target == 0.4

    def test_observe_uniform_distribution(self):
        """Test observing uniform distribution."""
        observer = EnhancedTopologicalObserver()
        
        counts = {"00": 250, "01": 250, "10": 250, "11": 250}
        observation = observer.observe(counts)
        
        assert observation.observation_id != ""
        assert observation.collapse_metrics.collapse_index == 0.25
        assert observation.collapse_metrics.effective_support == 4
        assert observation.collapse_metrics.entropy_ratio > 0.9

    def test_observe_collapsed_distribution(self):
        """Test observing collapsed distribution."""
        observer = EnhancedTopologicalObserver()
        
        # 90% in "00", others all >1% so effective support is 4
        # (900/1000=90%, 50/1000=5%, 30/1000=3%, 20/1000=2%)
        counts = {"00": 900, "01": 50, "10": 30, "11": 20}
        observation = observer.observe(counts)
        
        assert observation.collapse_metrics.collapse_index == 0.9
        # All 4 outcomes have probability >1%, so effective_support is 4
        assert observation.collapse_metrics.effective_support == 4
        assert observation.collapse_metrics.dominant_outcomes[0] == "00"
        # Entropy should be low due to concentration
        assert observation.collapse_metrics.entropy_ratio < 0.5

    def test_observe_with_expected_distribution(self):
        """Test observing with expected distribution for fidelity."""
        observer = EnhancedTopologicalObserver()
        
        counts = {"00": 500, "11": 500}
        expected = {"00": 0.5, "11": 0.5}
        
        observation = observer.observe(counts, expected_distribution=expected)
        
        assert observation.fidelity_metrics.simulation_fidelity > 0.99
        assert observation.fidelity_metrics.error_rate < 0.01

    def test_observe_with_mismatched_distribution(self):
        """Test observing with mismatched expected distribution."""
        observer = EnhancedTopologicalObserver()
        
        counts = {"00": 800, "11": 200}
        expected = {"00": 0.5, "11": 0.5}
        
        observation = observer.observe(counts, expected_distribution=expected)
        
        # Should have lower fidelity due to mismatch
        assert observation.fidelity_metrics.simulation_fidelity < 0.999
        assert observation.fidelity_metrics.error_rate > 0.1

    def test_observe_generates_findings(self):
        """Test that observation generates diagnostic findings."""
        observer = EnhancedTopologicalObserver(fidelity_threshold=0.999)
        
        # Create distribution that will trigger findings
        counts = {"00": 900, "11": 100}
        expected = {"00": 0.5, "11": 0.5}
        
        observation = observer.observe(counts, expected_distribution=expected)
        
        # Should have findings for fidelity issue
        assert len(observation.findings) > 0
        fidelity_findings = [
            f for f in observation.findings 
            if f.category == DiagnosticCategory.FIDELITY
        ]
        assert len(fidelity_findings) > 0

    def test_observe_empty_counts(self):
        """Test observing empty counts."""
        observer = EnhancedTopologicalObserver()
        
        observation = observer.observe({})
        
        assert "error" in observation.raw_diagnostics

    def test_observation_history_tracking(self):
        """Test that observations are tracked in history."""
        observer = EnhancedTopologicalObserver()
        
        observer.observe({"00": 500, "11": 500})
        observer.observe({"00": 600, "11": 400})
        observer.observe({"00": 700, "11": 300})
        
        observations = observer.get_observations()
        assert len(observations) == 3

    def test_proposals_tracking(self):
        """Test that proposals are tracked from findings."""
        observer = EnhancedTopologicalObserver(fidelity_threshold=0.999)
        
        # Create observation that triggers findings
        counts = {"00": 900, "11": 100}
        expected = {"00": 0.5, "11": 0.5}
        
        observer.observe(counts, expected_distribution=expected)
        
        proposals = observer.get_proposals()
        # Should have generated proposals for findings
        assert len(proposals) > 0
        assert all(p["requires_approval"] for p in proposals)

    def test_soi_visualization_data(self):
        """Test SOI visualization data generation."""
        observer = EnhancedTopologicalObserver()
        
        observer.observe({"00": 500, "11": 500})
        observer.observe({"00": 600, "11": 400})
        
        viz_data = observer.generate_soi_visualization_data()
        
        assert viz_data["visualization_type"] == "soi_topological"
        assert viz_data["data_points"] == 2
        assert "collapse_trend" in viz_data
        assert "fidelity_trend" in viz_data
        assert "findings_summary" in viz_data

    def test_soi_visualization_specific_observation(self):
        """Test SOI visualization for specific observation."""
        observer = EnhancedTopologicalObserver()
        
        obs1 = observer.observe({"00": 500, "11": 500})
        obs2 = observer.observe({"00": 600, "11": 400})
        
        viz_data = observer.generate_soi_visualization_data(
            observation_id=obs1.observation_id
        )
        
        assert viz_data["data_points"] == 1

    def test_summary_report(self):
        """Test summary report generation."""
        observer = EnhancedTopologicalObserver()
        
        observer.observe({"00": 500, "11": 500})
        observer.observe({"00": 600, "11": 400})
        
        summary = observer.get_summary_report()
        
        assert summary["observation_count"] == 2
        assert "collapse_statistics" in summary
        assert "fidelity_statistics" in summary
        assert "mean" in summary["collapse_statistics"]
        assert "std" in summary["collapse_statistics"]

    def test_summary_report_empty(self):
        """Test summary report with no observations."""
        observer = EnhancedTopologicalObserver()
        
        summary = observer.get_summary_report()
        
        assert summary["count"] == 0

    def test_provenance_hash_computation(self):
        """Test provenance hash is computed for observation."""
        observer = EnhancedTopologicalObserver()
        
        counts = {"00": 500, "11": 500}
        observation = observer.observe(counts)
        
        assert observation.provenance_hash != ""
        assert len(observation.provenance_hash) == 64  # SHA-256 hex

    def test_read_only_behavior(self):
        """Test that observer is read-only (no state modification)."""
        observer = EnhancedTopologicalObserver()
        
        counts = {"00": 500, "11": 500}
        observation = observer.observe(counts)
        
        # Observer should not modify counts
        assert counts == {"00": 500, "11": 500}
        
        # Observations should be immutable snapshots
        original_id = observation.observation_id
        observer.observe({"00": 600, "11": 400})
        
        # Original observation unchanged
        obs_list = observer.get_observations()
        assert obs_list[0].observation_id == original_id


@pytest.mark.skipif(not OBSERVER_AVAILABLE, reason="Topological observer module not available")
class TestTopologicalObservation:
    """Test TopologicalObservation dataclass."""

    def test_observation_creation(self):
        """Test creating observation."""
        observation = TopologicalObservation(observation_id="test-123")
        
        assert observation.observation_id == "test-123"
        assert observation.timestamp != ""
        assert isinstance(observation.collapse_metrics, CollapseMetrics)
        assert isinstance(observation.fidelity_metrics, FidelityMetrics)


@pytest.mark.skipif(not OBSERVER_AVAILABLE, reason="Topological observer module not available")
class TestDiagnosticEnums:
    """Test diagnostic enumeration types."""

    def test_severity_values(self):
        """Test DiagnosticSeverity enum values."""
        assert DiagnosticSeverity.INFO.value == "info"
        assert DiagnosticSeverity.WARNING.value == "warning"
        assert DiagnosticSeverity.CRITICAL.value == "critical"
        assert DiagnosticSeverity.FATAL.value == "fatal"

    def test_category_values(self):
        """Test DiagnosticCategory enum values."""
        assert DiagnosticCategory.NOISE.value == "noise"
        assert DiagnosticCategory.DECOHERENCE.value == "decoherence"
        assert DiagnosticCategory.TOPOLOGY.value == "topology"
        assert DiagnosticCategory.FIDELITY.value == "fidelity"
        assert DiagnosticCategory.CONSISTENCY.value == "consistency"
        assert DiagnosticCategory.ENTROPY.value == "entropy"


@pytest.mark.skipif(not OBSERVER_AVAILABLE, reason="Topological observer module not available")
class TestReferenceConsistency:
    """Test consistency checks with reference results."""

    def test_observe_with_reference_counts(self):
        """Test observing with reference counts for consistency."""
        observer = EnhancedTopologicalObserver()
        
        counts = {"00": 500, "11": 500}
        reference = [
            {"00": 490, "11": 510},
            {"00": 505, "11": 495},
        ]
        
        observation = observer.observe(counts, reference_counts=reference)
        
        # With similar results, fidelity should be high
        assert observation.fidelity_metrics.classical_fidelity > 0.95

    def test_observe_with_inconsistent_reference(self):
        """Test observing with inconsistent reference results."""
        observer = EnhancedTopologicalObserver()
        
        counts = {"00": 800, "11": 200}
        reference = [
            {"00": 200, "11": 800},  # Opposite distribution
        ]
        
        observation = observer.observe(counts, reference_counts=reference)
        
        # With opposite results, classical fidelity should be lower
        assert observation.fidelity_metrics.classical_fidelity < 0.9
