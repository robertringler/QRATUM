"""Tests for Discovery Projections Engine.

Tests quantitative forecasting and timeline simulation.
"""

import pytest

from qratum_asi.discovery_acceleration.projections import (
    DiscoveryProjectionsEngine,
)
from qratum_asi.discovery_acceleration.types import DiscoveryType


class TestDiscoveryProjectionsEngine:
    """Tests for DiscoveryProjectionsEngine."""

    def test_engine_initialization(self):
        """Test engine initializes correctly."""
        engine = DiscoveryProjectionsEngine()

        assert len(engine._projection_cache) == 0

    def test_get_projections_all_types(self):
        """Test projections available for all discovery types."""
        engine = DiscoveryProjectionsEngine()

        for dt in DiscoveryType:
            projection = engine.get_projections(dt)

            assert projection.discovery_type == dt
            assert 0 <= projection.discovery_probability <= 1.0
            assert projection.time_savings_factor >= 1.0
            assert 0 <= projection.risk_mitigation_score <= 1.0
            assert projection.estimated_timeline_months > 0
            assert projection.legacy_timeline_months > 0
            assert projection.legacy_timeline_months >= projection.estimated_timeline_months

    def test_projections_caching(self):
        """Test projections are cached after first retrieval."""
        engine = DiscoveryProjectionsEngine()

        dt = DiscoveryType.COMPLEX_DISEASE_GENETICS

        proj1 = engine.get_projections(dt)
        proj2 = engine.get_projections(dt)

        assert proj1 is proj2  # Same object (cached)
        assert len(engine._projection_cache) == 1

    def test_specific_projection_values(self):
        """Test specific projection values match baseline."""
        engine = DiscoveryProjectionsEngine()

        # Test COMPLEX_DISEASE_GENETICS
        proj = engine.get_projections(DiscoveryType.COMPLEX_DISEASE_GENETICS)
        assert proj.discovery_probability == 0.75
        assert proj.time_savings_factor == 10.0
        assert proj.risk_mitigation_score == 0.95
        assert proj.estimated_timeline_months == 6
        assert proj.legacy_timeline_months == 60

    def test_simulate_timeline(self):
        """Test timeline simulation with parameters."""
        engine = DiscoveryProjectionsEngine()

        dt = DiscoveryType.PERSONALIZED_DRUG_DESIGN
        params = {
            "team_size": 5,
            "data_quality": 0.8,
            "resource_availability": 0.9,
            "complexity_factor": 1.0,
        }

        simulation = engine.simulate_timeline(dt, params)

        assert simulation.discovery_type == dt
        assert simulation.baseline_months > 0
        assert simulation.optimistic_months < simulation.baseline_months
        assert simulation.pessimistic_months > simulation.baseline_months
        assert len(simulation.confidence_interval) == 2
        assert simulation.confidence_interval[0] < simulation.confidence_interval[1]

    def test_simulate_timeline_team_size_variance(self):
        """Test timeline varies with team size."""
        engine = DiscoveryProjectionsEngine()

        dt = DiscoveryType.CLIMATE_GENE_CONNECTIONS

        # Small team
        sim_small = engine.simulate_timeline(dt, {"team_size": 2})
        # Large team
        sim_large = engine.simulate_timeline(dt, {"team_size": 10})

        # Larger team should have shorter optimistic timeline
        assert sim_large.optimistic_months < sim_small.optimistic_months

    def test_simulate_timeline_data_quality_impact(self):
        """Test timeline impacted by data quality."""
        engine = DiscoveryProjectionsEngine()

        dt = DiscoveryType.NATURAL_DRUG_DISCOVERY

        # Low quality
        sim_low = engine.simulate_timeline(dt, {"data_quality": 0.5})
        # High quality
        sim_high = engine.simulate_timeline(dt, {"data_quality": 0.95})

        # Low quality should have more pessimistic timeline
        assert sim_low.pessimistic_months > sim_high.pessimistic_months
        # Risk factors should mention low quality
        assert any("quality" in rf.lower() for rf in sim_low.risk_factors)

    def test_simulate_timeline_complexity_factor(self):
        """Test timeline scales with complexity."""
        engine = DiscoveryProjectionsEngine()

        dt = DiscoveryType.ECONOMIC_BIOLOGICAL_MODEL

        # Normal complexity
        sim_normal = engine.simulate_timeline(dt, {"complexity_factor": 1.0})
        # High complexity
        sim_complex = engine.simulate_timeline(dt, {"complexity_factor": 2.0})

        # Complex should have longer baseline
        assert sim_complex.baseline_months > sim_normal.baseline_months

    def test_calculate_risk_score(self):
        """Test risk score calculation."""
        engine = DiscoveryProjectionsEngine()

        workflow_id = "wf_test_001"

        assessment = engine.calculate_risk_score(workflow_id)

        assert assessment.workflow_id == workflow_id
        assert 0 <= assessment.overall_risk_score <= 1.0
        assert 0 <= assessment.vulnerability_score <= 1.0
        assert 0 <= assessment.trajectory_compliance <= 1.0
        assert isinstance(assessment.risk_factors, list)
        assert isinstance(assessment.mitigation_recommendations, list)
        assert len(assessment.mitigation_recommendations) > 0

    def test_calculate_risk_score_deterministic(self):
        """Test risk score is deterministic for same workflow ID."""
        engine = DiscoveryProjectionsEngine()

        workflow_id = "wf_test_deterministic"

        assessment1 = engine.calculate_risk_score(workflow_id)
        assessment2 = engine.calculate_risk_score(workflow_id)

        assert assessment1.overall_risk_score == assessment2.overall_risk_score
        assert assessment1.vulnerability_score == assessment2.vulnerability_score
        assert assessment1.trajectory_compliance == assessment2.trajectory_compliance

    def test_calculate_risk_score_varied_ids(self):
        """Test risk scores vary for different workflow IDs."""
        engine = DiscoveryProjectionsEngine()

        assessment1 = engine.calculate_risk_score("wf_001")
        assessment2 = engine.calculate_risk_score("wf_002")

        # Scores should be different for different workflows
        assert (
            assessment1.vulnerability_score != assessment2.vulnerability_score
            or assessment1.trajectory_compliance != assessment2.trajectory_compliance
        )

    def test_get_all_projections(self):
        """Test getting projections for all discovery types."""
        engine = DiscoveryProjectionsEngine()

        all_projections = engine.get_all_projections()

        assert len(all_projections) == len(DiscoveryType)
        for dt in DiscoveryType:
            assert dt.value in all_projections
            assert all_projections[dt.value].discovery_type == dt

    def test_compare_discovery_types_time_savings(self):
        """Test comparing discovery types by time savings."""
        engine = DiscoveryProjectionsEngine()

        comparisons = engine.compare_discovery_types("time_savings_factor")

        assert len(comparisons) == len(DiscoveryType)
        # Should be sorted descending
        for i in range(len(comparisons) - 1):
            assert comparisons[i][1] >= comparisons[i + 1][1]

        # Anti-aging should have highest time savings (25.0x)
        assert comparisons[0][0] == DiscoveryType.ANTI_AGING_PATHWAYS

    def test_compare_discovery_types_probability(self):
        """Test comparing discovery types by discovery probability."""
        engine = DiscoveryProjectionsEngine()

        comparisons = engine.compare_discovery_types("discovery_probability")

        assert len(comparisons) > 0
        # Complex disease genetics should have high probability (0.75)
        complex_disease_comp = [
            c for c in comparisons if c[0] == DiscoveryType.COMPLEX_DISEASE_GENETICS
        ][0]
        assert complex_disease_comp[1] == 0.75

    def test_compare_discovery_types_invalid_metric(self):
        """Test comparing with invalid metric returns empty list."""
        engine = DiscoveryProjectionsEngine()

        comparisons = engine.compare_discovery_types("nonexistent_metric")

        # Should return empty list for invalid metric
        assert len(comparisons) == 0

    def test_projection_serialization(self):
        """Test projection can be serialized to dict."""
        engine = DiscoveryProjectionsEngine()

        projection = engine.get_projections(DiscoveryType.ANTI_AGING_PATHWAYS)
        proj_dict = projection.to_dict()

        assert proj_dict["discovery_type"] == "anti_aging_pathways"
        assert proj_dict["discovery_probability"] == projection.discovery_probability
        assert proj_dict["time_savings_factor"] == projection.time_savings_factor
        # Additional metrics should be included
        assert "reversibility_score" in proj_dict

    def test_timeline_simulation_serialization(self):
        """Test timeline simulation can be serialized to dict."""
        engine = DiscoveryProjectionsEngine()

        simulation = engine.simulate_timeline(
            DiscoveryType.PERSONALIZED_DRUG_DESIGN, {"team_size": 5}
        )
        sim_dict = simulation.to_dict()

        assert sim_dict["discovery_type"] == "personalized_drug_design"
        assert sim_dict["baseline_months"] == simulation.baseline_months
        assert sim_dict["optimistic_months"] == simulation.optimistic_months
        assert sim_dict["pessimistic_months"] == simulation.pessimistic_months
        assert "confidence_interval" in sim_dict
        assert "risk_factors" in sim_dict

    def test_risk_assessment_serialization(self):
        """Test risk assessment can be serialized to dict."""
        engine = DiscoveryProjectionsEngine()

        assessment = engine.calculate_risk_score("wf_test_serialize")
        assess_dict = assessment.to_dict()

        assert assess_dict["workflow_id"] == "wf_test_serialize"
        assert assess_dict["overall_risk_score"] == assessment.overall_risk_score
        assert assess_dict["vulnerability_score"] == assessment.vulnerability_score
        assert "risk_factors" in assess_dict
        assert "mitigation_recommendations" in assess_dict


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
