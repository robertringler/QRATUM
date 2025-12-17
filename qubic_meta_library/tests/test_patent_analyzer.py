"""Tests for PatentAnalyzer service."""

import pytest

from qubic_meta_library.models import Prompt
from qubic_meta_library.services import PatentAnalyzer


class TestPatentAnalyzer:
    """Test PatentAnalyzer service."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance."""

        return PatentAnalyzer()

    @pytest.fixture
    def sample_prompts(self):
        """Create sample prompts for testing."""

        return {
            1: Prompt(
                id=1,
                category="Quantum Simulation",
                description="High-value quantum simulation",
                domain="D1",
                patentability_score=0.92,
                commercial_potential=0.95,
                keystone_nodes=["Node1", "Node2"],
                synergy_connections=["D2", "D3"],
            ),
            2: Prompt(
                id=2,
                category="Materials",
                description="Advanced material design",
                domain="D1",
                patentability_score=0.85,
                commercial_potential=0.88,
                keystone_nodes=["Node1"],
                synergy_connections=["D2"],
            ),
            3: Prompt(
                id=3,
                category="Low Value",
                description="Low value prompt",
                domain="D2",
                patentability_score=0.70,
                commercial_potential=0.65,
            ),
        }

    def test_extract_high_value_prompts(self, analyzer, sample_prompts):
        """Test extracting high-value prompts."""

        high_value = analyzer.extract_high_value_prompts(sample_prompts)

        assert len(high_value) == 2
        assert high_value[0].id == 1  # Highest combined score
        assert high_value[1].id == 2

    def test_extract_high_value_with_threshold(self, analyzer, sample_prompts):
        """Test extracting with custom threshold."""

        premium = analyzer.extract_high_value_prompts(sample_prompts, threshold=0.9)

        assert len(premium) == 1
        assert premium[0].id == 1

    def test_identify_patent_clusters(self, analyzer, sample_prompts):
        """Test identifying patent clusters by domain."""

        clusters = analyzer.identify_patent_clusters(sample_prompts)

        assert "D1" in clusters
        assert len(clusters["D1"]) == 2
        assert clusters["D1"][0].id == 1  # Sorted by patentability

    def test_generate_patent_claim_template(self, analyzer, sample_prompts):
        """Test generating patent claim template."""

        template = analyzer.generate_patent_claim_template(sample_prompts[1])

        assert template["prompt_id"] == 1
        assert "title" in template
        assert "abstract" in template
        assert "claims" in template
        assert len(template["detailed_description"]["components"]) == 2

    def test_analyze_cross_domain_opportunities(self, analyzer, sample_prompts):
        """Test analyzing cross-domain opportunities."""

        opportunities = analyzer.analyze_cross_domain_opportunities(sample_prompts)

        # Only prompt 1 has 2+ synergy connections
        assert len(opportunities) == 1
        assert opportunities[0]["prompt_id"] == 1  # Highest novelty
        assert "novelty_score" in opportunities[0]
        assert "patent_template" in opportunities[0]

    def test_generate_patent_pipeline_report(self, analyzer, sample_prompts):
        """Test generating patent pipeline report."""

        report = analyzer.generate_patent_pipeline_report(sample_prompts)

        assert report["total_prompts"] == 3
        assert report["high_value_count"] == 2
        assert "top_10_opportunities" in report
        assert "average_patentability" in report
        assert "recommendations" in report

    def test_calculate_novelty_score(self, analyzer, sample_prompts):
        """Test novelty score calculation."""

        score = analyzer._calculate_novelty_score(sample_prompts[1])

        assert 0.0 <= score <= 1.0
        assert isinstance(score, float)

    def test_generate_recommendations(self, analyzer, sample_prompts):
        """Test generating recommendations."""

        high_value = analyzer.extract_high_value_prompts(sample_prompts)
        recommendations = analyzer._generate_recommendations(sample_prompts, high_value)

        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
