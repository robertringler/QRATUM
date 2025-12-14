"""Tests for Dashboard service."""

import pytest

from qubic_meta_library.models import Domain, Pipeline, Prompt, SynergyCluster
from qubic_meta_library.services import Dashboard


class TestDashboard:
    """Test Dashboard service."""

    @pytest.fixture
    def dashboard(self):
        """Create dashboard instance."""
        return Dashboard()

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        prompts = {
            1: Prompt(
                id=1,
                category="Test",
                description="Test 1",
                domain="D1",
                patentability_score=0.90,
                commercial_potential=0.92,
                phase_deployment=1,
                output_type="simulation",
            ),
            2: Prompt(
                id=2,
                category="Test",
                description="Test 2",
                domain="D2",
                patentability_score=0.85,
                commercial_potential=0.88,
                phase_deployment=2,
                output_type="model",
            ),
        }

        domains = {
            "D1": Domain(
                id="D1",
                name="Domain 1",
                tier=1,
                id_range=(1, 100),
                primary_platform="QuASIM",
                commercial_sector="Test",
            ),
            "D2": Domain(
                id="D2",
                name="Domain 2",
                tier=1,
                id_range=(101, 200),
                primary_platform="QStack",
                commercial_sector="Test",
            ),
        }

        clusters = {
            "A": SynergyCluster(
                id="A",
                name="Cluster A",
                domains=["D1", "D2"],
                prompts=[1, 2],
                revenue_projection={2026: 1000000, 2027: 2000000},
            )
        }

        pipelines = {
            "P1": Pipeline(id="P1", name="Pipeline 1", phase=1, prompts=[1]),
            "P2": Pipeline(
                id="P2",
                name="Pipeline 2",
                phase=2,
                prompts=[2],
                status="completed",
            ),
        }

        return prompts, domains, clusters, pipelines

    def test_calculate_kpis(self, dashboard, sample_data):
        """Test calculating all KPIs."""
        prompts, domains, clusters, pipelines = sample_data

        kpis = dashboard.calculate_kpis(prompts, domains, clusters, pipelines)

        assert "prompt_metrics" in kpis
        assert "domain_metrics" in kpis
        assert "cluster_metrics" in kpis
        assert "pipeline_metrics" in kpis
        assert "patent_metrics" in kpis
        assert "commercial_metrics" in kpis

    def test_calculate_prompt_metrics(self, dashboard, sample_data):
        """Test calculating prompt metrics."""
        prompts, _, _, _ = sample_data

        metrics = dashboard._calculate_prompt_metrics(prompts)

        assert metrics["total_prompts"] == 2
        assert metrics["high_value_prompts"] == 2
        assert metrics["high_value_percentage"] == 100.0
        assert 0.0 <= metrics["average_patentability"] <= 1.0

    def test_calculate_domain_metrics(self, dashboard, sample_data):
        """Test calculating domain metrics."""
        prompts, domains, _, _ = sample_data

        metrics = dashboard._calculate_domain_metrics(prompts, domains)

        assert metrics["total_domains"] == 2
        assert "prompts_per_domain" in metrics
        assert "avg_patentability_per_domain" in metrics

    def test_calculate_cluster_metrics(self, dashboard, sample_data):
        """Test calculating cluster metrics."""
        _, _, clusters, _ = sample_data

        metrics = dashboard._calculate_cluster_metrics(clusters)

        assert metrics["total_clusters"] == 1
        assert metrics["total_revenue_projection"] == 3000000
        assert metrics["clusters_with_prompts"] == 1

    def test_calculate_pipeline_metrics(self, dashboard, sample_data):
        """Test calculating pipeline metrics."""
        _, _, _, pipelines = sample_data

        metrics = dashboard._calculate_pipeline_metrics(pipelines)

        assert metrics["total_pipelines"] == 2
        assert "status_breakdown" in metrics
        assert metrics["completion_rate"] == 50.0

    def test_calculate_patent_metrics(self, dashboard, sample_data):
        """Test calculating patent metrics."""
        prompts, _, _, _ = sample_data

        metrics = dashboard._calculate_patent_metrics(prompts)

        assert metrics["total_patent_candidates"] == 2
        assert metrics["high_value_patents"] == 2
        assert 0.0 <= metrics["patent_readiness_score"] <= 100.0

    def test_calculate_commercial_metrics(self, dashboard, sample_data):
        """Test calculating commercial metrics."""
        prompts, _, clusters, _ = sample_data

        metrics = dashboard._calculate_commercial_metrics(prompts, clusters)

        assert 0.0 <= metrics["average_commercial_score"] <= 1.0
        assert metrics["market_ready_prompts"] >= 0
        assert metrics["total_revenue_projection"] == 3000000

    def test_generate_executive_summary(self, dashboard, sample_data):
        """Test generating executive summary."""
        prompts, domains, clusters, pipelines = sample_data

        summary = dashboard.generate_executive_summary(prompts, domains, clusters, pipelines)

        assert isinstance(summary, str)
        assert "Qubic Meta Library Executive Summary" in summary
        assert "Overview" in summary
        assert "Patent Pipeline" in summary
        assert "Commercial Metrics" in summary

    def test_empty_data(self, dashboard):
        """Test with empty data."""
        metrics = dashboard._calculate_prompt_metrics({})
        assert metrics["total_prompts"] == 0

        metrics = dashboard._calculate_cluster_metrics({})
        assert metrics["total_clusters"] == 0
