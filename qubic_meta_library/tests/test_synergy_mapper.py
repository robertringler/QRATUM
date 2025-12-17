"""Tests for SynergyMapper service."""

import pytest

from qubic_meta_library.models import Domain, Prompt
from qubic_meta_library.services import SynergyMapper


class TestSynergyMapper:
    """Test SynergyMapper service."""

    @pytest.fixture
    def mapper(self, tmp_path):
        """Create mapper with test configuration."""

        # Create test clusters config
        config_dir = tmp_path / "config"
        config_dir.mkdir()

        clusters_content = """

clusters:
  - id: A
    name: Test Cluster A
    domains: [D1, D2]
    application: Test application
    execution_mode: parallel
    revenue_projection:
      2026: 1000000
      2027: 2000000

  - id: B
    name: Test Cluster B
    domains: [D2, D3]
    application: Another test
    execution_mode: hybrid
    revenue_projection:
      2026: 1500000
      2027: 2500000
"""

        (config_dir / "clusters.yaml").write_text(clusters_content)

        return SynergyMapper(config_dir=config_dir)

    def test_load_clusters(self, mapper):
        """Test loading cluster configurations."""

        clusters = mapper.load_clusters()

        assert len(clusters) == 2
        assert "A" in clusters
        assert "B" in clusters
        assert clusters["A"].name == "Test Cluster A"

    def test_get_clusters_by_domain(self, mapper):
        """Test getting clusters by domain."""

        mapper.load_clusters()

        d1_clusters = mapper.get_clusters_by_domain("D1")
        assert len(d1_clusters) == 1
        assert d1_clusters[0].id == "A"

        d2_clusters = mapper.get_clusters_by_domain("D2")
        assert len(d2_clusters) == 2

    def test_get_clusters_by_type(self, mapper):
        """Test getting clusters by type."""

        mapper.load_clusters()

        two_domain = mapper.get_clusters_by_type("two-domain")
        assert len(two_domain) == 2

    def test_find_synergies(self, mapper):
        """Test finding synergies across prompts."""

        mapper.load_clusters()

        prompts = {
            1: Prompt(
                id=1,
                category="Test",
                description="Test 1",
                domain="D1",
                patentability_score=0.8,
                commercial_potential=0.8,
                synergy_connections=["D2"],
            ),
            2: Prompt(
                id=2,
                category="Test",
                description="Test 2",
                domain="D2",
                patentability_score=0.8,
                commercial_potential=0.8,
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
                primary_platform="QuASIM",
                commercial_sector="Test",
            ),
        }

        synergies = mapper.find_synergies(prompts, domains)

        assert "A" in synergies
        assert len(synergies["A"]) > 0

    def test_calculate_cluster_value(self, mapper):
        """Test calculating cluster value metrics."""

        mapper.load_clusters()

        value = mapper.calculate_cluster_value("A")

        assert value["cluster_id"] == "A"
        assert value["domain_count"] == 2
        assert value["total_revenue_projection"] == 3000000

    def test_get_cross_domain_opportunities(self, mapper):
        """Test identifying cross-domain opportunities."""

        prompts = {
            1: Prompt(
                id=1,
                category="Test",
                description="Cross-domain test",
                domain="D1",
                patentability_score=0.9,
                commercial_potential=0.85,
                synergy_connections=["D2", "D3"],
            ),
            2: Prompt(
                id=2,
                category="Test",
                description="Single domain",
                domain="D1",
                patentability_score=0.8,
                commercial_potential=0.8,
            ),
        }

        opportunities = mapper.get_cross_domain_opportunities(prompts, min_connections=2)

        assert len(opportunities) == 1
        assert opportunities[0]["prompt_id"] == 1
        assert opportunities[0]["connection_count"] == 2

    def test_generate_cluster_report(self, mapper):
        """Test generating cluster report."""

        mapper.load_clusters()

        report = mapper.generate_cluster_report()

        assert report["total_clusters"] == 2
        assert report["total_revenue_projection"] == 7000000
        assert "cluster_types" in report
