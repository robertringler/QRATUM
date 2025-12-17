"""Tests for Qubic Meta Library data models."""

import pytest

from qubic_meta_library.models import Domain, Pipeline, Prompt, SynergyCluster


class TestPrompt:
    """Test Prompt model."""

    def test_create_valid_prompt(self):
        """Test creating a valid prompt."""

        prompt = Prompt(
            id=1,
            category="Test",
            description="Test prompt",
            domain="D1",
            patentability_score=0.85,
            commercial_potential=0.90,
            keystone_nodes=["Node1", "Node2"],
            synergy_connections=["D2", "D3"],
            execution_layers=["QuASIM"],
            phase_deployment=1,
            output_type="simulation",
        )

        assert prompt.id == 1
        assert prompt.category == "Test"
        assert prompt.patentability_score == 0.85
        assert prompt.commercial_potential == 0.90
        assert len(prompt.keystone_nodes) == 2
        assert len(prompt.synergy_connections) == 2

    def test_invalid_id(self):
        """Test that invalid ID raises ValueError."""

        with pytest.raises(ValueError, match="Prompt ID must be between 1 and 10000"):
            Prompt(
                id=10001,
                category="Test",
                description="Test",
                domain="D1",
                patentability_score=0.85,
                commercial_potential=0.90,
            )

    def test_invalid_patentability_score(self):
        """Test that invalid patentability score raises ValueError."""

        with pytest.raises(ValueError, match="Patentability score must be between"):
            Prompt(
                id=1,
                category="Test",
                description="Test",
                domain="D1",
                patentability_score=1.5,
                commercial_potential=0.90,
            )

    def test_is_high_value(self):
        """Test high-value prompt detection."""

        prompt = Prompt(
            id=1,
            category="Test",
            description="Test",
            domain="D1",
            patentability_score=0.85,
            commercial_potential=0.90,
        )

        assert prompt.is_high_value(threshold=0.8) is True
        assert prompt.is_high_value(threshold=0.9) is False

    def test_to_dict(self):
        """Test prompt serialization to dict."""

        prompt = Prompt(
            id=1,
            category="Test",
            description="Test",
            domain="D1",
            patentability_score=0.85,
            commercial_potential=0.90,
        )

        data = prompt.to_dict()
        assert data["id"] == 1
        assert data["patentability_score"] == 0.85

    def test_from_dict(self):
        """Test prompt deserialization from dict."""

        data = {
            "id": 1,
            "category": "Test",
            "description": "Test prompt",
            "domain": "D1",
            "patentability_score": 0.85,
            "commercial_potential": 0.90,
        }

        prompt = Prompt.from_dict(data)
        assert prompt.id == 1
        assert prompt.patentability_score == 0.85


class TestDomain:
    """Test Domain model."""

    def test_create_valid_domain(self):
        """Test creating a valid domain."""

        domain = Domain(
            id="D1",
            name="Advanced Materials",
            tier=1,
            id_range=(1, 100),
            primary_platform="QuASIM",
            commercial_sector="Aerospace",
            keystones=["Keystone1"],
        )

        assert domain.id == "D1"
        assert domain.name == "Advanced Materials"
        assert domain.tier == 1
        assert domain.id_range == (1, 100)

    def test_invalid_domain_id(self):
        """Test that invalid domain ID raises ValueError."""

        with pytest.raises(ValueError, match="Domain ID must start with 'D'"):
            Domain(
                id="X1",
                name="Test",
                tier=1,
                id_range=(1, 100),
                primary_platform="QuASIM",
                commercial_sector="Test",
            )

    def test_invalid_tier(self):
        """Test that invalid tier raises ValueError."""

        with pytest.raises(ValueError, match="Domain tier must be between 1 and 5"):
            Domain(
                id="D1",
                name="Test",
                tier=6,
                id_range=(1, 100),
                primary_platform="QuASIM",
                commercial_sector="Test",
            )

    def test_contains_id(self):
        """Test prompt ID range checking."""

        domain = Domain(
            id="D1",
            name="Test",
            tier=1,
            id_range=(1, 100),
            primary_platform="QuASIM",
            commercial_sector="Test",
        )

        assert domain.contains_id(50) is True
        assert domain.contains_id(101) is False


class TestSynergyCluster:
    """Test SynergyCluster model."""

    def test_create_valid_cluster(self):
        """Test creating a valid synergy cluster."""

        cluster = SynergyCluster(
            id="A",
            name="Test Cluster",
            domains=["D1", "D2"],
            prompts=[1, 2, 3],
            application="Test application",
            execution_mode="parallel",
        )

        assert cluster.id == "A"
        assert len(cluster.domains) == 2
        assert len(cluster.prompts) == 3
        assert cluster.cluster_type == "two-domain"

    def test_auto_detect_cluster_type(self):
        """Test automatic cluster type detection."""

        # Two-domain
        cluster2 = SynergyCluster(id="A", name="Test", domains=["D1", "D2"])
        assert cluster2.cluster_type == "two-domain"

        # Multi-domain
        cluster_multi = SynergyCluster(id="B", name="Test", domains=["D1", "D2", "D3"])
        assert cluster_multi.cluster_type == "multi-domain"

        # Full-stack
        all_domains = [f"D{i}" for i in range(1, 21)]
        cluster_full = SynergyCluster(id="DK", name="Full Stack", domains=all_domains)
        assert cluster_full.cluster_type == "full-stack"

    def test_revenue_projection(self):
        """Test revenue projection calculation."""

        cluster = SynergyCluster(
            id="A",
            name="Test",
            domains=["D1", "D2"],
            revenue_projection={2026: 1000000, 2027: 2000000},
        )

        assert cluster.get_total_revenue_projection() == 3000000


class TestPipeline:
    """Test Pipeline model."""

    def test_create_valid_pipeline(self):
        """Test creating a valid pipeline."""

        pipeline = Pipeline(
            id="P1",
            name="Test Pipeline",
            phase=1,
            prompts=[1, 2, 3],
            keystones=[1],
            platform="QuASIM",
        )

        assert pipeline.id == "P1"
        assert pipeline.phase == 1
        assert len(pipeline.prompts) == 3

    def test_invalid_phase(self):
        """Test that invalid phase raises ValueError."""

        with pytest.raises(ValueError, match="Phase must be between 1 and 4"):
            Pipeline(id="P1", name="Test", phase=5)

    def test_is_ready(self):
        """Test pipeline readiness checking."""

        pipeline = Pipeline(id="P2", name="Test", phase=2, dependencies=["P1"])

        assert pipeline.is_ready(set()) is False
        assert pipeline.is_ready({"P1"}) is True

    def test_add_prompt(self):
        """Test adding prompts to pipeline."""

        pipeline = Pipeline(id="P1", name="Test", phase=1)

        pipeline.add_prompt(1, is_keystone=True)
        assert 1 in pipeline.prompts
        assert 1 in pipeline.keystones

        pipeline.add_prompt(2, is_keystone=False)
        assert 2 in pipeline.prompts
        assert 2 not in pipeline.keystones
