"""Tests for platform integration connectors."""

import pytest

from qubic_meta_library.integrations import (
    QNimbusConnector,
    QStackConnector,
    QuASIMConnector,
)
from qubic_meta_library.integrations.orchestrator import UnifiedOrchestrator
from qubic_meta_library.models import Prompt


class TestQuASIMConnector:
    """Test QuASIM connector."""

    @pytest.fixture
    def connector(self):
        """Create connector instance."""

        return QuASIMConnector(seed=42, deterministic_mode=True)

    @pytest.fixture
    def quasim_prompt(self):
        """Create QuASIM-compatible prompt."""

        return Prompt(
            id=1,
            category="Quantum Simulation",
            description="Test quantum simulation",
            domain="D1",
            patentability_score=0.9,
            commercial_potential=0.85,
            execution_layers=["QuASIM"],
        )

    def test_can_execute_by_domain(self, connector):
        """Test domain-based execution check."""

        prompt = Prompt(
            id=1,
            category="Test",
            description="Test",
            domain="D1",
            patentability_score=0.8,
            commercial_potential=0.8,
        )
        assert connector.can_execute(prompt) is True

    def test_can_execute_by_layer(self, connector):
        """Test execution layer check."""

        prompt = Prompt(
            id=1,
            category="Test",
            description="Test",
            domain="D99",  # Invalid domain
            patentability_score=0.8,
            commercial_potential=0.8,
            execution_layers=["QuASIM"],
        )
        assert connector.can_execute(prompt) is True

    def test_cannot_execute_unsupported(self, connector):
        """Test rejection of unsupported prompts."""

        prompt = Prompt(
            id=1,
            category="Test",
            description="Test",
            domain="D3",  # AI domain, not QuASIM
            patentability_score=0.8,
            commercial_potential=0.8,
        )
        assert connector.can_execute(prompt) is False

    def test_execute_dry_run(self, connector, quasim_prompt):
        """Test dry run execution."""

        result = connector.execute(quasim_prompt, dry_run=True)

        assert result.is_successful()
        assert result.status == "completed"
        assert result.output_data["mode"] == "dry_run"

    def test_execute_simulation(self, connector, quasim_prompt):
        """Test simulated execution."""

        result = connector.execute(quasim_prompt, dry_run=False)

        assert result.is_successful()
        assert result.execution_time_ms > 0
        assert result.reproducibility_hash != ""

    def test_execute_batch(self, connector):
        """Test batch execution."""

        prompts = [
            Prompt(
                id=i,
                category="Test",
                description=f"Test {i}",
                domain="D1",
                patentability_score=0.8,
                commercial_potential=0.8,
            )
            for i in range(1, 4)
        ]

        results = connector.execute_batch(prompts, dry_run=True)

        assert len(results) == 3
        assert all(r.is_successful() for r in results)


class TestQStackConnector:
    """Test QStack connector."""

    @pytest.fixture
    def connector(self):
        """Create connector instance."""

        return QStackConnector(device="cpu")

    def test_can_execute_ai_domain(self, connector):
        """Test AI domain support."""

        prompt = Prompt(
            id=1,
            category="Test",
            description="Test",
            domain="D3",  # Multi-Agent AI
            patentability_score=0.8,
            commercial_potential=0.8,
        )
        assert connector.can_execute(prompt) is True

    def test_execute_dry_run(self, connector):
        """Test dry run execution."""

        prompt = Prompt(
            id=1,
            category="ML Training",
            description="Test ML",
            domain="D8",
            patentability_score=0.8,
            commercial_potential=0.8,
        )
        result = connector.execute(prompt, dry_run=True)

        assert result.is_successful()
        assert result.model_metrics["accuracy"] == 0.0


class TestQNimbusConnector:
    """Test QNimbus connector."""

    @pytest.fixture
    def connector(self):
        """Create connector instance."""

        return QNimbusConnector(default_provider="aws")

    def test_can_execute_cloud_domain(self, connector):
        """Test cloud domain support."""

        prompt = Prompt(
            id=1,
            category="Test",
            description="Test",
            domain="D12",  # IoT & Sensor Networks
            patentability_score=0.8,
            commercial_potential=0.8,
        )
        assert connector.can_execute(prompt) is True

    def test_execute_with_provider(self, connector):
        """Test execution with custom provider."""

        prompt = Prompt(
            id=1,
            category="Cloud Deployment",
            description="Test cloud",
            domain="D20",
            patentability_score=0.8,
            commercial_potential=0.8,
        )
        result = connector.execute(prompt, dry_run=True, provider="gcp")

        assert result.is_successful()
        assert result.cloud_provider == "gcp"


class TestUnifiedOrchestrator:
    """Test unified orchestrator."""

    @pytest.fixture
    def orchestrator(self):
        """Create orchestrator instance."""

        return UnifiedOrchestrator(quasim_seed=42)

    def test_route_quasim(self, orchestrator):
        """Test routing to QuASIM."""

        prompt = Prompt(
            id=1,
            category="Test",
            description="Test",
            domain="D1",
            patentability_score=0.8,
            commercial_potential=0.8,
        )
        assert orchestrator.route_prompt(prompt) == "QuASIM"

    def test_route_qstack(self, orchestrator):
        """Test routing to QStack."""

        prompt = Prompt(
            id=1,
            category="Test",
            description="Test",
            domain="D3",
            patentability_score=0.8,
            commercial_potential=0.8,
        )
        assert orchestrator.route_prompt(prompt) == "QStack"

    def test_route_qnimbus(self, orchestrator):
        """Test routing to QNimbus."""

        prompt = Prompt(
            id=1,
            category="Test",
            description="Test",
            domain="D12",
            patentability_score=0.8,
            commercial_potential=0.8,
        )
        assert orchestrator.route_prompt(prompt) == "QNimbus"

    def test_execute_with_routing(self, orchestrator):
        """Test execution with automatic routing."""

        prompt = Prompt(
            id=1,
            category="Test",
            description="Test",
            domain="D1",
            patentability_score=0.8,
            commercial_potential=0.8,
        )
        result = orchestrator.execute(prompt, dry_run=True)

        assert result.is_successful()
        assert result.platform == "QuASIM"

    def test_execute_batch(self, orchestrator):
        """Test batch execution across platforms."""

        prompts = [
            Prompt(
                id=1,
                category="Test",
                description="QuASIM test",
                domain="D1",
                patentability_score=0.8,
                commercial_potential=0.8,
            ),
            Prompt(
                id=2,
                category="Test",
                description="QStack test",
                domain="D3",
                patentability_score=0.8,
                commercial_potential=0.8,
            ),
            Prompt(
                id=3,
                category="Test",
                description="QNimbus test",
                domain="D12",
                patentability_score=0.8,
                commercial_potential=0.8,
            ),
        ]

        results = orchestrator.execute_batch(prompts, dry_run=True)

        assert len(results) == 3
        platforms = {r.platform for r in results}
        assert platforms == {"QuASIM", "QStack", "QNimbus"}

    def test_get_routing_summary(self, orchestrator):
        """Test routing summary generation."""

        prompts = [
            Prompt(
                id=1,
                category="Test",
                description="Test",
                domain="D1",
                patentability_score=0.8,
                commercial_potential=0.8,
            ),
            Prompt(
                id=2,
                category="Test",
                description="Test",
                domain="D3",
                patentability_score=0.8,
                commercial_potential=0.8,
            ),
        ]

        summary = orchestrator.get_routing_summary(prompts)

        assert 1 in summary["QuASIM"]
        assert 2 in summary["QStack"]

    def test_execution_stats(self, orchestrator):
        """Test execution statistics."""

        prompt = Prompt(
            id=1,
            category="Test",
            description="Test",
            domain="D1",
            patentability_score=0.8,
            commercial_potential=0.8,
        )
        orchestrator.execute(prompt, dry_run=True)

        stats = orchestrator.get_execution_stats()

        assert stats["total_executions"] == 1
        assert stats["successful_executions"] == 1
        assert "quasim_stats" in stats
