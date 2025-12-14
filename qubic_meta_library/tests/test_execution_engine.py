"""Tests for ExecutionEngine service."""

import pytest

from qubic_meta_library.models import Prompt
from qubic_meta_library.services import ExecutionEngine


class TestExecutionEngine:
    """Test ExecutionEngine service."""

    @pytest.fixture
    def engine(self, tmp_path):
        """Create engine with test configuration."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()

        pipeline_content = """
pipelines:
  - id: P1
    name: Pipeline 1
    phase: 1
    platform: QuASIM
    start_date: "2026-01-01"
    end_date: "2026-06-30"
    keystones: []
    dependencies: []

  - id: P2
    name: Pipeline 2
    phase: 2
    platform: QStack
    start_date: "2026-07-01"
    end_date: "2027-06-30"
    keystones: []
    dependencies: [P1]
"""
        (config_dir / "pipeline_v12.yaml").write_text(pipeline_content)

        return ExecutionEngine(config_dir=config_dir)

    def test_load_pipelines(self, engine):
        """Test loading pipeline configurations."""
        pipelines = engine.load_pipelines()

        assert len(pipelines) == 2
        assert "P1" in pipelines
        assert "P2" in pipelines
        assert pipelines["P1"].phase == 1

    def test_assign_prompts_to_pipelines(self, engine):
        """Test assigning prompts to pipelines."""
        engine.load_pipelines()

        prompts = {
            1: Prompt(
                id=1,
                category="Test",
                description="Test 1",
                domain="D1",
                patentability_score=0.8,
                commercial_potential=0.8,
                phase_deployment=1,
                execution_layers=["QuASIM"],
            ),
            2: Prompt(
                id=2,
                category="Test",
                description="Test 2",
                domain="D2",
                patentability_score=0.8,
                commercial_potential=0.8,
                phase_deployment=2,
                execution_layers=["QStack"],
            ),
        }

        assignments = engine.assign_prompts_to_pipelines(prompts)

        assert "P1" in assignments
        assert "P2" in assignments
        assert 1 in assignments["P1"]
        assert 2 in assignments["P2"]

    def test_get_ready_pipelines(self, engine):
        """Test getting ready pipelines."""
        engine.load_pipelines()

        # Initially, only P1 should be ready (no dependencies)
        ready = engine.get_ready_pipelines()
        assert len(ready) == 1
        assert ready[0].id == "P1"

        # After P1 completes, P2 should be ready
        engine.pipelines["P1"].status = "completed"
        engine.completed_pipelines.add("P1")
        ready = engine.get_ready_pipelines()
        assert len(ready) == 1
        assert ready[0].id == "P2"

    def test_execute_pipeline_dry_run(self, engine):
        """Test dry run pipeline execution."""
        engine.load_pipelines()

        result = engine.execute_pipeline("P1", dry_run=True)

        assert result["status"] == "simulated"
        assert result["pipeline_id"] == "P1"
        assert result["execution_mode"] == "dry_run"

    def test_execute_pipeline_blocked(self, engine):
        """Test executing blocked pipeline."""
        engine.load_pipelines()

        result = engine.execute_pipeline("P2", dry_run=False)

        assert result["status"] == "blocked"
        assert "P1" in result["missing_dependencies"]

    def test_get_execution_timeline(self, engine):
        """Test generating execution timeline."""
        engine.load_pipelines()

        timeline = engine.get_execution_timeline()

        assert len(timeline) == 2
        assert timeline[0]["phase"] == 1
        assert timeline[1]["phase"] == 2

    def test_generate_execution_report(self, engine):
        """Test generating execution report."""
        engine.load_pipelines()

        report = engine.generate_execution_report()

        assert report["total_pipelines"] == 2
        assert report["completed_pipelines"] == 0
        assert "timeline" in report

    def test_validate_pipeline_configuration(self, engine):
        """Test validating pipeline configuration."""
        engine.load_pipelines()

        validation = engine.validate_pipeline_configuration()

        assert validation["valid"] is True
        assert validation["error_count"] == 0

    def test_validate_invalid_dependency(self, tmp_path):
        """Test validation with invalid dependency."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()

        pipeline_content = """
pipelines:
  - id: P1
    name: Pipeline 1
    phase: 1
    platform: QuASIM
    dependencies: [P_INVALID]
"""
        (config_dir / "pipeline_v12.yaml").write_text(pipeline_content)

        engine = ExecutionEngine(config_dir=config_dir)
        engine.load_pipelines()

        validation = engine.validate_pipeline_configuration()

        assert validation["valid"] is False
        assert validation["error_count"] > 0
