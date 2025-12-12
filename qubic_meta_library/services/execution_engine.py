"""Execution engine service for Qubic Meta Library."""

from pathlib import Path
from typing import Any

import yaml

from qubic_meta_library.models import Pipeline, Prompt


class ExecutionEngine:
    """Service for managing pipeline execution and orchestration."""

    def __init__(self, config_dir: Path | None = None):
        """
        Initialize execution engine.

        Args:
            config_dir: Directory containing configuration files
        """
        if config_dir is None:
            config_dir = Path(__file__).parent.parent / "config"

        self.config_dir = Path(config_dir)
        self.pipelines: dict[str, Pipeline] = {}
        self.completed_pipelines: set[str] = set()

    def load_pipelines(self) -> dict[str, Pipeline]:
        """
        Load pipeline configurations.

        Returns:
            Dictionary mapping pipeline IDs to Pipeline objects
        """
        pipeline_file = self.config_dir / "pipeline_v12.yaml"
        if not pipeline_file.exists():
            raise FileNotFoundError(f"Pipeline configuration not found: {pipeline_file}")

        with open(pipeline_file) as f:
            data = yaml.safe_load(f)

        self.pipelines = {}
        for pipeline_data in data.get("pipelines", []):
            pipeline = Pipeline.from_dict(pipeline_data)
            self.pipelines[pipeline.id] = pipeline

        return self.pipelines

    def assign_prompts_to_pipelines(self, prompts: dict[int, Prompt]) -> dict[str, list[int]]:
        """
        Assign prompts to appropriate pipelines based on phase and domain.

        Args:
            prompts: Dictionary of all prompts

        Returns:
            Dictionary mapping pipeline IDs to prompt lists
        """
        assignments: dict[str, list[int]] = {pid: [] for pid in self.pipelines}

        for prompt in prompts.values():
            # Find matching pipeline by phase
            for pipeline in self.pipelines.values():
                if pipeline.phase == prompt.phase_deployment and (
                    not prompt.execution_layers or pipeline.platform in prompt.execution_layers
                ):
                    assignments[pipeline.id].append(prompt.id)
                    pipeline.add_prompt(prompt.id)
                    break

        return assignments

    def get_ready_pipelines(self) -> list[Pipeline]:
        """
        Get pipelines that are ready to execute.

        Returns:
            List of ready pipelines
        """
        ready = []
        for pipeline in self.pipelines.values():
            if pipeline.status == "pending" and pipeline.is_ready(self.completed_pipelines):
                ready.append(pipeline)

        return ready

    def execute_pipeline(self, pipeline_id: str, dry_run: bool = True) -> dict[str, Any]:
        """
        Execute a pipeline (or simulate execution).

        Args:
            pipeline_id: Pipeline identifier
            dry_run: If True, simulate execution without actual processing

        Returns:
            Dictionary with execution results
        """
        if pipeline_id not in self.pipelines:
            raise ValueError(f"Pipeline not found: {pipeline_id}")

        pipeline = self.pipelines[pipeline_id]

        if not pipeline.is_ready(self.completed_pipelines):
            return {
                "status": "blocked",
                "message": f"Pipeline {pipeline_id} has unmet dependencies",
                "missing_dependencies": [
                    dep for dep in pipeline.dependencies if dep not in self.completed_pipelines
                ],
            }

        # Simulate execution
        if dry_run:
            result = {
                "pipeline_id": pipeline_id,
                "pipeline_name": pipeline.name,
                "phase": pipeline.phase,
                "platform": pipeline.platform,
                "prompt_count": len(pipeline.prompts),
                "keystone_count": len(pipeline.keystones),
                "status": "simulated",
                "prompts_to_execute": pipeline.prompts[:5],  # Preview
                "execution_mode": "dry_run",
            }
        else:
            # Mark as running
            pipeline.status = "running"
            # In real implementation, this would trigger actual execution
            # For now, simulate success
            pipeline.status = "completed"
            self.completed_pipelines.add(pipeline_id)

            result = {
                "pipeline_id": pipeline_id,
                "pipeline_name": pipeline.name,
                "phase": pipeline.phase,
                "platform": pipeline.platform,
                "prompt_count": len(pipeline.prompts),
                "keystone_count": len(pipeline.keystones),
                "status": "completed",
                "prompts_executed": len(pipeline.prompts),
            }

        return result

    def get_execution_timeline(self) -> list[dict[str, Any]]:
        """
        Generate execution timeline for all pipelines.

        Returns:
            List of pipeline execution schedule
        """
        timeline = []
        for pipeline in sorted(self.pipelines.values(), key=lambda p: p.phase):
            timeline.append(
                {
                    "pipeline_id": pipeline.id,
                    "pipeline_name": pipeline.name,
                    "phase": pipeline.phase,
                    "start_date": pipeline.start_date,
                    "end_date": pipeline.end_date,
                    "status": pipeline.status,
                    "dependencies": pipeline.dependencies,
                    "prompt_count": len(pipeline.prompts),
                    "keystone_count": len(pipeline.keystones),
                }
            )

        return timeline

    def generate_execution_report(self) -> dict[str, Any]:
        """
        Generate comprehensive execution report.

        Returns:
            Dictionary with execution metrics and status
        """
        total_prompts = sum(len(p.prompts) for p in self.pipelines.values())
        total_keystones = sum(len(p.keystones) for p in self.pipelines.values())

        status_counts = {}
        for pipeline in self.pipelines.values():
            status_counts[pipeline.status] = status_counts.get(pipeline.status, 0) + 1

        ready_pipelines = self.get_ready_pipelines()

        return {
            "total_pipelines": len(self.pipelines),
            "completed_pipelines": len(self.completed_pipelines),
            "ready_pipelines": len(ready_pipelines),
            "status_breakdown": status_counts,
            "total_prompts": total_prompts,
            "total_keystones": total_keystones,
            "completion_percentage": (
                len(self.completed_pipelines) / len(self.pipelines) * 100 if self.pipelines else 0
            ),
            "next_pipelines": [
                {"id": p.id, "name": p.name, "phase": p.phase} for p in ready_pipelines
            ],
            "timeline": self.get_execution_timeline(),
        }

    def validate_pipeline_configuration(self) -> dict[str, Any]:
        """
        Validate pipeline configuration for consistency.

        Returns:
            Dictionary with validation results
        """
        errors = []
        warnings = []

        # Check for circular dependencies
        for pipeline in self.pipelines.values():
            if pipeline.id in pipeline.dependencies:
                errors.append(f"Pipeline {pipeline.id} has circular dependency on itself")

        # Check for invalid dependencies
        for pipeline in self.pipelines.values():
            for dep in pipeline.dependencies:
                if dep not in self.pipelines:
                    errors.append(f"Pipeline {pipeline.id} depends on non-existent pipeline {dep}")

        # Check phase ordering
        for pipeline in self.pipelines.values():
            for dep in pipeline.dependencies:
                if dep in self.pipelines:
                    dep_pipeline = self.pipelines[dep]
                    if dep_pipeline.phase >= pipeline.phase:
                        warnings.append(
                            f"Pipeline {pipeline.id} (phase {pipeline.phase}) "
                            f"depends on {dep} (phase {dep_pipeline.phase})"
                        )

        # Check for empty pipelines
        for pipeline in self.pipelines.values():
            if not pipeline.prompts:
                warnings.append(f"Pipeline {pipeline.id} has no prompts assigned")

        return {
            "valid": len(errors) == 0,
            "error_count": len(errors),
            "warning_count": len(warnings),
            "errors": errors,
            "warnings": warnings,
        }
