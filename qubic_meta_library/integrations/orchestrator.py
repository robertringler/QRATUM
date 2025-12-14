"""Unified execution orchestrator for Q-Stack platforms.

Routes prompts to the appropriate platform (QuASIM, QStack, QNimbus)
based on domain and execution layer assignments.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from qubic_meta_library.integrations.qnimbus_connector import (
    CloudExecutionResult,
    QNimbusConnector,
)
from qubic_meta_library.integrations.qstack_connector import (
    MLExecutionResult,
    QStackConnector,
)
from qubic_meta_library.integrations.quasim_connector import (
    QuASIMConnector,
    SimulationResult,
)
from qubic_meta_library.models import Prompt


@dataclass
class UnifiedExecutionResult:
    """Unified result from any platform execution."""

    prompt_id: int
    platform: str  # "QuASIM", "QStack", "QNimbus"
    status: str
    execution_time_ms: float = 0.0
    output_data: dict[str, Any] = field(default_factory=dict)
    error_message: str = ""

    @classmethod
    def from_simulation_result(cls, result: SimulationResult) -> UnifiedExecutionResult:
        """Create from QuASIM SimulationResult."""
        return cls(
            prompt_id=result.prompt_id,
            platform="QuASIM",
            status=result.status,
            execution_time_ms=result.execution_time_ms,
            output_data={
                **result.output_data,
                "seed": result.seed,
                "reproducibility_hash": result.reproducibility_hash,
            },
            error_message=result.error_message,
        )

    @classmethod
    def from_ml_result(cls, result: MLExecutionResult) -> UnifiedExecutionResult:
        """Create from QStack MLExecutionResult."""
        return cls(
            prompt_id=result.prompt_id,
            platform="QStack",
            status=result.status,
            execution_time_ms=result.execution_time_ms,
            output_data={
                **result.output_data,
                "model_metrics": result.model_metrics,
            },
            error_message=result.error_message,
        )

    @classmethod
    def from_cloud_result(cls, result: CloudExecutionResult) -> UnifiedExecutionResult:
        """Create from QNimbus CloudExecutionResult."""
        return cls(
            prompt_id=result.prompt_id,
            platform="QNimbus",
            status=result.status,
            execution_time_ms=result.execution_time_ms,
            output_data={
                **result.output_data,
                "cloud_provider": result.cloud_provider,
                "region": result.region,
                "resource_usage": result.resource_usage,
            },
            error_message=result.error_message,
        )

    def is_successful(self) -> bool:
        """Check if execution completed successfully."""
        return self.status == "completed"


class UnifiedOrchestrator:
    """Unified orchestrator for executing prompts across Q-Stack platforms.

    Automatically routes prompts to the most appropriate platform based on:
    1. Explicit execution_layers assignment
    2. Domain-platform mapping
    3. Fallback to default platform
    """

    def __init__(
        self,
        quasim_seed: int = 42,
        deterministic_mode: bool = True,
        precision: str = "FP32",
        ml_device: str = "cpu",
        cloud_provider: str = "aws",
    ):
        """Initialize unified orchestrator.

        Args:
            quasim_seed: Seed for QuASIM reproducibility
            deterministic_mode: Enable deterministic execution
            precision: Floating point precision for QuASIM
            ml_device: Device for QStack ML workloads
            cloud_provider: Default cloud provider for QNimbus
        """
        self.quasim = QuASIMConnector(
            seed=quasim_seed,
            deterministic_mode=deterministic_mode,
            precision=precision,
        )
        self.qstack = QStackConnector(device=ml_device)
        self.qnimbus = QNimbusConnector(default_provider=cloud_provider)

        self._execution_history: list[UnifiedExecutionResult] = []

    def route_prompt(self, prompt: Prompt) -> str:
        """Determine the best platform for a prompt.

        Args:
            prompt: Prompt to route

        Returns:
            Platform name ("QuASIM", "QStack", or "QNimbus")
        """
        # Check explicit execution layers first
        if prompt.execution_layers:
            for layer in prompt.execution_layers:
                if layer in ["QuASIM", "QStack", "QNimbus"]:
                    return layer

        # Route based on domain
        if self.quasim.can_execute(prompt):
            return "QuASIM"
        elif self.qstack.can_execute(prompt):
            return "QStack"
        elif self.qnimbus.can_execute(prompt):
            return "QNimbus"

        # Default fallback
        return "QNimbus"

    def execute(
        self,
        prompt: Prompt,
        dry_run: bool = False,
        force_platform: str | None = None,
        **kwargs: Any,
    ) -> UnifiedExecutionResult:
        """Execute prompt on the appropriate platform.

        Args:
            prompt: Prompt to execute
            dry_run: If True, simulate execution
            force_platform: Override automatic routing
            **kwargs: Platform-specific parameters

        Returns:
            UnifiedExecutionResult with execution status
        """
        platform = force_platform or self.route_prompt(prompt)

        if platform == "QuASIM":
            result = self.quasim.execute(prompt, dry_run=dry_run, **kwargs)
            unified = UnifiedExecutionResult.from_simulation_result(result)
        elif platform == "QStack":
            result = self.qstack.execute(prompt, dry_run=dry_run, **kwargs)
            unified = UnifiedExecutionResult.from_ml_result(result)
        else:
            result = self.qnimbus.execute(prompt, dry_run=dry_run, **kwargs)
            unified = UnifiedExecutionResult.from_cloud_result(result)

        self._execution_history.append(unified)
        return unified

    def execute_batch(
        self,
        prompts: list[Prompt],
        dry_run: bool = False,
    ) -> list[UnifiedExecutionResult]:
        """Execute multiple prompts with automatic routing.

        Args:
            prompts: List of prompts to execute
            dry_run: If True, simulate execution

        Returns:
            List of UnifiedExecutionResults
        """
        results = []
        for prompt in prompts:
            results.append(self.execute(prompt, dry_run=dry_run))
        return results

    def get_routing_summary(self, prompts: list[Prompt]) -> dict[str, list[int]]:
        """Get routing summary for a set of prompts.

        Args:
            prompts: List of prompts to analyze

        Returns:
            Dictionary mapping platforms to prompt IDs
        """
        routing: dict[str, list[int]] = {
            "QuASIM": [],
            "QStack": [],
            "QNimbus": [],
        }

        for prompt in prompts:
            platform = self.route_prompt(prompt)
            routing[platform].append(prompt.id)

        return routing

    def get_execution_stats(self) -> dict[str, Any]:
        """Get combined execution statistics.

        Returns:
            Dictionary with execution metrics from all platforms
        """
        successful = sum(1 for r in self._execution_history if r.is_successful())
        total = len(self._execution_history)

        platform_counts: dict[str, int] = {}
        for result in self._execution_history:
            platform_counts[result.platform] = platform_counts.get(result.platform, 0) + 1

        return {
            "total_executions": total,
            "successful_executions": successful,
            "success_rate": (successful / total * 100) if total > 0 else 0.0,
            "executions_by_platform": platform_counts,
            "quasim_stats": self.quasim.get_execution_stats(),
            "qstack_stats": self.qstack.get_execution_stats(),
            "qnimbus_stats": self.qnimbus.get_execution_stats(),
        }
