"""Causal profiling with perturbation analysis."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List


@dataclass
class PerturbationResult:
    """Result of a perturbation experiment."""

    function_name: str
    baseline_latency_ms: float
    perturbed_latency_ms: float
    injected_delay_ms: float
    causal_impact: float = 0.0  # Downstream latency shift

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "function_name": self.function_name,
            "baseline_latency_ms": self.baseline_latency_ms,
            "perturbed_latency_ms": self.perturbed_latency_ms,
            "injected_delay_ms": self.injected_delay_ms,
            "causal_impact": self.causal_impact,
        }


@dataclass
class CausalInfluenceMap:
    """Map of causal contributions to total runtime."""

    total_runtime_ms: float
    perturbations: List[PerturbationResult] = field(default_factory=list)
    influence_scores: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "total_runtime_ms": self.total_runtime_ms,
            "perturbations": [p.to_dict() for p in self.perturbations],
            "influence_scores": self.influence_scores,
        }


class CausalProfiler:
    """
    Implements causal profiling using perturbation experiments.
    Injects micro-delays and measures downstream effects.
    """

    def __init__(self, delay_increment_ms: float = 1.0):
        self.delay_increment_ms = delay_increment_ms
        self.results: List[PerturbationResult] = []

    def inject_delay(self, duration_ms: float) -> None:
        """Inject a micro-delay (sleep)."""
        time.sleep(duration_ms / 1000.0)

    def measure_baseline(self, workload: Callable[[], None], repeat: int = 3) -> float:
        """Measure baseline latency without perturbations."""
        latencies = []

        for _ in range(repeat):
            start = time.perf_counter()
            workload()
            end = time.perf_counter()
            latencies.append((end - start) * 1000.0)  # Convert to ms

        return sum(latencies) / len(latencies)

    def measure_with_perturbation(
        self,
        workload: Callable[[], None],
        inject_point: Callable[[], None],
        delay_ms: float,
        repeat: int = 3,
    ) -> float:
        """
        Measure latency with injected delay at a specific point.
        inject_point should be a function that calls inject_delay internally.
        """
        latencies = []

        for _ in range(repeat):
            start = time.perf_counter()
            workload()
            end = time.perf_counter()
            latencies.append((end - start) * 1000.0)

        return sum(latencies) / len(latencies)

    def profile_function(
        self,
        function_name: str,
        baseline_workload: Callable[[], None],
        perturbed_workload: Callable[[], None],
    ) -> PerturbationResult:
        """
        Profile a function by comparing baseline and perturbed execution.
        """
        # Measure baseline
        baseline_latency = self.measure_baseline(baseline_workload)

        # Measure with perturbation
        perturbed_latency = self.measure_baseline(perturbed_workload)

        # Compute causal impact
        causal_impact = perturbed_latency - baseline_latency

        result = PerturbationResult(
            function_name=function_name,
            baseline_latency_ms=baseline_latency,
            perturbed_latency_ms=perturbed_latency,
            injected_delay_ms=self.delay_increment_ms,
            causal_impact=causal_impact,
        )

        self.results.append(result)
        return result

    def compute_influence_map(self, total_runtime_ms: float) -> CausalInfluenceMap:
        """
        Compute causal influence scores for all profiled functions.
        Higher score = more impact on total runtime.
        """
        influence_scores = {}

        for result in self.results:
            # Influence score: ratio of causal impact to injected delay
            # Normalized by total runtime
            if result.injected_delay_ms > 0:
                amplification = result.causal_impact / result.injected_delay_ms
                influence = (amplification * result.injected_delay_ms) / total_runtime_ms
            else:
                influence = 0.0

            influence_scores[result.function_name] = influence

        influence_map = CausalInfluenceMap(
            total_runtime_ms=total_runtime_ms,
            perturbations=self.results,
            influence_scores=influence_scores,
        )

        return influence_map

    def save_influence_map(
        self,
        influence_map: CausalInfluenceMap,
        output_path: str = "profiles/causal/influence_map.json",
    ) -> Path:
        """Save causal influence map to disk."""
        map_path = Path(output_path)
        map_path.parent.mkdir(parents=True, exist_ok=True)

        with open(map_path, "w") as f:
            json.dump(influence_map.to_dict(), f, indent=2)

        return map_path

    def generate_report(self, influence_map: CausalInfluenceMap) -> str:
        """Generate human-readable profiling report."""
        lines = [
            "Causal Profiling Report",
            "=" * 60,
            f"Total Runtime: {influence_map.total_runtime_ms:.2f} ms",
            "",
            "Function Influence Scores (sorted by impact):",
            "-" * 60,
        ]

        # Sort by influence score
        sorted_scores = sorted(
            influence_map.influence_scores.items(),
            key=lambda x: abs(x[1]),
            reverse=True,
        )

        for func_name, score in sorted_scores:
            lines.append(f"  {func_name:30s} {score:+.4f}")

        lines.append("")
        lines.append("Perturbation Details:")
        lines.append("-" * 60)

        for result in influence_map.perturbations:
            lines.append(f"Function: {result.function_name}")
            lines.append(f"  Baseline:   {result.baseline_latency_ms:.2f} ms")
            lines.append(f"  Perturbed:  {result.perturbed_latency_ms:.2f} ms")
            lines.append(f"  Delay:      {result.injected_delay_ms:.2f} ms")
            lines.append(f"  Impact:     {result.causal_impact:+.2f} ms")
            lines.append("")

        return "\n".join(lines)
