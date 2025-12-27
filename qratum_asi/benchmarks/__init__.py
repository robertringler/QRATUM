"""QRATUM Universal Performance Benchmark Suite.

Provides benchmarks across all cognitive domains to measure
progress toward universal superhuman performance, a key
criterion for superintelligence (SI).

Key Components:
- BenchmarkRegistry: Registry of benchmark tasks across domains
- PerformanceEvaluator: Evaluate system performance
- HumanBaselineComparator: Compare against human expert baselines

Version: 1.0.0
Status: Prototype (SI Transition Phase 4)
"""

from qratum_asi.benchmarks.types import (
    BenchmarkCategory,
    BenchmarkTask,
    BenchmarkResult,
    HumanBaseline,
    PerformanceLevel,
)
from qratum_asi.benchmarks.registry import (
    BenchmarkRegistry,
    TaskDefinition,
)
from qratum_asi.benchmarks.evaluator import (
    PerformanceEvaluator,
    EvaluationSession,
)

__all__ = [
    # Types
    "BenchmarkCategory",
    "BenchmarkTask",
    "BenchmarkResult",
    "HumanBaseline",
    "PerformanceLevel",
    # Registry
    "BenchmarkRegistry",
    "TaskDefinition",
    # Evaluator
    "PerformanceEvaluator",
    "EvaluationSession",
]
