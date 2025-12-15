"""QuASIM Ansys Performance Evaluation Framework.

This package provides automated benchmark execution and performance comparison
between Ansys Mechanical and QuASIM solvers.
"""

from .performance_runner import (
    # Executor classes
    AnsysBaselineExecutor,
    # Data structures
    BenchmarkDefinition,
    BenchmarkResult,
    ComparisonResult,
    # Analysis classes
    PerformanceComparer,
    QuasimExecutor,
    ReportGenerator,
)

__version__ = "1.0.0"
__all__ = [
    "BenchmarkDefinition",
    "BenchmarkResult",
    "ComparisonResult",
    "AnsysBaselineExecutor",
    "QuasimExecutor",
    "PerformanceComparer",
    "ReportGenerator",
]
