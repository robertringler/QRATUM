"""QuASIM Ansys Performance Evaluation Framework.

This package provides automated benchmark execution and performance comparison
between Ansys Mechanical and QuASIM solvers.
"""

from .performance_runner import (
    # Data structures
    BenchmarkDefinition,
    BenchmarkResult,
    ComparisonResult,
    
    # Executor classes
    AnsysBaselineExecutor,
    QuasimExecutor,
    
    # Analysis classes
    PerformanceComparer,
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
