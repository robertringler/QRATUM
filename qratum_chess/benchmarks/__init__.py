"""Benchmarking and Load-Testing Protocol for QRATUM-Chess (Stage IV).

Implements comprehensive performance certification framework:
1. Performance metrics (nodes/sec, latency, hash hit rate)
2. Strategic torture suite (zugzwang, fortress, trapped pieces)
3. Engine adversarial gauntlet
4. Elo certification protocol
5. Load-failure injection and recovery testing
6. Telemetry output (heatmaps, distributions, entropy curves)
7. Stage III certification verification
"""

from __future__ import annotations

from qratum_chess.benchmarks.elo import EloCertification
from qratum_chess.benchmarks.gauntlet import AdversarialGauntlet
from qratum_chess.benchmarks.metrics import PerformanceMetrics
from qratum_chess.benchmarks.resilience import ResilienceTest
from qratum_chess.benchmarks.runner import (
    BenchmarkConfig,
    BenchmarkRunner,
    BenchmarkSummary,
    CertificationResult,
)
from qratum_chess.benchmarks.telemetry import TelemetryOutput
from qratum_chess.benchmarks.torture import StrategicTortureSuite

# Kaggle integration imports disabled due to file corruption in kaggle_integration.py
# The file contains duplicate class definitions and mismatched parameters that need cleanup.
# TODO: Fix in separate PR - see code review comments from benchmark PR
# Required fixes:
#   1. Remove duplicate KaggleBenchmarkPosition class definitions
#   2. Resolve parameter name inconsistencies (position_id vs test_id)
#   3. Clean up merged/corrupted docstrings
# from qratum_chess.benchmarks.kaggle_config import KaggleConfig, load_config
# from qratum_chess.benchmarks.kaggle_integration import (...)
# from qratum_chess.benchmarks.kaggle_submission import (...)
# from qratum_chess.benchmarks.benchmark_kaggle import (...)

__all__ = [
    "BenchmarkRunner",
    "BenchmarkConfig",
    "BenchmarkSummary",
    "CertificationResult",
    "PerformanceMetrics",
    "StrategicTortureSuite",
    "AdversarialGauntlet",
    "EloCertification",
    "ResilienceTest",
    "TelemetryOutput",
]
