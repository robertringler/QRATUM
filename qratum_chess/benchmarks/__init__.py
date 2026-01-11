"""Benchmarking and Load-Testing Protocol for QRATUM-Chess (Stage IV).

Implements comprehensive performance certification framework:
1. Performance metrics (nodes/sec, latency, hash hit rate)
2. Strategic torture suite (zugzwang, fortress, trapped pieces)
3. Engine adversarial gauntlet
4. Elo certification protocol
5. Load-failure injection and recovery testing
6. Telemetry output (heatmaps, distributions, entropy curves)
7. Stage III certification verification
8. Kaggle Chess Leaderboard integration
"""

from __future__ import annotations

from qratum_chess.benchmarks.runner import (
    BenchmarkRunner,
    BenchmarkConfig,
    BenchmarkSummary,
    CertificationResult,
)
from qratum_chess.benchmarks.metrics import PerformanceMetrics
from qratum_chess.benchmarks.torture import StrategicTortureSuite
from qratum_chess.benchmarks.gauntlet import AdversarialGauntlet
from qratum_chess.benchmarks.elo import EloCertification
from qratum_chess.benchmarks.resilience import ResilienceTest
from qratum_chess.benchmarks.telemetry import TelemetryOutput
from qratum_chess.benchmarks.kaggle_config import KaggleConfig, load_config
from qratum_chess.benchmarks.kaggle_integration import (
    KaggleIntegration,
    KaggleBenchmarkPosition,
    KaggleLeaderboardData,
)
from qratum_chess.benchmarks.kaggle_submission import (
    KaggleSubmission,
    SubmissionResult,
from qratum_chess.benchmarks.kaggle_integration import (
    KaggleLeaderboardLoader,
    KaggleLeaderboard,
    KaggleBenchmarkPosition,
    KaggleSubmission,
    download_kaggle_leaderboard,
)
from qratum_chess.benchmarks.benchmark_kaggle import (
    KaggleBenchmarkRunner,
    KaggleBenchmarkResult,
    KaggleBenchmarkSummary,
)

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
    "KaggleConfig",
    "load_config",
    "KaggleIntegration",
    "KaggleBenchmarkPosition",
    "KaggleLeaderboardData",
    "KaggleSubmission",
    "SubmissionResult",
    "KaggleLeaderboardLoader",
    "KaggleLeaderboard",
    "KaggleBenchmarkPosition",
    "KaggleSubmission",
    "KaggleBenchmarkRunner",
    "KaggleBenchmarkResult",
    "KaggleBenchmarkSummary",
    "download_kaggle_leaderboard",
]
