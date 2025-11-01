"""Python bindings for the GB10 QuASIM runtime."""
from __future__ import annotations

from .runtime import Config, runtime
from .phase2_runtime import Phase2Config, Phase2Runtime

# Phase II modules
from . import ir
from . import meta_cache
from . import async_exec
from . import autotune
from . import hetero
from .adaptive_precision import AdaptivePrecisionManager, PrecisionConfig, PrecisionMode
from .verification import KernelVerifier, VerificationResult
from .visualization import BenchmarkResult, DashboardGenerator

__all__ = [
    "Config",
    "runtime",
    "Phase2Config",
    "Phase2Runtime",
    "ir",
    "meta_cache",
    "async_exec",
    "autotune",
    "hetero",
    "AdaptivePrecisionManager",
    "PrecisionConfig",
    "PrecisionMode",
    "KernelVerifier",
    "VerificationResult",
    "BenchmarkResult",
    "DashboardGenerator",
]
