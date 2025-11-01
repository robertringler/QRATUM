"""Meta-compilation cache for storing compiled kernels with versioning.

This module implements neural kernel fusion and learned cost models for
automatic kernel grouping and optimization.
"""
from __future__ import annotations

from .cache_manager import CacheManager, CacheEntry
from .fusion_engine import FusionEngine, KernelGraph

__all__ = [
    "CacheManager",
    "CacheEntry",
    "FusionEngine",
    "KernelGraph",
]
