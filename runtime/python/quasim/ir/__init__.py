"""MLIR/StableHLO intermediate representation layer for QuASIM.

This module provides unified IR for cross-backend compilation targeting
CUDA, HIP, Triton, and CPU backends.
"""
from __future__ import annotations

from .ir_builder import IRBuilder, IRNode, IRType
from .lowering import lower_to_backend, Backend

__all__ = [
    "IRBuilder",
    "IRNode",
    "IRType",
    "lower_to_backend",
    "Backend",
]
