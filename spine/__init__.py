"""Spine - Deterministic Execution Spine for QRATUM.

This module provides the contract executor that dispatches validated
contracts to substrate adapters.

Version: 1.0.0
Status: Production
"""

from __future__ import annotations

from spine.executor import ContractExecutor, ExecutionResult

__all__ = [
    "ContractExecutor",
    "ExecutionResult",
]

__version__ = "1.0.0"
