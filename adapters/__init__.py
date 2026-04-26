"""Adapters - Frankenstein Cluster Adapters for QRATUM.

This module provides substrate adapters for heterogeneous cluster execution.
CRITICAL: All adapters contain ZERO policy logic and accept ONLY valid contracts.

Version: 1.0.0
Status: Production
"""

from __future__ import annotations

from adapters.base import AdapterError, BaseAdapter, ExecutionProof
from adapters.cerebras import CerebrasAdapter
from adapters.cpu import CPUAdapter
from adapters.gaudi import Gaudi3Adapter
from adapters.gb200 import GB200Adapter
from adapters.ipu import IPUAdapter
from adapters.mi300x import MI300XAdapter
from adapters.qpu import QPUAdapter
from adapters.registry import (
    AdapterRegistry,
    get_adapter,
    get_global_adapter_registry,
)

__all__ = [
    # Base
    "BaseAdapter",
    "AdapterError",
    "ExecutionProof",
    # Adapters
    "CerebrasAdapter",
    "GB200Adapter",
    "MI300XAdapter",
    "QPUAdapter",
    "IPUAdapter",
    "Gaudi3Adapter",
    "CPUAdapter",
    # Registry
    "AdapterRegistry",
    "get_global_adapter_registry",
    "get_adapter",
]

__version__ = "1.0.0"
