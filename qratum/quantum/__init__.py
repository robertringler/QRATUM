"""Quantum integration module for QRATUM platform.

This module provides integration adapters connecting quantum modules
to the main QRATUM platform.
"""

from __future__ import annotations

from .integration import QuantumModuleAdapter

__all__ = ["QuantumModuleAdapter"]
"""QRATUM Quantum Integration Layer.

Provides adapters for quasim.quantum modules.
"""

from .integration import QuantumBackendAdapter

__all__ = ["QuantumBackendAdapter"]
