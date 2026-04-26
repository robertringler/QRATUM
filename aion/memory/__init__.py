"""AION Memory Module.

Implements region-based memory model with:
- Unified regions for stack, heap, thread-local, FPGA on-chip memory
- Lifetime enforcement
- GC, ownership, manual allocation mapping to region inference

Version: 1.0.0
Status: Production
"""

from __future__ import annotations

from .regions import (
    Allocation,
    BorrowChecker,
    MemoryBlock,
    OwnershipTransfer,
    Region,
    RegionKind,
    RegionLifetime,
    RegionManager,
)

__all__ = [
    "Region",
    "RegionKind",
    "RegionLifetime",
    "RegionManager",
    "MemoryBlock",
    "Allocation",
    "BorrowChecker",
    "OwnershipTransfer",
]
