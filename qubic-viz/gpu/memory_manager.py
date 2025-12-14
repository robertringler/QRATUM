"""GPU memory management."""

from __future__ import annotations

from typing import Optional


class GPUMemoryManager:
    """Manage GPU memory allocation and limits.

    Args:
        memory_limit_mb: Memory limit in MB
    """

    def __init__(self, memory_limit_mb: int = 4096) -> None:
        """Initialize memory manager."""
        self.memory_limit_bytes = memory_limit_mb * 1024 * 1024
        self.allocated_bytes = 0
        self.allocations = {}

    def allocate(self, name: str, size_bytes: int) -> bool:
        """Allocate memory.

        Args:
            name: Allocation name
            size_bytes: Size in bytes

        Returns:
            True if allocation succeeded
        """
        if self.allocated_bytes + size_bytes > self.memory_limit_bytes:
            return False

        self.allocations[name] = size_bytes
        self.allocated_bytes += size_bytes
        return True

    def free(self, name: str) -> None:
        """Free allocated memory.

        Args:
            name: Allocation name
        """
        if name in self.allocations:
            self.allocated_bytes -= self.allocations[name]
            del self.allocations[name]

    def can_allocate(self, size_bytes: int) -> bool:
        """Check if allocation is possible.

        Args:
            size_bytes: Size in bytes

        Returns:
            True if allocation is possible
        """
        return self.allocated_bytes + size_bytes <= self.memory_limit_bytes

    def get_available_memory(self) -> int:
        """Get available memory in bytes.

        Returns:
            Available memory in bytes
        """
        return self.memory_limit_bytes - self.allocated_bytes

    def get_utilization(self) -> float:
        """Get memory utilization percentage.

        Returns:
            Utilization as 0-1
        """
        return self.allocated_bytes / self.memory_limit_bytes

    def clear(self) -> None:
        """Clear all allocations."""
        self.allocations = {}
        self.allocated_bytes = 0

    def get_stats(self) -> dict[str, int]:
        """Get memory statistics.

        Returns:
            Dictionary with memory stats
        """
        return {
            "limit_mb": self.memory_limit_bytes // (1024 * 1024),
            "allocated_mb": self.allocated_bytes // (1024 * 1024),
            "available_mb": self.get_available_memory() // (1024 * 1024),
            "utilization_percent": int(self.get_utilization() * 100),
            "num_allocations": len(self.allocations),
        }
