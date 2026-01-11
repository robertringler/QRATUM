"""Resource Throttler for Dynamic Resource Management.

Implements dynamic resource throttling that adapts sandbox compute
based on production load.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable

from qradle.merkle import MerkleChain


class ThrottleLevel(Enum):
    """Level of resource throttling."""

    NONE = "none"  # No throttling
    LIGHT = "light"  # Light throttling (75% resources)
    MODERATE = "moderate"  # Moderate throttling (50% resources)
    HEAVY = "heavy"  # Heavy throttling (25% resources)
    CRITICAL = "critical"  # Minimal resources (10%)


class ThrottlePolicy(Enum):
    """Policy for throttle decisions."""

    ADAPTIVE = "adaptive"  # Adapt based on load
    FIXED = "fixed"  # Fixed throttle level
    SCHEDULED = "scheduled"  # Time-based scheduling
    MANUAL = "manual"  # Manual control only


@dataclass
class LoadMetrics:
    """Metrics for production load.

    Attributes:
        metric_id: Unique metric identifier
        cpu_utilization: CPU utilization (0-1)
        memory_utilization: Memory utilization (0-1)
        io_utilization: I/O utilization (0-1)
        queue_depth: Current queue depth
        latency_ms: Current latency
        throughput: Current throughput
    """

    metric_id: str
    cpu_utilization: float = 0.0
    memory_utilization: float = 0.0
    io_utilization: float = 0.0
    queue_depth: int = 0
    latency_ms: float = 0.0
    throughput: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def overall_load(self) -> float:
        """Compute overall load score (0-1)."""
        return max(
            self.cpu_utilization,
            self.memory_utilization,
            self.io_utilization,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize load metrics."""
        return {
            "metric_id": self.metric_id,
            "cpu_utilization": self.cpu_utilization,
            "memory_utilization": self.memory_utilization,
            "io_utilization": self.io_utilization,
            "queue_depth": self.queue_depth,
            "latency_ms": self.latency_ms,
            "throughput": self.throughput,
            "overall_load": self.overall_load,
            "timestamp": self.timestamp,
        }


@dataclass
class ThrottleState:
    """Current throttle state.

    Attributes:
        state_id: Unique state identifier
        level: Current throttle level
        resource_multiplier: Resource multiplier (0-1)
        reason: Reason for current state
        since: When state was entered
    """

    state_id: str
    level: ThrottleLevel
    resource_multiplier: float
    reason: str = ""
    since: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Serialize throttle state."""
        return {
            "state_id": self.state_id,
            "level": self.level.value,
            "resource_multiplier": self.resource_multiplier,
            "reason": self.reason,
            "since": self.since,
        }


@dataclass
class ResourceBudget:
    """Resource budget for sandbox operations.

    Attributes:
        cpu_cores: Available CPU cores
        memory_mb: Available memory in MB
        io_bandwidth: Available I/O bandwidth (0-1)
        concurrent_operations: Maximum concurrent operations
    """

    cpu_cores: float = 4.0
    memory_mb: int = 2048
    io_bandwidth: float = 1.0
    concurrent_operations: int = 10

    def apply_multiplier(self, multiplier: float) -> "ResourceBudget":
        """Apply throttle multiplier to budget.

        Args:
            multiplier: Multiplier to apply (0-1)

        Returns:
            New ResourceBudget with multiplier applied
        """
        return ResourceBudget(
            cpu_cores=max(0.5, self.cpu_cores * multiplier),
            memory_mb=max(256, int(self.memory_mb * multiplier)),
            io_bandwidth=max(0.1, self.io_bandwidth * multiplier),
            concurrent_operations=max(1, int(self.concurrent_operations * multiplier)),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize resource budget."""
        return {
            "cpu_cores": self.cpu_cores,
            "memory_mb": self.memory_mb,
            "io_bandwidth": self.io_bandwidth,
            "concurrent_operations": self.concurrent_operations,
        }


class ResourceThrottler:
    """Dynamic resource throttler for sandbox operations.

    Adapts sandbox compute resources based on production load
    to ensure production is never impacted by sandbox operations.
    """

    def __init__(
        self,
        throttler_id: str = "resource_throttler",
        policy: ThrottlePolicy = ThrottlePolicy.ADAPTIVE,
        base_budget: ResourceBudget | None = None,
        merkle_chain: MerkleChain | None = None,
    ):
        """Initialize resource throttler.

        Args:
            throttler_id: Unique throttler identifier
            policy: Throttle policy
            base_budget: Base resource budget
            merkle_chain: Merkle chain for audit trail
        """
        self.throttler_id = throttler_id
        self.policy = policy
        self.base_budget = base_budget or ResourceBudget()
        self.merkle_chain = merkle_chain or MerkleChain()

        # State
        self._current_state: ThrottleState | None = None
        self._state_counter = 0
        self._metric_counter = 0
        self._lock = threading.RLock()

        # Load history
        self._load_history: list[LoadMetrics] = []
        self._max_history = 100

        # Thresholds
        self._thresholds = {
            ThrottleLevel.NONE: 0.0,
            ThrottleLevel.LIGHT: 0.5,
            ThrottleLevel.MODERATE: 0.7,
            ThrottleLevel.HEAVY: 0.85,
            ThrottleLevel.CRITICAL: 0.95,
        }

        # Multipliers
        self._multipliers = {
            ThrottleLevel.NONE: 1.0,
            ThrottleLevel.LIGHT: 0.75,
            ThrottleLevel.MODERATE: 0.50,
            ThrottleLevel.HEAVY: 0.25,
            ThrottleLevel.CRITICAL: 0.10,
        }

        # Load provider
        self._load_provider: Callable[[], LoadMetrics] | None = None

        # Initialize state
        self._set_throttle_level(ThrottleLevel.NONE, "initial")

        # Log initialization
        self.merkle_chain.add_event(
            "resource_throttler_initialized",
            {
                "throttler_id": throttler_id,
                "policy": policy.value,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )

    def set_load_provider(
        self,
        provider: Callable[[], LoadMetrics],
    ) -> None:
        """Set the load metrics provider.

        Args:
            provider: Function that returns current load metrics
        """
        self._load_provider = provider

    def update_load(self, metrics: LoadMetrics | None = None) -> ThrottleState:
        """Update load metrics and adjust throttling.

        Args:
            metrics: Load metrics (uses provider if not specified)

        Returns:
            Current ThrottleState
        """
        with self._lock:
            # Get metrics
            if metrics is None and self._load_provider:
                metrics = self._load_provider()

            if metrics is None:
                metrics = self._create_default_metrics()

            # Store in history
            self._load_history.append(metrics)
            if len(self._load_history) > self._max_history:
                self._load_history.pop(0)

            # Determine throttle level based on policy
            if self.policy == ThrottlePolicy.ADAPTIVE:
                self._adapt_throttle(metrics)
            elif self.policy == ThrottlePolicy.MANUAL:
                pass  # Manual control, don't change

            return self._current_state

    def _adapt_throttle(self, metrics: LoadMetrics) -> None:
        """Adapt throttle level based on load."""
        load = metrics.overall_load

        # Determine appropriate level
        new_level = ThrottleLevel.NONE
        for level in [
            ThrottleLevel.CRITICAL,
            ThrottleLevel.HEAVY,
            ThrottleLevel.MODERATE,
            ThrottleLevel.LIGHT,
        ]:
            if load >= self._thresholds[level]:
                new_level = level
                break

        # Update if changed
        if self._current_state and new_level != self._current_state.level:
            self._set_throttle_level(
                new_level,
                f"load={load:.2f}",
            )

    def _set_throttle_level(
        self,
        level: ThrottleLevel,
        reason: str,
    ) -> None:
        """Set the throttle level.

        Args:
            level: New throttle level
            reason: Reason for change
        """
        self._state_counter += 1
        state_id = f"state_{self.throttler_id}_{self._state_counter:06d}"

        self._current_state = ThrottleState(
            state_id=state_id,
            level=level,
            resource_multiplier=self._multipliers[level],
            reason=reason,
        )

        # Log state change
        self.merkle_chain.add_event(
            "throttle_level_changed",
            {
                "state_id": state_id,
                "level": level.value,
                "multiplier": self._multipliers[level],
                "reason": reason,
            },
        )

    def set_throttle_level(
        self,
        level: ThrottleLevel,
        reason: str = "manual",
    ) -> ThrottleState:
        """Manually set the throttle level.

        Args:
            level: New throttle level
            reason: Reason for change

        Returns:
            New ThrottleState
        """
        with self._lock:
            self._set_throttle_level(level, reason)
            return self._current_state

    def get_available_budget(self) -> ResourceBudget:
        """Get currently available resource budget.

        Returns:
            ResourceBudget with throttling applied
        """
        if self._current_state is None:
            return self.base_budget

        return self.base_budget.apply_multiplier(
            self._current_state.resource_multiplier
        )

    def _create_default_metrics(self) -> LoadMetrics:
        """Create default load metrics."""
        self._metric_counter += 1
        return LoadMetrics(
            metric_id=f"metric_{self.throttler_id}_{self._metric_counter:08d}",
            cpu_utilization=0.3,
            memory_utilization=0.4,
            io_utilization=0.2,
        )

    def get_current_state(self) -> ThrottleState | None:
        """Get current throttle state."""
        return self._current_state

    def get_load_history(self, limit: int = 10) -> list[LoadMetrics]:
        """Get recent load history.

        Args:
            limit: Maximum entries to return

        Returns:
            List of recent LoadMetrics
        """
        return self._load_history[-limit:]

    def should_defer(self, resource_requirement: float = 0.1) -> bool:
        """Check if operation should be deferred due to load.

        Args:
            resource_requirement: Required resources (0-1)

        Returns:
            True if operation should be deferred
        """
        if self._current_state is None:
            return False

        available = self._current_state.resource_multiplier
        return resource_requirement > available

    def wait_for_resources(
        self,
        resource_requirement: float = 0.1,
        timeout_seconds: float = 30.0,
        check_interval: float = 1.0,
    ) -> bool:
        """Wait until resources are available.

        Args:
            resource_requirement: Required resources (0-1)
            timeout_seconds: Maximum wait time
            check_interval: Interval between checks

        Returns:
            True if resources became available
        """
        start_time = time.time()

        while time.time() - start_time < timeout_seconds:
            self.update_load()

            if not self.should_defer(resource_requirement):
                return True

            time.sleep(check_interval)

        return False

    def get_throttler_stats(self) -> dict[str, Any]:
        """Get throttler statistics."""
        current_budget = self.get_available_budget()

        level_counts: dict[str, int] = {}
        # Count level transitions (simplified)
        if self._current_state:
            level_counts[self._current_state.level.value] = 1

        return {
            "throttler_id": self.throttler_id,
            "policy": self.policy.value,
            "current_state": (
                self._current_state.to_dict() if self._current_state else None
            ),
            "base_budget": self.base_budget.to_dict(),
            "current_budget": current_budget.to_dict(),
            "load_history_size": len(self._load_history),
            "latest_load": (
                self._load_history[-1].to_dict() if self._load_history else None
            ),
            "level_counts": level_counts,
        }
