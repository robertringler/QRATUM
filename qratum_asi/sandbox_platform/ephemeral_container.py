"""Ephemeral Container Management for Lightweight Execution.

Implements ephemeral execution with minimal container overhead
and auto-destroy after evaluation.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable

from qradle.merkle import MerkleChain


class ContainerLifecyclePhase(Enum):
    """Phase of container lifecycle."""

    INITIALIZING = "initializing"
    READY = "ready"
    EXECUTING = "executing"
    COMPLETING = "completing"
    DESTROYING = "destroying"
    DESTROYED = "destroyed"


class AutoDestroyPolicy(Enum):
    """Policy for automatic container destruction."""

    IMMEDIATE = "immediate"  # Destroy immediately after execution
    DELAYED = "delayed"  # Destroy after a delay
    ON_IDLE = "on_idle"  # Destroy when idle
    MANUAL = "manual"  # Manual destruction only


@dataclass
class ContainerLifecycle:
    """Lifecycle tracking for an ephemeral container.

    Attributes:
        container_id: Unique container identifier
        phase: Current lifecycle phase
        created_at: Creation timestamp
        started_at: Execution start timestamp
        completed_at: Completion timestamp
        destroyed_at: Destruction timestamp
        auto_destroy_policy: Auto-destruction policy
        destroy_delay_ms: Delay before destruction (for DELAYED policy)
    """

    container_id: str
    phase: ContainerLifecyclePhase = ContainerLifecyclePhase.INITIALIZING
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    started_at: str | None = None
    completed_at: str | None = None
    destroyed_at: str | None = None
    auto_destroy_policy: AutoDestroyPolicy = AutoDestroyPolicy.IMMEDIATE
    destroy_delay_ms: int = 0
    execution_count: int = 0
    total_execution_time_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize lifecycle."""
        return {
            "container_id": self.container_id,
            "phase": self.phase.value,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "destroyed_at": self.destroyed_at,
            "auto_destroy_policy": self.auto_destroy_policy.value,
            "destroy_delay_ms": self.destroy_delay_ms,
            "execution_count": self.execution_count,
            "total_execution_time_ms": self.total_execution_time_ms,
        }


@dataclass
class ContainerMetrics:
    """Metrics for an ephemeral container.

    Attributes:
        container_id: Container identifier
        startup_time_ms: Time to start container
        execution_time_ms: Time spent executing
        teardown_time_ms: Time to tear down
        memory_peak_mb: Peak memory usage
        cpu_peak_percent: Peak CPU usage
    """

    container_id: str
    startup_time_ms: float = 0.0
    execution_time_ms: float = 0.0
    teardown_time_ms: float = 0.0
    memory_peak_mb: float = 0.0
    cpu_peak_percent: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Serialize metrics."""
        return {
            "container_id": self.container_id,
            "startup_time_ms": self.startup_time_ms,
            "execution_time_ms": self.execution_time_ms,
            "teardown_time_ms": self.teardown_time_ms,
            "total_overhead_ms": self.startup_time_ms + self.teardown_time_ms,
            "memory_peak_mb": self.memory_peak_mb,
            "cpu_peak_percent": self.cpu_peak_percent,
            "timestamp": self.timestamp,
        }


class EphemeralContainer:
    """Ephemeral container for lightweight sandbox execution.

    Provides:
    - Minimal startup/teardown overhead
    - Automatic resource cleanup
    - Lifecycle tracking
    - Metrics collection
    """

    def __init__(
        self,
        container_id: str,
        auto_destroy_policy: AutoDestroyPolicy = AutoDestroyPolicy.IMMEDIATE,
        destroy_delay_ms: int = 0,
        memory_limit_mb: int = 256,
        merkle_chain: MerkleChain | None = None,
    ):
        """Initialize ephemeral container.

        Args:
            container_id: Unique container identifier
            auto_destroy_policy: Policy for automatic destruction
            destroy_delay_ms: Delay before destruction
            memory_limit_mb: Memory limit in MB
            merkle_chain: Merkle chain for audit trail
        """
        self.container_id = container_id
        self.memory_limit_mb = memory_limit_mb
        self.merkle_chain = merkle_chain or MerkleChain()

        # Lifecycle tracking
        self.lifecycle = ContainerLifecycle(
            container_id=container_id,
            auto_destroy_policy=auto_destroy_policy,
            destroy_delay_ms=destroy_delay_ms,
        )

        # Metrics
        self.metrics = ContainerMetrics(container_id=container_id)

        # State
        self._lock = threading.RLock()
        self._destroy_timer: threading.Timer | None = None
        self._execution_context: dict[str, Any] = {}

        # Initialize
        self._initialize()

    def _initialize(self) -> None:
        """Initialize the container."""
        start_time = time.perf_counter()

        # Simulate container initialization
        self._execution_context = {
            "initialized": True,
            "memory_limit_mb": self.memory_limit_mb,
        }

        self.lifecycle.phase = ContainerLifecyclePhase.READY
        self.metrics.startup_time_ms = (time.perf_counter() - start_time) * 1000

        # Log initialization
        self.merkle_chain.add_event(
            "ephemeral_container_initialized",
            {
                "container_id": self.container_id,
                "startup_time_ms": self.metrics.startup_time_ms,
            },
        )

    def execute(
        self,
        func: Callable[[dict[str, Any]], Any],
        context: dict[str, Any] | None = None,
    ) -> Any:
        """Execute a function in the ephemeral container.

        Args:
            func: Function to execute
            context: Execution context

        Returns:
            Function result
        """
        with self._lock:
            if self.lifecycle.phase != ContainerLifecyclePhase.READY:
                raise RuntimeError(f"Container not ready: {self.lifecycle.phase.value}")

            self.lifecycle.phase = ContainerLifecyclePhase.EXECUTING
            self.lifecycle.started_at = datetime.now(timezone.utc).isoformat()

        start_time = time.perf_counter()

        try:
            # Merge context
            exec_context = {**self._execution_context, **(context or {})}

            # Execute function
            result = func(exec_context)

            execution_time = (time.perf_counter() - start_time) * 1000
            self.metrics.execution_time_ms += execution_time
            self.lifecycle.execution_count += 1
            self.lifecycle.total_execution_time_ms += execution_time

            return result

        finally:
            with self._lock:
                self.lifecycle.phase = ContainerLifecyclePhase.COMPLETING
                self.lifecycle.completed_at = datetime.now(timezone.utc).isoformat()

            # Log execution
            self.merkle_chain.add_event(
                "ephemeral_container_executed",
                {
                    "container_id": self.container_id,
                    "execution_time_ms": self.metrics.execution_time_ms,
                },
            )

            # Handle auto-destroy
            self._handle_auto_destroy()

    def _handle_auto_destroy(self) -> None:
        """Handle automatic destruction based on policy."""
        policy = self.lifecycle.auto_destroy_policy

        if policy == AutoDestroyPolicy.IMMEDIATE:
            self.destroy()
        elif policy == AutoDestroyPolicy.DELAYED:
            self._schedule_destroy(self.lifecycle.destroy_delay_ms)
        elif policy == AutoDestroyPolicy.ON_IDLE:
            self.lifecycle.phase = ContainerLifecyclePhase.READY
        # MANUAL policy does nothing

    def _schedule_destroy(self, delay_ms: int) -> None:
        """Schedule destruction after delay."""
        self._cancel_destroy_timer()
        self._destroy_timer = threading.Timer(
            delay_ms / 1000.0,
            self.destroy,
        )
        self._destroy_timer.start()

    def _cancel_destroy_timer(self) -> None:
        """Cancel scheduled destruction."""
        if self._destroy_timer:
            self._destroy_timer.cancel()
            self._destroy_timer = None

    def destroy(self) -> None:
        """Destroy the container and release resources."""
        with self._lock:
            if self.lifecycle.phase == ContainerLifecyclePhase.DESTROYED:
                return

            self.lifecycle.phase = ContainerLifecyclePhase.DESTROYING

        start_time = time.perf_counter()

        # Cancel any scheduled destruction
        self._cancel_destroy_timer()

        # Clear execution context
        self._execution_context.clear()

        self.lifecycle.phase = ContainerLifecyclePhase.DESTROYED
        self.lifecycle.destroyed_at = datetime.now(timezone.utc).isoformat()
        self.metrics.teardown_time_ms = (time.perf_counter() - start_time) * 1000

        # Log destruction
        self.merkle_chain.add_event(
            "ephemeral_container_destroyed",
            {
                "container_id": self.container_id,
                "teardown_time_ms": self.metrics.teardown_time_ms,
                "total_execution_time_ms": self.lifecycle.total_execution_time_ms,
            },
        )

    @property
    def is_alive(self) -> bool:
        """Check if container is still alive."""
        return self.lifecycle.phase not in (
            ContainerLifecyclePhase.DESTROYING,
            ContainerLifecyclePhase.DESTROYED,
        )

    def get_stats(self) -> dict[str, Any]:
        """Get container statistics."""
        return {
            "container_id": self.container_id,
            "lifecycle": self.lifecycle.to_dict(),
            "metrics": self.metrics.to_dict(),
            "is_alive": self.is_alive,
        }


class EphemeralContainerPool:
    """Pool of ephemeral containers for efficient reuse.

    Manages a pool of containers to reduce initialization overhead
    while maintaining ephemeral semantics.
    """

    def __init__(
        self,
        pool_id: str = "ephemeral_pool",
        pool_size: int = 10,
        auto_destroy_policy: AutoDestroyPolicy = AutoDestroyPolicy.ON_IDLE,
        memory_limit_mb: int = 256,
        merkle_chain: MerkleChain | None = None,
    ):
        """Initialize container pool.

        Args:
            pool_id: Unique pool identifier
            pool_size: Maximum pool size
            auto_destroy_policy: Default auto-destroy policy
            memory_limit_mb: Default memory limit
            merkle_chain: Merkle chain for audit trail
        """
        self.pool_id = pool_id
        self.pool_size = pool_size
        self.auto_destroy_policy = auto_destroy_policy
        self.memory_limit_mb = memory_limit_mb
        self.merkle_chain = merkle_chain or MerkleChain()

        self._available: list[EphemeralContainer] = []
        self._in_use: dict[str, EphemeralContainer] = {}
        self._container_counter = 0
        self._lock = threading.RLock()

        # Statistics
        self._total_created = 0
        self._total_destroyed = 0
        self._total_executions = 0

    def acquire(self) -> EphemeralContainer:
        """Acquire a container from the pool.

        Returns:
            Available or newly created container
        """
        with self._lock:
            # Try to get from available pool
            while self._available:
                container = self._available.pop()
                if (
                    container.is_alive
                    and container.lifecycle.phase == ContainerLifecyclePhase.READY
                ):
                    self._in_use[container.container_id] = container
                    return container

            # Create new container
            self._container_counter += 1
            container_id = f"container_{self.pool_id}_{self._container_counter:06d}"

            container = EphemeralContainer(
                container_id=container_id,
                auto_destroy_policy=AutoDestroyPolicy.MANUAL,  # Pool manages lifecycle
                memory_limit_mb=self.memory_limit_mb,
                merkle_chain=self.merkle_chain,
            )

            self._in_use[container.container_id] = container
            self._total_created += 1

            return container

    def release(self, container: EphemeralContainer) -> None:
        """Release a container back to the pool.

        Args:
            container: Container to release
        """
        with self._lock:
            self._in_use.pop(container.container_id, None)

            if container.is_alive and len(self._available) < self.pool_size:
                container.lifecycle.phase = ContainerLifecyclePhase.READY
                self._available.append(container)
            else:
                container.destroy()
                self._total_destroyed += 1

            self._total_executions += container.lifecycle.execution_count

    def execute(
        self,
        func: Callable[[dict[str, Any]], Any],
        context: dict[str, Any] | None = None,
    ) -> Any:
        """Execute a function using a pooled container.

        Args:
            func: Function to execute
            context: Execution context

        Returns:
            Function result
        """
        container = self.acquire()
        try:
            return container.execute(func, context)
        finally:
            self.release(container)

    def clear(self) -> None:
        """Clear all containers in the pool."""
        with self._lock:
            for container in self._available:
                container.destroy()
                self._total_destroyed += 1
            self._available.clear()

            for container in list(self._in_use.values()):
                container.destroy()
                self._total_destroyed += 1
            self._in_use.clear()

    def get_pool_stats(self) -> dict[str, Any]:
        """Get pool statistics."""
        return {
            "pool_id": self.pool_id,
            "pool_size": self.pool_size,
            "available_count": len(self._available),
            "in_use_count": len(self._in_use),
            "total_created": self._total_created,
            "total_destroyed": self._total_destroyed,
            "total_executions": self._total_executions,
            "utilization": (len(self._in_use) / self.pool_size if self.pool_size > 0 else 0),
        }
