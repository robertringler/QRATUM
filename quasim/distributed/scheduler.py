"""Task scheduling for distributed execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class TaskPriority(Enum):
    """Task priority levels for scheduling."""

    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


class TaskStatus(Enum):
    """Task execution status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Task:
    """Represents a schedulable task.

    Attributes:
        task_id: Unique task identifier
        func_name: Name of function to execute
        args: Function arguments
        kwargs: Function keyword arguments
        priority: Task priority
        status: Current task status
        result: Task result (when completed)
        gpu_required: Whether task requires GPU
    """

    task_id: str
    func_name: str
    args: tuple[Any, ...] = field(default_factory=tuple)
    kwargs: dict[str, Any] = field(default_factory=dict)
    priority: TaskPriority = TaskPriority.NORMAL
    status: TaskStatus = TaskStatus.PENDING
    result: Any = None
    gpu_required: bool = True


@dataclass
class TaskScheduler:
    """Intelligent task scheduler for distributed GPU workloads.

    Schedules tasks across GPU workers based on:
    - Task priority
    - GPU availability
    - Data locality
    - Load balancing

    Attributes:
        max_concurrent_tasks: Maximum tasks to run concurrently
        gpu_memory_limit_gb: GPU memory limit per worker
        enable_preemption: Whether to allow task preemption
    """

    max_concurrent_tasks: int = 16
    gpu_memory_limit_gb: int = 32
    enable_preemption: bool = False
    _task_queue: list[Task] = field(default_factory=list)
    _running_tasks: dict[str, Task] = field(default_factory=dict)

    def submit(self, task: Task) -> None:
        """Submit a task for scheduling.

        Args:
            task: Task to schedule
        """
        self._task_queue.append(task)
        self._sort_queue()

    def _sort_queue(self) -> None:
        """Sort task queue by priority."""
        self._task_queue.sort(key=lambda t: t.priority.value, reverse=True)

    def schedule_next(self) -> Task | None:
        """Schedule the next task for execution.

        Returns:
            Next task to execute, or None if queue is empty or at capacity
        """
        # Check if we can schedule more tasks
        if len(self._running_tasks) >= self.max_concurrent_tasks:
            return None

        # Get highest priority pending task
        for task in self._task_queue:
            if task.status == TaskStatus.PENDING:
                # Check GPU availability
                if task.gpu_required and not self._has_available_gpu():
                    continue

                # Mark as running
                task.status = TaskStatus.RUNNING
                self._running_tasks[task.task_id] = task
                self._task_queue.remove(task)
                return task

        return None

    def _has_available_gpu(self) -> bool:
        """Check if GPU resources are available.

        Returns:
            True if GPU is available
        """
        # Simplified - production would query actual GPU availability
        gpu_tasks = sum(1 for t in self._running_tasks.values() if t.gpu_required)
        return gpu_tasks < self.max_concurrent_tasks

    def complete_task(self, task_id: str, result: Any) -> None:
        """Mark a task as completed.

        Args:
            task_id: ID of completed task
            result: Task result
        """
        if task_id in self._running_tasks:
            task = self._running_tasks[task_id]
            task.status = TaskStatus.COMPLETED
            task.result = result
            del self._running_tasks[task_id]

    def fail_task(self, task_id: str, error: str) -> None:
        """Mark a task as failed.

        Args:
            task_id: ID of failed task
            error: Error message
        """
        if task_id in self._running_tasks:
            task = self._running_tasks[task_id]
            task.status = TaskStatus.FAILED
            task.result = {"error": error}
            del self._running_tasks[task_id]

    def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending or running task.

        Args:
            task_id: ID of task to cancel

        Returns:
            True if task was cancelled
        """
        # Check running tasks
        if task_id in self._running_tasks:
            if self.enable_preemption:
                task = self._running_tasks[task_id]
                task.status = TaskStatus.CANCELLED
                del self._running_tasks[task_id]
                return True
            return False

        # Check pending tasks
        for task in self._task_queue:
            if task.task_id == task_id:
                task.status = TaskStatus.CANCELLED
                self._task_queue.remove(task)
                return True

        return False

    def get_queue_status(self) -> dict[str, Any]:
        """Get current queue statistics.

        Returns:
            Dictionary with queue metrics
        """
        return {
            "pending": len([t for t in self._task_queue if t.status == TaskStatus.PENDING]),
            "running": len(self._running_tasks),
            "total_capacity": self.max_concurrent_tasks,
            "utilization": len(self._running_tasks) / self.max_concurrent_tasks,
        }
