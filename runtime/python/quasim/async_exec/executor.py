"""Asynchronous task executor with graph-based scheduling."""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, List, Optional


class TaskStatus(Enum):
    """Status of an async task."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class AsyncTask:
    """Represents an asynchronous task in the execution graph."""
    task_id: str
    func: Callable[..., Any]
    args: tuple[Any, ...] = field(default_factory=tuple)
    kwargs: dict[str, Any] = field(default_factory=dict)
    dependencies: List[AsyncTask] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[Any] = None
    error: Optional[Exception] = None
    start_time: float = 0.0
    end_time: float = 0.0
    
    def __hash__(self) -> int:
        return hash(self.task_id)
        
    @property
    def duration(self) -> float:
        """Get task execution duration in seconds."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return 0.0


@dataclass
class ExecutionGraph:
    """Graph of async tasks with dependency tracking."""
    tasks: List[AsyncTask] = field(default_factory=list)
    
    def add_task(
        self,
        task_id: str,
        func: Callable[..., Any],
        dependencies: List[AsyncTask] | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> AsyncTask:
        """Add a task to the execution graph."""
        task = AsyncTask(
            task_id=task_id,
            func=func,
            args=args,
            kwargs=kwargs,
            dependencies=dependencies or [],
        )
        self.tasks.append(task)
        return task
        
    def get_ready_tasks(self) -> List[AsyncTask]:
        """Get tasks that are ready to execute (all dependencies completed)."""
        ready = []
        for task in self.tasks:
            if task.status != TaskStatus.PENDING:
                continue
            if all(dep.status == TaskStatus.COMPLETED for dep in task.dependencies):
                ready.append(task)
        return ready
        
    def is_complete(self) -> bool:
        """Check if all tasks are completed."""
        return all(
            task.status in (TaskStatus.COMPLETED, TaskStatus.FAILED)
            for task in self.tasks
        )


class AsyncExecutor:
    """Asynchronous executor with CUDA/ROCm graph-like scheduling."""
    
    def __init__(self, max_concurrent: int = 4) -> None:
        self.max_concurrent = max_concurrent
        self._execution_history: List[AsyncTask] = []
        self._enable_prefetch = True
        
    async def _execute_task(self, task: AsyncTask) -> None:
        """Execute a single task."""
        task.status = TaskStatus.RUNNING
        task.start_time = time.perf_counter()
        
        try:
            # Simulate async execution
            if asyncio.iscoroutinefunction(task.func):
                task.result = await task.func(*task.args, **task.kwargs)
            else:
                task.result = task.func(*task.args, **task.kwargs)
            task.status = TaskStatus.COMPLETED
        except Exception as e:
            task.error = e
            task.status = TaskStatus.FAILED
        finally:
            task.end_time = time.perf_counter()
            self._execution_history.append(task)
            
    async def execute_graph(self, graph: ExecutionGraph) -> dict[str, Any]:
        """Execute tasks in the graph with maximum parallelism."""
        running_tasks: set[asyncio.Task] = set()
        
        while not graph.is_complete():
            # Get tasks ready to execute
            ready_tasks = graph.get_ready_tasks()
            
            # Launch tasks up to concurrency limit
            while ready_tasks and len(running_tasks) < self.max_concurrent:
                task = ready_tasks.pop(0)
                async_task = asyncio.create_task(self._execute_task(task))
                running_tasks.add(async_task)
                
            # Wait for any task to complete
            if running_tasks:
                done, running_tasks = await asyncio.wait(
                    running_tasks,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                
        # Collect results
        results = {
            task.task_id: task.result
            for task in graph.tasks
            if task.status == TaskStatus.COMPLETED
        }
        
        return results
        
    def enable_prefetch(self, enable: bool = True) -> None:
        """Enable/disable memory prefetching for latency hiding."""
        self._enable_prefetch = enable
        
    def get_statistics(self) -> dict[str, Any]:
        """Get execution statistics."""
        if not self._execution_history:
            return {
                "total_tasks": 0,
                "avg_duration": 0.0,
                "total_time": 0.0,
            }
            
        total_time = sum(task.duration for task in self._execution_history)
        return {
            "total_tasks": len(self._execution_history),
            "completed": sum(1 for t in self._execution_history if t.status == TaskStatus.COMPLETED),
            "failed": sum(1 for t in self._execution_history if t.status == TaskStatus.FAILED),
            "avg_duration": total_time / len(self._execution_history),
            "total_time": total_time,
            "max_duration": max(t.duration for t in self._execution_history),
            "min_duration": min(t.duration for t in self._execution_history),
        }
