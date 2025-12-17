"""Task scheduler for distributing render jobs."""

from __future__ import annotations

import asyncio
from typing import Any, Optional

from .gpu_worker import RenderWorker


class TaskScheduler:
    """Scheduler for distributing render jobs to workers.

    Args:
        num_workers: Number of worker instances
    """

    def __init__(self, num_workers: int = 1) -> None:
        """Initialize task scheduler."""

        self.workers: list[RenderWorker] = []
        self.job_queue: asyncio.Queue = asyncio.Queue()

        # Create workers
        for i in range(num_workers):
            worker = RenderWorker(f"worker-{i}", gpu_device=i % 8)
            self.workers.append(worker)

    async def submit_job(self, job: dict[str, Any]) -> str:
        """Submit a job to the queue.

        Args:
            job: Job specification

        Returns:
            Job ID
        """

        job_id = job.get("job_id")
        await self.job_queue.put(job)
        return job_id

    async def process_queue(self) -> None:
        """Process jobs from the queue."""

        while True:
            # Get next job
            job = await self.job_queue.get()

            # Find available worker
            worker = self._get_available_worker()
            if worker:
                await worker.process_job(job)

            self.job_queue.task_done()

    def _get_available_worker(self) -> Optional[RenderWorker]:
        """Get an available worker.

        Returns:
            Available worker or None
        """

        for worker in self.workers:
            if worker.is_available():
                return worker
        return None

    def get_stats(self) -> dict[str, Any]:
        """Get scheduler statistics.

        Returns:
            Statistics dictionary
        """

        return {
            "num_workers": len(self.workers),
            "busy_workers": sum(1 for w in self.workers if not w.is_available()),
            "queue_size": self.job_queue.qsize(),
        }
