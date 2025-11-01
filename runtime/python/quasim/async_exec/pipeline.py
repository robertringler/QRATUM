"""Pipeline execution with overlapped stages for latency hiding."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional


@dataclass
class PipelineStage:
    """Represents a stage in an execution pipeline."""
    name: str
    func: Callable[[Any], Any]
    buffer_size: int = 2
    
    async def process(self, input_data: Any) -> Any:
        """Process data through this pipeline stage."""
        if asyncio.iscoroutinefunction(self.func):
            return await self.func(input_data)
        else:
            return self.func(input_data)


class Pipeline:
    """Multi-stage pipeline with overlapped execution."""
    
    def __init__(self, stages: Optional[List[PipelineStage]] = None) -> None:
        self.stages = stages or []
        self._statistics: dict[str, dict[str, float]] = {}
        
    def add_stage(self, name: str, func: Callable[[Any], Any], buffer_size: int = 2) -> None:
        """Add a stage to the pipeline."""
        stage = PipelineStage(name=name, func=func, buffer_size=buffer_size)
        self.stages.append(stage)
        
    async def execute(self, input_stream: List[Any]) -> List[Any]:
        """Execute pipeline on input stream with overlapped stages."""
        if not self.stages:
            return input_stream
            
        # Create queues for inter-stage communication
        queues: List[asyncio.Queue] = [
            asyncio.Queue(maxsize=stage.buffer_size)
            for stage in self.stages
        ]
        queues.append(asyncio.Queue())  # Output queue
        
        # Start producer
        async def producer() -> None:
            for item in input_stream:
                await queues[0].put(item)
            await queues[0].put(None)  # Sentinel
            
        # Start stage processors
        async def process_stage(stage_idx: int) -> None:
            stage = self.stages[stage_idx]
            input_q = queues[stage_idx]
            output_q = queues[stage_idx + 1]
            
            stage_stats: dict[str, float] = {"processed": 0, "total_time": 0.0}
            
            while True:
                item = await input_q.get()
                if item is None:  # Sentinel
                    await output_q.put(None)
                    break
                    
                import time
                start = time.perf_counter()
                result = await stage.process(item)
                elapsed = time.perf_counter() - start
                
                stage_stats["processed"] += 1
                stage_stats["total_time"] += elapsed
                
                await output_q.put(result)
                
            self._statistics[stage.name] = stage_stats
            
        # Start consumer
        async def consumer() -> List[Any]:
            results = []
            output_q = queues[-1]
            while True:
                item = await output_q.get()
                if item is None:  # Sentinel
                    break
                results.append(item)
            return results
            
        # Run all stages concurrently
        tasks = [asyncio.create_task(producer())]
        tasks.extend(asyncio.create_task(process_stage(i)) for i in range(len(self.stages)))
        consumer_task = asyncio.create_task(consumer())
        
        # Wait for all stages
        await asyncio.gather(*tasks)
        results = await consumer_task
        
        return results
        
    def get_statistics(self) -> dict[str, dict[str, float]]:
        """Get pipeline execution statistics."""
        return self._statistics.copy()
