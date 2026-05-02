from __future__ import annotations

from typing import Any

from .tracing import TraceRecorder


class QNXVM:
    """Deterministic execution engine that combines scheduling with safety validation."""

    def __init__(self, scheduler, safety, tracer: TraceRecorder | None = None):
        self._scheduler = scheduler
        self._safety = safety
        self._tracer = tracer or TraceRecorder()

    def run_cycle(self, state, goal) -> dict[str, Any]:
        if not self._safety.pre_check(state, goal):
            self._tracer.record("pre_check_failed", {"goal": goal})
            return {"error": "safety_pre_check_failed"}

        trace = self._scheduler.schedule(state, goal)
        self._tracer.record("execution", {"goal": goal, "trace": trace})

        if not self._safety.post_check(state, trace):
            self._tracer.record("post_check_failed", {"state": state.data})
            return {"error": "safety_post_check_failed"}

        return {"trace": trace, "recorded": self._tracer.snapshot()}
