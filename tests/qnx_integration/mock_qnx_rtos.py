from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, MutableSequence


@dataclass
class SchedulerEvent:
    tick: int
    payload: Any


class MockQNXRTOS:
    """Lightweight RTOS stub to exercise lifecycle hooks without hardware."""

    def __init__(self) -> None:
        self.booted: bool = False
        self.tick_count: int = 0
        self.ipc_bus: MutableSequence[dict[str, Any]] = []
        self.scheduler_log: list[SchedulerEvent] = []

    def boot(self) -> dict[str, Any]:
        self.booted = True
        self.scheduler_log.append(SchedulerEvent(self.tick_count, "boot"))
        return {"status": "booted", "ticks": self.tick_count}

    def dispatch_ticks(self, *, iterations: int = 1, handler: Callable[[int], Any] | None = None) -> list[Any]:
        outputs: list[Any] = []
        for _ in range(iterations):
            self.tick_count += 1
            payload = handler(self.tick_count) if handler else {"tick": self.tick_count}
            self.scheduler_log.append(SchedulerEvent(self.tick_count, payload))
            outputs.append(payload)
            time.sleep(0.001)
        return outputs

    def send_ipc(self, channel: str, payload: Any) -> dict[str, Any]:
        message = {"channel": channel, "payload": payload, "tick": self.tick_count}
        self.ipc_bus.append(message)
        return message

    def teardown(self) -> dict[str, Any]:
        self.booted = False
        return {"status": "stopped", "ticks": self.tick_count, "messages": len(self.ipc_bus)}


__all__ = ["MockQNXRTOS", "SchedulerEvent"]
