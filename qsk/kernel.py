"""Deterministic sovereign kernel."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .scheduler import DeterministicScheduler
from .syscall import SyscallTable


@dataclass
class ExecutionTrace:
    steps: list[str]

    def record(self, message: str):
        self.steps.append(message)


class Kernel:
    def __init__(self):
        self.scheduler = DeterministicScheduler()
        self.syscalls = SyscallTable()
        self.trace = ExecutionTrace([])

    def register_syscall(self, name: str, func: Callable):
        self.syscalls.register(name, func)

    def run(self, dag: list[dict[str, Any]]) -> list[Any]:
        ordered = self.scheduler.order(dag)
        results = []
        for node in ordered:
            self.trace.record(f"execute:{node['id']}")
            if node['type'] == 'syscall':
                result = self.syscalls.invoke(node['name'], *node.get('args', []), **node.get('kwargs', {}))
                results.append(result)
            else:
                results.append(node.get('value'))
        return results

    def verify_trace(self) -> bool:
        return self.scheduler.verify(self.trace.steps)
