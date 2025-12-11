"""Syscall table bridging QDL/QIR to kernel."""
from __future__ import annotations

from typing import Any, Callable


class SyscallTable:
    def __init__(self):
        self._handlers: dict[str, Callable[..., Any]] = {}

    def register(self, name: str, func: Callable[..., Any]):
        self._handlers[name] = func

    def invoke(self, name: str, *args, **kwargs):
        if name not in self._handlers:
            raise KeyError(f"Syscall {name} not registered")
        return self._handlers[name](*args, **kwargs)
