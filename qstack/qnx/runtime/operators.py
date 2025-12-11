from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class Operator:
    """Wraps a deterministic callable with optional name and description."""

    fn: Callable[[Any, Any], Any]
    description: str = ""

    def execute(self, state: Any, goal: Any) -> Any:
        return self.fn(state, goal)


class OperatorLibrary:
    def __init__(self):
        self._ops: dict[str, Operator] = {}

    def register(self, name: str, fn: Callable[[Any, Any], Any], description: str = "") -> None:
        if name in self._ops:
            raise ValueError(f"operator '{name}' already registered")
        self._ops[name] = Operator(fn=fn, description=description)

    def extend(self, mapping: dict[str, Callable[[Any, Any], Any]]) -> None:
        for name, fn in mapping.items():
            self.register(name, fn)

    def available(self) -> dict[str, Operator]:
        return dict(self._ops)

    def describe(self, name: str) -> str | None:
        op = self._ops.get(name)
        return op.description if op else None
