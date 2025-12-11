"""Registry for reusable scenario templates."""
from __future__ import annotations

from typing import Callable

from qscenario.scenario import Scenario


class ScenarioRegistry:
    def __init__(self) -> None:
        self._registry: dict[str, Callable[[], Scenario]] = {}

    def register(self, name: str, builder: Callable[[], Scenario]) -> None:
        self._registry[name] = builder

    def build(self, name: str) -> Scenario:
        if name not in self._registry:
            raise KeyError(f"scenario '{name}' is not registered")
        return self._registry[name]()

    def list(self) -> dict[str, str]:
        return {name: builder.__doc__ or "" for name, builder in self._registry.items()}
