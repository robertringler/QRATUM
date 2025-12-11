"""Global registry for Q-Stack components."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, TypeVar

from .contracts import Engine, Kernel, Module

T = TypeVar("T", Kernel, Module, Engine)


@dataclass
class Registry:
    """Simple in-memory registry for kernels, modules, and engines."""

    kernels: dict[str, Kernel] = field(default_factory=dict)
    modules: dict[str, Module] = field(default_factory=dict)
    engines: dict[str, Engine] = field(default_factory=dict)

    def _register(self, store: dict[str, T], component: T) -> None:
        name = component.name
        if name in store:
            raise ValueError(f"Component with name '{name}' is already registered")
        store[name] = component

    def register_kernel(self, kernel: Kernel) -> None:
        """Register a kernel instance."""
        self._register(self.kernels, kernel)

    def register_module(self, module: Module) -> None:
        """Register a module instance."""
        self._register(self.modules, module)

    def register_engine(self, engine: Engine) -> None:
        """Register an engine instance."""
        self._register(self.engines, engine)

    def get_kernel(self, name: str) -> Kernel:
        return self.kernels[name]

    def get_module(self, name: str) -> Module:
        return self.modules[name]

    def get_engine(self, name: str) -> Engine:
        return self.engines[name]

    def list_kernels(self) -> Iterable[str]:
        return tuple(self.kernels.keys())

    def list_modules(self) -> Iterable[str]:
        return tuple(self.modules.keys())

    def list_engines(self) -> Iterable[str]:
        return tuple(self.engines.keys())


GLOBAL_REGISTRY = Registry()

__all__ = ["GLOBAL_REGISTRY", "Registry"]
