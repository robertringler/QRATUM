"""Core contracts for the qstack-superrepo components.

This module defines the abstract interfaces for kernels, modules, and engines
used throughout the Q-Stack ecosystem. Implementations must provide a stable
name, a human-readable description, and an execution entry point.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Identifiable(Protocol):
    """Protocol describing identifiable components."""

    @property
    @abstractmethod
    def name(self) -> str:  # pragma: no cover - interface declaration
        """Unique identifier for the component."""


class Kernel(ABC):
    """Base contract for computational kernels."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the unique kernel name."""

    @abstractmethod
    def describe(self) -> str:
        """Return a human-readable kernel description."""

    @abstractmethod
    def run(self, **kwargs: Any) -> Any:
        """Execute the kernel with deterministic semantics."""


class Module(ABC):
    """Base contract for higher-order modules built atop kernels."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the unique module name."""

    @abstractmethod
    def describe(self) -> str:
        """Return a human-readable module description."""

    @abstractmethod
    def execute(self, context: dict[str, Any] | None = None) -> Any:
        """Execute the module using the provided context."""


class Engine(ABC):
    """Base contract for orchestrating engines."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the unique engine name."""

    @abstractmethod
    def describe(self) -> str:
        """Return a human-readable engine description."""

    @abstractmethod
    def step(self, state: dict[str, Any] | None = None) -> Any:
        """Execute a deterministic engine step."""


__all__ = ["Kernel", "Module", "Engine", "Identifiable"]
