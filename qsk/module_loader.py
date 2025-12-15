"""Deterministic module loader."""

from __future__ import annotations

from importlib import import_module
from typing import Any


class ModuleLoader:
    def load(self, path: str) -> Any:
        module_name, _, attr = path.partition(":")
        module = import_module(module_name)
        return getattr(module, attr) if attr else module
