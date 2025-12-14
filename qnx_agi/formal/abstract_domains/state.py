"""Symbolic state abstraction."""

from __future__ import annotations

from dataclasses import dataclass

from .interval import Interval


@dataclass
class SymbolicState:
    intervals: dict[str, Interval]
