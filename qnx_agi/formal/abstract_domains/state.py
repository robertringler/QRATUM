"""Symbolic state abstraction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from .interval import Interval


@dataclass
class SymbolicState:
    intervals: Dict[str, Interval]
