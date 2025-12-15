"""Snapshot differ utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from qtime.snapshot import Snapshot


@dataclass
class SnapshotDiffer:
    def diff(self, base: Snapshot, target: Snapshot) -> Dict[str, Dict[str, object]]:
        return base.diff(target)
