"""Snapshot differ utilities."""

from __future__ import annotations

from dataclasses import dataclass

from qtime.snapshot import Snapshot


@dataclass
class SnapshotDiffer:
    def diff(self, base: Snapshot, target: Snapshot) -> dict[str, dict[str, object]]:
        return base.diff(target)
