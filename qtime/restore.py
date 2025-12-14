"""Snapshot restore helpers."""

from __future__ import annotations

from qtime.snapshot import Snapshot


def restore_snapshot(serialized: dict) -> Snapshot:
    return Snapshot.restore(serialized)
