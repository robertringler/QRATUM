"""Registry for snapshots."""

from __future__ import annotations

from dataclasses import dataclass, field

from qtime.snapshot import Snapshot


@dataclass
class SnapshotRegistry:
    snapshots: dict[str, Snapshot] = field(default_factory=dict)

    def register(self, snapshot: Snapshot) -> str:
        sid = snapshot.snapshot_id()
        self.snapshots[sid] = snapshot
        return sid

    def get(self, snapshot_id: str) -> Snapshot:
        return self.snapshots[snapshot_id]

    def latest_for_tick(self, tick: int) -> Snapshot:
        candidates = [s for s in self.snapshots.values() if s.tick <= tick]
        if not candidates:
            raise KeyError("no snapshot for tick")
        return sorted(candidates, key=lambda s: s.tick)[-1]
