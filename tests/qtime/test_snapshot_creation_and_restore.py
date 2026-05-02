from qtime.registry import SnapshotRegistry
from qtime.restore import restore_snapshot
from qtime.snapshot import Snapshot


def test_snapshot_creation_and_restore_roundtrip():
    snap = Snapshot(state={"a": 1}, tick=3, metadata={"tag": "t"})
    serialized = snap.serialize()
    restored = restore_snapshot(serialized)
    assert restored.state == {"a": 1}
    assert restored.tick == 3
    registry = SnapshotRegistry()
    sid = registry.register(restored)
    assert registry.get(sid).serialize() == serialized
