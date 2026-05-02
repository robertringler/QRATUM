from qtime.differ import SnapshotDiffer
from qtime.snapshot import Snapshot


def test_snapshot_diff_detects_changes():
    base = Snapshot(state={"a": 1, "b": 2}, tick=1)
    target = Snapshot(state={"a": 1, "b": 3, "c": 4}, tick=2)
    differ = SnapshotDiffer()
    diff = differ.diff(base, target)
    assert diff == {"b": {"from": 2, "to": 3}, "c": {"from": None, "to": 4}}
