from qsk.distributed.real_mirror.mirror_diff import diff
from qsk.distributed.real_mirror.mirror_state import MirrorState


def test_mirror_diff_detects_changes():
    mirror = MirrorState({"grid": {"frequency": 59.9}}, tick=2)
    cluster_state = {"grid": {"frequency": 60.0}}
    deltas = diff(mirror, cluster_state)
    assert "grid" in deltas
    assert deltas["grid"]["frequency"]["mirror"] == 59.9
