from qsk.distributed.real_mirror.mirror_state import MirrorState


def test_mirror_state_apply_and_view():
    state = MirrorState()
    state.apply("market", {"price": 10}, tick=1)
    assert state.view("market") == {"price": 10}
    assert state.tick == 1
