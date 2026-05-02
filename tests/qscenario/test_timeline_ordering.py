from qscenario.events import SystemEvent
from qscenario.timeline import Timeline, TimelineEntry


def test_timeline_streams_in_order():
    entries = [
        TimelineEntry(2, [SystemEvent(2, "sys", "b", {})]),
        TimelineEntry(0, [SystemEvent(0, "sys", "a", {})]),
    ]
    tl = Timeline(entries)
    ticks = [tick for tick, _ in tl.stream()]
    assert ticks == [0, 2]
