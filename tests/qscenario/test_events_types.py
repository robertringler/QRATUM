from qscenario.events import (Event, MarketEvent, MissionEvent, NodeEvent,
                              SystemEvent)


def test_event_describe():
    event = MarketEvent(1, "finance", "shock", {"impact": 3})
    desc = event.describe()
    assert desc["domain"] == "finance"
    assert desc["kind"] == "shock"


def test_event_subclasses():
    assert issubclass(MissionEvent, Event)
    assert issubclass(NodeEvent, Event)
    assert issubclass(SystemEvent, Event)
