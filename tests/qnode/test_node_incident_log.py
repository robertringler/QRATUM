from qnode.incident_log import IncidentLog


def test_incident_log_ordering():
    log = IncidentLog()
    log.record("a", {"x": 1})
    log.record("b", {"y": 2})
    recent = log.recent(2)
    assert [i.category for i in recent] == ["a", "b"]
    assert recent[-1].sequence == 2
