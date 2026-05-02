from qscenario.domains.aerospace import aerospace_launch_anomaly


def test_aerospace_template_incidents():
    scenario = aerospace_launch_anomaly()
    state = scenario.run()
    labels = [inc["label"] for inc in state.incidents]
    assert "engine_issue" in labels
