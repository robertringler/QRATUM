
from qscenario.domains.aerospace import aerospace_launch_anomaly


def test_scenario_runs_and_records_metrics():
    scenario = aerospace_launch_anomaly()
    state = scenario.run()
    assert state.metrics["mission_steps"] == 5
    assert any(inc["label"] == "engine_issue" for inc in state.incidents)
