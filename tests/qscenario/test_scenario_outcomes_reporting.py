from qscenario.domains.aerospace import aerospace_launch_anomaly
from qscenario.outcomes import evaluate_outcomes
from qscenario.reporting import ScenarioReport


def test_reporting_contains_narrative():
    scenario = aerospace_launch_anomaly()
    state = scenario.run()
    outcome = evaluate_outcomes(state, ["mission_steps"])
    report = ScenarioReport(scenario.config, scenario.timeline, outcome, state, scenario.results)
    serialized = report.serialize()
    assert serialized["outcome"]["classification"] in {"success", "mixed", "failure"}
    assert any("Outcome classified" in line for line in serialized["narrative"])
