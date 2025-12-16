from qintervention.actions import InterventionAction, ScheduledAction
from qintervention.rollout import rollout_plan
from qscenario.scenario import ScenarioState


def test_rollout_updates_metrics():
    state = ScenarioState()
    plan = [
        ScheduledAction(
            tick=0, action=InterventionAction(kind="grid", target="n1", params={"impact": 1.0})
        ),
        ScheduledAction(
            tick=1, action=InterventionAction(kind="finance", target="n2", params={"impact": 2.0})
        ),
    ]

    applied = rollout_plan(plan, state)

    assert state.metrics["interventions"] == 2
    assert applied[0]["impact"] == 1.0
    assert applied[1]["impact"] == 2.0
