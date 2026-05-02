from qintervention.actions import InterventionAction, ScheduledAction
from qsk.distributed.interventions import deterministic_allocation


def test_cluster_intervention_allocation_round_robin():
    nodes = ["n1", "n2"]
    actions = [
        ScheduledAction(tick=0, action=InterventionAction(kind="finance", target="a")),
        ScheduledAction(tick=1, action=InterventionAction(kind="finance", target="b")),
        ScheduledAction(tick=2, action=InterventionAction(kind="finance", target="c")),
    ]

    plan = deterministic_allocation(nodes, actions)
    schedule = plan.schedule()

    assert schedule[0].node_id == "n1"
    assert len(schedule[0].actions) == 2
    assert schedule[1].node_id == "n2"
    assert len(schedule[1].actions) == 1
