from qintervention.actions import InterventionAction
from qintervention.planner import InterventionPlanner


def test_planner_orders_actions_by_tick():
    planner = InterventionPlanner()
    proposals = {
        2: [InterventionAction(kind="finance", target="mkt")],
        1: [InterventionAction(kind="grid", target="node")],
    }

    plan = planner.build_plan(proposals)
    ordered = [a.key() for a in plan.ordered()]

    assert ordered[0][0] == 1
    assert ordered[1][0] == 2
