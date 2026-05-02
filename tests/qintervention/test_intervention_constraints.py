from qintervention.actions import InterventionAction
from qintervention.constraints import ConstraintSet, domain_whitelist
from qintervention.planner import InterventionPlanner


def test_domain_whitelist_blocks_unknown():
    constraints = ConstraintSet([domain_whitelist({"finance"})])
    planner = InterventionPlanner(constraints)
    proposals = {
        0: [
            InterventionAction(kind="finance", target="a"),
            InterventionAction(kind="grid", target="b"),
        ]
    }

    plan = planner.build_plan(proposals)

    assert len(plan.actions) == 1
    assert plan.actions[0].action.kind == "finance"
