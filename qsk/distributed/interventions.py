"""Cluster-level intervention coordination."""
from __future__ import annotations

from dataclasses import dataclass, field

from qintervention.actions import ScheduledAction


@dataclass
class NodeAssignment:
    node_id: str
    actions: list[ScheduledAction] = field(default_factory=list)


@dataclass
class ClusterInterventionPlan:
    assignments: dict[str, NodeAssignment] = field(default_factory=dict)

    def assign(self, node_id: str, action: ScheduledAction) -> None:
        assignment = self.assignments.setdefault(node_id, NodeAssignment(node_id=node_id))
        assignment.actions.append(action)
        assignment.actions.sort(key=lambda a: a.key())

    def schedule(self) -> list[NodeAssignment]:
        return [self.assignments[k] for k in sorted(self.assignments)]


def deterministic_allocation(nodes: list[str], actions: list[ScheduledAction]) -> ClusterInterventionPlan:
    plan = ClusterInterventionPlan()
    sorted_actions = sorted(actions, key=lambda a: a.key())
    for idx, action in enumerate(sorted_actions):
        node = nodes[idx % len(nodes)]
        plan.assign(node, action)
    return plan
