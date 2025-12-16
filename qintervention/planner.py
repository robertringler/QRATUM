"""Deterministic intervention planning."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List

from qintervention.actions import InterventionAction, ScheduledAction
from qintervention.constraints import ConstraintSet


@dataclass
class InterventionPlan:
    actions: List[ScheduledAction] = field(default_factory=list)

    def ordered(self) -> List[ScheduledAction]:
        return sorted(self.actions, key=lambda a: a.key())

    def add(self, scheduled: ScheduledAction) -> None:
        self.actions.append(scheduled)
        self.actions.sort(key=lambda a: a.key())


class InterventionPlanner:
    """Plans interventions for a scenario deterministically."""

    def __init__(self, constraints: ConstraintSet | None = None) -> None:
        self.constraints = constraints or ConstraintSet()

    def build_plan(self, proposals: Dict[int, List[InterventionAction]]) -> InterventionPlan:
        plan = InterventionPlan()
        for tick in sorted(proposals):
            for action in proposals[tick]:
                scheduled = ScheduledAction(tick=tick, action=action)
                if self.constraints.allows(scheduled):
                    plan.add(scheduled)
        return plan

    def merge_plans(self, plans: Iterable[InterventionPlan]) -> InterventionPlan:
        merged = InterventionPlan()
        for plan in plans:
            for action in plan.actions:
                if self.constraints.allows(action):
                    merged.add(action)
        return merged
