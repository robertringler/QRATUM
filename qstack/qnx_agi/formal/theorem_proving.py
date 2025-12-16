"""Isabelle/HOL proof obligation scaffolding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class IsabelleGoal:
    name: str
    context: Dict[str, float]
    statement: str


@dataclass(frozen=True)
class ProofObligation:
    goal: IsabelleGoal
    assumptions: List[str]
    conclusion: str

    def render(self) -> str:
        assumptions = "\n".join(self.assumptions)
        return f"theorem {self.goal.name}:\nassumes {assumptions}\nshows {self.conclusion}"


def build_goal(
    name: str, context: Dict[str, float], statement: str, assumptions: List[str]
) -> ProofObligation:
    goal = IsabelleGoal(name=name, context=context, statement=statement)
    return ProofObligation(goal=goal, assumptions=assumptions, conclusion=statement)
