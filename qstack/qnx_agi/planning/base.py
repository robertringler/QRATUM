from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from ...qnx.runtime.safety import SafetyEnvelope


@dataclass
class PlanStep:
    action: str
    parameters: Dict[str, Any]
    cost: float
    heuristic: float
    parent: Optional[PlanStep] = None

    def total(self) -> float:
        return self.cost + self.heuristic


class Planner:
    """Abstract planner interface."""

    def __init__(
        self, heuristic: Optional[Callable[[Dict[str, Any], Dict[str, Any]], float]] = None
    ):
        self._heuristic = heuristic or (lambda goal, state: 0.0)

    def plan(
        self, goal: Dict[str, Any], state: Dict[str, Any]
    ) -> List[PlanStep]:  # pragma: no cover - interface
        """Default deterministic plan: mirror goal into a single noop step."""
        return [
            PlanStep(
                action="noop",
                parameters=dict(state),
                cost=0.0,
                heuristic=self._heuristic(goal, state),
            )
        ]


class GreedyPlanner(Planner):
    def __init__(self):
        super().__init__(lambda goal, state: 0.0)

    def plan(self, goal: Dict[str, Any], state: Dict[str, Any]) -> List[PlanStep]:
        steps: List[PlanStep] = []
        for key, value in sorted(goal.items(), key=lambda kv: kv[0]):
            cost = 0.0 if state.get(key) == value else 1.0
            steps.append(
                PlanStep(
                    action="set", parameters={"key": key, "value": value}, cost=cost, heuristic=0.0
                )
            )
        return steps


class HeuristicSearchPlanner(Planner):
    def __init__(
        self, heuristic: Optional[Callable[[Dict[str, Any], Dict[str, Any]], float]] = None
    ):
        super().__init__(heuristic or (lambda goal, state: float(len(goal))))

    def plan(self, goal: Dict[str, Any], state: Dict[str, Any]) -> List[PlanStep]:
        ordered = sorted(goal.items(), key=lambda kv: kv[0])
        return [
            PlanStep(
                action="set",
                parameters={"key": key, "value": value},
                cost=float(idx),
                heuristic=self._heuristic(goal, state),
            )
            for idx, (key, value) in enumerate(ordered)
        ]


class AStarPlanner(Planner):
    """Deterministic A* over symbolic key-value states."""

    def __init__(self, heuristic: Callable[[Dict[str, Any], Dict[str, Any]], float]):
        super().__init__(heuristic)

    def _neighbors(
        self, state: Dict[str, Any], goal: Dict[str, Any]
    ) -> Iterable[Tuple[str, Dict[str, Any]]]:
        for key, value in goal.items():
            if state.get(key) != value:
                new_state = dict(state)
                new_state[key] = value
                yield f"set_{key}", new_state

    def plan(self, goal: Dict[str, Any], state: Dict[str, Any]) -> List[PlanStep]:
        start = PlanStep(
            action="start", parameters=state, cost=0.0, heuristic=self._heuristic(goal, state)
        )
        open_set: List[PlanStep] = [start]
        closed: Dict[str, float] = {}
        while open_set:
            open_set.sort(key=lambda step: (step.total(), step.action))
            current = open_set.pop(0)
            state_digest = str(sorted(current.parameters.items()))
            if current.heuristic == 0.0 and all(
                goal.get(k) == current.parameters.get(k) for k in goal
            ):
                path: List[PlanStep] = []
                cursor: Optional[PlanStep] = current
                while cursor:
                    path.append(cursor)
                    cursor = cursor.parent
                return list(reversed(path))
            if state_digest in closed and closed[state_digest] <= current.total():
                continue
            closed[state_digest] = current.total()
            for action_label, neighbor_state in self._neighbors(current.parameters, goal):
                cost = current.cost + 1.0
                heuristic = self._heuristic(goal, neighbor_state)
                neighbor_step = PlanStep(
                    action=action_label,
                    parameters=neighbor_state,
                    cost=cost,
                    heuristic=heuristic,
                    parent=current,
                )
                open_set.append(neighbor_step)
        return []


class BeamSearchPlanner(Planner):
    def __init__(
        self, heuristic: Callable[[Dict[str, Any], Dict[str, Any]], float], width: int = 3
    ):
        super().__init__(heuristic)
        self._width = width

    def plan(self, goal: Dict[str, Any], state: Dict[str, Any]) -> List[PlanStep]:
        frontier: List[PlanStep] = [
            PlanStep(
                action="start", parameters=state, cost=0.0, heuristic=self._heuristic(goal, state)
            )
        ]
        for _ in range(len(goal)):
            expanded: List[PlanStep] = []
            for step in frontier:
                for key, value in goal.items():
                    if step.parameters.get(key) != value:
                        new_state = dict(step.parameters)
                        new_state[key] = value
                        expanded.append(
                            PlanStep(
                                action=f"set_{key}",
                                parameters=new_state,
                                cost=step.cost + 1.0,
                                heuristic=self._heuristic(goal, new_state),
                                parent=step,
                            )
                        )
            expanded.sort(key=lambda s: (s.total(), s.action))
            frontier = expanded[: self._width] or frontier
        best = min(frontier, key=lambda s: s.total())
        path: List[PlanStep] = []
        cursor: Optional[PlanStep] = best
        while cursor:
            path.append(cursor)
            cursor = cursor.parent
        return list(reversed(path))


class MPCPlanner(Planner):
    """Deterministic model predictive controller using provided dynamics."""

    def __init__(
        self,
        predict_fn: Callable[[Dict[str, Any]], Dict[str, Any]],
        cost_fn: Callable[[Dict[str, Any], Dict[str, Any]], float],
        envelope: Optional[SafetyEnvelope] = None,
        horizon: int = 3,
    ):
        super().__init__(lambda goal, state: cost_fn(goal, state))
        self._predict_fn = predict_fn
        self._cost_fn = cost_fn
        self._envelope = envelope
        self._horizon = horizon

    def plan(self, goal: Dict[str, Any], state: Dict[str, Any]) -> List[PlanStep]:
        trajectory: List[PlanStep] = []
        current_state = dict(state)
        for step_idx in range(self._horizon):
            predicted = self._predict_fn(current_state)
            if self._envelope and not self._envelope.inside(type("State", (), {"data": predicted})):
                break
            cost = self._cost_fn(goal, predicted)
            plan_step = PlanStep(
                action="predict",
                parameters=predicted,
                cost=cost,
                heuristic=0.0,
                parent=trajectory[-1] if trajectory else None,
            )
            trajectory.append(plan_step)
            current_state = predicted
            if all(predicted.get(k) == goal.get(k) for k in goal):
                break
        return trajectory


class PlanningSystem:
    def __init__(self, planner: Planner):
        self._planner = planner

    def evaluate(self, goal: Dict[str, Any], state: Dict[str, Any]) -> List[PlanStep]:
        return self._planner.plan(goal, state)
