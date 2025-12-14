from __future__ import annotations

from typing import Any

from ..perception.base import PerceptionLayer
from ..worldmodel.base import WorldModel
from ..memory.base import MemorySystem
from ..planning.base import PlanningSystem
from ..utils.audit_log import AuditLog
from ..utils.ids import deterministic_id


class Orchestrator:
    """Deterministic orchestration loop gluing perception, world model, memory, and planning."""

    def __init__(
        self,
        perception: PerceptionLayer,
        world_model: WorldModel,
        memory: MemorySystem,
        planning: PlanningSystem,
        audit_log: AuditLog | None = None,
    ):
        self._perception = perception
        self._world_model = world_model
        self._memory = memory
        self._planning = planning
        self._audit = audit_log or AuditLog()

    def cycle(self, raw_input: Any, goal: dict[str, Any]) -> dict[str, Any]:
        percepts = self._perception.process(raw_input)
        state = self._world_model.ingest(percepts)
        for percept in percepts:
            self._memory.record(percept.modality, percept.value)
        plan = self._planning.evaluate(goal, state.facts)
        run_id = deterministic_id("cycle", {"goal": goal, "state": state.facts})
        self._audit.record("cycle", {"id": run_id, "goal": goal, "state": state.facts, "plan": plan})
        return {
            "id": run_id,
            "percepts": percepts,
            "state": state.facts,
            "memory": list(self._memory._buffer),
            "plan": plan,
            "audit": self._audit.export(),
        }
