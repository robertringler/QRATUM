"""Intervention action definitions."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class InterventionAction:
    """A deterministic intervention description."""

    kind: str
    target: str
    params: dict[str, object] = field(default_factory=dict)

    def describe(self) -> dict[str, object]:
        return {"kind": self.kind, "target": self.target, "params": dict(self.params)}


@dataclass(frozen=True)
class ScheduledAction:
    """Action bound to a scenario tick."""

    tick: int
    action: InterventionAction

    def key(self) -> tuple:
        return (self.tick, self.action.kind, self.action.target)
