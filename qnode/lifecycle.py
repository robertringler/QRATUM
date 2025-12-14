"""Node lifecycle manager."""
from __future__ import annotations

from dataclasses import dataclass, field

from qnode.monitor import HealthMonitor


@dataclass
class NodeLifecycle:
    monitor: HealthMonitor
    config: "NodeConfig | None" = None
    state: str = "init"
    history: list[str] = field(default_factory=list)

    def _transition(self, new_state: str) -> None:
        allowed = {
            "init": {"running", "shutdown"},
            "running": {"paused", "shutdown"},
            "paused": {"running", "shutdown"},
            "shutdown": set(),
        }
        if new_state not in allowed[self.state]:
            raise ValueError(f"invalid transition {self.state}->{new_state}")
        self.history.append(f"{self.state}->{new_state}")
        self.state = new_state

    def start(self) -> None:
        from qconstitution.validator import validate_node_config
        from qconstitution.charter import default_charter

        if self.config is not None:
            validate_node_config(self.config, charter=default_charter())
        self._transition("running")
        self.monitor.observe_event("start")

    def pause(self) -> None:
        self._transition("paused")
        self.monitor.observe_event("pause")

    def resume(self) -> None:
        self._transition("running")
        self.monitor.observe_event("resume")

    def shutdown(self) -> None:
        if self.state != "shutdown":
            self._transition("shutdown")
            self.monitor.observe_event("shutdown")

    def trace(self) -> list[str]:
        return list(self.history)
