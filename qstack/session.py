"""SystemSession binds configuration, kernel, telemetry, and event bus."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from qstack.config import QStackConfig
from qstack.events import EventBus
from qstack.kernel import QStackKernel
from qstack.system import QStackSystem
from qstack.telemetry import Telemetry


@dataclass
class RunContext:
    """Deterministic run metadata."""

    scenario_name: str = "default"
    node_id: str = "node-0"
    timestamp_seed: str = "0"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "scenario_name": self.scenario_name,
            "node_id": self.node_id,
            "timestamp_seed": self.timestamp_seed,
        }


@dataclass
class SystemSession:
    """Top-level entrypoint for interacting with the Q-Stack."""

    config: QStackConfig
    event_bus: EventBus
    telemetry: Telemetry
    system: QStackSystem
    kernel: QStackKernel
    run_context: RunContext

    @classmethod
    def build(
        cls,
        config: QStackConfig | None = None,
        scenario_name: str = "default",
        node_id: str = "node-0",
        timestamp_seed: str | int = "0",
    ) -> "SystemSession":
        resolved_config = config or QStackConfig()
        run_context = RunContext(
            scenario_name=scenario_name,
            node_id=node_id,
            timestamp_seed=str(timestamp_seed),
        )
        event_bus = EventBus(timestamp_seed=run_context.timestamp_seed)
        telemetry = Telemetry()
        system = QStackSystem(resolved_config)
        kernel = QStackKernel(config=resolved_config, system=system, event_bus=event_bus, telemetry=telemetry)
        return cls(
            config=resolved_config,
            event_bus=event_bus,
            telemetry=telemetry,
            system=system,
            kernel=kernel,
            run_context=run_context,
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "run_context": self.run_context.as_dict(),
            "events": [event.event_id for event in self.event_bus.events],
            "telemetry": self.telemetry.as_dict(),
        }

    def describe(self) -> Dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "run_context": self.run_context.as_dict(),
            "event_count": len(self.event_bus.events),
            "telemetry_entries": len(self.telemetry.entries),
        }
