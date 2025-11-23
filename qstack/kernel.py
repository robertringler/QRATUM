"""System kernel orchestrating Q-Stack subsystems deterministically."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping

from qstack.config import QStackConfig
from qstack.events import EventBus, EventType
from qstack.system import QStackSystem
from qstack.telemetry import Telemetry


def _safe_repr(value: Any) -> str:
    """Provide a stable textual representation for telemetry payloads."""

    try:
        return repr(value)
    except Exception:  # pragma: no cover - defensive
        return f"<unreprable:{type(value).__name__}>"


@dataclass
class QStackKernel:
    """Deterministic kernel responsible for subsystem lifecycle."""

    config: QStackConfig
    system: QStackSystem
    event_bus: EventBus
    telemetry: Telemetry

    def boot(self) -> Dict[str, Any]:
        config_snapshot = self.config.to_dict()
        event = self.event_bus.publish(EventType.SYSTEM_BOOT, {"config": config_snapshot})
        self.telemetry.record(
            "kernel", {"status": "booted", "event_id": event.event_id}, {"config": config_snapshot}
        )
        return {"status": "booted", "event_id": event.event_id, "config": config_snapshot}

    def run_qnx_cycles(self, steps: int) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for step in range(max(0, steps)):
            start_event = self.event_bus.publish(EventType.QNX_CYCLE_STARTED, {"step": step})
            self.telemetry.record("qnx", {"phase": "start", "step": step, "event_id": start_event.event_id}, {})

            result = self.system.run_qnx_simulation()
            serialized_result = _safe_repr(result)

            complete_event = self.event_bus.publish(
                EventType.QNX_CYCLE_COMPLETED, {"step": step, "result": serialized_result}
            )
            self.telemetry.record(
                "qnx",
                {"phase": "complete", "step": step, "event_id": complete_event.event_id},
                {"result": serialized_result},
            )
            results.append({"step": step, "event_id": complete_event.event_id, "result": serialized_result})
        return results

    def run_quasim(self, circuit: List[List[complex]]) -> List[complex]:
        simulation_result = self.system.simulate_circuit(circuit)
        event = self.event_bus.publish(
            EventType.QUASIM_SIMULATION_RUN,
            {"circuit": _safe_repr(circuit), "result": _safe_repr(simulation_result)},
        )
        self.telemetry.record(
            "quasim", {"event_id": event.event_id, "result_length": len(simulation_result)}, {}
        )
        return simulation_result

    def run_qunimbus(self, agents: Any, shocks: Any, steps: int) -> Dict[str, Any]:
        market_result = self.system.run_synthetic_market(agents, shocks, steps)
        event = self.event_bus.publish(
            EventType.QUNIMBUS_EVAL_COMPLETED,
            {"steps": steps, "result": _safe_repr(market_result)},
        )
        self.telemetry.record(
            "qunimbus", {"event_id": event.event_id, "steps": steps}, {"result": market_result}
        )
        return market_result

    def run_scenario(
        self, name: str, scenario_steps: int, circuit: List[List[complex]] | None = None, report: Mapping[str, Any] | None = None
    ) -> Dict[str, Any]:
        scenario_event = self.event_bus.publish(EventType.SCENARIO_STARTED, {"name": name, "steps": scenario_steps})
        self.telemetry.record("scenario", {"name": name, "event_id": scenario_event.event_id}, {})

        qnx_results = self.run_qnx_cycles(scenario_steps)
        quasim_result: List[complex] | None = None
        if circuit is not None:
            quasim_result = self.run_quasim(circuit)

        node_score: Dict[str, Any] | None = None
        if report is not None:
            node_score = self.score_node(report)

        end_event = self.event_bus.publish(
            EventType.SCENARIO_ENDED,
            {
                "name": name,
                "qnx_events": [res.get("event_id") for res in qnx_results],
                "quasim_ran": quasim_result is not None,
                "node_scored": node_score is not None,
            },
        )
        self.telemetry.record(
            "scenario",
            {"name": name, "event_id": end_event.event_id},
            {"qnx_results": qnx_results, "quasim_result": quasim_result, "node_score": node_score},
        )

        return {
            "scenario": name,
            "events": {
                "start": scenario_event.event_id,
                "end": end_event.event_id,
            },
            "qnx_results": qnx_results,
            "quasim_result": quasim_result,
            "node_score": node_score,
        }

    def score_node(self, report: Mapping[str, Any]) -> Dict[str, Any]:
        score = self.system.score_node_from_report(report)
        event = self.event_bus.publish(EventType.NODE_SCORED, {"score": _safe_repr(score)})
        self.telemetry.record("qunimbus", {"event_id": event.event_id, "score": score}, {})
        return score

    def record_error(self, message: str, details: Mapping[str, Any] | None = None) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"message": message}
        if details:
            payload.update(details)
        event = self.event_bus.publish(EventType.ERROR_RAISED, payload)
        self.telemetry.record("error", {"event_id": event.event_id, "message": message}, details or {})
        return {"event_id": event.event_id, "message": message}
