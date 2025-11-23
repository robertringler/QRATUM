"""System kernel orchestrating Q-Stack subsystems deterministically with alignment."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping

from qstack.alignment.constitution import DEFAULT_CONSTITUTION
from qstack.alignment.evaluator import AlignmentEvaluator
from qstack.alignment.violations import AlignmentViolation
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
    """Deterministic kernel responsible for subsystem lifecycle with alignment checks."""

    config: QStackConfig
    system: QStackSystem
    event_bus: EventBus
    telemetry: Telemetry
    alignment_evaluator: AlignmentEvaluator | None = None

    def _resolve_alignment(self) -> AlignmentEvaluator:
        if self.alignment_evaluator is None:
            self.alignment_evaluator = AlignmentEvaluator(self.config, DEFAULT_CONSTITUTION)
        return self.alignment_evaluator

    def _record_alignment_event(
        self, event_type: EventType, violations: List[AlignmentViolation], phase: str, operation: str
    ) -> None:
        payload = {
            "operation": operation,
            "phase": phase,
            "violations": [violation.as_dict() for violation in violations],
        }
        event = self.event_bus.publish(event_type, payload)
        self.telemetry.record(
            "alignment",
            {"event_id": event.event_id, "phase": phase, "violation_count": len(violations)},
            payload,
        )

    def _precheck_or_raise(self, operation: str, context: Mapping[str, Any]) -> None:
        evaluator = self._resolve_alignment()
        violations = evaluator.pre_operation_check(operation, dict(context))
        if violations:
            self._record_alignment_event(EventType.ALIGNMENT_PRE_CHECK_FAILED, violations, "pre", operation)
        if evaluator.has_fatal(violations):
            raise ValueError(f"Alignment pre-check failed for {operation}")

    def _postcheck(self, operation: str, context: Mapping[str, Any]) -> List[AlignmentViolation]:
        evaluator = self._resolve_alignment()
        violations = evaluator.post_operation_check(operation, dict(context))
        if violations:
            self._record_alignment_event(EventType.ALIGNMENT_POST_CHECK_VIOLATION, violations, "post", operation)
        return violations

    def boot(self) -> Dict[str, Any]:
        config_snapshot = self.config.to_dict()
        event = self.event_bus.publish(EventType.SYSTEM_BOOT, {"config": config_snapshot})
        self.telemetry.record(
            "kernel", {"status": "booted", "event_id": event.event_id}, {"config": config_snapshot}
        )
        return {"status": "booted", "event_id": event.event_id, "config": config_snapshot}

    def run_qnx_cycles(self, steps: int) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        self._precheck_or_raise("qnx.simulation", {"requested_steps": steps})
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
        self._postcheck("qnx.simulation", {"steps": steps, "results": _safe_repr(results)})
        return results

    def run_quasim(self, circuit: List[List[complex]]) -> List[complex]:
        self._precheck_or_raise("quasim.simulation", {"circuit": _safe_repr(circuit)})
        simulation_result = self.system.simulate_circuit(circuit)
        event = self.event_bus.publish(
            EventType.QUASIM_SIMULATION_RUN,
            {"circuit": _safe_repr(circuit), "result": _safe_repr(simulation_result)},
        )
        self.telemetry.record(
            "quasim", {"event_id": event.event_id, "result_length": len(simulation_result)}, {},
        )
        self._postcheck("quasim.simulation", {"circuit": _safe_repr(circuit), "result": _safe_repr(simulation_result)})
        return simulation_result

    def run_qunimbus(self, agents: Any, shocks: Any, steps: int) -> Dict[str, Any]:
        self._precheck_or_raise("qunimbus.synthetic_market", {"agents": _safe_repr(agents), "steps": steps})
        market_result = self.system.run_synthetic_market(agents, shocks, steps)
        event = self.event_bus.publish(
            EventType.QUNIMBUS_EVAL_COMPLETED,
            {"steps": steps, "result": _safe_repr(market_result)},
        )
        self.telemetry.record(
            "qunimbus", {"event_id": event.event_id, "steps": steps}, {"result": market_result}
        )
        self._postcheck("qunimbus.synthetic_market", {"steps": steps, "result": _safe_repr(market_result)})
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

        self._postcheck(
            "qnx.simulation", {"steps": scenario_steps, "results": _safe_repr(qnx_results), "scenario": name}
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
