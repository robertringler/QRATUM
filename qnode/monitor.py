"""Health and integrity monitor."""

from __future__ import annotations

from typing import Dict, List

from infra.alerts import AlertEngine
from infra.metrics import MetricRegistry
from qnode.incident_log import IncidentLog


class HealthMonitor:
    def __init__(self, incident_log: IncidentLog) -> None:
        self.metrics = MetricRegistry()
        self.alerts = AlertEngine()
        self.incidents = incident_log
        self.alerts.register("errors", 1)

    def observe_event(self, category: str) -> None:
        counter = self.metrics.counter(category)
        counter.inc()

    def record_error(self, detail: Dict[str, object]) -> None:
        self.metrics.counter("errors").inc()
        self.incidents.record("error", detail)

    def evaluate(self) -> Dict[str, object]:
        snapshot = self.metrics.snapshot()
        triggered = self.alerts.evaluate(snapshot)
        if triggered:
            self.incidents.record("alert", {"alerts": triggered})
        return {"metrics": snapshot, "alerts": triggered}

    def health_score(self) -> int:
        errors = self.metrics.counter("errors").value
        return max(0, 100 - errors * 10)

    def incident_history(self) -> List[object]:
        return [i.__dict__ for i in self.incidents.all()]
