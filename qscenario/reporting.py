"""Structured scenario reporting."""
from __future__ import annotations

from dataclasses import dataclass

from qscenario.outcomes import OutcomeSummary
from qscenario.timeline import Timeline
from qscenario.scenario import ScenarioConfig, ScenarioState


@dataclass
class ScenarioReport:
    config: ScenarioConfig
    timeline: Timeline
    outcome: OutcomeSummary
    state: ScenarioState
    results: list[dict[str, object]]

    def serialize(self) -> dict[str, object]:
        return {
            "config": self.config.__dict__,
            "timeline": self.timeline.describe(),
            "outcome": {
                "success": self.outcome.success,
                "reasons": list(self.outcome.reasons),
                "metrics": dict(self.outcome.metrics),
                "incidents": list(self.outcome.incidents),
                "classification": self.outcome.classify(),
            },
            "state": {
                "tick": self.state.tick,
                "metrics": dict(self.state.metrics),
                "incidents": list(self.state.incidents),
            },
            "results": [self._format_result(res) for res in self.results],
            "narrative": self.narrative(),
        }

    def narrative(self) -> list[str]:
        lines: list[str] = []
        lines.append(f"Scenario {self.config.name} executed over {len(self.timeline.describe())} ticks.")
        if self.outcome.incidents:
            lines.append(f"Incidents observed: {len(self.outcome.incidents)}")
        if self.outcome.metrics:
            metrics_summary = ", ".join(f"{k}={v}" for k, v in sorted(self.outcome.metrics.items()))
            lines.append(f"Metrics: {metrics_summary}")
        lines.append(f"Outcome classified as {self.outcome.classify()}.")
        return lines

    def _format_result(self, result: dict[str, object]) -> dict[str, object]:
        event = result["event"]
        formatted = {
            "tick": result["tick"],
            "event": event.describe(),
            "outcome": result["outcome"],
        }
        return formatted
