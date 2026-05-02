"""Compare scenario reports across variants."""

from __future__ import annotations

from dataclasses import dataclass

from qscenario.reporting import ScenarioReport


@dataclass
class ComparisonReport:
    reports: list[ScenarioReport]

    def best_outcome(self) -> ScenarioReport:
        sorted_reports = sorted(
            self.reports,
            key=lambda r: (
                0 if r.outcome.success else 1,
                len(r.outcome.incidents),
                -sum(r.outcome.metrics.values()) if r.outcome.metrics else 0,
            ),
        )
        return sorted_reports[0]

    def summarize(self) -> dict[str, object]:
        return {
            "variants": [r.config.policies for r in self.reports],
            "classifications": [r.outcome.classify() for r in self.reports],
        }
