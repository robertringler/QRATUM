"""Run scenarios under multiple policy variants."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from qscenario.outcomes import evaluate_outcomes
from qscenario.policy_sandbox.comparison import ComparisonReport
from qscenario.policy_sandbox.constraints import enforce_core_constraints
from qscenario.policy_sandbox.policy_variant import PolicyVariant
from qscenario.reporting import ScenarioReport
from qscenario.scenario import Scenario


@dataclass
class PolicySandbox:
    base_policies: Dict[str, object]
    variants: List[PolicyVariant] = field(default_factory=list)

    def run(self, scenario_builder) -> ComparisonReport:
        reports: List[ScenarioReport] = []
        for variant in self.variants:
            scenario: Scenario = scenario_builder()
            policies = enforce_core_constraints(variant.apply(self.base_policies))
            scenario.config.policies = policies
            state = scenario.run()
            outcome = evaluate_outcomes(state, list(policies.keys()))
            report = ScenarioReport(
                scenario.config, scenario.timeline, outcome, state, scenario.results
            )
            reports.append(report)
        return ComparisonReport(reports)
