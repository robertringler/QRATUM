"""Node-level orchestration across subsystems."""

from __future__ import annotations

from typing import Callable, Dict, List

from qnode.config import NodeConfig
from qnode.incident_log import IncidentLog
from qnode.policies import NodePolicy
from qnode.syscalls import SyscallRouter
from qscenario.outcomes import evaluate_outcomes
from qscenario.reporting import ScenarioReport
from qscenario.scenario import Scenario


class NodeOrchestrator:
    def __init__(
        self,
        config: NodeConfig,
        policies: Dict[str, NodePolicy],
        dispatcher: Dict[str, Callable[[Dict[str, object]], object]],
    ) -> None:
        self.incidents = IncidentLog()
        policy_list = list(policies.values())
        self.syscalls = SyscallRouter(config, policy_list, self.incidents)
        for name, handler in dispatcher.items():
            self.syscalls.register(name, handler)

    def execute(self, syscall: str, payload: Dict[str, object]) -> object:
        return self.syscalls.dispatch(syscall, payload)

    def log(self) -> Dict[str, object]:
        return {"incidents": [i.__dict__ for i in self.incidents.all()]}

    def attach_scenario(self, scenario: Scenario) -> None:
        scenario.state.context.setdefault("node_config", self.syscalls.config.__dict__)

    def run_scenario(
        self, scenario: Scenario, required_metrics: List[str] | None = None
    ) -> ScenarioReport:
        required = required_metrics or []
        self.attach_scenario(scenario)
        state = scenario.run()
        outcome = evaluate_outcomes(state, required)
        report = ScenarioReport(
            scenario.config, scenario.timeline, outcome, state, scenario.results
        )
        for incident in state.incidents:
            self.incidents.record("scenario", incident)
        return report
