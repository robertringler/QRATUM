"""System-level facade providing a unified Q-Stack API surface."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from qstack.config import QStackConfig
from qstack.qnx_adapter import QNXAdapter
from qstack.quasim_adapter import QuASIMAdapter
from qstack.qunimbus_adapter import QuNimbusAdapter


@dataclass
class QStackSystem:
    """Deterministic orchestration facade over QNX, QuASIM, and QuNimbus."""

    config: QStackConfig

    def run_qnx_lifecycle(self, rtos: Any | None = None) -> Dict[str, Any]:
        adapter = QNXAdapter(self.config.qnx)
        return dict(adapter.run_lifecycle(rtos=rtos))

    def run_qnx_simulation(self) -> Any:
        adapter = QNXAdapter(self.config.qnx)
        return adapter.run_simulation()

    def simulate_circuit(self, circuit: List[List[complex]]) -> List[complex]:
        adapter = QuASIMAdapter(self.config.quasim)
        return adapter.simulate_circuit(circuit)

    def simulate_circuits_batch(self, circuits: List[List[List[complex]]]) -> List[List[complex]]:
        adapter = QuASIMAdapter(self.config.quasim)
        return adapter.simulate_batch(circuits)

    def run_synthetic_market(
        self, agents: Any, shocks: Any, steps: int
    ) -> Dict[str, Any]:  # type: ignore[override]
        adapter = QuNimbusAdapter(self.config.qunimbus)
        return adapter.run_synthetic_market(agents, shocks, steps)

    def score_node_from_report(self, report: Any) -> Dict[str, Any]:
        adapter = QuNimbusAdapter(self.config.qunimbus)
        return adapter.score_node_from_report(report)
