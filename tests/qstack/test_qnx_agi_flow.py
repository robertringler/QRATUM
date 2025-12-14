from qstack.qnx import (
    QNXVM,
    DeterministicScheduler,
    OperatorLibrary,
    QNXState,
    SafetyConstraints,
    SafetyEnvelope,
    SafetyValidator,
)
from qstack.qnx_agi.memory.working import WorkingMemory
from qstack.qnx_agi.orchestrator.base import Orchestrator
from qstack.qnx_agi.perception.encoders.finance import FinanceEncoder
from qstack.qnx_agi.planning.base import PlanningSystem
from qstack.qnx_agi.planning.planners.greedy import GreedyPlanner
from qstack.qnx_agi.worldmodel.base import WorldModel
from qstack.quasim.core.engine import SimulationEngine
from qstack.quasim.integration.qnx_adapter import build_qnx_operator_library


def test_agi_cycle_and_qnx_execution():
    encoder = FinanceEncoder()
    world = WorldModel()
    memory = WorkingMemory()
    planner = PlanningSystem(GreedyPlanner())
    orchestrator = Orchestrator(encoder, world, memory, planner)

    cycle = orchestrator.cycle({"price": 10.0, "volume": 5.0}, {"target": "rebalance"})
    assert cycle["state"]
    assert cycle["plan"]

    engine = SimulationEngine()
    ops = OperatorLibrary()
    ops.extend(build_qnx_operator_library(engine))
    constraints = SafetyConstraints(rules=[lambda s, g: True])
    envelope = SafetyEnvelope(bounds={})
    validator = SafetyValidator(constraints, envelope)
    vm = QNXVM(DeterministicScheduler(ops), validator)

    res = vm.run_cycle(QNXState(), {"price": 2.0, "volume": 3.0})
    assert "trace" in res
