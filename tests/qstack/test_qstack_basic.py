from qstack.q import Attestor, KeyManager, QIdentity, Signer, SovereignObject
from qstack.qnx import (QNXVM, DeterministicScheduler, OperatorLibrary,
                        QNXState, SafetyConstraints, SafetyEnvelope,
                        SafetyValidator)
from qstack.qnx_agi.memory.base import MemorySystem
from qstack.qnx_agi.orchestrator.base import Orchestrator
from qstack.qnx_agi.perception.base import PerceptionLayer
from qstack.qnx_agi.planning.base import Planner, PlanningSystem
from qstack.qnx_agi.worldmodel.base import WorldModel
from qstack.quasim.core.engine import SimulationEngine
from qstack.qunimbus.core.engine import QuNimbusEngine, ValuationInput


def test_identity_and_attestation():
    km = KeyManager(seed="seed")
    signer = Signer(key=km.derive_key("device"))
    identity = QIdentity(name="device", key=signer.key)
    sovereign = SovereignObject(identity=identity, claims={"role": "test"})
    attestor = Attestor(signer=signer)
    attestation = attestor.attest(sovereign.describe())
    assert attestor.verify(attestation)


def test_runtime_cycle():
    state = QNXState()
    ops = OperatorLibrary()
    ops.register("set_goal", lambda s, g: s.update("goal", g))
    ops.register("echo", lambda s, g: g)
    scheduler = DeterministicScheduler(operators=ops)
    safety = SafetyValidator(
        constraints=SafetyConstraints(rules=[lambda _s, _g: True]),
        envelope=SafetyEnvelope(bounds={}),
    )
    vm = QNXVM(scheduler=scheduler, safety=safety)
    result = vm.run_cycle(state, goal={"target": 1})
    assert "trace" in result
    assert state.read("goal") == {"target": 1}


def test_orchestrator_cycle():
    perception = PerceptionLayer()
    world = WorldModel()
    memory = MemorySystem(capacity=4)
    planner = PlanningSystem(Planner())
    orchestrator = Orchestrator(perception, world, memory, planner)
    output = orchestrator.cycle(raw_input={"sensor": 5}, goal={"target": 10})
    assert output["plan"]
    assert memory.latest() is not None


def test_simulation_and_qunimbus():
    engine = SimulationEngine()
    engine.add_step(lambda x: x + 1)
    engine.add_step(lambda x: x * 2)
    outputs = engine.run(1)
    assert outputs == [2, 4]

    economics = QuNimbusEngine(weights={"a": 0.5, "b": 1.5})
    valuation = economics.evaluate(ValuationInput(metrics={"a": 2, "b": 4}))
    assert valuation == 2 * 0.5 + 4 * 1.5
