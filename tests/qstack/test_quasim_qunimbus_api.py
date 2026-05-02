import pytest

from qstack.quasim import QuantumCircuit
from qstack.quasim.quantum.gates import identity
from qstack.qunimbus import QuNimbusEngine, valuation_operator


def test_quasim_and_qunimbus_interfaces():
    circuit = QuantumCircuit()
    circuit.add_gate(identity())
    assert circuit.evaluate(1.0) == 1.0

    engine = QuNimbusEngine(weights={"risk": 0.5, "reward": 1.0})
    op = valuation_operator(engine)
    result = op({}, {"risk": 2.0, "reward": 4.0})
    assert result == pytest.approx(5.0)
