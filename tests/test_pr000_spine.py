"""Tests for PR-000 architectural spine normalization.

This test suite validates the new architectural layers (QCORE, QoS, QSTACK, QNX)
while ensuring backward compatibility with existing functionality.

Version: 1.0.0
Status: Production
"""

from __future__ import annotations

import pytest


def test_qcore_imports():
    """Test that QCORE layer modules can be imported and contain expected classes."""
    from qcore import Hamiltonian, SemanticState
    
    assert SemanticState is not None
    assert Hamiltonian is not None


def test_qos_imports():
    """Test that QoS layer modules can be imported and contain expected classes."""
    from qos import QoSPolicy, SafetyEnvelope
    
    assert QoSPolicy is not None
    assert SafetyEnvelope is not None


def test_qstack_dag_imports():
    """Test that QSTACK DAG primitives can be imported."""
    from qstack.dag import DAGNode, NodeType
    
    assert DAGNode is not None
    assert NodeType is not None
    assert NodeType.CLASSICAL.value == "classical"
    assert NodeType.QUANTUM.value == "quantum"


def test_qnx_executor_imports():
    """Test that QNX executor can be imported."""
    from qnx.executor import QNXExecutor
    
    assert QNXExecutor is not None


def test_backward_compatibility():
    """Test that existing SystemSession.build() still works."""
    from qstack.session import SystemSession
    
    # Build session with defaults - this should not raise any errors
    session = SystemSession.build()
    
    assert session is not None
    assert session.kernel is not None
    assert session.config is not None


def test_semantic_state_validation():
    """Test SemanticState validation rules."""
    from qcore import SemanticState
    
    # Valid state
    state = SemanticState(
        variables={"x": 1, "y": 2},
        constraints={"c1": "x + y <= 10"},
        objective="minimize x + y",
        domain="optimization",
    )
    assert state is not None
    
    # Invalid: empty variables
    with pytest.raises(ValueError, match="variables must be non-empty"):
        SemanticState(
            variables={},
            constraints={"c1": "test"},
            objective="test",
            domain="test",
        )
    
    # Invalid: empty objective
    with pytest.raises(ValueError, match="objective must be non-empty"):
        SemanticState(
            variables={"x": 1},
            constraints={"c1": "test"},
            objective=None,
            domain="test",
        )
    
    # Invalid: empty domain
    with pytest.raises(ValueError, match="domain must be non-empty"):
        SemanticState(
            variables={"x": 1},
            constraints={"c1": "test"},
            objective="test",
            domain="",
        )


def test_semantic_state_hashing():
    """Test deterministic hash computation for SemanticState."""
    from qcore import SemanticState
    
    state1 = SemanticState(
        variables={"x": 1, "y": 2},
        constraints={"c1": "test"},
        objective="minimize",
        domain="optimization",
    )
    
    state2 = SemanticState(
        variables={"x": 1, "y": 2},
        constraints={"c1": "test"},
        objective="minimize",
        domain="optimization",
    )
    
    # Same state should produce same hash
    assert state1.compute_hash() == state2.compute_hash()
    
    # Different state should produce different hash
    state3 = SemanticState(
        variables={"x": 1, "y": 3},
        constraints={"c1": "test"},
        objective="minimize",
        domain="optimization",
    )
    
    assert state1.compute_hash() != state3.compute_hash()
    
    # Hash should be SHA-256 (64 hex characters)
    hash_value = state1.compute_hash()
    assert len(hash_value) == 64
    assert all(c in "0123456789abcdef" for c in hash_value)


def test_domain_validators():
    """Test chemistry, optimization, and finance domain validators."""
    from qcore import SemanticState
    
    # Chemistry domain validator
    with pytest.raises(ValueError, match="Chemistry domain requires 'molecule' in metadata"):
        SemanticState(
            variables={"x": 1},
            constraints={"c1": "test"},
            objective="test",
            domain="chemistry",
            metadata={},
        )
    
    # Valid chemistry state
    chem_state = SemanticState(
        variables={"x": 1},
        constraints={"c1": "test"},
        objective="test",
        domain="chemistry",
        metadata={"molecule": "H2O"},
    )
    assert chem_state is not None
    
    # Optimization domain validator
    with pytest.raises(ValueError, match="Optimization domain requires non-empty constraints"):
        SemanticState(
            variables={"x": 1},
            constraints={},
            objective="test",
            domain="optimization",
        )
    
    # Valid optimization state
    opt_state = SemanticState(
        variables={"x": 1},
        constraints={"c1": "x >= 0"},
        objective="minimize x",
        domain="optimization",
    )
    assert opt_state is not None
    
    # Finance domain validator
    with pytest.raises(ValueError, match="Finance domain requires 'risk_tolerance' in metadata"):
        SemanticState(
            variables={"x": 1},
            constraints={"c1": "test"},
            objective="test",
            domain="finance",
            metadata={},
        )
    
    # Valid finance state
    finance_state = SemanticState(
        variables={"x": 1},
        constraints={"c1": "test"},
        objective="test",
        domain="finance",
        metadata={"risk_tolerance": 0.5},
    )
    assert finance_state is not None


def test_hamiltonian_structure():
    """Test Hamiltonian structure and basic operations."""
    from qcore import Hamiltonian, PauliTerm
    
    # Create some Pauli terms
    term1 = PauliTerm(coefficient=1.0 + 0.5j, operators=[(0, "X"), (1, "Y")])
    term2 = PauliTerm(coefficient=2.0, operators=[(0, "Z")])
    
    hamiltonian = Hamiltonian([term1, term2])
    
    # Test num_qubits
    assert hamiltonian.num_qubits() == 2
    
    # Test encode
    encoded = hamiltonian.encode()
    assert len(encoded) == 2
    assert encoded[0]["coefficient"]["real"] == 1.0
    assert encoded[0]["coefficient"]["imag"] == 0.5
    
    # Test to_json
    json_str = hamiltonian.to_json()
    assert "num_qubits" in json_str
    assert "terms" in json_str


def test_hamiltonian_stub_methods():
    """Test that Hamiltonian stub methods raise NotImplementedError."""
    from qcore import Hamiltonian, PauliTerm
    
    hamiltonian = Hamiltonian([PauliTerm(coefficient=1.0, operators=[(0, "X")])])
    
    # Test energy() stub
    with pytest.raises(NotImplementedError, match="PR-003"):
        hamiltonian.energy(None)
    
    # Test from_semantic_state() stub
    with pytest.raises(NotImplementedError, match="PR-004"):
        Hamiltonian.from_semantic_state(None)


def test_qos_policy_validation():
    """Test QoS policy validation."""
    from qos import QoSPolicy
    
    # Valid policy
    policy = QoSPolicy(name="test_policy", constraints={"max_time": 60}, priority=1)
    policy.validate()
    
    # Invalid: empty name
    with pytest.raises(ValueError, match="Policy name must be non-empty"):
        policy_bad = QoSPolicy(name="", constraints={}, priority=0)
        policy_bad.validate()
    
    # Invalid: negative priority
    with pytest.raises(ValueError, match="Priority must be non-negative"):
        policy_bad = QoSPolicy(name="test", constraints={}, priority=-1)
        policy_bad.validate()


def test_safety_envelope_validation():
    """Test SafetyEnvelope validation."""
    from qos import SafetyEnvelope
    
    # Valid envelope
    envelope = SafetyEnvelope(
        max_qubits=100,
        max_depth=1000,
        max_runtime=3600.0,
        resource_limits={"memory": "8GB"},
    )
    envelope.validate()
    
    # Invalid: non-positive max_qubits
    with pytest.raises(ValueError, match="max_qubits must be positive"):
        envelope_bad = SafetyEnvelope(
            max_qubits=0, max_depth=100, max_runtime=60.0, resource_limits={}
        )
        envelope_bad.validate()
    
    # Invalid: non-positive max_depth
    with pytest.raises(ValueError, match="max_depth must be positive"):
        envelope_bad = SafetyEnvelope(
            max_qubits=10, max_depth=-1, max_runtime=60.0, resource_limits={}
        )
        envelope_bad.validate()
    
    # Invalid: non-positive max_runtime
    with pytest.raises(ValueError, match="max_runtime must be positive"):
        envelope_bad = SafetyEnvelope(
            max_qubits=10, max_depth=100, max_runtime=0.0, resource_limits={}
        )
        envelope_bad.validate()


def test_dag_node_creation():
    """Test DAG node creation."""
    from qstack.dag import DAGNode, NodeType
    
    node = DAGNode(
        id="node1",
        type=NodeType.QUANTUM,
        dependencies=["node0"],
        payload={"circuit": "test"},
    )
    
    assert node.id == "node1"
    assert node.type == NodeType.QUANTUM
    assert node.dependencies == ["node0"]


def test_scheduler_stub():
    """Test scheduler stub implementation."""
    from qstack.scheduler import Scheduler
    
    scheduler = Scheduler()
    job = {"task": "test"}
    
    # Stub should return job unchanged
    result = scheduler.schedule(job)
    assert result == job


def test_orchestrator_delegation():
    """Test that Orchestrator delegates to kernel."""
    from qstack.orchestrator import Orchestrator
    from qstack.session import SystemSession
    
    session = SystemSession.build()
    orchestrator = Orchestrator(session)
    
    # Verify orchestrator has kernel
    assert orchestrator.kernel is not None
    assert orchestrator.kernel == session.kernel


def test_qnx_executor_creation():
    """Test QNXExecutor creation."""
    from qnx.executor import QNXExecutor
    
    executor = QNXExecutor()
    
    # Verify executor has substrate
    assert executor.substrate is not None


def test_backend_stubs():
    """Test that backend stubs raise NotImplementedError."""
    from qnx.backends import CPUBackend, GPUBackend, QPUBackend
    
    cpu = CPUBackend()
    gpu = GPUBackend()
    qpu = QPUBackend()
    
    with pytest.raises(NotImplementedError):
        cpu.run(None)
    
    with pytest.raises(NotImplementedError):
        gpu.run(None)
    
    with pytest.raises(NotImplementedError):
        qpu.run(None)


def test_pauli_term_serialization():
    """Test PauliTerm serialization and validation."""
    from qcore import PauliTerm
    
    term = PauliTerm(coefficient=2.5 + 1.5j, operators=[(0, "X"), (1, "Y"), (2, "Z")])
    
    # Test to_dict
    term_dict = term.to_dict()
    assert term_dict["coefficient"]["real"] == 2.5
    assert term_dict["coefficient"]["imag"] == 1.5
    assert term_dict["operators"] == [(0, "X"), (1, "Y"), (2, "Z")]
    
    # Test __str__
    str_repr = str(term)
    assert "2.5" in str_repr or "(2.5+1.5j)" in str_repr
    
    # Test invalid Pauli operator
    with pytest.raises(ValueError, match="Invalid Pauli operator"):
        PauliTerm(coefficient=1.0, operators=[(0, "A")])


def test_semantic_state_serialization():
    """Test SemanticState serialization."""
    from qcore import SemanticState
    
    state = SemanticState(
        variables={"x": 1, "y": 2},
        constraints={"c1": "x + y <= 10"},
        objective="minimize x + y",
        domain="optimization",
        metadata={"solver": "cplex"},
    )
    
    # Test serialize
    serialized = state.serialize()
    assert "variables" in serialized
    assert "constraints" in serialized
    assert "objective" in serialized
    assert "domain" in serialized
    assert "metadata" in serialized
    
    # Test to_json
    json_str = state.to_json()
    assert '"domain": "optimization"' in json_str
