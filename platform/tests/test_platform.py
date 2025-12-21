"""Tests for QRATUM Platform Core Infrastructure.

Tests for intent system, event chain, base module, orchestrator, and substrates.
"""

import hashlib
import json
from datetime import datetime, timedelta, timezone

import pytest

from platform.core.base import ExecutionResult, VerticalModuleBase
from platform.core.events import EventType, ExecutionEvent, MerkleEventChain
from platform.core.intent import PlatformContract, PlatformIntent, VerticalType
from platform.core.orchestrator import QRATUMPlatform
from platform.core.substrates import ComputeSubstrate, select_optimal_substrate


class TestPlatformIntent:
    """Test PlatformIntent class."""

    def test_intent_creation(self):
        """Test creating a platform intent."""
        intent = PlatformIntent(
            vertical=VerticalType.JURIS,
            operation="analyze_contract",
            parameters={"contract_text": "Sample contract"},
            compliance_attestations=frozenset(["not_legal_advice_acknowledged"]),
            user_id="test_user",
        )

        assert intent.vertical == VerticalType.JURIS
        assert intent.operation == "analyze_contract"
        assert intent.parameters["contract_text"] == "Sample contract"
        assert "not_legal_advice_acknowledged" in intent.compliance_attestations

    def test_intent_immutability(self):
        """Test that intent is immutable."""
        intent = PlatformIntent(
            vertical=VerticalType.JURIS,
            operation="test",
            parameters={"key": "value"},
        )

        # Frozen dataclasses raise FrozenInstanceError on attribute modification
        with pytest.raises((AttributeError, Exception)):  # FrozenInstanceError is a subclass
            intent.operation = "modified"

    def test_intent_validation(self):
        """Test intent validation."""
        with pytest.raises(ValueError):
            PlatformIntent(
                vertical=VerticalType.JURIS,
                operation="",  # Empty operation should fail
                parameters={},
            )


class TestPlatformContract:
    """Test PlatformContract class."""

    def test_contract_creation(self):
        """Test creating a platform contract."""
        intent = PlatformIntent(
            vertical=VerticalType.JURIS,
            operation="test",
            parameters={},
        )

        contract = PlatformContract(
            intent=intent,
            contract_hash="test_hash",
            authorized=True,
        )

        assert contract.intent == intent
        assert contract.authorized is True
        assert contract.is_valid() is True

    def test_contract_expiry(self):
        """Test contract expiration."""
        intent = PlatformIntent(
            vertical=VerticalType.JURIS,
            operation="test",
            parameters={},
        )

        # Create expired contract
        contract = PlatformContract(
            intent=intent,
            contract_hash="test_hash",
            authorized=True,
            expiry_timestamp=datetime.now(timezone.utc) - timedelta(hours=1),
        )

        assert contract.is_valid() is False

    def test_contract_attestation_check(self):
        """Test attestation checking."""
        intent = PlatformIntent(
            vertical=VerticalType.JURIS,
            operation="test",
            parameters={},
            compliance_attestations=frozenset(["attestation_a", "attestation_b"]),
        )

        contract = PlatformContract(
            intent=intent,
            contract_hash="test_hash",
            authorized=True,
        )

        assert contract.has_attestation("attestation_a") is True
        assert contract.has_attestation("attestation_c") is False


class TestExecutionEvent:
    """Test ExecutionEvent class."""

    def test_event_creation(self):
        """Test creating an execution event."""
        event = ExecutionEvent(
            event_type=EventType.EXECUTION_STARTED,
            vertical="TEST",
            operation="test_op",
            payload={"key": "value"},
        )

        assert event.event_type == EventType.EXECUTION_STARTED
        assert event.vertical == "TEST"
        assert event.operation == "test_op"

    def test_event_hash_computation(self):
        """Test event hash computation."""
        event = ExecutionEvent(
            event_type=EventType.EXECUTION_STARTED,
            vertical="TEST",
            operation="test_op",
        )

        hash1 = event.compute_hash()
        hash2 = event.compute_hash()

        # Same event should produce same hash
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA-256 hex length


class TestMerkleEventChain:
    """Test MerkleEventChain class."""

    def test_chain_initialization(self):
        """Test initializing event chain."""
        chain = MerkleEventChain()

        assert len(chain) == 0
        assert chain.current_hash is None

    def test_chain_append(self):
        """Test appending events to chain."""
        chain = MerkleEventChain()

        event1 = ExecutionEvent(
            event_type=EventType.EXECUTION_STARTED,
            vertical="TEST",
            operation="test1",
        )

        chained_event = chain.append(event1)

        assert len(chain) == 1
        assert chained_event.previous_hash is None
        assert chained_event.event_hash is not None
        assert chain.current_hash == chained_event.event_hash

    def test_chain_linking(self):
        """Test that events are properly linked."""
        chain = MerkleEventChain()

        event1 = ExecutionEvent(
            event_type=EventType.EXECUTION_STARTED,
            vertical="TEST",
            operation="test1",
        )
        event2 = ExecutionEvent(
            event_type=EventType.EXECUTION_COMPLETED,
            vertical="TEST",
            operation="test2",
        )

        chained1 = chain.append(event1)
        chained2 = chain.append(event2)

        assert chained2.previous_hash == chained1.event_hash

    def test_chain_verification(self):
        """Test chain integrity verification."""
        chain = MerkleEventChain()

        for i in range(5):
            event = ExecutionEvent(
                event_type=EventType.COMPUTATION_STEP,
                vertical="TEST",
                operation=f"test{i}",
            )
            chain.append(event)

        assert chain.verify_chain() is True

    def test_merkle_root(self):
        """Test Merkle root computation."""
        chain = MerkleEventChain()

        event1 = ExecutionEvent(
            event_type=EventType.EXECUTION_STARTED,
            vertical="TEST",
            operation="test1",
        )
        event2 = ExecutionEvent(
            event_type=EventType.EXECUTION_COMPLETED,
            vertical="TEST",
            operation="test2",
        )

        chain.append(event1)
        chain.append(event2)

        root = chain.get_merkle_root()
        assert len(root) == 64  # SHA-256 hex length

    def test_event_filtering(self):
        """Test filtering events by type and operation."""
        chain = MerkleEventChain()

        chain.append(
            ExecutionEvent(
                event_type=EventType.EXECUTION_STARTED,
                vertical="TEST",
                operation="op1",
            )
        )
        chain.append(
            ExecutionEvent(
                event_type=EventType.COMPUTATION_STEP,
                vertical="TEST",
                operation="op1",
            )
        )
        chain.append(
            ExecutionEvent(
                event_type=EventType.EXECUTION_COMPLETED,
                vertical="TEST",
                operation="op2",
            )
        )

        started_events = chain.get_events_by_type(EventType.EXECUTION_STARTED)
        assert len(started_events) == 1

        op1_events = chain.get_events_by_operation("op1")
        assert len(op1_events) == 2


class TestComputeSubstrate:
    """Test compute substrate selection."""

    def test_substrate_selection_small(self):
        """Test substrate selection for small problems."""
        substrate = select_optimal_substrate(
            problem_size=100,
            task_type="general",
            required_availability=0.9,
        )

        assert substrate in [ComputeSubstrate.CPU_SERIAL, ComputeSubstrate.CPU_PARALLEL]

    def test_substrate_selection_large(self):
        """Test substrate selection for large problems."""
        substrate = select_optimal_substrate(
            problem_size=10000000,
            task_type="matrix_ops",
            required_availability=0.7,
        )

        assert substrate in [ComputeSubstrate.GPU_CUDA, ComputeSubstrate.GPU_METAL, ComputeSubstrate.CPU_PARALLEL]

    def test_substrate_selection_quantum(self):
        """Test substrate selection for quantum algorithms."""
        substrate = select_optimal_substrate(
            problem_size=20,
            task_type="quantum_algorithms",
            required_availability=0.9,
        )

        assert substrate == ComputeSubstrate.QUANTUM_SIM


class MockVerticalModule(VerticalModuleBase):
    """Mock vertical module for testing."""

    def get_safety_disclaimer(self) -> str:
        return "TEST DISCLAIMER"

    def get_prohibited_uses(self) -> frozenset:
        return frozenset(["prohibited_action"])

    def get_required_attestations(self, operation: str) -> frozenset:
        return frozenset(["test_attestation"])

    def _execute_operation(self, contract, substrate):
        return {"result": "success", "test_data": 42}


class TestVerticalModuleBase:
    """Test VerticalModuleBase class."""

    def test_module_initialization(self):
        """Test initializing a vertical module."""
        module = MockVerticalModule("TEST", seed=42)

        assert module.vertical_name == "TEST"
        assert module.seed == 42
        assert len(module.event_chain) == 0

    def test_module_execution_success(self):
        """Test successful module execution."""
        module = MockVerticalModule("TEST", seed=42)

        intent = PlatformIntent(
            vertical=VerticalType.JURIS,
            operation="test",
            parameters={"problem_size": 100},
            compliance_attestations=frozenset(["test_attestation"]),
        )

        contract = PlatformContract(
            intent=intent,
            contract_hash="test_hash",
            authorized=True,
        )

        result = module.execute(contract)

        assert result.success is True
        assert result.result_data["result"] == "success"
        assert len(module.event_chain) > 0
        assert module.verify_audit_trail() is True

    def test_module_execution_failure(self):
        """Test module execution with invalid contract."""
        module = MockVerticalModule("TEST", seed=42)

        intent = PlatformIntent(
            vertical=VerticalType.JURIS,
            operation="test",
            parameters={},
            # Missing required attestation
        )

        contract = PlatformContract(
            intent=intent,
            contract_hash="test_hash",
            authorized=True,
        )

        result = module.execute(contract)

        assert result.success is False
        assert "attestation" in result.result_data["error"].lower()


class TestQRATUMPlatform:
    """Test QRATUMPlatform orchestrator."""

    def test_platform_initialization(self):
        """Test initializing platform."""
        platform = QRATUMPlatform(seed=42)

        assert platform.platform_seed == 42
        assert len(platform.get_registered_verticals()) == 0

    def test_module_registration(self):
        """Test registering modules."""
        platform = QRATUMPlatform(seed=42)
        module = MockVerticalModule("TEST", seed=42)

        platform.register_module(VerticalType.JURIS, module)

        verticals = platform.get_registered_verticals()
        assert VerticalType.JURIS in verticals

    def test_contract_creation(self):
        """Test creating contracts."""
        platform = QRATUMPlatform(seed=42)

        intent = PlatformIntent(
            vertical=VerticalType.JURIS,
            operation="test",
            parameters={},
        )

        contract = platform.create_contract(intent, authorized=True)

        assert contract.intent == intent
        assert contract.authorized is True
        assert len(contract.contract_hash) == 64

    def test_intent_execution(self):
        """Test executing intents through platform."""
        platform = QRATUMPlatform(seed=42)
        module = MockVerticalModule("TEST", seed=42)

        platform.register_module(VerticalType.JURIS, module)

        intent = PlatformIntent(
            vertical=VerticalType.JURIS,
            operation="test",
            parameters={"problem_size": 100},
            compliance_attestations=frozenset(["test_attestation"]),
        )

        result = platform.execute_intent(intent)

        assert result.success is True
        assert result.result_data["result"] == "success"

    def test_platform_stats(self):
        """Test getting platform statistics."""
        platform = QRATUMPlatform(seed=42)
        module = MockVerticalModule("TEST", seed=42)

        platform.register_module(VerticalType.JURIS, module)

        stats = platform.get_platform_stats()

        assert stats["registered_modules"] == 1
        assert "JURIS" in stats["verticals"]


class TestDeterminism:
    """Test deterministic execution."""

    def test_deterministic_execution(self):
        """Test that same inputs produce same outputs."""
        module1 = MockVerticalModule("TEST", seed=42)
        module2 = MockVerticalModule("TEST", seed=42)

        intent = PlatformIntent(
            vertical=VerticalType.JURIS,
            operation="test",
            parameters={"problem_size": 100},
            compliance_attestations=frozenset(["test_attestation"]),
        )

        contract = PlatformContract(
            intent=intent,
            contract_hash="test_hash",
            authorized=True,
        )

        result1 = module1.execute(contract)
        result2 = module2.execute(contract)

        # Results should be identical
        assert result1.result_data == result2.result_data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
