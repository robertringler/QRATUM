"""Tests for QRATUM Platform core functionality."""

import time

import pytest

from qratum_platform.core import (
    ComputeSubstrate,
    ExecutionEvent,
    MerkleEventChain,
    PlatformContract,
    PlatformIntent,
    QRATUMPlatform,
    SafetyViolation,
    VerticalModule,
    VerticalModuleBase,
)


class TestVerticalModuleEnum:
    """Test VerticalModule enumeration."""

    def test_all_verticals_defined(self):
        """Test that all 14 verticals are defined."""
        verticals = list(VerticalModule)
        assert len(verticals) == 14

    def test_vertical_values(self):
        """Test vertical enum values."""
        assert VerticalModule.JURIS.value == "juris"
        assert VerticalModule.VITRA.value == "vitra"
        assert VerticalModule.ECORA.value == "ecora"


class TestComputeSubstrateEnum:
    """Test ComputeSubstrate enumeration."""

    def test_all_substrates_defined(self):
        """Test that all substrates are defined."""
        substrates = list(ComputeSubstrate)
        assert len(substrates) >= 7

    def test_substrate_values(self):
        """Test substrate enum values."""
        assert ComputeSubstrate.GB200.value == "gb200"
        assert ComputeSubstrate.QPU.value == "qpu"


class TestPlatformIntent:
    """Test PlatformIntent dataclass."""

    def test_intent_creation(self):
        """Test creating a platform intent."""
        intent = PlatformIntent(
            vertical=VerticalModule.JURIS,
            operation="legal_reasoning",
            parameters={"facts": "test facts"},
            user_id="user_001",
        )

        assert intent.vertical == VerticalModule.JURIS
        assert intent.operation == "legal_reasoning"
        assert intent.user_id == "user_001"

    def test_intent_immutability(self):
        """Test that intent is immutable (frozen)."""
        intent = PlatformIntent(
            vertical=VerticalModule.JURIS,
            operation="test",
            parameters={},
            user_id="user_001",
        )

        with pytest.raises(AttributeError):
            intent.operation = "modified"

    def test_intent_to_dict(self):
        """Test intent serialization to dict."""
        intent = PlatformIntent(
            vertical=VerticalModule.JURIS,
            operation="test",
            parameters={"key": "value"},
            user_id="user_001",
        )

        data = intent.to_dict()
        assert data["vertical"] == "juris"
        assert data["operation"] == "test"
        assert data["user_id"] == "user_001"

    def test_intent_hash_deterministic(self):
        """Test that intent hash is deterministic."""
        intent1 = PlatformIntent(
            vertical=VerticalModule.JURIS,
            operation="test",
            parameters={"key": "value"},
            user_id="user_001",
            timestamp=1234567890.0,
        )

        intent2 = PlatformIntent(
            vertical=VerticalModule.JURIS,
            operation="test",
            parameters={"key": "value"},
            user_id="user_001",
            timestamp=1234567890.0,
        )

        assert intent1.compute_hash() == intent2.compute_hash()


class TestPlatformContract:
    """Test PlatformContract dataclass."""

    def test_contract_creation(self):
        """Test creating a platform contract."""
        intent = PlatformIntent(
            vertical=VerticalModule.JURIS,
            operation="test",
            parameters={},
            user_id="user_001",
        )

        contract = PlatformContract(
            intent=intent,
            contract_id="contract_001",
            substrate=ComputeSubstrate.CPU,
            estimated_cost=10.0,
            estimated_duration=5.0,
            safety_checks_passed=True,
        )

        assert contract.contract_id == "contract_001"
        assert contract.substrate == ComputeSubstrate.CPU

    def test_contract_immutability(self):
        """Test that contract is immutable (frozen)."""
        intent = PlatformIntent(
            vertical=VerticalModule.JURIS,
            operation="test",
            parameters={},
            user_id="user_001",
        )

        contract = PlatformContract(
            intent=intent,
            contract_id="contract_001",
            substrate=ComputeSubstrate.CPU,
            estimated_cost=10.0,
            estimated_duration=5.0,
            safety_checks_passed=True,
        )

        with pytest.raises(AttributeError):
            contract.estimated_cost = 20.0


class TestExecutionEvent:
    """Test ExecutionEvent dataclass."""

    def test_event_creation(self):
        """Test creating an execution event."""
        event = ExecutionEvent(
            event_type="test_event",
            contract_id="contract_001",
            timestamp=time.time(),
            data={"key": "value"},
        )

        assert event.event_type == "test_event"
        assert event.contract_id == "contract_001"
        assert event.event_hash is not None

    def test_event_hash_computed(self):
        """Test that event hash is computed automatically."""
        event = ExecutionEvent(
            event_type="test_event",
            contract_id="contract_001",
            timestamp=1234567890.0,
            data={"key": "value"},
        )

        assert event.event_hash is not None
        assert len(event.event_hash) == 64  # SHA-256 hex length


class TestMerkleEventChain:
    """Test MerkleEventChain class."""

    def test_chain_creation(self):
        """Test creating an event chain."""
        chain = MerkleEventChain()
        assert len(chain.events) == 0
        assert chain.last_hash is None

    def test_append_event(self):
        """Test appending events to chain."""
        chain = MerkleEventChain()

        event1 = chain.append("event_type_1", "contract_001", {"data": "1"})
        assert event1.previous_hash is None

        event2 = chain.append("event_type_2", "contract_001", {"data": "2"})
        assert event2.previous_hash == event1.event_hash

    def test_chain_integrity(self):
        """Test verifying chain integrity."""
        chain = MerkleEventChain()

        chain.append("event_1", "contract_001", {})
        chain.append("event_2", "contract_001", {})
        chain.append("event_3", "contract_001", {})

        assert chain.verify_chain() is True

    def test_get_events_for_contract(self):
        """Test filtering events by contract."""
        chain = MerkleEventChain()

        chain.append("event_1", "contract_001", {})
        chain.append("event_2", "contract_002", {})
        chain.append("event_3", "contract_001", {})

        events = chain.get_events_for_contract("contract_001")
        assert len(events) == 2


class MockModule(VerticalModuleBase):
    """Mock module for testing."""

    MODULE_NAME = "MOCK"
    MODULE_VERSION = "1.0.0"
    SAFETY_DISCLAIMER = "Mock disclaimer"
    PROHIBITED_USES = ["prohibited_action"]

    def execute(self, contract):
        return {"result": "success"}

    def get_optimal_substrate(self, operation, parameters):
        return ComputeSubstrate.CPU


class TestVerticalModuleBase:
    """Test VerticalModuleBase abstract class."""

    def test_module_creation(self):
        """Test creating a module instance."""
        module = MockModule()
        assert module.MODULE_NAME == "MOCK"
        assert module.event_chain is not None

    def test_safety_check_passes(self):
        """Test safety check with valid intent."""
        module = MockModule()
        intent = PlatformIntent(
            vertical=VerticalModule.JURIS,
            operation="safe_operation",
            parameters={},
            user_id="user_001",
        )

        assert module.check_safety(intent) is True

    def test_safety_check_fails(self):
        """Test safety check with prohibited use."""
        module = MockModule()
        intent = PlatformIntent(
            vertical=VerticalModule.JURIS,
            operation="prohibited_action",
            parameters={},
            user_id="user_001",
        )

        with pytest.raises(SafetyViolation):
            module.check_safety(intent)

    def test_emit_event(self):
        """Test emitting events."""
        module = MockModule()
        event = module.emit_event("test_event", "contract_001", {"data": "test"})

        assert event.event_type == "test_event"
        assert len(module.event_chain.events) == 1


class TestQRATUMPlatform:
    """Test QRATUMPlatform orchestrator."""

    def test_platform_creation(self):
        """Test creating platform instance."""
        platform = QRATUMPlatform()
        assert platform.event_chain is not None
        assert len(platform.modules) == 0

    def test_register_module(self):
        """Test registering a module."""
        platform = QRATUMPlatform()
        module = MockModule()

        platform.register_module(VerticalModule.JURIS, module)
        assert VerticalModule.JURIS in platform.modules

    def test_create_contract(self):
        """Test creating a contract."""
        platform = QRATUMPlatform()
        module = MockModule()
        platform.register_module(VerticalModule.JURIS, module)

        intent = PlatformIntent(
            vertical=VerticalModule.JURIS,
            operation="test",
            parameters={},
            user_id="user_001",
        )

        contract = platform.create_contract(intent)
        assert contract.contract_id.startswith("juris_")
        assert contract.safety_checks_passed is True

    def test_create_contract_unregistered_module(self):
        """Test creating contract for unregistered module fails."""
        platform = QRATUMPlatform()

        intent = PlatformIntent(
            vertical=VerticalModule.JURIS,
            operation="test",
            parameters={},
            user_id="user_001",
        )

        with pytest.raises(ValueError):
            platform.create_contract(intent)

    def test_execute_contract(self):
        """Test executing a contract."""
        platform = QRATUMPlatform()
        module = MockModule()
        platform.register_module(VerticalModule.JURIS, module)

        intent = PlatformIntent(
            vertical=VerticalModule.JURIS,
            operation="test",
            parameters={},
            user_id="user_001",
        )

        contract = platform.create_contract(intent)
        result = platform.execute_contract(contract.contract_id)

        assert result["result"] == "success"

    def test_platform_integrity(self):
        """Test platform event chain integrity."""
        platform = QRATUMPlatform()
        module = MockModule()
        platform.register_module(VerticalModule.JURIS, module)

        intent = PlatformIntent(
            vertical=VerticalModule.JURIS,
            operation="test",
            parameters={},
            user_id="user_001",
        )

        contract = platform.create_contract(intent)
        platform.execute_contract(contract.contract_id)

        assert platform.verify_integrity() is True
