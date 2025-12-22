"""Unit tests for contract system.

Tests all contract types (Intent, Capability, Temporal, Event) and base contract.
"""

from datetime import datetime, timedelta

import pytest

from contracts import (
    BaseContract,
    ClusterTopology,
    create_capability_contract,
    create_event_contract,
    create_intent_contract,
    create_temporal_contract,
    get_current_timestamp,
)
from qil import parse_intent


class TestBaseContract:
    """Test suite for base contract."""

    def test_base_contract_creation(self):
        """Test creating a base contract."""
        timestamp = get_current_timestamp()
        contract = BaseContract(
            contract_id="test_id",
            contract_type="TestContract",
            created_at=timestamp,
        )

        assert contract.contract_id == "test_id"
        assert contract.contract_type == "TestContract"
        assert contract.created_at == timestamp

    def test_base_contract_serialization(self):
        """Test base contract serialization."""
        timestamp = get_current_timestamp()
        contract = BaseContract(
            contract_id="test_id",
            contract_type="TestContract",
            created_at=timestamp,
        )

        serialized = contract.serialize()
        assert serialized["contract_id"] == "test_id"
        assert serialized["contract_type"] == "TestContract"
        assert "created_at" in serialized

    def test_base_contract_immutability(self):
        """Test that base contracts are immutable."""
        timestamp = get_current_timestamp()
        contract = BaseContract(
            contract_id="test_id",
            contract_type="TestContract",
            created_at=timestamp,
        )

        with pytest.raises(AttributeError):  # FrozenInstanceError from dataclasses
            contract.contract_id = "new_id"


class TestIntentContract:
    """Test suite for intent contract."""

    def test_create_intent_contract(self):
        """Test creating an intent contract."""
        qil_text = """
        INTENT test_intent {
            OBJECTIVE test_objective
            AUTHORITY user: alice
        }
        """
        intent = parse_intent(qil_text)

        contract = create_intent_contract(
            intent=intent,
            authorization_proof="AUTH_PROOF_123",
        )

        assert contract.intent_name == "test_intent"
        assert contract.objective == "test_objective"
        assert contract.authorization_proof == "AUTH_PROOF_123"
        assert contract.contract_type == "IntentContract"

    def test_intent_contract_deadline(self):
        """Test getting deadline from intent contract."""
        qil_text = """
        INTENT timed_intent {
            OBJECTIVE test
            TIME deadline: 3600s
            AUTHORITY user: alice
        }
        """
        intent = parse_intent(qil_text)
        contract = create_intent_contract(intent, "AUTH_PROOF")

        deadline = contract.get_deadline_seconds()
        assert deadline == 3600.0

    def test_intent_contract_requires_cluster(self):
        """Test cluster requirement checking."""
        qil_text = """
        INTENT gpu_intent {
            OBJECTIVE test
            HARDWARE ONLY GB200
            AUTHORITY user: alice
        }
        """
        intent = parse_intent(qil_text)
        contract = create_intent_contract(intent, "AUTH_PROOF")

        assert contract.requires_cluster("GB200")
        assert not contract.requires_cluster("CPU")


class TestCapabilityContract:
    """Test suite for capability contract."""

    def test_create_capability_contract(self):
        """Test creating a capability contract."""
        topology = ClusterTopology(
            cluster_type="GB200",
            node_count=2,
            accelerators_per_node=8,
            memory_per_node_gb=1536,
            interconnect="NVLink5",
        )

        contract = create_capability_contract(
            intent_contract_id="INTENT_123",
            topology=topology,
            allocated_resources={"test": "data"},
            capability_proof="CAP_PROOF",
        )

        assert contract.intent_contract_id == "INTENT_123"
        assert contract.get_cluster_type() == "GB200"
        assert contract.capability_proof == "CAP_PROOF"

    def test_cluster_topology(self):
        """Test cluster topology calculations."""
        topology = ClusterTopology(
            cluster_type="GB200",
            node_count=4,
            accelerators_per_node=8,
            memory_per_node_gb=1536,
            interconnect="NVLink5",
        )

        assert topology.total_accelerators() == 32  # 4 nodes * 8 accel
        assert topology.total_memory_gb() == 6144  # 4 nodes * 1536 GB

    def test_cluster_topology_validation(self):
        """Test cluster topology validation."""
        with pytest.raises(ValueError, match="node_count must be positive"):
            ClusterTopology(
                cluster_type="GB200",
                node_count=0,  # Invalid!
                accelerators_per_node=8,
                memory_per_node_gb=1536,
                interconnect="NVLink5",
            )


class TestTemporalContract:
    """Test suite for temporal contract."""

    def test_create_temporal_contract(self):
        """Test creating a temporal contract."""
        contract = create_temporal_contract(
            intent_contract_id="INTENT_123",
            deadline_seconds=3600,
            budget_seconds=1800,
            temporal_proof="TIME_PROOF",
        )

        assert contract.intent_contract_id == "INTENT_123"
        assert contract.deadline_seconds == 3600
        assert contract.budget_seconds == 1800

    def test_temporal_contract_absolute_deadline(self):
        """Test calculating absolute deadline."""
        contract = create_temporal_contract(
            intent_contract_id="INTENT_123",
            deadline_seconds=3600,
            temporal_proof="TIME_PROOF",
        )

        deadline = contract.get_absolute_deadline()
        created = datetime.fromisoformat(contract.created_at.replace("Z", "+00:00"))
        expected = created + timedelta(seconds=3600)

        # Should be within 1 second (account for processing time)
        assert abs((deadline - expected).total_seconds()) < 1

    def test_temporal_contract_expiration(self):
        """Test checking contract expiration."""
        contract = create_temporal_contract(
            intent_contract_id="INTENT_123",
            deadline_seconds=3600,
            temporal_proof="TIME_PROOF",
        )

        # Should not be expired immediately
        assert not contract.is_expired()

        # Test with future time (should not be expired)
        from datetime import timezone
        future_time = datetime.now(timezone.utc) + timedelta(seconds=1000)
        assert not contract.is_expired(future_time)

        # Test with far future time (should be expired)
        far_future = datetime.now(timezone.utc) + timedelta(seconds=7200)
        assert contract.is_expired(far_future)

    def test_temporal_contract_time_remaining(self):
        """Test calculating remaining time."""
        contract = create_temporal_contract(
            intent_contract_id="INTENT_123",
            deadline_seconds=3600,
            temporal_proof="TIME_PROOF",
        )

        remaining = contract.time_remaining_seconds()
        # Should be close to 3600 seconds
        assert 3590 < remaining <= 3600

    def test_temporal_contract_window(self):
        """Test execution window checking."""
        from datetime import timezone
        now = datetime.now(timezone.utc)
        window_start = (now + timedelta(hours=-1)).isoformat().replace("+00:00", "Z")
        window_end = (now + timedelta(hours=1)).isoformat().replace("+00:00", "Z")

        contract = create_temporal_contract(
            intent_contract_id="INTENT_123",
            deadline_seconds=7200,
            window_start=window_start,
            window_end=window_end,
            temporal_proof="TIME_PROOF",
        )

        # Current time should be within window
        assert contract.is_within_window()

        # Past should not be within window
        past = now - timedelta(hours=2)
        assert not contract.is_within_window(past)

        # Far future should not be within window
        future = now + timedelta(hours=2)
        assert not contract.is_within_window(future)


class TestEventContract:
    """Test suite for event contract."""

    def test_create_event_contract(self):
        """Test creating an event contract."""
        expected_events = [
            "IntentReceived",
            "IntentAuthorized",
            "ExecutionStarted",
            "ExecutionCompleted",
        ]

        contract = create_event_contract(
            intent_contract_id="INTENT_123",
            expected_events=expected_events,
            event_proof="EVENT_PROOF",
        )

        assert contract.intent_contract_id == "INTENT_123"
        assert len(contract.expected_events) == 4
        assert contract.has_event("IntentReceived")

    def test_event_contract_sequence_validation(self):
        """Test event sequence validation."""
        expected_events = [
            "EventA",
            "EventB",
            "EventC",
        ]

        contract = create_event_contract(
            intent_contract_id="INTENT_123",
            expected_events=expected_events,
            event_proof="EVENT_PROOF",
        )

        # Valid sequence
        assert contract.is_ordered_correctly(["EventA", "EventB"])
        assert contract.is_ordered_correctly(["EventA", "EventC"])

        # Invalid sequence (out of order)
        assert not contract.is_ordered_correctly(["EventB", "EventA"])

        # Invalid sequence (unexpected event)
        assert not contract.is_ordered_correctly(["EventX"])

    def test_event_contract_next_expected(self):
        """Test getting next expected event."""
        expected_events = ["EventA", "EventB", "EventC"]

        contract = create_event_contract(
            intent_contract_id="INTENT_123",
            expected_events=expected_events,
            event_proof="EVENT_PROOF",
        )

        # First event
        assert contract.get_next_expected_event([]) == "EventA"

        # Second event
        assert contract.get_next_expected_event(["EventA"]) == "EventB"

        # Third event
        assert contract.get_next_expected_event(["EventA", "EventB"]) == "EventC"

        # No more events
        assert contract.get_next_expected_event(["EventA", "EventB", "EventC"]) is None


class TestContractIntegration:
    """Integration tests for contract system."""

    def test_full_contract_flow(self):
        """Test creating all contracts for an intent."""
        qil_text = """
        INTENT integration_test {
            OBJECTIVE test_objective
            HARDWARE ONLY GB200
            TIME deadline: 3600s
            AUTHORITY user: alice
        }
        """
        intent = parse_intent(qil_text)

        # Create intent contract
        intent_contract = create_intent_contract(intent, "AUTH_PROOF")

        # Create capability contract
        topology = ClusterTopology(
            cluster_type="GB200",
            node_count=1,
            accelerators_per_node=8,
            memory_per_node_gb=1536,
            interconnect="NVLink5",
        )
        capability_contract = create_capability_contract(
            intent_contract_id=intent_contract.contract_id,
            topology=topology,
            allocated_resources={},
            capability_proof="CAP_PROOF",
        )

        # Create temporal contract
        temporal_contract = create_temporal_contract(
            intent_contract_id=intent_contract.contract_id,
            deadline_seconds=3600,
            temporal_proof="TIME_PROOF",
        )

        # Create event contract
        event_contract = create_event_contract(
            intent_contract_id=intent_contract.contract_id,
            expected_events=["Start", "Complete"],
            event_proof="EVENT_PROOF",
        )

        # Verify all contracts reference same intent
        assert capability_contract.intent_contract_id == intent_contract.contract_id
        assert temporal_contract.intent_contract_id == intent_contract.contract_id
        assert event_contract.intent_contract_id == intent_contract.contract_id
