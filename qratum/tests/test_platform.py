"""
Tests for QRATUM Platform Core Infrastructure

Tests for PlatformIntent, PlatformContract, MerkleEventChain,
and PlatformOrchestrator.
"""

import pytest
from qratum.platform import (
    PlatformIntent,
    PlatformContract,
    Event,
    EventType,
    create_contract_from_intent,
    create_event,
    MerkleEventChain,
    PlatformOrchestrator,
    ContractStatus,
)


class TestPlatformIntent:
    """Test PlatformIntent creation and immutability"""
    
    def test_intent_creation(self):
        """Test basic intent creation"""
        intent = PlatformIntent(
            vertical="JURIS",
            task="analyze_contract",
            parameters={"contract_text": "Sample contract"},
            requester_id="user_123",
        )
        
        assert intent.vertical == "JURIS"
        assert intent.task == "analyze_contract"
        assert intent.requester_id == "user_123"
        assert intent.intent_id.startswith("intent_")
    
    def test_intent_immutability(self):
        """Test that intents are immutable"""
        intent = PlatformIntent(
            vertical="JURIS",
            task="analyze_contract",
            parameters={"contract_text": "Sample"},
            requester_id="user_123",
        )
        
        with pytest.raises(Exception):
            intent.vertical = "VITRA"
    
    def test_intent_deterministic_id(self):
        """Test that intent ID is deterministic"""
        intent1 = PlatformIntent(
            vertical="JURIS",
            task="analyze_contract",
            parameters={"text": "same"},
            requester_id="user_123",
            timestamp="2024-01-01T00:00:00Z",
        )
        intent2 = PlatformIntent(
            vertical="JURIS",
            task="analyze_contract",
            parameters={"text": "same"},
            requester_id="user_123",
            timestamp="2024-01-01T00:00:00Z",
        )
        
        assert intent1.intent_id == intent2.intent_id


class TestPlatformContract:
    """Test PlatformContract creation and validation"""
    
    def test_contract_creation(self):
        """Test contract creation from intent"""
        intent = PlatformIntent(
            vertical="JURIS",
            task="analyze_contract",
            parameters={"contract_text": "Sample"},
            requester_id="user_123",
        )
        
        contract = create_contract_from_intent(intent)
        
        assert contract.intent == intent
        assert contract.authorized_by == "Q-Core"
        assert contract.status == ContractStatus.AUTHORIZED
        assert contract.signature != ""
    
    def test_contract_signature_verification(self):
        """Test contract signature can be verified"""
        intent = PlatformIntent(
            vertical="JURIS",
            task="analyze_contract",
            parameters={"text": "Sample"},
            requester_id="user_123",
        )
        
        contract = create_contract_from_intent(intent)
        
        assert contract.verify_signature() is True
    
    def test_contract_immutability(self):
        """Test that contracts are immutable"""
        intent = PlatformIntent(
            vertical="JURIS",
            task="analyze_contract",
            parameters={"text": "Sample"},
            requester_id="user_123",
        )
        
        contract = create_contract_from_intent(intent)
        
        with pytest.raises(Exception):
            contract.status = ContractStatus.COMPLETED


class TestMerkleEventChain:
    """Test MerkleEventChain integrity and operations"""
    
    def test_chain_initialization(self):
        """Test empty chain initialization"""
        chain = MerkleEventChain()
        
        assert len(chain) == 0
        assert chain.verify_integrity() is True
    
    def test_append_event(self):
        """Test appending events to chain"""
        chain = MerkleEventChain()
        
        event = create_event(
            event_type=EventType.CONTRACT_CREATED,
            contract_id="contract_123",
            data={"test": "data"},
            emitter="test",
        )
        
        node_hash = chain.append(event)
        
        assert len(chain) == 1
        assert node_hash != ""
        assert chain.verify_integrity() is True
    
    def test_chain_integrity(self):
        """Test chain maintains cryptographic integrity"""
        chain = MerkleEventChain()
        
        # Add multiple events
        for i in range(5):
            event = create_event(
                event_type=EventType.TASK_COMPLETED,
                contract_id=f"contract_{i}",
                data={"index": i},
                emitter="test",
            )
            chain.append(event)
        
        assert len(chain) == 5
        assert chain.verify_integrity() is True
    
    def test_get_events_by_contract(self):
        """Test filtering events by contract ID"""
        chain = MerkleEventChain()
        
        # Add events for different contracts
        for i in range(3):
            event = create_event(
                event_type=EventType.TASK_COMPLETED,
                contract_id="contract_1",
                data={"index": i},
                emitter="test",
            )
            chain.append(event)
        
        for i in range(2):
            event = create_event(
                event_type=EventType.TASK_COMPLETED,
                contract_id="contract_2",
                data={"index": i},
                emitter="test",
            )
            chain.append(event)
        
        contract_1_events = chain.get_events(contract_id="contract_1")
        assert len(contract_1_events) == 3
        
        contract_2_events = chain.get_events(contract_id="contract_2")
        assert len(contract_2_events) == 2
    
    def test_replay_events(self):
        """Test event replay functionality"""
        chain = MerkleEventChain()
        
        contract_id = "contract_replay"
        
        for i in range(3):
            event = create_event(
                event_type=EventType.TASK_COMPLETED,
                contract_id=contract_id,
                data={"step": i},
                emitter="test",
            )
            chain.append(event)
        
        replay = chain.replay_events(contract_id)
        
        assert len(replay) == 3
        assert replay[0]["data"]["step"] == 0
        assert replay[2]["data"]["step"] == 2


class TestPlatformOrchestrator:
    """Test PlatformOrchestrator routing and execution"""
    
    def test_orchestrator_initialization(self):
        """Test orchestrator initialization"""
        orchestrator = PlatformOrchestrator()
        
        assert len(orchestrator.vertical_registry) == 0
        assert orchestrator.contracts_created == 0
    
    def test_register_vertical(self):
        """Test registering a vertical module"""
        from qratum.verticals import JurisModule
        
        orchestrator = PlatformOrchestrator()
        juris = JurisModule()
        
        orchestrator.register_vertical("JURIS", juris)
        
        assert "JURIS" in orchestrator.vertical_registry
    
    def test_submit_intent(self):
        """Test submitting an intent creates a contract"""
        from qratum.verticals import JurisModule
        
        orchestrator = PlatformOrchestrator()
        juris = JurisModule()
        orchestrator.register_vertical("JURIS", juris)
        
        intent = PlatformIntent(
            vertical="JURIS",
            task="analyze_contract",
            parameters={"contract_text": "Sample"},
            requester_id="user_123",
        )
        
        contract = orchestrator.submit_intent(intent)
        
        assert contract.intent == intent
        assert orchestrator.contracts_created == 1
        assert len(orchestrator.event_chain) == 1
    
    def test_execute_contract(self):
        """Test executing a contract through orchestrator"""
        from qratum.verticals import JurisModule
        
        orchestrator = PlatformOrchestrator()
        juris = JurisModule()
        orchestrator.register_vertical("JURIS", juris)
        
        intent = PlatformIntent(
            vertical="JURIS",
            task="analyze_contract",
            parameters={"contract_text": "Sample contract text"},
            requester_id="user_123",
        )
        
        contract = orchestrator.submit_intent(intent)
        result = orchestrator.execute_contract(contract)
        
        assert "output" in result
        assert result["vertical"] == "JURIS"
        assert "safety_disclaimer" in result
        assert orchestrator.contracts_executed == 1
    
    def test_replay_contract(self):
        """Test replaying a contract execution"""
        from qratum.verticals import JurisModule
        
        orchestrator = PlatformOrchestrator()
        juris = JurisModule()
        orchestrator.register_vertical("JURIS", juris)
        
        intent = PlatformIntent(
            vertical="JURIS",
            task="analyze_contract",
            parameters={"contract_text": "Sample"},
            requester_id="user_123",
        )
        
        contract = orchestrator.submit_intent(intent)
        orchestrator.execute_contract(contract)
        
        replay = orchestrator.replay_contract(contract.contract_id)
        
        assert "contract_id" in replay
        assert replay["event_count"] > 0
        assert replay["replay_verified"] is True
    
    def test_unknown_vertical_raises_error(self):
        """Test that unknown vertical raises error"""
        orchestrator = PlatformOrchestrator()
        
        intent = PlatformIntent(
            vertical="UNKNOWN",
            task="some_task",
            parameters={},
            requester_id="user_123",
        )
        
        with pytest.raises(ValueError, match="Unknown vertical"):
            orchestrator.submit_intent(intent)
    
    def test_get_platform_status(self):
        """Test getting platform status"""
        orchestrator = PlatformOrchestrator()
        
        status = orchestrator.get_platform_status()
        
        assert "registered_verticals" in status
        assert "contracts_created" in status
        assert "event_chain_integrity" in status
        assert status["event_chain_integrity"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
