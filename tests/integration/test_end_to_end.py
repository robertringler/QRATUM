"""End-to-end integration tests for QRATUM Canonical Architecture.

Tests the complete flow: Intent → Authority → Contract → Time → Execution → Event → Audit
"""

import pytest

from adapters import AdapterRegistry
from events import get_global_event_log
from qcore import AuthorizationEngine, CapabilityResolver, ContractIssuer, PolicyEngine
from qil import parse_intent
from spine import ContractExecutor


class TestEndToEndFlow:
    """End-to-end integration tests."""

    def setup_method(self):
        """Setup for each test."""
        # Clear global event log
        event_log = get_global_event_log()
        event_log._events = []
        event_log._event_index = {}
        event_log._contract_events = {}

    def test_complete_flow_gb200(self):
        """Test complete flow for GB200 intent."""
        # 1. Parse QIL Intent
        qil_text = """
        INTENT llm_training_gb200 {
            OBJECTIVE train_llama3_70b
            HARDWARE ONLY GB200
            CONSTRAINT GPU_VRAM >= 1000
            CAPABILITY llm_training
            TIME deadline: 7200s
            AUTHORITY user: mlops_team
            TRUST level: verified
        }
        """
        intent = parse_intent(qil_text)
        assert intent.name == "llm_training_gb200"

        # 2. Authorize Intent (Authority Engine)
        auth_engine = AuthorizationEngine()
        auth_result = auth_engine.authorize_intent(intent)
        assert auth_result.authorized
        assert auth_result.proof

        # 3. Evaluate Policies
        policy_engine = PolicyEngine()
        policy_result = policy_engine.evaluate(intent)
        assert policy_result.passed

        # 4. Resolve Capabilities
        resolver = CapabilityResolver()
        resolved_caps = resolver.resolve_capabilities(intent)
        assert len(resolved_caps) > 0
        assert resolved_caps[0].cluster_type == "GB200"

        # 5. Issue All 4 Contracts
        issuer = ContractIssuer(resolver=resolver)
        contract_bundle = issuer.issue_contracts(intent, auth_result)

        assert contract_bundle.intent_contract is not None
        assert contract_bundle.capability_contract is not None
        assert contract_bundle.temporal_contract is not None
        assert contract_bundle.event_contract is not None

        # Verify all contracts reference same intent
        intent_id = contract_bundle.intent_contract.contract_id
        assert contract_bundle.capability_contract.intent_contract_id == intent_id
        assert contract_bundle.temporal_contract.intent_contract_id == intent_id
        assert contract_bundle.event_contract.intent_contract_id == intent_id

        # 6. Execute via Adapter
        adapter_registry = AdapterRegistry()
        executor = ContractExecutor(adapter_registry=adapter_registry)

        execution_result = executor.execute(contract_bundle)
        assert execution_result.success
        assert execution_result.cluster_type == "GB200"

        # 7. Verify Event Log
        event_log = get_global_event_log()
        events = event_log.get_contract_events(intent_id)

        assert len(events) > 0
        event_types = [e.event_type for e in events]
        assert "ExecutionStarted" in event_types
        assert "ExecutionCompleted" in event_types
        assert "AuditLogged" in event_types

        # 8. Verify Causal Chain
        assert event_log.verify_causal_chain(intent_id)

    def test_complete_flow_qpu(self):
        """Test complete flow for QPU intent."""
        qil_text = """
        INTENT quantum_chemistry {
            OBJECTIVE simulate_h2o_molecule
            HARDWARE ONLY QPU
            CAPABILITY molecular_simulation
            TIME deadline: 3600s
            AUTHORITY user: quantum_research
            TRUST level: verified
        }
        """
        intent = parse_intent(qil_text)

        # Authorize
        auth_engine = AuthorizationEngine()
        auth_result = auth_engine.authorize_intent(intent)
        assert auth_result.authorized

        # Issue contracts
        resolver = CapabilityResolver()
        issuer = ContractIssuer(resolver=resolver)
        contract_bundle = issuer.issue_contracts(intent, auth_result)

        assert contract_bundle.capability_contract.get_cluster_type() == "QPU"

        # Execute
        adapter_registry = AdapterRegistry()
        executor = ContractExecutor(adapter_registry=adapter_registry)
        execution_result = executor.execute(contract_bundle)

        assert execution_result.success
        assert execution_result.cluster_type == "QPU"

    def test_complete_flow_hybrid(self):
        """Test complete flow for hybrid intent."""
        qil_text = """
        INTENT hybrid_ai_quantum {
            OBJECTIVE optimize_molecular_design
            HARDWARE ONLY GB200 AND QPU NOT IPU
            CONSTRAINT GPU_VRAM >= 500
            CAPABILITY llm_training
            CAPABILITY quantum_optimizer
            TIME deadline: 10800s
            AUTHORITY user: research_lead
            TRUST level: verified
        }
        """
        intent = parse_intent(qil_text)

        # Authorize
        auth_engine = AuthorizationEngine()
        auth_result = auth_engine.authorize_intent(intent)
        assert auth_result.authorized

        # Issue contracts
        resolver = CapabilityResolver()
        issuer = ContractIssuer(resolver=resolver)
        contract_bundle = issuer.issue_contracts(intent, auth_result)

        # First capability should select GB200 or QPU
        cluster_type = contract_bundle.capability_contract.get_cluster_type()
        assert cluster_type in ["GB200", "QPU"]

        # Execute
        adapter_registry = AdapterRegistry()
        executor = ContractExecutor(adapter_registry=adapter_registry)
        execution_result = executor.execute(contract_bundle)

        assert execution_result.success

    def test_authorization_failure(self):
        """Test authorization failure handling."""
        from qcore import AuthorizationError

        qil_text = """
        INTENT untrusted_intent {
            OBJECTIVE malicious_task
            TRUST level: untrusted
            AUTHORITY user: unknown
        }
        """
        intent = parse_intent(qil_text)

        auth_engine = AuthorizationEngine()

        # Should raise AuthorizationError (FATAL)
        with pytest.raises(AuthorizationError):
            auth_engine.authorize_intent(intent)

    def test_temporal_contract_expiration(self):
        """Test temporal contract expiration handling."""

        qil_text = """
        INTENT expired_intent {
            OBJECTIVE test
            TIME deadline: 1s
            AUTHORITY user: test
            TRUST level: verified
        }
        """
        intent = parse_intent(qil_text)

        # Authorize and create contracts
        auth_engine = AuthorizationEngine()
        auth_result = auth_engine.authorize_intent(intent)

        resolver = CapabilityResolver()
        issuer = ContractIssuer(resolver=resolver)
        contract_bundle = issuer.issue_contracts(intent, auth_result)

        # Wait for expiration (simulate)
        import time

        time.sleep(1.1)

        # Should fail execution due to expiration
        adapter_registry = AdapterRegistry()
        executor = ContractExecutor(adapter_registry=adapter_registry)

        with pytest.raises(ValueError, match="expired"):
            executor.execute(contract_bundle)

    def test_event_sequence_verification(self):
        """Test event sequence verification."""
        qil_text = """
        INTENT event_test {
            OBJECTIVE test
            AUTHORITY user: test
            TRUST level: verified
        }
        """
        intent = parse_intent(qil_text)

        # Execute full flow
        auth_engine = AuthorizationEngine()
        auth_result = auth_engine.authorize_intent(intent)

        resolver = CapabilityResolver()
        issuer = ContractIssuer(resolver=resolver)
        contract_bundle = issuer.issue_contracts(intent, auth_result)

        adapter_registry = AdapterRegistry()
        executor = ContractExecutor(adapter_registry=adapter_registry)
        executor.execute(contract_bundle)

        # Verify event sequence
        event_log = get_global_event_log()
        observed_events = event_log.get_event_sequence(contract_bundle.intent_contract.contract_id)

        # Check that execution events are in the sequence
        assert "ExecutionStarted" in observed_events
        assert "ExecutionCompleted" in observed_events

        # ExecutionStarted should come before ExecutionCompleted
        start_idx = observed_events.index("ExecutionStarted")
        complete_idx = observed_events.index("ExecutionCompleted")
        assert start_idx < complete_idx

    def test_adapter_validation(self):
        """Test adapter contract validation."""

        qil_text = """
        INTENT valid_intent {
            OBJECTIVE test
            HARDWARE ONLY GB200
            AUTHORITY user: test
            TRUST level: verified
        }
        """
        intent = parse_intent(qil_text)

        # Create contracts
        auth_engine = AuthorizationEngine()
        auth_result = auth_engine.authorize_intent(intent)

        resolver = CapabilityResolver()
        issuer = ContractIssuer(resolver=resolver)
        contract_bundle = issuer.issue_contracts(intent, auth_result)

        # Try to execute on wrong adapter (simulate by modifying contract)
        # This tests that adapters validate contracts
        adapter_registry = AdapterRegistry()
        adapter = adapter_registry.get_adapter("GB200")

        # Adapter should validate successfully
        adapter.validate_contract(contract_bundle)

    def test_multiple_intents_isolation(self):
        """Test that multiple intents are isolated."""
        qil_text1 = """
        INTENT intent1 {
            OBJECTIVE task1
            AUTHORITY user: user1
            TRUST level: verified
        }
        """

        qil_text2 = """
        INTENT intent2 {
            OBJECTIVE task2
            AUTHORITY user: user2
            TRUST level: verified
        }
        """

        # Execute first intent
        intent1 = parse_intent(qil_text1)
        auth_engine = AuthorizationEngine()
        auth_result1 = auth_engine.authorize_intent(intent1)

        resolver = CapabilityResolver()
        issuer = ContractIssuer(resolver=resolver)
        bundle1 = issuer.issue_contracts(intent1, auth_result1)

        adapter_registry = AdapterRegistry()
        executor = ContractExecutor(adapter_registry=adapter_registry)
        result1 = executor.execute(bundle1)

        # Execute second intent
        intent2 = parse_intent(qil_text2)
        auth_result2 = auth_engine.authorize_intent(intent2)
        bundle2 = issuer.issue_contracts(intent2, auth_result2)
        result2 = executor.execute(bundle2)

        # Verify both succeeded
        assert result1.success
        assert result2.success

        # Verify separate event chains
        event_log = get_global_event_log()
        events1 = event_log.get_contract_events(bundle1.intent_contract.contract_id)
        events2 = event_log.get_contract_events(bundle2.intent_contract.contract_id)

        assert len(events1) > 0
        assert len(events2) > 0

        # Verify causal chains are separate and valid
        assert event_log.verify_causal_chain(bundle1.intent_contract.contract_id)
        assert event_log.verify_causal_chain(bundle2.intent_contract.contract_id)
