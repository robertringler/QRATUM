"""Tests for QRATUM-OMNILEX main engine.

Version: 1.0.0
Status: Production
"""

import time

import pytest

from omnilex.engine import QRATUMOmniLexEngine
from omnilex.qil_legal import LegalQILIntent, generate_intent_id


class TestQRATUMOmniLexEngine:
    """Test main OMNILEX engine."""

    def test_initialization(self):
        """Test engine initialization."""
        engine = QRATUMOmniLexEngine()

        assert engine.knowledge_base is not None
        assert engine.reasoning_engine is not None
        assert engine.adversarial_simulator is not None
        assert engine.conflict_resolver is not None
        assert engine.prediction_engine is not None
        assert engine.contract_engine is not None
        assert engine.VERSION == "1.0.0"

    def test_upl_disclaimer_present(self):
        """Test that UPL disclaimer is defined."""
        assert len(QRATUMOmniLexEngine.UPL_DISCLAIMER) > 0
        assert "INFORMATIONAL PURPOSES ONLY" in QRATUMOmniLexEngine.UPL_DISCLAIMER
        assert "attorney" in QRATUMOmniLexEngine.UPL_DISCLAIMER.lower()

    def test_submit_irac_analysis(self):
        """Test submitting IRAC analysis intent."""
        engine = QRATUMOmniLexEngine()

        intent = LegalQILIntent(
            intent_id=generate_intent_id("irac_analysis", "US", time.time()),
            compute_task="irac_analysis",
            jurisdiction_primary="US",
            jurisdictions_secondary=(),
            legal_domain="contract",
            reasoning_framework="irac",
            attorney_supervised=True,
            raw_facts="Test facts about a contract dispute.",
            legal_question="Was there a breach of contract?",
        )

        response = engine.submit_legal_intent(intent)

        assert response["intent_id"] == intent.intent_id
        assert "intent_hash" in response
        assert "result" in response
        assert "result_hash" in response
        assert "timestamp" in response
        assert response["version"] == engine.VERSION
        assert response["attorney_supervised"] is True
        assert response["disclaimer"] == engine.UPL_DISCLAIMER

        # Check IRAC result structure
        result = response["result"]
        assert result["analysis_type"] == "irac"
        assert "issue" in result
        assert "rule" in result
        assert "application" in result
        assert "conclusion" in result
        assert "confidence" in result

    def test_submit_adversarial_simulation(self):
        """Test submitting adversarial simulation intent."""
        engine = QRATUMOmniLexEngine()

        intent = LegalQILIntent(
            intent_id=generate_intent_id("adversarial_simulation", "US", time.time()),
            compute_task="adversarial_simulation",
            jurisdiction_primary="US",
            jurisdictions_secondary=(),
            legal_domain="contract",
            reasoning_framework="irac",
            attorney_supervised=True,
            raw_facts="Contract dispute facts.",
            legal_question="Who will prevail?",
        )

        response = engine.submit_legal_intent(intent)
        result = response["result"]

        assert "plaintiff_position" in result
        assert "defendant_position" in result
        assert "outcome_prediction" in result

    def test_submit_conflict_resolution(self):
        """Test submitting conflict of laws intent."""
        engine = QRATUMOmniLexEngine()

        intent = LegalQILIntent(
            intent_id=generate_intent_id("conflict_of_laws", "US", time.time()),
            compute_task="conflict_of_laws",
            jurisdiction_primary="US-CA",
            jurisdictions_secondary=("US-NY", "US-TX"),
            legal_domain="contract",
            reasoning_framework="irac",
            attorney_supervised=True,
            raw_facts="Multi-state contract dispute.",
            legal_question="Which law applies?",
        )

        response = engine.submit_legal_intent(intent)
        result = response["result"]

        assert "governing_law" in result
        assert "methodology" in result

    def test_submit_litigation_prediction(self):
        """Test submitting litigation prediction intent."""
        engine = QRATUMOmniLexEngine()

        intent = LegalQILIntent(
            intent_id=generate_intent_id("litigation_prediction", "US", time.time()),
            compute_task="litigation_prediction",
            jurisdiction_primary="US",
            jurisdictions_secondary=(),
            legal_domain="contract",
            reasoning_framework="irac",
            attorney_supervised=True,
            raw_facts="Litigation facts.",
            legal_question="What is the likely outcome?",
        )

        response = engine.submit_legal_intent(intent)
        result = response["result"]

        assert "plaintiff_win_probability" in result
        assert "settlement_range" in result

    def test_submit_contract_review(self):
        """Test submitting contract review intent."""
        engine = QRATUMOmniLexEngine()

        contract_text = """
        SERVICE AGREEMENT
        Provider shall provide services.
        Client shall pay fees.
        Either party may terminate for cause.
        """

        intent = LegalQILIntent(
            intent_id=generate_intent_id("contract_review", "US", time.time()),
            compute_task="contract_review",
            jurisdiction_primary="US",
            jurisdictions_secondary=(),
            legal_domain="service_agreement",
            reasoning_framework="irac",
            attorney_supervised=True,
            raw_facts=contract_text,
            legal_question="Review this contract.",
        )

        response = engine.submit_legal_intent(intent)
        result = response["result"]

        assert result["analysis_type"] == "contract_review"
        assert "overall_risk" in result
        assert "red_flags" in result
        assert "recommendations" in result

    def test_submit_unknown_task(self):
        """Test submitting unknown compute task."""
        engine = QRATUMOmniLexEngine()

        intent = LegalQILIntent(
            intent_id="test-unknown",
            compute_task="unknown_task",
            jurisdiction_primary="US",
            jurisdictions_secondary=(),
            legal_domain="contract",
            reasoning_framework="irac",
            attorney_supervised=True,
            raw_facts="Test",
            legal_question="Test",
        )

        with pytest.raises(ValueError, match="Unknown compute_task"):
            engine.submit_legal_intent(intent)

    def test_replay_analysis(self):
        """Test replaying a previous analysis."""
        engine = QRATUMOmniLexEngine()

        # First, submit an analysis
        intent = LegalQILIntent(
            intent_id="test-replay",
            compute_task="irac_analysis",
            jurisdiction_primary="US",
            jurisdictions_secondary=(),
            legal_domain="contract",
            reasoning_framework="irac",
            attorney_supervised=True,
            raw_facts="Test facts.",
            legal_question="Test question.",
        )

        original_response = engine.submit_legal_intent(intent)

        # Now replay it
        replayed_response = engine.replay_analysis("test-replay")

        # Should return the same response
        assert replayed_response["intent_id"] == original_response["intent_id"]
        assert replayed_response["result_hash"] == original_response["result_hash"]

    def test_replay_nonexistent_analysis(self):
        """Test replaying analysis that doesn't exist."""
        engine = QRATUMOmniLexEngine()

        with pytest.raises(ValueError, match="not found"):
            engine.replay_analysis("nonexistent-id")

    def test_audit_analysis(self):
        """Test auditing an analysis."""
        engine = QRATUMOmniLexEngine()

        # Submit an analysis
        intent = LegalQILIntent(
            intent_id="test-audit",
            compute_task="irac_analysis",
            jurisdiction_primary="US",
            jurisdictions_secondary=(),
            legal_domain="contract",
            reasoning_framework="irac",
            attorney_supervised=True,
            raw_facts="Test facts.",
            legal_question="Test question.",
        )

        engine.submit_legal_intent(intent)

        # Audit it
        audit_report = engine.audit_analysis("test-audit")

        assert audit_report["intent_id"] == "test-audit"
        assert "hash_integrity" in audit_report
        assert audit_report["hash_integrity"]["valid"] is True
        assert "invariants_satisfied" in audit_report
        assert audit_report["invariants_satisfied"]["hash_chain_integrity"] is True

    def test_audit_nonexistent_analysis(self):
        """Test auditing analysis that doesn't exist."""
        engine = QRATUMOmniLexEngine()

        with pytest.raises(ValueError, match="not found"):
            engine.audit_analysis("nonexistent-id")

    def test_hash_chain_integrity(self):
        """Test that hash chain integrity is maintained."""
        engine = QRATUMOmniLexEngine()

        intent = LegalQILIntent(
            intent_id="test-hash",
            compute_task="irac_analysis",
            jurisdiction_primary="US",
            jurisdictions_secondary=(),
            legal_domain="contract",
            reasoning_framework="irac",
            attorney_supervised=True,
            raw_facts="Test facts.",
            legal_question="Test question.",
        )

        response = engine.submit_legal_intent(intent)

        # Compute hash independently
        expected_hash = engine._compute_result_hash(response["result"])

        # Should match stored hash
        assert response["result_hash"] == expected_hash

    def test_deterministic_serialization(self):
        """Test that serialization is deterministic."""
        engine = QRATUMOmniLexEngine()

        result = {"key2": "value2", "key1": "value1", "key3": {"nested2": "n2", "nested1": "n1"}}

        hash1 = engine._compute_result_hash(result)
        hash2 = engine._compute_result_hash(result)

        # Should produce same hash
        assert hash1 == hash2

    def test_contract_immutability_enforcement(self):
        """Test that contract immutability is enforced."""
        engine = QRATUMOmniLexEngine()

        # Create valid frozen intent
        intent = LegalQILIntent(
            intent_id="test-immutable",
            compute_task="irac_analysis",
            jurisdiction_primary="US",
            jurisdictions_secondary=(),
            legal_domain="contract",
            reasoning_framework="irac",
            attorney_supervised=True,
            raw_facts="Test",
            legal_question="Test",
        )

        # Should not raise error (frozen dataclass)
        engine._enforce_contract_immutability(intent)

        # Attempt to modify should fail
        with pytest.raises(AttributeError):
            intent.intent_id = "modified"  # type: ignore
