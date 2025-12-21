"""Tests for JURIS Legal AI Vertical Module."""

import pytest

from platform.core.intent import PlatformContract, PlatformIntent, VerticalType
from platform.verticals.juris import JurisModule


class TestJurisModule:
    """Test JURIS Legal AI module."""

    def test_module_initialization(self):
        """Test JURIS module initialization."""
        module = JurisModule(seed=42)

        assert module.vertical_name == "JURIS"
        assert module.seed == 42

    def test_safety_disclaimer(self):
        """Test safety disclaimer is present."""
        module = JurisModule(seed=42)
        disclaimer = module.get_safety_disclaimer()

        assert "NOT legal advice" in disclaimer
        assert "attorney" in disclaimer.lower()

    def test_prohibited_uses(self):
        """Test prohibited uses are defined."""
        module = JurisModule(seed=42)
        prohibited = module.get_prohibited_uses()

        assert len(prohibited) > 0
        assert "replace_attorney" in prohibited or any("attorney" in use for use in prohibited)

    def test_required_attestations(self):
        """Test required attestations."""
        module = JurisModule(seed=42)

        attestations = module.get_required_attestations("analyze_contract")
        assert "not_legal_advice_acknowledged" in attestations
        assert "attorney_review_required" in attestations

    def test_contract_analysis(self):
        """Test contract analysis operation."""
        module = JurisModule(seed=42)

        intent = PlatformIntent(
            vertical=VerticalType.JURIS,
            operation="analyze_contract",
            parameters={
                "contract_text": "Employment agreement with termination clause and liability provisions",
                "jurisdiction": "California",
                "problem_size": 100,
            },
            compliance_attestations=frozenset(
                [
                    "not_legal_advice_acknowledged",
                    "attorney_review_required",
                    "contract_jurisdiction_known",
                ]
            ),
        )

        contract = PlatformContract(
            intent=intent,
            contract_hash="test_hash",
            authorized=True,
        )

        result = module.execute(contract)

        assert result.success is True
        assert "analysis_id" in result.result_data
        assert "clauses_identified" in result.result_data
        assert result.result_data["jurisdiction"] == "California"
        assert module.verify_audit_trail() is True

    def test_legal_research(self):
        """Test legal research operation."""
        module = JurisModule(seed=42)

        intent = PlatformIntent(
            vertical=VerticalType.JURIS,
            operation="legal_research",
            parameters={
                "query": "intellectual property rights",
                "jurisdiction": "federal",
                "problem_size": 100,
            },
            compliance_attestations=frozenset(
                [
                    "not_legal_advice_acknowledged",
                    "attorney_review_required",
                    "research_purpose_only",
                ]
            ),
        )

        contract = PlatformContract(
            intent=intent,
            contract_hash="test_hash",
            authorized=True,
        )

        result = module.execute(contract)

        assert result.success is True
        assert "precedents" in result.result_data
        assert result.result_data["jurisdiction"] == "federal"

    def test_compliance_check(self):
        """Test compliance checking operation."""
        module = JurisModule(seed=42)

        intent = PlatformIntent(
            vertical=VerticalType.JURIS,
            operation="compliance_check",
            parameters={
                "framework": "GDPR",
                "requirements": ["data_subject_rights", "consent_management"],
                "problem_size": 100,
            },
            compliance_attestations=frozenset(
                [
                    "not_legal_advice_acknowledged",
                    "attorney_review_required",
                    "compliance_framework_specified",
                ]
            ),
        )

        contract = PlatformContract(
            intent=intent,
            contract_hash="test_hash",
            authorized=True,
        )

        result = module.execute(contract)

        assert result.success is True
        assert "compliance_results" in result.result_data
        assert result.result_data["framework"] == "GDPR"

    def test_risk_assessment(self):
        """Test risk assessment operation."""
        module = JurisModule(seed=42)

        intent = PlatformIntent(
            vertical=VerticalType.JURIS,
            operation="risk_assessment",
            parameters={
                "scenario": "Launching new data collection product",
                "problem_size": 100,
            },
            compliance_attestations=frozenset(
                [
                    "not_legal_advice_acknowledged",
                    "attorney_review_required",
                ]
            ),
        )

        contract = PlatformContract(
            intent=intent,
            contract_hash="test_hash",
            authorized=True,
        )

        result = module.execute(contract)

        assert result.success is True
        assert "risk_score" in result.result_data
        assert 0 <= result.result_data["risk_score"] <= 1

    def test_missing_attestation(self):
        """Test execution fails with missing attestation."""
        module = JurisModule(seed=42)

        intent = PlatformIntent(
            vertical=VerticalType.JURIS,
            operation="analyze_contract",
            parameters={
                "contract_text": "Sample contract",
                "problem_size": 100,
            },
            # Missing required attestations
        )

        contract = PlatformContract(
            intent=intent,
            contract_hash="test_hash",
            authorized=True,
        )

        result = module.execute(contract)

        assert result.success is False
        assert "attestation" in result.result_data["error"].lower()

    def test_deterministic_analysis(self):
        """Test that analysis is deterministic."""
        module1 = JurisModule(seed=42)
        module2 = JurisModule(seed=42)

        intent = PlatformIntent(
            vertical=VerticalType.JURIS,
            operation="analyze_contract",
            parameters={
                "contract_text": "Test contract with termination clause",
                "jurisdiction": "New York",
                "problem_size": 100,
            },
            compliance_attestations=frozenset(
                [
                    "not_legal_advice_acknowledged",
                    "attorney_review_required",
                    "contract_jurisdiction_known",
                ]
            ),
        )

        contract = PlatformContract(
            intent=intent,
            contract_hash="test_hash",
            authorized=True,
        )

        result1 = module1.execute(contract)
        result2 = module2.execute(contract)

        # Results should be identical (deterministic)
        assert result1.result_data["analysis_id"] == result2.result_data["analysis_id"]
        assert result1.result_data["clauses_identified"] == result2.result_data["clauses_identified"]

    def test_audit_trail(self):
        """Test that audit trail is properly maintained."""
        module = JurisModule(seed=42)

        intent = PlatformIntent(
            vertical=VerticalType.JURIS,
            operation="analyze_contract",
            parameters={
                "contract_text": "Sample contract",
                "jurisdiction": "California",
                "problem_size": 100,
            },
            compliance_attestations=frozenset(
                [
                    "not_legal_advice_acknowledged",
                    "attorney_review_required",
                    "contract_jurisdiction_known",
                ]
            ),
        )

        contract = PlatformContract(
            intent=intent,
            contract_hash="test_hash",
            authorized=True,
        )

        result = module.execute(contract)

        # Check event chain
        assert len(module.event_chain) > 0
        assert module.verify_audit_trail() is True

        # Check that result includes event chain root
        assert len(result.event_chain_root) == 64  # SHA-256 hex length

    def test_invalid_operation(self):
        """Test handling of invalid operation."""
        module = JurisModule(seed=42)

        intent = PlatformIntent(
            vertical=VerticalType.JURIS,
            operation="invalid_operation",
            parameters={"problem_size": 100},
            compliance_attestations=frozenset(
                [
                    "not_legal_advice_acknowledged",
                    "attorney_review_required",
                ]
            ),
        )

        contract = PlatformContract(
            intent=intent,
            contract_hash="test_hash",
            authorized=True,
        )

        result = module.execute(contract)

        assert result.success is False
        assert "Unknown" in result.result_data["error"] or "invalid" in result.result_data["error"].lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
