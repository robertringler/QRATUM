"""Tests for JURIS Legal AI module."""

import pytest

from qratum_platform.core import (ComputeSubstrate, PlatformContract,
                                  PlatformIntent, SafetyViolation,
                                  VerticalModule)
from verticals.juris import JURISModule


class TestJURISModule:
    """Test JURIS Legal AI module."""

    def test_module_metadata(self):
        """Test module has required metadata."""
        module = JURISModule()
        assert module.MODULE_NAME == "JURIS"
        assert module.MODULE_VERSION == "2.0.0"
        assert module.SAFETY_DISCLAIMER
        assert len(module.PROHIBITED_USES) > 0

    def test_legal_reasoning(self):
        """Test IRAC legal reasoning."""
        module = JURISModule()
        intent = PlatformIntent(
            vertical=VerticalModule.JURIS,
            operation="legal_reasoning",
            parameters={"facts": "Party A and Party B entered a contract", "area_of_law": "contract"},
            user_id="user_001",
        )
        contract = PlatformContract(
            intent=intent,
            contract_id="test_001",
            substrate=ComputeSubstrate.CPU,
            estimated_cost=0,
            estimated_duration=0,
            safety_checks_passed=True,
        )

        result = module.execute(contract)
        assert result["method"] == "IRAC"
        assert "issue" in result
        assert "rule" in result
        assert "application" in result
        assert "conclusion" in result

    def test_contract_analysis(self):
        """Test contract analysis."""
        module = JURISModule()
        intent = PlatformIntent(
            vertical=VerticalModule.JURIS,
            operation="contract_analysis",
            parameters={"contract_text": "This contract includes termination and liability clauses"},
            user_id="user_001",
        )
        contract = PlatformContract(
            intent=intent,
            contract_id="test_002",
            substrate=ComputeSubstrate.CPU,
            estimated_cost=0,
            estimated_duration=0,
            safety_checks_passed=True,
        )

        result = module.execute(contract)
        assert result["analysis_type"] == "contract_analysis"
        assert "identified_clauses" in result
        assert "risk_factors" in result

    def test_compliance_checking(self):
        """Test compliance checking."""
        module = JURISModule()
        intent = PlatformIntent(
            vertical=VerticalModule.JURIS,
            operation="compliance_checking",
            parameters={
                "policy_text": "We collect and process personal data with consent",
                "frameworks": ["gdpr"],
            },
            user_id="user_001",
        )
        contract = PlatformContract(
            intent=intent,
            contract_id="test_003",
            substrate=ComputeSubstrate.CPU,
            estimated_cost=0,
            estimated_duration=0,
            safety_checks_passed=True,
        )

        result = module.execute(contract)
        assert result["compliance_check"] == "regulatory_frameworks"
        assert "GDPR" in result["compliance_status"]

    def test_prohibited_use_detection(self):
        """Test that prohibited uses are detected."""
        module = JURISModule()
        intent = PlatformIntent(
            vertical=VerticalModule.JURIS,
            operation="legal advice without license",
            parameters={},
            user_id="user_001",
        )

        with pytest.raises(SafetyViolation):
            module.check_safety(intent)

    def test_get_optimal_substrate(self):
        """Test substrate selection."""
        module = JURISModule()
        substrate = module.get_optimal_substrate("legal_reasoning", {})
        assert isinstance(substrate, ComputeSubstrate)
