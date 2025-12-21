"""Tests for contract analysis engine.

Version: 1.0.0
Status: Production
"""

import pytest

from omnilex.contracts import ContractAnalysisEngine, ContractClause


class TestContractClause:
    """Test contract clause dataclass."""

    def test_contract_clause_valid(self):
        """Test valid contract clause creation."""
        clause = ContractClause(
            clause_id="test-1",
            clause_type="indemnification",
            text="Test clause text",
            risk_level="high",
            issues=["Issue 1"],
            recommendations=["Rec 1"]
        )

        assert clause.risk_level == "high"
        assert len(clause.issues) == 1

    def test_contract_clause_invalid_risk(self):
        """Test contract clause with invalid risk level."""
        with pytest.raises(ValueError, match="Invalid risk_level"):
            ContractClause(
                clause_id="test",
                clause_type="test",
                text="test",
                risk_level="extreme",  # Invalid
                issues=[],
                recommendations=[]
            )


class TestContractAnalysisEngine:
    """Test contract analysis engine."""

    def test_initialization(self):
        """Test engine initialization."""
        engine = ContractAnalysisEngine()
        assert engine._red_flag_patterns is not None
        assert engine._standard_provisions is not None

    def test_analyze_contract_basic(self):
        """Test basic contract analysis."""
        engine = ContractAnalysisEngine()

        text = """
        SERVICE AGREEMENT

        1. Services: Provider shall provide software development services.
        2. Payment: Client shall pay within 30 days.
        3. Termination: Either party may terminate for cause with notice.
        4. Governing Law: This agreement shall be governed by California law.
        """

        result = engine.analyze_contract(
            text=text,
            contract_type="service_agreement",
            jurisdiction="US-CA"
        )

        assert result.contract_type == "service_agreement"
        assert result.overall_risk in {"low", "medium", "high", "critical"}
        assert isinstance(result.clauses, list)
        assert isinstance(result.red_flags, list)
        assert isinstance(result.missing_provisions, list)
        assert isinstance(result.recommendations, list)

    def test_identify_indemnification_clause(self):
        """Test identification of indemnification clause."""
        engine = ContractAnalysisEngine()

        text = "Client shall indemnify Provider from all claims and liabilities."
        clauses = engine._identify_clauses(text, "service_agreement")

        indemnification_clauses = [c for c in clauses if c.clause_type == "indemnification"]
        assert len(indemnification_clauses) > 0

    def test_identify_termination_clause(self):
        """Test identification of termination clause."""
        engine = ContractAnalysisEngine()

        text = "Either party may terminate this agreement for cause with 30 days notice."
        clauses = engine._identify_clauses(text, "service_agreement")

        termination_clauses = [c for c in clauses if c.clause_type == "termination"]
        assert len(termination_clauses) > 0
        assert termination_clauses[0].risk_level == "low"  # Has "for cause"

    def test_identify_ip_clause(self):
        """Test identification of IP clause."""
        engine = ContractAnalysisEngine()

        text = "All intellectual property rights shall be owned by Provider."
        clauses = engine._identify_clauses(text, "service_agreement")

        ip_clauses = [c for c in clauses if c.clause_type == "intellectual_property"]
        assert len(ip_clauses) > 0

    def test_identify_red_flags(self):
        """Test red flag identification."""
        engine = ContractAnalysisEngine()

        text = """
        Provider has sole discretion over all decisions.
        Client waives all rights to dispute resolution.
        This agreement is binding in perpetuity.
        """

        red_flags = engine._identify_red_flags(text)

        assert len(red_flags) > 0
        assert any("sole discretion" in flag.lower() for flag in red_flags)
        assert any("perpetuity" in flag.lower() or "perpetual" in flag.lower() for flag in red_flags)

    def test_identify_automatic_renewal_flag(self):
        """Test identification of automatic renewal red flag."""
        engine = ContractAnalysisEngine()

        text = "This agreement will automatically renew for successive terms."
        red_flags = engine._identify_red_flags(text)

        assert any("automatic" in flag.lower() and "renew" in flag.lower() for flag in red_flags)

    def test_check_missing_provisions_general(self):
        """Test checking for missing provisions in general contract."""
        engine = ContractAnalysisEngine()

        text = "This is a contract for services."
        missing = engine._check_missing_provisions(text, "general")

        # Should flag missing standard provisions
        assert len(missing) > 0

    def test_check_missing_provisions_complete(self):
        """Test checking for missing provisions with complete contract."""
        engine = ContractAnalysisEngine()

        text = """
        This agreement shall be governed by California law.
        Disputes shall be resolved through binding arbitration.
        The parties agree to maintain confidentiality.
        This constitutes the entire agreement.
        Amendments must be in writing.
        Notices shall be sent to specified addresses.
        """

        missing = engine._check_missing_provisions(text, "general")

        # Should have few or no missing provisions
        assert len(missing) < 3

    def test_calculate_overall_risk_low(self):
        """Test overall risk calculation for low-risk contract."""
        engine = ContractAnalysisEngine()

        clauses = [
            ContractClause("1", "test", "text", "low", [], []),
            ContractClause("2", "test", "text", "low", [], [])
        ]
        red_flags = []
        missing = []

        risk = engine._calculate_overall_risk(clauses, red_flags, missing)
        assert risk == "low"

    def test_calculate_overall_risk_high(self):
        """Test overall risk calculation for high-risk contract."""
        engine = ContractAnalysisEngine()

        clauses = [
            ContractClause("1", "test", "text", "high", [], []),
            ContractClause("2", "test", "text", "high", [], []),
            ContractClause("3", "test", "text", "high", [], [])
        ]
        red_flags = ["Flag 1", "Flag 2", "Flag 3"]
        missing = ["Missing 1", "Missing 2"]

        risk = engine._calculate_overall_risk(clauses, red_flags, missing)
        assert risk in {"high", "critical"}

    def test_calculate_overall_risk_critical(self):
        """Test overall risk calculation for critical-risk contract."""
        engine = ContractAnalysisEngine()

        clauses = [
            ContractClause("1", "test", "text", "critical", [], [])
        ]
        red_flags = []
        missing = []

        risk = engine._calculate_overall_risk(clauses, red_flags, missing)
        assert risk == "critical"

    def test_generate_recommendations(self):
        """Test recommendation generation."""
        engine = ContractAnalysisEngine()

        clauses = [ContractClause("1", "test", "text", "high", [], [])]
        red_flags = ["Flag 1"]
        missing = ["Missing 1"]

        recommendations = engine._generate_recommendations(
            clauses, red_flags, missing, "service_agreement"
        )

        assert len(recommendations) > 0
        assert any("legal counsel" in rec.lower() for rec in recommendations)

    def test_red_flag_patterns_loaded(self):
        """Test that red flag patterns are loaded."""
        engine = ContractAnalysisEngine()
        patterns = engine._red_flag_patterns

        assert len(patterns) > 0
        assert "sole discretion" in patterns
        assert "unlimited liability" in patterns

    def test_standard_provisions_loaded(self):
        """Test that standard provisions are loaded."""
        engine = ContractAnalysisEngine()
        provisions = engine._standard_provisions

        assert len(provisions) > 0
        assert "general" in provisions
        assert len(provisions["general"]) > 0

    def test_analyze_one_sided_contract(self):
        """Test analysis of one-sided unfavorable contract."""
        engine = ContractAnalysisEngine()

        text = """
        Provider may terminate at any time for any reason.
        Client shall indemnify Provider from all claims.
        Provider has sole discretion over all deliverables.
        All intellectual property belongs to Provider.
        Client waives all rights to dispute.
        This agreement renews automatically in perpetuity.
        """

        result = engine.analyze_contract(text, "service_agreement", "US")

        # Should identify as high-risk
        assert result.overall_risk in {"high", "critical"}

        # Should have multiple red flags
        assert len(result.red_flags) >= 3

        # Should have recommendations
        assert len(result.recommendations) > 0
