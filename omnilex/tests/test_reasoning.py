"""Tests for IRAC legal reasoning engine.

Version: 1.0.0
Status: Production
"""

import pytest

from omnilex.knowledge import LegalKnowledgeBase
from omnilex.reasoning import IRACAnalysis, LegalReasoningEngine


class TestIRACAnalysis:
    """Test IRAC analysis dataclass."""

    def test_irac_analysis_valid(self):
        """Test valid IRAC analysis creation."""
        analysis = IRACAnalysis(
            issue="Test issue",
            rule="Test rule",
            rule_sources=["Source 1"],
            application="Test application",
            conclusion="Test conclusion",
            confidence=0.75,
            caveats=["Caveat 1"]
        )

        assert analysis.issue == "Test issue"
        assert analysis.confidence == 0.75

    def test_irac_analysis_invalid_confidence(self):
        """Test IRAC analysis with invalid confidence."""
        with pytest.raises(ValueError, match="Confidence must be between"):
            IRACAnalysis(
                issue="Test",
                rule="Test",
                rule_sources=[],
                application="Test",
                conclusion="Test",
                confidence=1.5,  # Invalid
                caveats=[]
            )


class TestLegalReasoningEngine:
    """Test legal reasoning engine."""

    def test_initialization(self):
        """Test engine initialization."""
        engine = LegalReasoningEngine()
        assert engine.knowledge_base is not None

    def test_initialization_with_kb(self):
        """Test engine initialization with provided knowledge base."""
        kb = LegalKnowledgeBase()
        engine = LegalReasoningEngine(kb)
        assert engine.knowledge_base is kb

    def test_analyze_irac_contract_breach(self):
        """Test IRAC analysis for contract breach."""
        engine = LegalReasoningEngine()

        facts = """
        AlphaCorp and BetaServices entered into a service agreement requiring
        4-hour response time. BetaServices responded after 9 hours, causing
        $50,000 in damages.
        """

        question = "Did BetaServices breach the contract?"

        analysis = engine.analyze_irac(
            facts=facts,
            question=question,
            jurisdiction="US",
            domain="contract"
        )

        assert isinstance(analysis, IRACAnalysis)
        assert "breach" in analysis.issue.lower()
        assert len(analysis.rule) > 0
        assert len(analysis.application) > 0
        assert len(analysis.conclusion) > 0
        assert 0.0 <= analysis.confidence <= 1.0
        assert len(analysis.caveats) > 0

    def test_analyze_irac_tort(self):
        """Test IRAC analysis for tort case."""
        engine = LegalReasoningEngine()

        facts = "Driver A collided with Driver B's vehicle while texting."
        question = "Is Driver A liable for negligence?"

        analysis = engine.analyze_irac(
            facts=facts,
            question=question,
            jurisdiction="US",
            domain="tort"
        )

        assert isinstance(analysis, IRACAnalysis)
        assert "negligence" in analysis.issue.lower() or "liability" in analysis.issue.lower()
        assert analysis.confidence > 0

    def test_identify_issue_contract(self):
        """Test issue identification for contract questions."""
        engine = LegalReasoningEngine()

        issue = engine._identify_issue(
            "Is the contract enforceable?",
            "contract"
        )

        assert "enforceable" in issue.lower()

    def test_identify_issue_tort(self):
        """Test issue identification for tort questions."""
        engine = LegalReasoningEngine()

        issue = engine._identify_issue(
            "Is there negligence?",
            "tort"
        )

        assert "negligence" in issue.lower()

    def test_synthesize_rule_with_authorities(self):
        """Test rule synthesis when authorities are found."""
        engine = LegalReasoningEngine()

        issue = "breach of contract damages"
        rule, sources = engine._synthesize_rule(issue, "US", "contract")

        assert len(rule) > 0
        assert len(sources) > 0

    def test_synthesize_rule_no_authorities(self):
        """Test rule synthesis when no authorities found."""
        engine = LegalReasoningEngine()

        issue = "obscure legal question xyz123"
        rule, sources = engine._synthesize_rule(issue, "US", "contract")

        assert len(rule) > 0
        assert len(sources) > 0
        assert "General legal principles" in sources or len(sources) > 0

    def test_extract_keywords(self):
        """Test keyword extraction."""
        engine = LegalReasoningEngine()

        text = "The breach of contract caused damages due to negligence"
        keywords = engine._extract_keywords(text)

        assert "breach" in keywords
        assert "contract" in keywords
        assert "damages" in keywords
        assert "negligence" in keywords

    def test_get_default_rule_contract(self):
        """Test default rule for contract domain."""
        engine = LegalReasoningEngine()

        rule = engine._get_default_rule("contract")

        assert "contract" in rule.lower()
        assert "offer" in rule.lower() or "breach" in rule.lower()

    def test_get_default_rule_tort(self):
        """Test default rule for tort domain."""
        engine = LegalReasoningEngine()

        rule = engine._get_default_rule("tort")

        assert "negligence" in rule.lower() or "duty" in rule.lower()

    def test_apply_rule_to_facts(self):
        """Test applying rule to facts."""
        engine = LegalReasoningEngine()

        rule = "Contracts require offer, acceptance, and consideration"
        facts = "Party A offered to sell goods to Party B for $1000"
        issue = "Is there a valid contract?"

        application = engine._apply_rule_to_facts(rule, facts, issue)

        assert len(application) > 0
        assert "rule" in application.lower()
        assert "facts" in application.lower()

    def test_draw_conclusion(self):
        """Test conclusion drawing."""
        engine = LegalReasoningEngine()

        issue = "breach of contract"
        rule = "test rule"
        application = "test application"

        conclusion, confidence, caveats = engine._draw_conclusion(
            issue, rule, application, "contract"
        )

        assert len(conclusion) > 0
        assert 0.0 <= confidence <= 1.0
        assert len(caveats) > 0
        assert any("informational" in c.lower() or "attorney" in c.lower() for c in caveats)
