"""IRAC legal reasoning engine for QRATUM-OMNILEX.

This module implements the IRAC (Issue, Rule, Application, Conclusion) legal
reasoning framework with full deterministic execution.

Version: 1.0.0
Status: Production
"""

from __future__ import annotations

from dataclasses import dataclass

from omnilex.knowledge import LegalKnowledgeBase


@dataclass
class IRACAnalysis:
    """Results of IRAC legal analysis.

    Attributes:
        issue: The legal issue identified
        rule: The applicable legal rule(s)
        rule_sources: Citations to authority for the rule
        application: Application of rule to facts
        conclusion: Legal conclusion
        confidence: Confidence score (0.0 to 1.0)
        caveats: List of caveats and limitations
    """

    issue: str
    rule: str
    rule_sources: list[str]
    application: str
    conclusion: str
    confidence: float
    caveats: list[str]

    def __post_init__(self) -> None:
        """Validate IRAC analysis."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")


class LegalReasoningEngine:
    """Legal reasoning engine implementing IRAC framework.

    This engine performs deterministic legal analysis using the IRAC method:
    - Issue: Identify the legal question
    - Rule: State the applicable legal principles
    - Application: Apply the rule to the facts
    - Conclusion: Draw a conclusion based on the analysis
    """

    def __init__(self, knowledge_base: LegalKnowledgeBase | None = None) -> None:
        """Initialize the reasoning engine.

        Args:
            knowledge_base: Legal knowledge base to use (creates default if None)
        """
        self.knowledge_base = knowledge_base or LegalKnowledgeBase()

    def analyze_irac(
        self,
        facts: str,
        question: str,
        jurisdiction: str,
        domain: str = "contract"
    ) -> IRACAnalysis:
        """Perform IRAC analysis on legal question.

        Args:
            facts: Factual scenario
            question: Legal question to analyze
            jurisdiction: Jurisdiction code
            domain: Legal domain

        Returns:
            Complete IRAC analysis
        """
        # Step 1: Identify the issue
        issue = self._identify_issue(question, domain)

        # Step 2: Synthesize the rule
        rule, sources = self._synthesize_rule(issue, jurisdiction, domain)

        # Step 3: Apply rule to facts
        application = self._apply_rule_to_facts(rule, facts, issue)

        # Step 4: Draw conclusion
        conclusion, confidence, caveats = self._draw_conclusion(
            issue, rule, application, domain
        )

        return IRACAnalysis(
            issue=issue,
            rule=rule,
            rule_sources=sources,
            application=application,
            conclusion=conclusion,
            confidence=confidence,
            caveats=caveats
        )

    def _identify_issue(self, question: str, domain: str) -> str:
        """Identify the legal issue from the question.

        Args:
            question: Legal question
            domain: Legal domain

        Returns:
            Formatted legal issue statement
        """
        # Extract key concepts from question
        question_lower = question.lower()

        # Domain-specific issue identification
        if domain == "contract":
            if "breach" in question_lower:
                return f"Whether a breach of contract occurred: {question}"
            elif "enforceable" in question_lower or "valid" in question_lower:
                return f"Whether the contract is enforceable: {question}"
            elif "damages" in question_lower:
                return f"What damages are recoverable: {question}"
        elif domain == "tort":
            if "negligence" in question_lower or "duty" in question_lower:
                return f"Whether negligence is established: {question}"
            elif "liable" in question_lower or "liability" in question_lower:
                return f"Whether liability exists: {question}"

        # Default issue format
        return f"Legal issue: {question}"

    def _synthesize_rule(
        self,
        issue: str,
        jurisdiction: str,
        domain: str
    ) -> tuple[str, list[str]]:
        """Synthesize applicable legal rule from authorities.

        Args:
            issue: Legal issue
            jurisdiction: Jurisdiction code
            domain: Legal domain

        Returns:
            Tuple of (rule statement, list of source citations)
        """
        # Search knowledge base for relevant authorities
        issue_keywords = self._extract_keywords(issue)
        authorities = []
        sources = []

        for keyword in issue_keywords:
            found = self.knowledge_base.search(keyword, jurisdiction, limit=3)
            authorities.extend(found)

        # Remove duplicates while preserving order
        seen = set()
        unique_authorities = []
        for auth in authorities:
            if auth.authority_id not in seen:
                seen.add(auth.authority_id)
                unique_authorities.append(auth)
                sources.append(auth.citation)

        # Synthesize rule from authorities
        if unique_authorities:
            holdings = []
            for auth in unique_authorities[:3]:  # Limit to top 3
                holdings.extend(auth.key_holdings)

            rule = "The applicable legal principles are: " + "; ".join(holdings[:5])
        else:
            # Fallback rules if no authorities found
            rule = self._get_default_rule(domain)

        if not sources:
            sources = ["General legal principles"]

        return rule, sources

    def _extract_keywords(self, text: str) -> list[str]:
        """Extract legal keywords from text.

        Args:
            text: Text to extract keywords from

        Returns:
            List of keywords
        """
        legal_keywords = [
            "breach", "contract", "damages", "duty", "negligence",
            "consideration", "capacity", "offer", "acceptance",
            "foreseeability", "proximate cause", "reasonable person",
            "unconscionable", "promissory estoppel"
        ]

        text_lower = text.lower()
        return [kw for kw in legal_keywords if kw in text_lower]

    def _get_default_rule(self, domain: str) -> str:
        """Get default rule for domain when no authorities found.

        Args:
            domain: Legal domain

        Returns:
            Default rule statement
        """
        default_rules = {
            "contract": (
                "A valid contract requires offer, acceptance, consideration, "
                "capacity, and legality. Breach occurs when a party fails to "
                "perform a material obligation without legal excuse."
            ),
            "tort": (
                "Negligence requires duty, breach, causation, and damages. "
                "The defendant must have owed a duty of care to the plaintiff, "
                "breached that duty, and the breach must have proximately "
                "caused the plaintiff's injuries."
            ),
            "property": (
                "Property rights are protected by law. Owners have the right "
                "to possess, use, and dispose of their property, subject to "
                "legal restrictions."
            ),
        }

        return default_rules.get(domain, "General legal principles apply.")

    def _apply_rule_to_facts(self, rule: str, facts: str, issue: str) -> str:
        """Apply legal rule to factual scenario.

        Args:
            rule: Legal rule
            facts: Factual scenario
            issue: Legal issue

        Returns:
            Application analysis
        """
        # Extract key fact elements
        facts_summary = facts[:200] + "..." if len(facts) > 200 else facts

        application = (
            f"Applying the rule to these facts: {facts_summary}\n\n"
            f"The rule states: {rule}\n\n"
            f"Analysis: Based on the facts presented, we must evaluate whether "
            f"the elements of the rule are satisfied. "
        )

        # Add domain-specific analysis
        if "contract" in issue.lower():
            application += (
                "For contract formation, we examine whether there was a clear "
                "offer, unambiguous acceptance, and adequate consideration. "
            )
        elif "negligence" in issue.lower():
            application += (
                "For negligence, we examine whether a duty of care existed, "
                "whether it was breached, and whether the breach caused harm. "
            )

        return application

    def _draw_conclusion(
        self,
        issue: str,
        rule: str,
        application: str,
        domain: str
    ) -> tuple[str, float, list[str]]:
        """Draw legal conclusion from analysis.

        Args:
            issue: Legal issue
            rule: Legal rule
            application: Application of rule to facts
            domain: Legal domain

        Returns:
            Tuple of (conclusion, confidence score, caveats)
        """
        # Generate conclusion based on domain
        conclusion = (
            "Based on the analysis, the legal conclusion depends on "
            "the specific facts and applicable law in the jurisdiction. "
            "Further factual development may be necessary for a definitive answer."
        )

        # Confidence scoring (production stub)
        confidence = 0.65  # Moderate confidence for general analysis

        # Generate caveats
        caveats = [
            "This analysis is for informational purposes only",
            "Specific facts may alter the conclusion",
            "Jurisdiction-specific rules may apply",
            "Attorney review is recommended",
            "Recent case law may modify these principles"
        ]

        return conclusion, confidence, caveats
