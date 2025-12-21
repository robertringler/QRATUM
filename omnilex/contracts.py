"""Contract analysis engine for QRATUM-OMNILEX.

This module provides automated contract review and risk assessment
with red flag detection and missing provision identification.

Version: 1.0.0
Status: Production
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ContractClause:
    """Represents a contract clause with risk assessment.

    Attributes:
        clause_id: Unique identifier for the clause
        clause_type: Type of clause (e.g., 'indemnification', 'limitation_of_liability')
        text: Text of the clause
        risk_level: Risk level (low, medium, high, critical)
        issues: List of identified issues
        recommendations: List of recommendations
    """

    clause_id: str
    clause_type: str
    text: str
    risk_level: str
    issues: list[str]
    recommendations: list[str]

    def __post_init__(self) -> None:
        """Validate contract clause."""
        valid_risk_levels = {"low", "medium", "high", "critical"}
        if self.risk_level not in valid_risk_levels:
            raise ValueError(f"Invalid risk_level: {self.risk_level}")


@dataclass
class ContractAnalysisResult:
    """Results of contract analysis.

    Attributes:
        contract_type: Type of contract analyzed
        overall_risk: Overall risk assessment
        clauses: List of analyzed clauses
        red_flags: List of red flags identified
        missing_provisions: List of recommended missing provisions
        recommendations: Overall recommendations
    """

    contract_type: str
    overall_risk: str
    clauses: list[ContractClause]
    red_flags: list[str]
    missing_provisions: list[str]
    recommendations: list[str]


class ContractAnalysisEngine:
    """Analyzes contracts for risks, red flags, and missing provisions.

    This engine performs automated contract review to identify potential
    issues and provide recommendations for improvement.
    """

    def __init__(self) -> None:
        """Initialize the contract analysis engine."""
        self._red_flag_patterns = self._load_red_flag_patterns()
        self._standard_provisions = self._load_standard_provisions()

    def analyze_contract(
        self,
        text: str,
        contract_type: str,
        jurisdiction: str
    ) -> ContractAnalysisResult:
        """Analyze a contract.

        Args:
            text: Full text of the contract
            contract_type: Type of contract (e.g., 'service_agreement', 'purchase_agreement')
            jurisdiction: Jurisdiction code

        Returns:
            Complete contract analysis results
        """
        # Identify clauses
        clauses = self._identify_clauses(text, contract_type)

        # Identify red flags
        red_flags = self._identify_red_flags(text)

        # Check for missing provisions
        missing = self._check_missing_provisions(text, contract_type)

        # Determine overall risk
        overall_risk = self._calculate_overall_risk(clauses, red_flags, missing)

        # Generate recommendations
        recommendations = self._generate_recommendations(
            clauses, red_flags, missing, contract_type
        )

        return ContractAnalysisResult(
            contract_type=contract_type,
            overall_risk=overall_risk,
            clauses=clauses,
            red_flags=red_flags,
            missing_provisions=missing,
            recommendations=recommendations
        )

    def _identify_clauses(self, text: str, contract_type: str) -> list[ContractClause]:
        """Identify and analyze clauses in contract.

        Args:
            text: Contract text
            contract_type: Type of contract

        Returns:
            List of analyzed clauses
        """
        clauses = []
        text_lower = text.lower()

        # Check for indemnification clause
        if "indemnif" in text_lower or "hold harmless" in text_lower:
            risk_level = "high" if "indemnif" in text_lower and "sole" not in text_lower else "medium"
            clauses.append(ContractClause(
                clause_id="indemnification-1",
                clause_type="indemnification",
                text="[Indemnification clause detected]",
                risk_level=risk_level,
                issues=["Broad indemnification may create unlimited liability"],
                recommendations=["Consider mutual indemnification", "Add liability cap"]
            ))

        # Check for limitation of liability
        if "limit" in text_lower and "liability" in text_lower:
            clauses.append(ContractClause(
                clause_id="liability-1",
                clause_type="limitation_of_liability",
                text="[Limitation of liability clause detected]",
                risk_level="low",
                issues=[],
                recommendations=["Ensure limitation is reasonable and enforceable"]
            ))

        # Check for termination clause
        if "terminat" in text_lower:
            has_cause = "for cause" in text_lower
            risk_level = "low" if has_cause else "medium"
            clauses.append(ContractClause(
                clause_id="termination-1",
                clause_type="termination",
                text="[Termination clause detected]",
                risk_level=risk_level,
                issues=[] if has_cause else ["Termination rights may be one-sided"],
                recommendations=["Ensure mutual termination rights", "Specify notice period"]
            ))

        # Check for IP assignment
        if ("intellectual property" in text_lower or "ip" in text_lower or
            "copyright" in text_lower or "patent" in text_lower):
            clauses.append(ContractClause(
                clause_id="ip-1",
                clause_type="intellectual_property",
                text="[IP clause detected]",
                risk_level="medium",
                issues=["IP ownership must be clearly defined"],
                recommendations=["Clarify ownership of pre-existing and new IP"]
            ))

        return clauses

    def _identify_red_flags(self, text: str) -> list[str]:
        """Identify red flags in contract text.

        Args:
            text: Contract text

        Returns:
            List of red flags
        """
        red_flags = []
        text_lower = text.lower()

        # Check against red flag patterns
        for pattern, description in self._red_flag_patterns.items():
            if pattern in text_lower:
                red_flags.append(description)

        # Check for one-sided terms
        if "party shall not" in text_lower and text_lower.count("party shall not") == 1:
            red_flags.append("One-sided restriction - obligations may be unbalanced")

        # Check for automatic renewal
        if "automatic" in text_lower and "renew" in text_lower:
            red_flags.append("Automatic renewal clause - may create unintended long-term commitment")

        # Check for penalty clauses
        if "penalty" in text_lower or "liquidated damages" in text_lower:
            red_flags.append("Liquidated damages or penalty clause - may be unenforceable if excessive")

        return red_flags

    def _check_missing_provisions(
        self,
        text: str,
        contract_type: str
    ) -> list[str]:
        """Check for missing standard provisions.

        Args:
            text: Contract text
            contract_type: Type of contract

        Returns:
            List of missing provisions
        """
        missing = []
        text_lower = text.lower()

        # Get standard provisions for contract type
        standard = self._standard_provisions.get(contract_type, self._standard_provisions["general"])

        for provision in standard:
            keyword = provision["keyword"]
            if keyword not in text_lower:
                missing.append(provision["description"])

        return missing

    def _calculate_overall_risk(
        self,
        clauses: list[ContractClause],
        red_flags: list[str],
        missing: list[str]
    ) -> str:
        """Calculate overall contract risk.

        Args:
            clauses: List of analyzed clauses
            red_flags: List of red flags
            missing: List of missing provisions

        Returns:
            Overall risk level (low, medium, high, critical)
        """
        # Count high-risk indicators
        critical_clauses = sum(1 for c in clauses if c.risk_level == "critical")
        high_risk_clauses = sum(1 for c in clauses if c.risk_level == "high")
        num_red_flags = len(red_flags)
        num_missing = len(missing)

        # Calculate risk score
        risk_score = (critical_clauses * 10 + high_risk_clauses * 5 +
                     num_red_flags * 3 + num_missing * 2)

        # Classify overall risk
        if risk_score >= 20 or critical_clauses > 0:
            return "critical"
        elif risk_score >= 10 or high_risk_clauses > 2:
            return "high"
        elif risk_score >= 5:
            return "medium"
        else:
            return "low"

    def _generate_recommendations(
        self,
        clauses: list[ContractClause],
        red_flags: list[str],
        missing: list[str],
        contract_type: str
    ) -> list[str]:
        """Generate overall recommendations.

        Args:
            clauses: Analyzed clauses
            red_flags: Red flags
            missing: Missing provisions
            contract_type: Contract type

        Returns:
            List of recommendations
        """
        recommendations = []

        # High-level recommendations based on findings
        if red_flags:
            recommendations.append(
                f"Address {len(red_flags)} red flag(s) identified in contract"
            )

        if missing:
            recommendations.append(
                f"Consider adding {len(missing)} missing standard provision(s)"
            )

        # Clause-specific recommendations
        high_risk_clauses = [c for c in clauses if c.risk_level in ("high", "critical")]
        if high_risk_clauses:
            recommendations.append(
                f"Review and negotiate {len(high_risk_clauses)} high-risk clause(s)"
            )

        # Always recommend legal review
        recommendations.append("Have contract reviewed by qualified legal counsel")

        return recommendations

    def _load_red_flag_patterns(self) -> dict:
        """Load red flag patterns.

        Returns:
            Dictionary of patterns to descriptions
        """
        return {
            "sole discretion": "One party has sole discretion - creates power imbalance",
            "waive all rights": "Rights waiver - may be overly broad",
            "in perpetuity": "Perpetual term - no defined end date",
            "unlimited liability": "Unlimited liability exposure",
            "non-compete": "Non-compete clause - may restrict future employment",
            "binding arbitration": "Mandatory arbitration - limits access to courts",
            "class action waiver": "Class action waiver - limits collective action",
            "unilateral": "Unilateral rights - one-sided provision",
        }

    def _load_standard_provisions(self) -> dict:
        """Load standard provisions by contract type.

        Returns:
            Dictionary of contract types to standard provisions
        """
        return {
            "general": [
                {"keyword": "govern", "description": "Governing law clause"},
                {"keyword": "dispute", "description": "Dispute resolution mechanism"},
                {"keyword": "confiden", "description": "Confidentiality provisions"},
                {"keyword": "entire agreement", "description": "Entire agreement clause"},
                {"keyword": "amend", "description": "Amendment procedures"},
                {"keyword": "notice", "description": "Notice provisions"},
            ],
            "service_agreement": [
                {"keyword": "scope", "description": "Scope of services"},
                {"keyword": "deliverable", "description": "Deliverables definition"},
                {"keyword": "payment", "description": "Payment terms"},
                {"keyword": "warrant", "description": "Warranties"},
                {"keyword": "terminat", "description": "Termination rights"},
            ],
            "purchase_agreement": [
                {"keyword": "price", "description": "Purchase price"},
                {"keyword": "delivery", "description": "Delivery terms"},
                {"keyword": "inspect", "description": "Inspection rights"},
                {"keyword": "warrant", "description": "Warranties"},
                {"keyword": "risk of loss", "description": "Risk of loss allocation"},
            ]
        }
