"""JURIS - Legal AI Module for QRATUM Platform.

Provides legal reasoning, contract analysis, litigation prediction,
and compliance checking using IRAC methodology.
"""

import re
from typing import Any, Dict, List

from qratum_platform.core import (
    ComputeSubstrate,
    PlatformContract,
    VerticalModuleBase,
)
from qratum_platform.substrates import VerticalModule, get_optimal_substrate


class JURISModule(VerticalModuleBase):
    """Legal AI module using IRAC (Issue, Rule, Application, Conclusion) reasoning."""

    MODULE_NAME = "JURIS"
    MODULE_VERSION = "2.0.0"
    SAFETY_DISCLAIMER = """
    JURIS Legal AI Disclaimer:
    - NOT a substitute for licensed legal counsel
    - Output is for informational purposes only
    - Does not establish attorney-client relationship
    - Always consult qualified attorney for legal advice
    - Jurisdiction-specific rules may apply
    - AI may not reflect latest legal changes
    """

    PROHIBITED_USES = [
        "criminal defense without attorney",
        "unauthorized practice of law",
        "legal advice without license",
        "court filing without attorney",
        "contract signing without review",
    ]

    def __init__(self):
        super().__init__()
        self.legal_database = self._initialize_legal_database()

    def _initialize_legal_database(self) -> Dict[str, Any]:
        """Initialize legal rules and precedents database."""
        return {
            "contract_law": {
                "offer": "A proposal to enter into a contract",
                "acceptance": "Unqualified agreement to offer terms",
                "consideration": "Something of value exchanged",
                "capacity": "Legal ability to enter contract",
                "legality": "Contract purpose must be legal",
            },
            "tort_law": {
                "duty": "Legal obligation to act reasonably",
                "breach": "Failure to meet duty standard",
                "causation": "Breach caused the harm",
                "damages": "Actual harm or injury occurred",
            },
            "compliance_frameworks": {
                "gdpr": "EU data protection regulation",
                "ccpa": "California Consumer Privacy Act",
                "hipaa": "Health Insurance Portability Act",
                "sox": "Sarbanes-Oxley Act",
            },
        }

    def execute(self, contract: PlatformContract) -> Dict[str, Any]:
        """Execute legal AI operation.

        Args:
            contract: Immutable execution contract

        Returns:
            Results of legal analysis

        Raises:
            SafetyViolation: If prohibited use detected
        """
        operation = contract.intent.operation
        parameters = contract.intent.parameters

        self.emit_event("juris_execution_start", contract.contract_id, {"operation": operation})

        try:
            if operation == "legal_reasoning":
                result = self._perform_irac_analysis(parameters)
            elif operation == "contract_analysis":
                result = self._analyze_contract(parameters)
            elif operation == "litigation_prediction":
                result = self._predict_litigation_outcome(parameters)
            elif operation == "compliance_checking":
                result = self._check_compliance(parameters)
            else:
                raise ValueError(f"Unknown operation: {operation}")

            self.emit_event(
                "juris_execution_complete",
                contract.contract_id,
                {"operation": operation, "success": True},
            )

            return result

        except Exception as e:
            self.emit_event(
                "juris_execution_failed",
                contract.contract_id,
                {"operation": operation, "error": str(e)},
            )
            raise

    def _perform_irac_analysis(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Perform IRAC legal reasoning.

        Args:
            parameters: Must contain 'facts' and optional 'area_of_law'

        Returns:
            IRAC analysis with Issue, Rule, Application, Conclusion
        """
        facts = parameters.get("facts", "")
        area_of_law = parameters.get("area_of_law", "general")

        # Issue identification
        issues = self._identify_legal_issues(facts, area_of_law)

        # Rule determination
        rules = self._determine_applicable_rules(issues, area_of_law)

        # Application
        application = self._apply_rules_to_facts(facts, rules)

        # Conclusion
        conclusion = self._draw_conclusion(application)

        return {
            "method": "IRAC",
            "issue": issues,
            "rule": rules,
            "application": application,
            "conclusion": conclusion,
            "confidence": 0.75,  # Simplified confidence
            "disclaimer": self.SAFETY_DISCLAIMER,
        }

    def _identify_legal_issues(self, facts: str, area_of_law: str) -> List[str]:
        """Identify legal issues from facts."""
        issues = []

        # Contract law issues
        if area_of_law == "contract" or "contract" in facts.lower():
            contract_keywords = ["offer", "acceptance", "consideration", "breach"]
            for keyword in contract_keywords:
                if keyword in facts.lower():
                    issues.append(f"Whether valid {keyword} exists")

        # Tort law issues
        if area_of_law == "tort" or any(
            word in facts.lower() for word in ["injury", "damage", "negligence"]
        ):
            issues.append("Whether duty of care was owed")
            issues.append("Whether breach of duty occurred")
            issues.append("Whether causation exists")

        if not issues:
            issues.append("Legal issue requires further analysis")

        return issues

    def _determine_applicable_rules(self, issues: List[str], area_of_law: str) -> Dict[str, str]:
        """Determine applicable legal rules."""
        rules = {}

        if area_of_law == "contract":
            rules.update(self.legal_database["contract_law"])
        elif area_of_law == "tort":
            rules.update(self.legal_database["tort_law"])
        else:
            # General rules
            rules["good_faith"] = "Parties must act in good faith"
            rules["reasonableness"] = "Actions judged by reasonable person standard"

        return rules

    def _apply_rules_to_facts(self, facts: str, rules: Dict[str, str]) -> str:
        """Apply legal rules to facts."""
        application = "Based on the facts provided:\n"

        for rule_name, rule_text in rules.items():
            if rule_name.lower() in facts.lower():
                application += f"- {rule_text} appears relevant\n"

        application += "\nFurther analysis required with full legal context."
        return application

    def _draw_conclusion(self, application: str) -> str:
        """Draw legal conclusion."""
        conclusion = (
            "Preliminary analysis suggests further legal review is warranted. "
            "Consult with qualified legal counsel for definitive advice."
        )
        return conclusion

    def _analyze_contract(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze contract text for key provisions and risks.

        Args:
            parameters: Must contain 'contract_text'

        Returns:
            Contract analysis with identified clauses and risks
        """
        contract_text = parameters.get("contract_text", "")

        # Extract key clauses
        clauses = self._extract_clauses(contract_text)

        # Identify risks
        risks = self._identify_contract_risks(contract_text)

        # Check for missing provisions
        missing = self._check_missing_provisions(contract_text)

        return {
            "analysis_type": "contract_analysis",
            "identified_clauses": clauses,
            "risk_factors": risks,
            "missing_provisions": missing,
            "overall_risk_level": "medium" if risks else "low",
            "recommendations": [
                "Review with legal counsel",
                "Verify all parties have capacity",
                "Ensure consideration is adequate",
            ],
            "disclaimer": self.SAFETY_DISCLAIMER,
        }

    def _extract_clauses(self, text: str) -> List[str]:
        """Extract contract clauses."""
        clauses = []

        # Look for common clause patterns
        clause_patterns = [
            r"termination",
            r"liability",
            r"indemnification",
            r"warranty",
            r"confidentiality",
            r"arbitration",
            r"jurisdiction",
        ]

        for pattern in clause_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                clauses.append(pattern.title())

        return clauses

    def _identify_contract_risks(self, text: str) -> List[str]:
        """Identify potential contract risks."""
        risks = []

        # Check for risk indicators
        risk_keywords = {
            "unlimited liability": "Unlimited liability exposure",
            "no limitation": "No limitation of liability",
            "perpetual": "Perpetual obligations",
            "exclusive": "Exclusive rights granted",
            "non-compete": "Non-compete restrictions",
        }

        for keyword, risk_desc in risk_keywords.items():
            if keyword in text.lower():
                risks.append(risk_desc)

        return risks

    def _check_missing_provisions(self, text: str) -> List[str]:
        """Check for missing standard provisions."""
        missing = []

        standard_provisions = {
            "termination": "Termination clause",
            "liability": "Limitation of liability",
            "dispute resolution": "Dispute resolution mechanism",
            "governing law": "Governing law clause",
        }

        for keyword, provision in standard_provisions.items():
            if keyword not in text.lower():
                missing.append(provision)

        return missing

    def _predict_litigation_outcome(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Predict litigation outcome based on case facts.

        Args:
            parameters: Must contain 'case_facts' and 'case_type'

        Returns:
            Litigation prediction with probability estimates
        """
        case_facts = parameters.get("case_facts", "")
        case_type = parameters.get("case_type", "civil")

        # Simplified prediction model
        # In production, this would use ML models trained on case law
        strength_score = self._assess_case_strength(case_facts)

        return {
            "prediction_type": "litigation_outcome",
            "case_type": case_type,
            "plaintiff_win_probability": strength_score,
            "defendant_win_probability": 1.0 - strength_score,
            "settlement_likelihood": 0.60,  # Most cases settle
            "estimated_duration_months": 18,
            "key_factors": [
                "Strength of evidence",
                "Legal precedents",
                "Jurisdiction",
                "Party resources",
            ],
            "disclaimer": self.SAFETY_DISCLAIMER,
            "note": "Prediction based on simplified model - not legal advice",
        }

    def _assess_case_strength(self, facts: str) -> float:
        """Assess case strength (simplified)."""
        # Simple heuristic based on keywords
        strength = 0.5  # Neutral baseline

        positive_indicators = ["evidence", "documented", "witness", "contract", "agreement"]
        negative_indicators = ["disputed", "unclear", "alleged", "unverified"]

        for indicator in positive_indicators:
            if indicator in facts.lower():
                strength += 0.05

        for indicator in negative_indicators:
            if indicator in facts.lower():
                strength -= 0.05

        return max(0.1, min(0.9, strength))

    def _check_compliance(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Check compliance with regulatory frameworks.

        Args:
            parameters: Must contain 'policy_text' and 'frameworks'

        Returns:
            Compliance analysis with gaps identified
        """
        policy_text = parameters.get("policy_text", "")
        frameworks = parameters.get("frameworks", ["gdpr", "ccpa"])

        compliance_status = {}
        gaps = []

        for framework in frameworks:
            status = self._check_framework_compliance(policy_text, framework)
            compliance_status[framework.upper()] = status

            if not status["compliant"]:
                gaps.extend(status["gaps"])

        return {
            "compliance_check": "regulatory_frameworks",
            "frameworks_checked": frameworks,
            "compliance_status": compliance_status,
            "identified_gaps": gaps,
            "overall_compliant": all(s["compliant"] for s in compliance_status.values()),
            "recommendations": [
                "Conduct detailed legal review",
                "Update policies to address gaps",
                "Implement compliance monitoring",
            ],
            "disclaimer": self.SAFETY_DISCLAIMER,
        }

    def _check_framework_compliance(self, policy_text: str, framework: str) -> Dict[str, Any]:
        """Check compliance with specific framework."""
        framework_requirements = {
            "gdpr": [
                "data protection",
                "consent",
                "right to erasure",
                "data portability",
                "privacy by design",
            ],
            "ccpa": ["consumer rights", "opt-out", "data disclosure", "non-discrimination"],
            "hipaa": ["patient privacy", "security", "breach notification", "minimum necessary"],
        }

        requirements = framework_requirements.get(framework.lower(), [])
        gaps = []

        for req in requirements:
            if req not in policy_text.lower():
                gaps.append(f"Missing: {req}")

        return {
            "compliant": len(gaps) == 0,
            "requirements_checked": len(requirements),
            "gaps": gaps,
        }

    def get_optimal_substrate(self, operation: str, parameters: Dict[str, Any]) -> ComputeSubstrate:
        """Get optimal compute substrate for legal operation.

        Args:
            operation: Operation name
            parameters: Operation parameters

        Returns:
            Optimal compute substrate
        """
        return get_optimal_substrate(VerticalModule.JURIS, operation)
