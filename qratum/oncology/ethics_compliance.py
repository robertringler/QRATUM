"""
Ethics and Compliance Module - FDA SaMD and AI Safety

This module implements ethics and regulatory compliance for the
QRATUM oncology research engine.

Key components:
1. FDA Software as Medical Device (SaMD) considerations
2. AI safety constraints and guardrails
3. Bias and dataset risk assessment
4. Explainability requirements
5. Clinical decision support system (CDSS) classification

CRITICAL: QRATUM is a decision-support system, NOT a clinician replacement.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)


class SaMDRiskCategory(Enum):
    """FDA Software as Medical Device risk categories."""

    CATEGORY_I = "category_i"  # Low risk
    CATEGORY_II = "category_ii"  # Moderate risk
    CATEGORY_III = "category_iii"  # High risk
    CATEGORY_IV = "category_iv"  # Highest risk - not applicable to decision support


class IntendedUse(Enum):
    """Intended use categories for medical software."""

    RESEARCH_ONLY = "research_only"
    CLINICAL_DECISION_SUPPORT = "clinical_decision_support"
    DIAGNOSTIC = "diagnostic"
    THERAPEUTIC = "therapeutic"


class AISafetyLevel(Enum):
    """AI safety risk levels."""

    INFORMATIONAL = "informational"
    ADVISORY = "advisory"
    DECISION_SUPPORT = "decision_support"
    AUTONOMOUS = "autonomous"  # NOT SUPPORTED


@dataclass
class BiasRiskAssessment:
    """Assessment of bias and dataset risks.

    Attributes:
        assessment_id: Unique identifier
        data_sources: Data sources used
        population_representation: Demographics represented
        known_biases: Identified biases
        mitigation_strategies: Bias mitigation approaches
        monitoring_plan: Ongoing bias monitoring
    """

    assessment_id: str
    data_sources: list[str] = field(default_factory=list)
    population_representation: dict[str, float] = field(default_factory=dict)
    known_biases: list[dict[str, Any]] = field(default_factory=list)
    mitigation_strategies: list[str] = field(default_factory=list)
    monitoring_plan: str = ""
    last_reviewed: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "assessment_id": self.assessment_id,
            "data_sources": self.data_sources,
            "population_representation": self.population_representation,
            "known_biases": self.known_biases,
            "mitigation_strategies": self.mitigation_strategies,
            "monitoring_plan": self.monitoring_plan,
            "last_reviewed": self.last_reviewed,
        }


@dataclass
class ExplainabilityRequirement:
    """Explainability requirements for AI outputs.

    Attributes:
        requirement_id: Unique identifier
        output_type: Type of output requiring explanation
        explanation_method: Method for generating explanations
        minimum_detail_level: Minimum detail level required
        target_audience: Intended audience for explanations
        validation_approach: How explanations are validated
    """

    requirement_id: str
    output_type: str
    explanation_method: str
    minimum_detail_level: str = "intermediate"
    target_audience: str = "clinician"
    validation_approach: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "requirement_id": self.requirement_id,
            "output_type": self.output_type,
            "explanation_method": self.explanation_method,
            "minimum_detail_level": self.minimum_detail_level,
            "target_audience": self.target_audience,
            "validation_approach": self.validation_approach,
        }


@dataclass
class AISafetyConstraints:
    """AI safety constraints for the oncology engine.

    Attributes:
        constraint_id: Unique identifier
        safety_level: AI safety level
        hard_constraints: Constraints that must never be violated
        soft_constraints: Constraints that should generally be followed
        human_oversight_requirements: Human oversight requirements
        kill_switch_conditions: Conditions for system shutdown
    """

    constraint_id: str
    safety_level: AISafetyLevel = AISafetyLevel.DECISION_SUPPORT
    hard_constraints: list[str] = field(default_factory=list)
    soft_constraints: list[str] = field(default_factory=list)
    human_oversight_requirements: list[str] = field(default_factory=list)
    kill_switch_conditions: list[str] = field(default_factory=list)
    audit_requirements: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "constraint_id": self.constraint_id,
            "safety_level": self.safety_level.value,
            "hard_constraints": self.hard_constraints,
            "soft_constraints": self.soft_constraints,
            "human_oversight_requirements": self.human_oversight_requirements,
            "kill_switch_conditions": self.kill_switch_conditions,
            "audit_requirements": self.audit_requirements,
        }


@dataclass
class FDACompliance:
    """FDA compliance documentation for SaMD.

    Attributes:
        compliance_id: Unique identifier
        intended_use: Intended use classification
        risk_category: SaMD risk category
        clinical_evaluation: Clinical evaluation summary
        quality_management_system: QMS reference
        labeling_requirements: Labeling requirements
    """

    compliance_id: str
    intended_use: IntendedUse = IntendedUse.RESEARCH_ONLY
    risk_category: SaMDRiskCategory = SaMDRiskCategory.CATEGORY_I
    clinical_evaluation: str = ""
    quality_management_system: str = ""
    labeling_requirements: list[str] = field(default_factory=list)
    premarket_submission_type: str = ""
    post_market_surveillance: str = ""
    unique_device_identifier: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "compliance_id": self.compliance_id,
            "intended_use": self.intended_use.value,
            "risk_category": self.risk_category.value,
            "clinical_evaluation": self.clinical_evaluation,
            "quality_management_system": self.quality_management_system,
            "labeling_requirements": self.labeling_requirements,
            "premarket_submission_type": self.premarket_submission_type,
            "post_market_surveillance": self.post_market_surveillance,
            "unique_device_identifier": self.unique_device_identifier,
        }


class EthicsComplianceModule:
    """
    Ethics and regulatory compliance management.

    Ensures QRATUM oncology outputs meet:
    - FDA SaMD requirements
    - AI safety constraints
    - Explainability standards
    - Bias monitoring requirements

    CRITICAL: QRATUM is a DECISION-SUPPORT SYSTEM.
    It does NOT replace clinical judgment or provide medical advice.
    """

    # System-wide disclaimers
    SYSTEM_DISCLAIMER = """
═══════════════════════════════════════════════════════════════════════════════
                    QRATUM ONCOLOGY RESEARCH ENGINE
                        ETHICS AND COMPLIANCE NOTICE
═══════════════════════════════════════════════════════════════════════════════

INTENDED USE:
This system is designed for RESEARCH USE ONLY. It provides computational
decision support for qualified researchers and clinicians, NOT direct
patient care recommendations.

LIMITATIONS:
• Outputs are HYPOTHESES requiring experimental validation
• NOT FDA-approved for clinical diagnosis or treatment
• NOT a substitute for clinical judgment
• Requires human expert review before any application

REGULATORY STATUS:
• Current status: Research-use only
• SaMD classification: Category I (Low risk decision support)
• Regulatory pathway: Not submitted for FDA clearance

AI SAFETY:
• Human oversight required for all outputs
• System cannot make autonomous treatment decisions
• All recommendations require clinician validation
• Audit trail maintained for all operations

DATA AND BIAS:
• Training data may not represent all populations
• Known biases documented in bias risk assessment
• Regular monitoring for algorithmic bias
• Users should consider population-specific factors

═══════════════════════════════════════════════════════════════════════════════
"""

    PROHIBITED_USES = [
        "Direct patient treatment decisions without clinical oversight",
        "Autonomous diagnosis without human review",
        "Patient-facing recommendations without clinician intermediary",
        "Replacement of qualified oncologist judgment",
        "Use in settings without appropriate ethical oversight",
        "Application to populations not represented in training data",
        "Clinical decisions in emergency/acute settings",
        "Pediatric oncology applications (not validated)",
    ]

    def __init__(
        self,
        name: str = "QRATUM_Oncology_Ethics",
        intended_use: IntendedUse = IntendedUse.RESEARCH_ONLY,
    ) -> None:
        """Initialize the ethics compliance module.

        Args:
            name: Module name
            intended_use: Intended use classification
        """
        self.name = name
        self.intended_use = intended_use

        self.safety_constraints: Optional[AISafetyConstraints] = None
        self.fda_compliance: Optional[FDACompliance] = None
        self.bias_assessment: Optional[BiasRiskAssessment] = None
        self.explainability_requirements: list[ExplainabilityRequirement] = []

        self.created_at = datetime.now(timezone.utc).isoformat()
        self.audit_log: list[dict[str, Any]] = []

        self._initialize_default_constraints()
        logger.info(f"Initialized EthicsComplianceModule: {name}")

    def _initialize_default_constraints(self) -> None:
        """Initialize default safety constraints and compliance."""
        # Default AI safety constraints
        self.safety_constraints = AISafetyConstraints(
            constraint_id="SAFETY_001",
            safety_level=AISafetyLevel.DECISION_SUPPORT,
            hard_constraints=[
                "NEVER provide direct treatment recommendations to patients",
                "ALWAYS require clinical expert review before application",
                "NEVER claim diagnostic or prognostic accuracy",
                "ALWAYS include uncertainty quantification with outputs",
                "NEVER access patient-identifiable information",
                "ALWAYS maintain audit trail of all operations",
                "NEVER operate autonomously without human oversight",
                "ALWAYS display appropriate disclaimers",
            ],
            soft_constraints=[
                "Prefer interpretable models when possible",
                "Provide confidence intervals with predictions",
                "Flag outputs with high uncertainty",
                "Recommend validation studies for novel findings",
            ],
            human_oversight_requirements=[
                "All treatment sequence recommendations require oncologist review",
                "Hypothesis ranking requires domain expert validation",
                "Causal graph interpretations require clinical input",
                "Validation pipeline designs require IRB review",
            ],
            kill_switch_conditions=[
                "Detection of inappropriate autonomous operation",
                "Evidence of bias-driven harm",
                "System attempts to bypass safety constraints",
                "Unauthorized access to protected data",
            ],
            audit_requirements=[
                "Log all queries and responses",
                "Track user acknowledgment of disclaimers",
                "Record all model version changes",
                "Document all safety constraint evaluations",
            ],
        )

        # Default FDA compliance
        self.fda_compliance = FDACompliance(
            compliance_id="FDA_001",
            intended_use=self.intended_use,
            risk_category=SaMDRiskCategory.CATEGORY_I,
            clinical_evaluation=(
                "Not yet submitted for clinical evaluation. "
                "Current status: Research-use only."
            ),
            quality_management_system="QRATUM QMS v1.0 (Research)",
            labeling_requirements=[
                "FOR RESEARCH USE ONLY - Not for clinical diagnosis",
                "Requires expert review before any clinical application",
                "Not FDA cleared or approved",
                "Outputs are computational hypotheses only",
            ],
            premarket_submission_type="N/A (Research use only)",
            post_market_surveillance="N/A (Not marketed)",
        )

        # Default bias assessment
        self.bias_assessment = BiasRiskAssessment(
            assessment_id="BIAS_001",
            data_sources=[
                "TCGA (The Cancer Genome Atlas)",
                "ICGC (International Cancer Genome Consortium)",
                "Published literature",
                "Computational simulations",
            ],
            population_representation={
                "european_descent": 0.60,
                "asian_descent": 0.20,
                "african_descent": 0.10,
                "hispanic": 0.07,
                "other": 0.03,
            },
            known_biases=[
                {
                    "bias_type": "Population representation",
                    "description": (
                        "Training data over-represents European ancestry populations"
                    ),
                    "severity": "medium",
                    "affected_outputs": "All population-level predictions",
                },
                {
                    "bias_type": "Cancer type coverage",
                    "description": (
                        "More data available for common cancer types"
                    ),
                    "severity": "low",
                    "affected_outputs": "Rare cancer type predictions",
                },
            ],
            mitigation_strategies=[
                "Explicit documentation of population limitations",
                "Confidence adjustment for underrepresented populations",
                "Recommendation for population-specific validation",
                "Regular bias audits with diverse expert panels",
            ],
            monitoring_plan=(
                "Quarterly bias audits with external review. "
                "Annual update of population representation metrics."
            ),
        )

        # Default explainability requirements
        self.explainability_requirements = [
            ExplainabilityRequirement(
                requirement_id="EXPLAIN_001",
                output_type="Treatment sequence recommendation",
                explanation_method="Causal pathway tracing with evidence citations",
                minimum_detail_level="detailed",
                target_audience="oncologist",
                validation_approach="Expert review of explanation accuracy",
            ),
            ExplainabilityRequirement(
                requirement_id="EXPLAIN_002",
                output_type="Hypothesis ranking",
                explanation_method="Factor contribution breakdown",
                minimum_detail_level="intermediate",
                target_audience="researcher",
                validation_approach="Peer review of ranking rationale",
            ),
            ExplainabilityRequirement(
                requirement_id="EXPLAIN_003",
                output_type="Causal graph inference",
                explanation_method="Evidence source attribution",
                minimum_detail_level="detailed",
                target_audience="biologist",
                validation_approach="Literature verification",
            ),
        ]

    def validate_output(
        self,
        output_type: str,
        output_content: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, Any]:
        """Validate an output against safety constraints.

        Args:
            output_type: Type of output being validated
            output_content: Content of the output
            context: Context of the output generation

        Returns:
            Validation result with any required modifications
        """
        validation_result = {
            "output_type": output_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "passed": True,
            "violations": [],
            "warnings": [],
            "required_modifications": [],
        }

        # Check hard constraints
        if self.safety_constraints:
            for constraint in self.safety_constraints.hard_constraints:
                violation = self._check_constraint_violation(
                    constraint, output_type, output_content, context
                )
                if violation:
                    validation_result["violations"].append(violation)
                    validation_result["passed"] = False

        # Check for required disclaimers
        if "disclaimer" not in output_content:
            validation_result["required_modifications"].append(
                "Add appropriate disclaimer to output"
            )

        # Check for uncertainty quantification
        if "uncertainty" not in output_content and "confidence" not in output_content:
            validation_result["warnings"].append(
                "Output lacks uncertainty quantification"
            )

        # Log validation
        self._log_audit(
            "output_validation",
            {
                "output_type": output_type,
                "passed": validation_result["passed"],
                "violations": len(validation_result["violations"]),
            },
        )

        return validation_result

    def _check_constraint_violation(
        self,
        constraint: str,
        output_type: str,
        output_content: dict[str, Any],
        context: dict[str, Any],
    ) -> Optional[dict[str, Any]]:
        """Check if a specific constraint is violated.

        NOTE: This is a simplified keyword-based check for demonstration.
        Production implementations should use:
        1. NLP-based semantic analysis for nuanced violation detection
        2. Formal verification methods for critical constraints
        3. Human-in-the-loop review for ambiguous cases
        4. Domain-specific rule engines for medical context

        This basic implementation provides a first-pass filter but
        is NOT sufficient for production medical AI systems.
        """
        # Keyword-based violation triggers (basic first-pass filter)
        # Maps constraint categories to potentially violating keywords
        violation_triggers = {
            "direct treatment": ["prescribe", "administer", "give patient", "take this"],
            "diagnostic accuracy": ["diagnose", "confirms diagnosis", "you have", "definitive"],
            "autonomous": ["automatically", "without review", "autonomous", "no oversight"],
        }

        content_str = json.dumps(output_content).lower()

        for trigger_type, keywords in violation_triggers.items():
            for keyword in keywords:
                if keyword in content_str:
                    if trigger_type in constraint.lower():
                        return {
                            "constraint": constraint,
                            "trigger": keyword,
                            "location": "output_content",
                            "note": "Basic keyword match - requires human review for confirmation",
                        }

        return None

    def _log_audit(self, event_type: str, details: dict[str, Any]) -> None:
        """Log an audit event."""
        self.audit_log.append(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "event_type": event_type,
                "details": details,
            }
        )

    def generate_compliance_report(self) -> dict[str, Any]:
        """Generate a comprehensive compliance report.

        Returns:
            Compliance report with all relevant information
        """
        return {
            "report_id": hashlib.md5(
                datetime.now(timezone.utc).isoformat().encode()
            ).hexdigest()[:12],
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "system_name": self.name,
            "disclaimer": self.SYSTEM_DISCLAIMER,
            "prohibited_uses": self.PROHIBITED_USES,
            "intended_use": self.intended_use.value,
            "safety_constraints": (
                self.safety_constraints.to_dict() if self.safety_constraints else None
            ),
            "fda_compliance": (
                self.fda_compliance.to_dict() if self.fda_compliance else None
            ),
            "bias_assessment": (
                self.bias_assessment.to_dict() if self.bias_assessment else None
            ),
            "explainability_requirements": [
                r.to_dict() for r in self.explainability_requirements
            ],
            "audit_summary": {
                "total_events": len(self.audit_log),
                "last_event": self.audit_log[-1] if self.audit_log else None,
            },
        }

    def get_required_disclaimers(
        self, output_type: str, target_audience: str = "researcher"
    ) -> list[str]:
        """Get required disclaimers for an output type.

        Args:
            output_type: Type of output
            target_audience: Target audience

        Returns:
            List of required disclaimer statements
        """
        disclaimers = [
            "FOR RESEARCH USE ONLY - Not for clinical diagnosis or treatment",
            "This output requires expert review before any application",
            "Computational hypothesis - experimental validation required",
        ]

        if target_audience == "clinician":
            disclaimers.extend(
                [
                    "Not FDA cleared or approved for clinical use",
                    "Should not replace clinical judgment or standard of care",
                    "Patient-specific factors must be considered",
                ]
            )

        if output_type == "treatment_recommendation":
            disclaimers.append(
                "Treatment suggestions are computational hypotheses only"
            )
            disclaimers.append(
                "Consult appropriate clinical guidelines and oncology expertise"
            )

        return disclaimers

    def acknowledge_disclaimer(self, user_id: str, disclaimer_type: str) -> bool:
        """Record user acknowledgment of disclaimer.

        Args:
            user_id: User identifier
            disclaimer_type: Type of disclaimer acknowledged

        Returns:
            True if acknowledgment recorded
        """
        self._log_audit(
            "disclaimer_acknowledgment",
            {
                "user_id": user_id,
                "disclaimer_type": disclaimer_type,
            },
        )
        return True

    def to_dict(self) -> dict[str, Any]:
        """Serialize module to dictionary."""
        return {
            "name": self.name,
            "intended_use": self.intended_use.value,
            "created_at": self.created_at,
            "safety_constraints": (
                self.safety_constraints.to_dict() if self.safety_constraints else None
            ),
            "fda_compliance": (
                self.fda_compliance.to_dict() if self.fda_compliance else None
            ),
            "bias_assessment": (
                self.bias_assessment.to_dict() if self.bias_assessment else None
            ),
            "explainability_requirements": [
                r.to_dict() for r in self.explainability_requirements
            ],
            "audit_log_size": len(self.audit_log),
            "disclaimer": self.SYSTEM_DISCLAIMER,
        }
