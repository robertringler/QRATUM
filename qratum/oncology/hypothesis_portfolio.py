"""
Hypothesis Portfolio - Research Hypothesis Management

This module manages ranked research hypotheses with risk analysis,
supporting the QRATUM oncology research framework.

Key capabilities:
1. Hypothesis generation and ranking
2. Risk analysis and failure mode identification
3. Counter-strategy development
4. Priority scoring for research allocation

RESEARCH USE ONLY - Not for clinical applications.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


class HypothesisCategory(Enum):
    """Categories of research hypotheses."""

    MECHANISM_DISCOVERY = "mechanism_discovery"
    RESISTANCE_SUPPRESSION = "resistance_suppression"
    IMMUNE_RESYNCHRONIZATION = "immune_resynchronization"
    METABOLIC_INTERVENTION = "metabolic_intervention"
    EPIGENETIC_REPROGRAMMING = "epigenetic_reprogramming"
    COMBINATION_THERAPY = "combination_therapy"
    ADAPTIVE_THERAPY = "adaptive_therapy"
    BIOMARKER_DISCOVERY = "biomarker_discovery"
    CROSS_CANCER_MOTIF = "cross_cancer_motif"


class EvidenceLevel(Enum):
    """Level of supporting evidence."""

    THEORETICAL = "theoretical"
    COMPUTATIONAL = "computational"
    IN_VITRO = "in_vitro"
    IN_VIVO = "in_vivo"
    CLINICAL_OBSERVATIONAL = "clinical_observational"
    CLINICAL_TRIAL = "clinical_trial"


class RiskLevel(Enum):
    """Risk level classification."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


@dataclass
class RiskAnalysis:
    """Risk analysis for a research hypothesis.

    Attributes:
        risk_id: Unique identifier
        overall_risk: Overall risk level
        failure_modes: List of potential failure modes
        counter_strategies: Strategies to mitigate failures
        probability_of_success: Estimated success probability
        impact_if_successful: Impact score if successful
        resource_requirement: Relative resource requirement
        time_to_validation: Estimated months to validation
    """

    risk_id: str
    overall_risk: RiskLevel = RiskLevel.MEDIUM
    failure_modes: list[dict[str, Any]] = field(default_factory=list)
    counter_strategies: list[dict[str, Any]] = field(default_factory=list)
    probability_of_success: float = 0.3
    impact_if_successful: float = 0.5
    resource_requirement: float = 0.5
    time_to_validation: int = 24
    ethical_considerations: list[str] = field(default_factory=list)
    regulatory_hurdles: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "risk_id": self.risk_id,
            "overall_risk": self.overall_risk.value,
            "failure_modes": self.failure_modes,
            "counter_strategies": self.counter_strategies,
            "probability_of_success": self.probability_of_success,
            "impact_if_successful": self.impact_if_successful,
            "resource_requirement": self.resource_requirement,
            "time_to_validation": self.time_to_validation,
            "ethical_considerations": self.ethical_considerations,
            "regulatory_hurdles": self.regulatory_hurdles,
        }


@dataclass
class ResearchHypothesis:
    """A research hypothesis with supporting evidence and risk analysis.

    Attributes:
        hypothesis_id: Unique identifier
        title: Hypothesis title
        description: Detailed description
        category: Hypothesis category
        cancer_types: Applicable cancer types
        evidence_level: Current evidence level
        supporting_evidence: List of supporting evidence
        contradicting_evidence: List of contradicting evidence
        testable_predictions: Specific testable predictions
        validation_approaches: Proposed validation methods
        risk_analysis: Associated risk analysis
        priority_score: Priority score for resource allocation
        related_hypotheses: IDs of related hypotheses
    """

    hypothesis_id: str
    title: str
    description: str
    category: HypothesisCategory
    cancer_types: list[str] = field(default_factory=list)
    evidence_level: EvidenceLevel = EvidenceLevel.THEORETICAL
    supporting_evidence: list[dict[str, Any]] = field(default_factory=list)
    contradicting_evidence: list[dict[str, Any]] = field(default_factory=list)
    testable_predictions: list[str] = field(default_factory=list)
    validation_approaches: list[dict[str, Any]] = field(default_factory=list)
    risk_analysis: Optional[RiskAnalysis] = None
    priority_score: float = 0.0
    related_hypotheses: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    provenance: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "hypothesis_id": self.hypothesis_id,
            "title": self.title,
            "description": self.description,
            "category": self.category.value,
            "cancer_types": self.cancer_types,
            "evidence_level": self.evidence_level.value,
            "supporting_evidence": self.supporting_evidence,
            "contradicting_evidence": self.contradicting_evidence,
            "testable_predictions": self.testable_predictions,
            "validation_approaches": self.validation_approaches,
            "risk_analysis": self.risk_analysis.to_dict() if self.risk_analysis else None,
            "priority_score": self.priority_score,
            "related_hypotheses": self.related_hypotheses,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "provenance": self.provenance,
        }


class HypothesisPortfolio:
    """
    Portfolio manager for research hypotheses.

    Maintains a ranked portfolio of research hypotheses with
    risk analysis and resource allocation support.

    RESEARCH USE ONLY - Not for clinical applications.
    """

    RESEARCH_DISCLAIMER = """
    This hypothesis portfolio is for RESEARCH PURPOSES ONLY.
    Hypotheses require experimental validation before any clinical application.
    Not approved for clinical decision-making.
    """

    def __init__(self, name: str, seed: int = 42) -> None:
        """Initialize the hypothesis portfolio.

        Args:
            name: Portfolio name
            seed: Random seed for reproducibility
        """
        self.name = name
        self.seed = seed
        self.rng = np.random.RandomState(seed)

        self._hypotheses: dict[str, ResearchHypothesis] = {}
        self.created_at = datetime.now(timezone.utc).isoformat()

        logger.info(f"Initialized HypothesisPortfolio: {name}")

    def add_hypothesis(self, hypothesis: ResearchHypothesis) -> None:
        """Add a hypothesis to the portfolio.

        Args:
            hypothesis: ResearchHypothesis to add
        """
        self._hypotheses[hypothesis.hypothesis_id] = hypothesis
        self._update_priority_scores()
        logger.debug(f"Added hypothesis {hypothesis.hypothesis_id}")

    def get_hypothesis(self, hypothesis_id: str) -> Optional[ResearchHypothesis]:
        """Get a hypothesis by ID."""
        return self._hypotheses.get(hypothesis_id)

    def remove_hypothesis(self, hypothesis_id: str) -> bool:
        """Remove a hypothesis from the portfolio."""
        if hypothesis_id in self._hypotheses:
            del self._hypotheses[hypothesis_id]
            self._update_priority_scores()
            return True
        return False

    def _update_priority_scores(self) -> None:
        """Update priority scores for all hypotheses."""
        for hypothesis in self._hypotheses.values():
            hypothesis.priority_score = self._compute_priority_score(hypothesis)

    def _compute_priority_score(self, hypothesis: ResearchHypothesis) -> float:
        """Compute priority score for a hypothesis.

        Score considers:
        - Probability of success
        - Impact if successful
        - Resource requirements
        - Time to validation
        - Evidence level

        Args:
            hypothesis: Hypothesis to score

        Returns:
            Priority score (0.0 to 1.0)
        """
        if hypothesis.risk_analysis is None:
            return 0.5

        risk = hypothesis.risk_analysis

        # Expected value = P(success) * Impact
        expected_value = risk.probability_of_success * risk.impact_if_successful

        # Efficiency = Expected value / Resources
        efficiency = expected_value / max(risk.resource_requirement, 0.1)

        # Time penalty (prefer faster validation)
        time_penalty = 1.0 / (1.0 + risk.time_to_validation / 36.0)

        # Evidence bonus (higher evidence = higher confidence)
        evidence_bonus = {
            EvidenceLevel.THEORETICAL: 0.0,
            EvidenceLevel.COMPUTATIONAL: 0.1,
            EvidenceLevel.IN_VITRO: 0.2,
            EvidenceLevel.IN_VIVO: 0.3,
            EvidenceLevel.CLINICAL_OBSERVATIONAL: 0.4,
            EvidenceLevel.CLINICAL_TRIAL: 0.5,
        }.get(hypothesis.evidence_level, 0.0)

        # Combined score
        score = efficiency * 0.4 + time_penalty * 0.2 + evidence_bonus * 0.4

        return min(max(score, 0.0), 1.0)

    def get_ranked_hypotheses(
        self,
        category: Optional[HypothesisCategory] = None,
        min_evidence_level: Optional[EvidenceLevel] = None,
        max_risk: Optional[RiskLevel] = None,
        limit: int = 10,
    ) -> list[ResearchHypothesis]:
        """Get hypotheses ranked by priority.

        Args:
            category: Filter by category
            min_evidence_level: Minimum evidence level
            max_risk: Maximum risk level
            limit: Maximum number to return

        Returns:
            List of hypotheses sorted by priority
        """
        # Define evidence ordering
        evidence_order = [
            EvidenceLevel.THEORETICAL,
            EvidenceLevel.COMPUTATIONAL,
            EvidenceLevel.IN_VITRO,
            EvidenceLevel.IN_VIVO,
            EvidenceLevel.CLINICAL_OBSERVATIONAL,
            EvidenceLevel.CLINICAL_TRIAL,
        ]

        # Define risk ordering
        risk_order = [
            RiskLevel.VERY_HIGH,
            RiskLevel.HIGH,
            RiskLevel.MEDIUM,
            RiskLevel.LOW,
        ]

        candidates = list(self._hypotheses.values())

        # Apply filters
        if category is not None:
            candidates = [h for h in candidates if h.category == category]

        if min_evidence_level is not None:
            min_idx = evidence_order.index(min_evidence_level)
            candidates = [
                h for h in candidates if evidence_order.index(h.evidence_level) >= min_idx
            ]

        if max_risk is not None and max_risk in risk_order:
            max_idx = risk_order.index(max_risk)
            candidates = [
                h
                for h in candidates
                if h.risk_analysis is not None
                and risk_order.index(h.risk_analysis.overall_risk) >= max_idx
            ]

        # Sort by priority
        candidates.sort(key=lambda h: h.priority_score, reverse=True)

        return candidates[:limit]

    def generate_failure_analysis_report(self) -> dict[str, Any]:
        """Generate a comprehensive failure analysis report.

        Analyzes why most cancer approaches fail and how the
        current hypotheses address these failure modes.

        Returns:
            Failure analysis report
        """
        common_failure_modes = [
            {
                "mode": "Tumor Heterogeneity",
                "description": (
                    "Tumors contain diverse subpopulations with different "
                    "vulnerabilities, allowing resistant clones to survive treatment"
                ),
                "frequency": 0.85,
                "mitigation_hypotheses": [],
            },
            {
                "mode": "Acquired Resistance",
                "description": (
                    "Tumor cells acquire mutations or epigenetic changes "
                    "that bypass targeted therapies"
                ),
                "frequency": 0.75,
                "mitigation_hypotheses": [],
            },
            {
                "mode": "Immune Evasion",
                "description": (
                    "Tumors develop mechanisms to evade immune surveillance, "
                    "limiting immunotherapy effectiveness"
                ),
                "frequency": 0.70,
                "mitigation_hypotheses": [],
            },
            {
                "mode": "Microenvironment Protection",
                "description": (
                    "The tumor microenvironment provides sanctuary for "
                    "cancer cells and promotes resistance"
                ),
                "frequency": 0.65,
                "mitigation_hypotheses": [],
            },
            {
                "mode": "Toxicity Limitations",
                "description": (
                    "Effective doses are limited by normal tissue toxicity, "
                    "preventing optimal treatment"
                ),
                "frequency": 0.60,
                "mitigation_hypotheses": [],
            },
            {
                "mode": "Biomarker Imprecision",
                "description": (
                    "Current biomarkers fail to accurately predict treatment "
                    "response, leading to suboptimal patient selection"
                ),
                "frequency": 0.55,
                "mitigation_hypotheses": [],
            },
        ]

        # Map hypotheses to failure modes they address
        for hypothesis in self._hypotheses.values():
            if hypothesis.category == HypothesisCategory.RESISTANCE_SUPPRESSION:
                common_failure_modes[1]["mitigation_hypotheses"].append(hypothesis.hypothesis_id)
                common_failure_modes[0]["mitigation_hypotheses"].append(hypothesis.hypothesis_id)
            elif hypothesis.category == HypothesisCategory.IMMUNE_RESYNCHRONIZATION:
                common_failure_modes[2]["mitigation_hypotheses"].append(hypothesis.hypothesis_id)
            elif hypothesis.category == HypothesisCategory.METABOLIC_INTERVENTION:
                common_failure_modes[3]["mitigation_hypotheses"].append(hypothesis.hypothesis_id)
            elif hypothesis.category == HypothesisCategory.ADAPTIVE_THERAPY:
                common_failure_modes[0]["mitigation_hypotheses"].append(hypothesis.hypothesis_id)
                common_failure_modes[1]["mitigation_hypotheses"].append(hypothesis.hypothesis_id)
            elif hypothesis.category == HypothesisCategory.BIOMARKER_DISCOVERY:
                common_failure_modes[5]["mitigation_hypotheses"].append(hypothesis.hypothesis_id)

        # Calculate coverage
        covered_modes = sum(1 for mode in common_failure_modes if mode["mitigation_hypotheses"])
        coverage_rate = covered_modes / len(common_failure_modes)

        return {
            "title": "Cancer Research Failure Analysis",
            "common_failure_modes": common_failure_modes,
            "coverage_rate": coverage_rate,
            "total_hypotheses": len(self._hypotheses),
            "recommendations": self._generate_recommendations(common_failure_modes),
            "disclaimer": self.RESEARCH_DISCLAIMER,
        }

    def _generate_recommendations(self, failure_modes: list[dict[str, Any]]) -> list[str]:
        """Generate recommendations based on gap analysis."""
        recommendations = []

        for mode in failure_modes:
            if not mode["mitigation_hypotheses"]:
                recommendations.append(
                    f"Gap identified: No hypotheses address '{mode['mode']}'. "
                    f"Consider developing hypotheses targeting this failure mode."
                )

        # Add general recommendations
        recommendations.extend(
            [
                "Prioritize combination approaches that target multiple failure modes",
                "Integrate adaptive therapy principles to address heterogeneity",
                "Develop predictive biomarkers for treatment response",
                "Consider tumor microenvironment as a therapeutic target",
            ]
        )

        return recommendations

    def compute_portfolio_hash(self) -> str:
        """Compute unique hash of portfolio contents."""
        portfolio_dict = {
            "name": self.name,
            "hypotheses": sorted(
                [h.to_dict() for h in self._hypotheses.values()],
                key=lambda x: x["hypothesis_id"],
            ),
        }
        portfolio_json = json.dumps(portfolio_dict, sort_keys=True)
        return hashlib.sha256(portfolio_json.encode()).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        """Serialize portfolio to dictionary."""
        return {
            "name": self.name,
            "seed": self.seed,
            "created_at": self.created_at,
            "hypotheses": [h.to_dict() for h in self._hypotheses.values()],
            "hash": self.compute_portfolio_hash(),
            "disclaimer": self.RESEARCH_DISCLAIMER,
        }


def create_example_hypothesis_portfolio() -> HypothesisPortfolio:
    """Create an example hypothesis portfolio.

    Returns:
        HypothesisPortfolio with example research hypotheses
    """
    portfolio = HypothesisPortfolio(name="QRATUM_Oncology_v1", seed=42)

    # Hypothesis 1: Adaptive Therapy
    h1_risk = RiskAnalysis(
        risk_id="R001",
        overall_risk=RiskLevel.MEDIUM,
        failure_modes=[
            {
                "mode": "Resistance emergence despite drug holidays",
                "probability": 0.3,
                "impact": "Treatment failure",
            },
            {
                "mode": "Optimal switching thresholds unknown",
                "probability": 0.4,
                "impact": "Suboptimal disease control",
            },
        ],
        counter_strategies=[
            {
                "strategy": "ctDNA-guided treatment decisions",
                "description": "Use circulating tumor DNA to detect early resistance",
            },
            {
                "strategy": "Multi-arm adaptive trials",
                "description": "Test multiple threshold strategies simultaneously",
            },
        ],
        probability_of_success=0.4,
        impact_if_successful=0.8,
        resource_requirement=0.6,
        time_to_validation=36,
        ethical_considerations=[
            "Patient consent for non-standard treatment paradigm",
            "Monitoring burden on patients",
        ],
    )

    h1 = ResearchHypothesis(
        hypothesis_id="H001",
        title="Adaptive Therapy Delays Resistance in EGFR-Mutant NSCLC",
        description=(
            "Intermittent treatment with EGFR inhibitors, guided by tumor burden "
            "monitoring, will delay resistance emergence compared to continuous "
            "maximum-tolerated dose treatment by maintaining a population of "
            "drug-sensitive cells that compete with resistant clones."
        ),
        category=HypothesisCategory.ADAPTIVE_THERAPY,
        cancer_types=["NSCLC", "Colorectal", "Melanoma"],
        evidence_level=EvidenceLevel.IN_VIVO,
        supporting_evidence=[
            {
                "source": "Zhang et al., Nature Communications 2017",
                "finding": "Mathematical models predict adaptive therapy delays resistance",
            },
            {
                "source": "Enriquez-Navas et al., Science Translational Medicine 2016",
                "finding": "Adaptive therapy improved survival in mouse models",
            },
        ],
        testable_predictions=[
            "Time to progression will increase by >50% with adaptive dosing",
            "Resistant clone frequency will remain lower in adaptive arm",
            "Overall survival will not be compromised by drug holidays",
        ],
        validation_approaches=[
            {
                "approach": "Randomized Phase II trial",
                "description": "Compare adaptive vs continuous osimertinib",
                "timeline_months": 36,
            },
            {
                "approach": "Patient-derived xenograft models",
                "description": "Test adaptive schedules in PDX",
                "timeline_months": 12,
            },
        ],
        risk_analysis=h1_risk,
        provenance=["QRATUM_oncology_generator"],
    )
    portfolio.add_hypothesis(h1)

    # Hypothesis 2: Immune Re-synchronization
    h2_risk = RiskAnalysis(
        risk_id="R002",
        overall_risk=RiskLevel.HIGH,
        failure_modes=[
            {
                "mode": "Immune-related adverse events",
                "probability": 0.35,
                "impact": "Treatment discontinuation",
            },
            {
                "mode": "Inadequate T-cell infiltration",
                "probability": 0.5,
                "impact": "No response",
            },
        ],
        counter_strategies=[
            {
                "strategy": "Biomarker-guided patient selection",
                "description": "Select patients with favorable immune profiles",
            },
            {
                "strategy": "Local immune priming",
                "description": "Combine with radiation or oncolytic virus",
            },
        ],
        probability_of_success=0.3,
        impact_if_successful=0.9,
        resource_requirement=0.7,
        time_to_validation=48,
    )

    h2 = ResearchHypothesis(
        hypothesis_id="H002",
        title="Temporal Sequencing Restores Immune Checkpoint Response",
        description=(
            "Sequential administration of immune checkpoint inhibitors following "
            "a period of targeted therapy will restore immune cell infiltration "
            "and improve response rates by exploiting the immunomodulatory effects "
            "of tumor cell death induced by targeted agents."
        ),
        category=HypothesisCategory.IMMUNE_RESYNCHRONIZATION,
        cancer_types=["NSCLC", "Melanoma", "Renal Cell Carcinoma"],
        evidence_level=EvidenceLevel.COMPUTATIONAL,
        testable_predictions=[
            "CD8+ T-cell infiltration increases after targeted therapy priming",
            "IFN-gamma signature predicts response to sequential immunotherapy",
            "Optimal sequencing window is 2-4 weeks post targeted therapy",
        ],
        risk_analysis=h2_risk,
        provenance=["QRATUM_oncology_generator"],
    )
    portfolio.add_hypothesis(h2)

    # Hypothesis 3: Metabolic Vulnerability
    h3_risk = RiskAnalysis(
        risk_id="R003",
        overall_risk=RiskLevel.MEDIUM,
        failure_modes=[
            {
                "mode": "Metabolic plasticity",
                "probability": 0.4,
                "impact": "Compensatory pathway activation",
            },
        ],
        counter_strategies=[
            {
                "strategy": "Multi-target metabolic inhibition",
                "description": "Simultaneously target multiple metabolic nodes",
            },
        ],
        probability_of_success=0.35,
        impact_if_successful=0.7,
        resource_requirement=0.5,
        time_to_validation=24,
    )

    h3 = ResearchHypothesis(
        hypothesis_id="H003",
        title="Glutamine Dependency Creates Therapeutic Window in KRAS-Mutant Cancers",
        description=(
            "KRAS-mutant cancers exhibit enhanced glutamine dependency for "
            "anaplerosis. Targeting glutaminase in combination with KRAS pathway "
            "inhibitors will create synthetic lethality by disrupting both "
            "signaling and metabolic adaptations simultaneously."
        ),
        category=HypothesisCategory.METABOLIC_INTERVENTION,
        cancer_types=["Pancreatic", "NSCLC", "Colorectal"],
        evidence_level=EvidenceLevel.IN_VITRO,
        testable_predictions=[
            "CB-839 + KRAS-G12C inhibitor shows synergy in KRAS-mutant cell lines",
            "Glutamine deprivation sensitizes to KRAS pathway inhibition",
            "Metabolic flux analysis reveals anaplerosis bottleneck",
        ],
        risk_analysis=h3_risk,
        provenance=["QRATUM_oncology_generator"],
    )
    portfolio.add_hypothesis(h3)

    # Hypothesis 4: Cross-Cancer Motif
    h4_risk = RiskAnalysis(
        risk_id="R004",
        overall_risk=RiskLevel.LOW,
        failure_modes=[
            {
                "mode": "Context-dependent effects",
                "probability": 0.3,
                "impact": "Limited generalizability",
            },
        ],
        counter_strategies=[
            {
                "strategy": "Pan-cancer validation cohort",
                "description": "Test across multiple tumor types",
            },
        ],
        probability_of_success=0.5,
        impact_if_successful=0.85,
        resource_requirement=0.4,
        time_to_validation=18,
    )

    h4 = ResearchHypothesis(
        hypothesis_id="H004",
        title="p53 Reactivation Restores Immune Visibility Across Cancer Types",
        description=(
            "Loss of p53 function is a universal cancer feature that contributes "
            "to immune evasion. Pharmacological reactivation of p53 will restore "
            "MHC class I expression and enhance tumor immunogenicity across "
            "multiple cancer types with p53 mutations."
        ),
        category=HypothesisCategory.CROSS_CANCER_MOTIF,
        cancer_types=["Pan-cancer"],
        evidence_level=EvidenceLevel.IN_VITRO,
        testable_predictions=[
            "APR-246 treatment restores MHC-I in p53-mutant cell lines",
            "p53 reactivation increases tumor mutational antigen presentation",
            "Combination with PD-1 inhibitors improves response across cancer types",
        ],
        risk_analysis=h4_risk,
        provenance=["QRATUM_oncology_generator"],
    )
    portfolio.add_hypothesis(h4)

    return portfolio
