"""
Research Roadmap - 5-10 Year Research Plan

This module implements the research roadmap for the QRATUM oncology
research framework, providing milestone-based planning.

Key components:
1. Research phases with milestones
2. Resource allocation planning
3. Risk-adjusted timeline estimation
4. Dependency management

RESEARCH USE ONLY - This is a planning framework, not a clinical implementation plan.
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


class MilestoneStatus(Enum):
    """Status of a research milestone."""

    PLANNED = "planned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    DELAYED = "delayed"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"


class ResearchDomain(Enum):
    """Research domain categories."""

    COMPUTATIONAL = "computational"
    PRECLINICAL = "preclinical"
    TRANSLATIONAL = "translational"
    CLINICAL = "clinical"
    REGULATORY = "regulatory"


class ResourceType(Enum):
    """Types of research resources."""

    FUNDING = "funding"
    PERSONNEL = "personnel"
    EQUIPMENT = "equipment"
    PARTNERSHIPS = "partnerships"
    DATA = "data"


@dataclass
class ResourceRequirement:
    """Resource requirement for a milestone.

    Attributes:
        resource_type: Type of resource
        amount: Amount required
        unit: Unit of measurement
        description: Description of requirement
        availability: Current availability status
    """

    resource_type: ResourceType
    amount: float
    unit: str
    description: str = ""
    availability: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "resource_type": self.resource_type.value,
            "amount": self.amount,
            "unit": self.unit,
            "description": self.description,
            "availability": self.availability,
        }


@dataclass
class Milestone:
    """A research milestone.

    Attributes:
        milestone_id: Unique identifier
        name: Milestone name
        description: Detailed description
        domain: Research domain
        target_date: Target completion date
        dependencies: IDs of prerequisite milestones
        deliverables: Expected deliverables
        success_criteria: Criteria for success
        resources: Required resources
        status: Current status
        risks: Associated risks
    """

    milestone_id: str
    name: str
    description: str
    domain: ResearchDomain
    target_date: str
    dependencies: list[str] = field(default_factory=list)
    deliverables: list[str] = field(default_factory=list)
    success_criteria: list[str] = field(default_factory=list)
    resources: list[ResourceRequirement] = field(default_factory=list)
    status: MilestoneStatus = MilestoneStatus.PLANNED
    risks: list[dict[str, Any]] = field(default_factory=list)
    actual_completion_date: Optional[str] = None
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "milestone_id": self.milestone_id,
            "name": self.name,
            "description": self.description,
            "domain": self.domain.value,
            "target_date": self.target_date,
            "dependencies": self.dependencies,
            "deliverables": self.deliverables,
            "success_criteria": self.success_criteria,
            "resources": [r.to_dict() for r in self.resources],
            "status": self.status.value,
            "risks": self.risks,
            "actual_completion_date": self.actual_completion_date,
            "notes": self.notes,
        }


@dataclass
class ResearchPhase:
    """A phase in the research roadmap.

    Attributes:
        phase_id: Unique identifier
        name: Phase name
        description: Phase description
        start_year: Start year
        end_year: End year
        objectives: Phase objectives
        milestones: Milestones in this phase
        success_metrics: Overall success metrics
        budget_usd: Total budget estimate
    """

    phase_id: str
    name: str
    description: str
    start_year: int
    end_year: int
    objectives: list[str] = field(default_factory=list)
    milestones: list[Milestone] = field(default_factory=list)
    success_metrics: dict[str, Any] = field(default_factory=dict)
    budget_usd: float = 0.0
    status: MilestoneStatus = MilestoneStatus.PLANNED

    def add_milestone(self, milestone: Milestone) -> None:
        """Add a milestone to this phase."""
        self.milestones.append(milestone)

    def get_progress(self) -> float:
        """Calculate phase progress based on milestone completion."""
        if not self.milestones:
            return 0.0

        completed = sum(1 for m in self.milestones if m.status == MilestoneStatus.COMPLETED)
        return completed / len(self.milestones)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "phase_id": self.phase_id,
            "name": self.name,
            "description": self.description,
            "start_year": self.start_year,
            "end_year": self.end_year,
            "objectives": self.objectives,
            "milestones": [m.to_dict() for m in self.milestones],
            "success_metrics": self.success_metrics,
            "budget_usd": self.budget_usd,
            "status": self.status.value,
            "progress": self.get_progress(),
        }


class ResearchRoadmap:
    """
    Research roadmap for QRATUM oncology research.

    Provides a 5-10 year planning framework with milestones,
    not hype. All timelines are estimates subject to scientific
    discovery and validation requirements.

    RESEARCH USE ONLY - This is a planning framework.
    """

    RESEARCH_DISCLAIMER = """
    This research roadmap is for PLANNING PURPOSES ONLY.
    All timelines are estimates subject to:
    - Scientific discovery uncertainties
    - Regulatory requirements
    - Funding availability
    - Clinical trial outcomes

    Success is not guaranteed and milestones may change
    based on research findings.
    """

    # Risk adjustment factor for timeline estimates (20% buffer)
    # Based on historical data showing research projects typically
    # experience 15-30% timeline expansion due to unforeseen challenges
    RISK_ADJUSTMENT_FACTOR = 1.2

    def __init__(self, name: str = "QRATUM_Oncology_Roadmap", seed: int = 42) -> None:
        """Initialize the research roadmap.

        Args:
            name: Roadmap name
            seed: Random seed for reproducibility
        """
        self.name = name
        self.seed = seed
        self.rng = np.random.RandomState(seed)

        self.phases: list[ResearchPhase] = []
        self.created_at = datetime.now(timezone.utc).isoformat()

        logger.info(f"Initialized ResearchRoadmap: {name}")

    def add_phase(self, phase: ResearchPhase) -> None:
        """Add a research phase."""
        self.phases.append(phase)

    def get_phase(self, phase_id: str) -> Optional[ResearchPhase]:
        """Get a phase by ID."""
        for phase in self.phases:
            if phase.phase_id == phase_id:
                return phase
        return None

    def get_milestone(self, milestone_id: str) -> Optional[Milestone]:
        """Get a milestone by ID."""
        for phase in self.phases:
            for milestone in phase.milestones:
                if milestone.milestone_id == milestone_id:
                    return milestone
        return None

    def get_critical_path(self) -> list[str]:
        """Identify critical path milestones.

        Returns:
            List of milestone IDs on the critical path
        """
        # Build dependency graph
        all_milestones = {}
        for phase in self.phases:
            for milestone in phase.milestones:
                all_milestones[milestone.milestone_id] = milestone

        # Find milestones with most downstream dependencies
        dependency_count = {}
        for mid, milestone in all_milestones.items():
            dependency_count[mid] = 0
            for other_id, other in all_milestones.items():
                if mid in other.dependencies:
                    dependency_count[mid] += 1

        # Critical path = milestones with high dependency count or blocking status
        critical = sorted(dependency_count.items(), key=lambda x: x[1], reverse=True)

        return [mid for mid, _ in critical[:10]]

    def estimate_total_budget(self) -> dict[str, Any]:
        """Estimate total budget across all phases.

        Returns:
            Budget estimate breakdown
        """
        phase_budgets = {}
        for phase in self.phases:
            phase_budgets[phase.name] = phase.budget_usd

        return {
            "total_usd": sum(phase_budgets.values()),
            "by_phase": phase_budgets,
            "average_per_year": sum(phase_budgets.values()) / max(len(self.phases) * 2, 1),
        }

    def estimate_timeline(self) -> dict[str, Any]:
        """Estimate timeline with risk adjustments.

        Returns:
            Timeline estimate
        """
        if not self.phases:
            return {"start_year": 0, "end_year": 0, "duration_years": 0}

        start_year = min(p.start_year for p in self.phases)
        end_year = max(p.end_year for p in self.phases)

        # Risk-adjusted estimate: add buffer to account for typical delays
        # in research timelines due to unforeseen challenges, regulatory
        # requirements, and scientific uncertainties
        risk_adjusted_years = (end_year - start_year) * self.RISK_ADJUSTMENT_FACTOR

        return {
            "start_year": start_year,
            "planned_end_year": end_year,
            "risk_adjusted_end_year": int(start_year + risk_adjusted_years),
            "planned_duration_years": end_year - start_year,
            "risk_adjusted_duration_years": risk_adjusted_years,
        }

    def get_roadmap_summary(self) -> dict[str, Any]:
        """Generate roadmap summary.

        Returns:
            Summary of roadmap status
        """
        total_milestones = sum(len(p.milestones) for p in self.phases)
        completed_milestones = sum(
            sum(1 for m in p.milestones if m.status == MilestoneStatus.COMPLETED)
            for p in self.phases
        )

        return {
            "name": self.name,
            "total_phases": len(self.phases),
            "total_milestones": total_milestones,
            "completed_milestones": completed_milestones,
            "overall_progress": completed_milestones / max(total_milestones, 1),
            "timeline": self.estimate_timeline(),
            "budget": self.estimate_total_budget(),
            "critical_path": self.get_critical_path(),
        }

    def compute_roadmap_hash(self) -> str:
        """Compute unique hash of roadmap contents."""
        roadmap_dict = {
            "name": self.name,
            "phases": [p.to_dict() for p in self.phases],
        }
        roadmap_json = json.dumps(roadmap_dict, sort_keys=True)
        return hashlib.sha256(roadmap_json.encode()).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        """Serialize roadmap to dictionary."""
        return {
            "name": self.name,
            "seed": self.seed,
            "created_at": self.created_at,
            "phases": [p.to_dict() for p in self.phases],
            "summary": self.get_roadmap_summary(),
            "hash": self.compute_roadmap_hash(),
            "disclaimer": self.RESEARCH_DISCLAIMER,
        }


def create_qratum_oncology_roadmap() -> ResearchRoadmap:
    """Create the QRATUM oncology research roadmap (5-10 years).

    Returns:
        ResearchRoadmap with phased milestones
    """
    roadmap = ResearchRoadmap(name="QRATUM_Oncology_5_10_Year_Roadmap", seed=42)

    # Phase 1: Foundation (Years 1-2)
    phase1 = ResearchPhase(
        phase_id="P1",
        name="Phase 1: Foundation",
        description=(
            "Establish computational framework, validate core algorithms, "
            "and build initial hypothesis portfolio"
        ),
        start_year=2025,
        end_year=2027,
        objectives=[
            "Validate VITRA causal graph algorithms on known pathways",
            "Develop XENON intervention search with benchmark datasets",
            "Build initial hypothesis portfolio for 3 cancer types",
            "Establish validation pipeline framework",
            "Secure initial research partnerships",
        ],
        budget_usd=5000000,
    )

    # Phase 1 milestones
    phase1.add_milestone(
        Milestone(
            milestone_id="M1.1",
            name="VITRA Algorithm Validation",
            description="Validate causal graph algorithms on KEGG/Reactome pathways",
            domain=ResearchDomain.COMPUTATIONAL,
            target_date="2025-06",
            deliverables=[
                "Validated causal inference algorithms",
                "Benchmark results on 10+ pathways",
                "Technical documentation",
            ],
            success_criteria=[
                "F1 score > 0.8 on pathway reconstruction",
                "Causal direction accuracy > 85%",
            ],
            resources=[
                ResourceRequirement(ResourceType.FUNDING, 500000, "USD", "Compute and personnel"),
                ResourceRequirement(ResourceType.PERSONNEL, 3, "FTE", "Computational biologists"),
            ],
        )
    )

    phase1.add_milestone(
        Milestone(
            milestone_id="M1.2",
            name="XENON Search Benchmarking",
            description="Benchmark intervention search on synthetic tumor models",
            domain=ResearchDomain.COMPUTATIONAL,
            target_date="2025-12",
            dependencies=["M1.1"],
            deliverables=[
                "Validated search algorithms",
                "Performance benchmarks",
                "Comparison with existing methods",
            ],
            success_criteria=[
                "Search efficiency > 2x baseline",
                "Resistance prediction accuracy > 75%",
            ],
        )
    )

    phase1.add_milestone(
        Milestone(
            milestone_id="M1.3",
            name="Initial Hypothesis Portfolio",
            description="Generate hypothesis portfolio for NSCLC, colorectal, melanoma",
            domain=ResearchDomain.COMPUTATIONAL,
            target_date="2026-06",
            dependencies=["M1.2"],
            deliverables=[
                "20+ ranked hypotheses per cancer type",
                "Risk analysis for top 10 hypotheses",
                "Validation pipeline designs",
            ],
            success_criteria=[
                "Portfolio covers major failure modes",
                "Expert validation of hypothesis quality",
            ],
        )
    )

    phase1.add_milestone(
        Milestone(
            milestone_id="M1.4",
            name="Research Partnership Establishment",
            description="Establish partnerships with 3+ academic cancer centers",
            domain=ResearchDomain.TRANSLATIONAL,
            target_date="2026-12",
            deliverables=[
                "Signed collaboration agreements",
                "Shared data access protocols",
                "Joint research proposals",
            ],
            success_criteria=[
                "3+ active partnerships",
                "Access to clinical data for validation",
            ],
        )
    )

    roadmap.add_phase(phase1)

    # Phase 2: Preclinical Validation (Years 2-4)
    phase2 = ResearchPhase(
        phase_id="P2",
        name="Phase 2: Preclinical Validation",
        description=(
            "Validate top hypotheses in preclinical models, "
            "refine algorithms based on experimental feedback"
        ),
        start_year=2027,
        end_year=2029,
        objectives=[
            "Validate 5+ hypotheses in organoid models",
            "Validate 3+ hypotheses in PDX models",
            "Demonstrate adaptive therapy benefit in preclinical models",
            "Publish foundational research papers",
            "Refine algorithms based on experimental feedback",
        ],
        budget_usd=15000000,
    )

    phase2.add_milestone(
        Milestone(
            milestone_id="M2.1",
            name="Organoid Validation Studies",
            description="Test top hypotheses in patient-derived organoid models",
            domain=ResearchDomain.PRECLINICAL,
            target_date="2027-12",
            dependencies=["M1.3"],
            deliverables=[
                "Validation results for 5+ hypotheses",
                "Mechanism confirmation data",
                "Publications",
            ],
            success_criteria=[
                "3+ hypotheses show predicted effects",
                "Mechanism of action confirmed",
            ],
            resources=[
                ResourceRequirement(ResourceType.FUNDING, 3000000, "USD", "Organoid studies"),
                ResourceRequirement(
                    ResourceType.PARTNERSHIPS,
                    2,
                    "centers",
                    "Cancer centers with organoid facilities",
                ),
            ],
        )
    )

    phase2.add_milestone(
        Milestone(
            milestone_id="M2.2",
            name="PDX Model Validation",
            description="Validate in patient-derived xenograft models",
            domain=ResearchDomain.PRECLINICAL,
            target_date="2028-12",
            dependencies=["M2.1"],
            deliverables=[
                "PDX validation results",
                "Efficacy and resistance data",
                "Biomarker candidates",
            ],
            success_criteria=[
                "2+ hypotheses validated in PDX",
                "Resistance delay demonstrated",
            ],
        )
    )

    phase2.add_milestone(
        Milestone(
            milestone_id="M2.3",
            name="Adaptive Therapy Preclinical Proof-of-Concept",
            description="Demonstrate adaptive therapy principle in PDX models",
            domain=ResearchDomain.PRECLINICAL,
            target_date="2029-06",
            dependencies=["M2.2"],
            deliverables=[
                "Adaptive vs continuous comparison",
                "Optimal scheduling parameters",
                "Translational biomarkers",
            ],
            success_criteria=[
                "PFS improvement > 50% with adaptive approach",
                "Biomarker-guided scheduling feasible",
            ],
        )
    )

    roadmap.add_phase(phase2)

    # Phase 3: Clinical Translation (Years 4-7)
    phase3 = ResearchPhase(
        phase_id="P3",
        name="Phase 3: Clinical Translation",
        description=(
            "Initiate clinical trials for validated hypotheses, " "establish regulatory pathway"
        ),
        start_year=2029,
        end_year=2032,
        objectives=[
            "Complete Phase I trial for adaptive therapy approach",
            "Establish FDA breakthrough therapy pathway if applicable",
            "Expand hypothesis portfolio to 5+ cancer types",
            "Develop clinical-grade decision support tool",
            "Build multi-center trial network",
        ],
        budget_usd=50000000,
    )

    phase3.add_milestone(
        Milestone(
            milestone_id="M3.1",
            name="IND Submission",
            description="Submit Investigational New Drug application",
            domain=ResearchDomain.REGULATORY,
            target_date="2029-12",
            dependencies=["M2.3"],
            deliverables=[
                "IND application package",
                "FDA pre-IND meeting summary",
                "Clinical protocol",
            ],
            success_criteria=[
                "IND cleared by FDA",
                "Phase I trial authorized",
            ],
        )
    )

    phase3.add_milestone(
        Milestone(
            milestone_id="M3.2",
            name="Phase I Clinical Trial",
            description="Conduct Phase I trial of adaptive therapy approach",
            domain=ResearchDomain.CLINICAL,
            target_date="2031-12",
            dependencies=["M3.1"],
            deliverables=[
                "Safety data",
                "Preliminary efficacy signals",
                "Optimal adaptive schedule",
            ],
            success_criteria=[
                "Acceptable safety profile",
                "Signals of clinical benefit",
            ],
            risks=[
                {
                    "risk": "Insufficient patient enrollment",
                    "probability": 0.3,
                    "mitigation": "Multi-center approach",
                },
                {
                    "risk": "Unexpected toxicity",
                    "probability": 0.2,
                    "mitigation": "Careful dose escalation",
                },
            ],
        )
    )

    phase3.add_milestone(
        Milestone(
            milestone_id="M3.3",
            name="Clinical Decision Support Tool Development",
            description="Develop clinical-grade version of QRATUM oncology",
            domain=ResearchDomain.TRANSLATIONAL,
            target_date="2032-06",
            dependencies=["M3.2"],
            deliverables=[
                "Clinical-grade software",
                "FDA 510(k) or De Novo submission",
                "User documentation",
            ],
            success_criteria=[
                "Software meets FDA SaMD requirements",
                "Validated for clinical use",
            ],
        )
    )

    roadmap.add_phase(phase3)

    # Phase 4: Expansion and Impact (Years 7-10)
    phase4 = ResearchPhase(
        phase_id="P4",
        name="Phase 4: Expansion and Impact",
        description=(
            "Expand clinical validation, scale platform, " "demonstrate population-level impact"
        ),
        start_year=2032,
        end_year=2035,
        objectives=[
            "Complete Phase II/III trials for lead indication",
            "Expand to 10+ cancer types",
            "Achieve regulatory clearance for decision support tool",
            "Demonstrate real-world evidence of clinical benefit",
            "Establish as standard of care decision support",
        ],
        budget_usd=100000000,
    )

    phase4.add_milestone(
        Milestone(
            milestone_id="M4.1",
            name="Phase II/III Clinical Trial",
            description="Pivotal trial for adaptive therapy approach",
            domain=ResearchDomain.CLINICAL,
            target_date="2034-06",
            dependencies=["M3.2"],
            deliverables=[
                "Pivotal trial results",
                "Regulatory submission",
                "Practice-changing evidence",
            ],
            success_criteria=[
                "Primary endpoint met",
                "Favorable benefit-risk profile",
            ],
        )
    )

    phase4.add_milestone(
        Milestone(
            milestone_id="M4.2",
            name="Multi-Cancer Platform Validation",
            description="Validate platform across 10+ cancer types",
            domain=ResearchDomain.TRANSLATIONAL,
            target_date="2034-12",
            dependencies=["M3.3"],
            deliverables=[
                "Validation results across cancer types",
                "Cancer-specific model refinements",
                "Expanded hypothesis portfolios",
            ],
            success_criteria=[
                "Platform validated in 10+ cancer types",
                "Consistent performance across types",
            ],
        )
    )

    phase4.add_milestone(
        Milestone(
            milestone_id="M4.3",
            name="Real-World Evidence Generation",
            description="Generate real-world evidence of clinical impact",
            domain=ResearchDomain.CLINICAL,
            target_date="2035-06",
            dependencies=["M4.1", "M4.2"],
            deliverables=[
                "Real-world data analysis",
                "Health economics evidence",
                "Patient outcomes data",
            ],
            success_criteria=[
                "Demonstrated improvement in patient outcomes",
                "Cost-effectiveness established",
            ],
        )
    )

    roadmap.add_phase(phase4)

    return roadmap
