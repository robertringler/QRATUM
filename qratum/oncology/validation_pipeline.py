"""
Validation Pipeline - Research Validation Framework

This module implements the validation pipeline from in-silico to clinical
translation, supporting rigorous hypothesis testing.

Pipeline stages:
1. In-silico: QRATUM simulations, synthetic tumor populations
2. In-vitro: Organoids, co-culture immune models
3. In-vivo: Mouse models (ethically constrained)
4. Clinical Translation: Phase 0/I trial design, biomarkers, endpoints

RESEARCH USE ONLY - Not for clinical applications.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class ValidationStage(Enum):
    """Stages in the validation pipeline."""

    IN_SILICO = "in_silico"
    IN_VITRO = "in_vitro"
    IN_VIVO = "in_vivo"
    CLINICAL_PHASE_0 = "clinical_phase_0"
    CLINICAL_PHASE_I = "clinical_phase_i"
    CLINICAL_PHASE_II = "clinical_phase_ii"
    CLINICAL_PHASE_III = "clinical_phase_iii"


class ValidationStatus(Enum):
    """Status of a validation step."""

    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED_SUCCESS = "completed_success"
    COMPLETED_FAILURE = "completed_failure"
    BLOCKED = "blocked"


class ModelSystem(Enum):
    """Experimental model systems."""

    # In-silico
    AGENT_BASED_MODEL = "agent_based_model"
    ODE_MODEL = "ode_model"
    STOCHASTIC_SIMULATION = "stochastic_simulation"
    MACHINE_LEARNING = "machine_learning"

    # In-vitro
    CELL_LINE_2D = "cell_line_2d"
    CELL_LINE_3D = "cell_line_3d"
    ORGANOID = "organoid"
    IMMUNE_CO_CULTURE = "immune_co_culture"
    PDO = "patient_derived_organoid"

    # In-vivo
    CDX = "cell_derived_xenograft"
    PDX = "patient_derived_xenograft"
    GENETICALLY_ENGINEERED = "genetically_engineered"
    HUMANIZED_MOUSE = "humanized_mouse"


@dataclass
class ValidationCriteria:
    """Criteria for validation success.

    Attributes:
        criterion_id: Unique identifier
        description: Criterion description
        metric: Metric to measure
        threshold: Threshold for success
        direction: 'greater' or 'less' than threshold
        weight: Weight in overall assessment
    """

    criterion_id: str
    description: str
    metric: str
    threshold: float
    direction: str = "greater"
    weight: float = 1.0

    def evaluate(self, value: float) -> bool:
        """Evaluate if criterion is met."""
        if self.direction == "greater":
            return value >= self.threshold
        else:
            return value <= self.threshold

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "criterion_id": self.criterion_id,
            "description": self.description,
            "metric": self.metric,
            "threshold": self.threshold,
            "direction": self.direction,
            "weight": self.weight,
        }


@dataclass
class ValidationResult:
    """Result of a validation step.

    Attributes:
        result_id: Unique identifier
        criterion_id: Associated criterion
        observed_value: Observed metric value
        passed: Whether criterion was met
        confidence_interval: 95% confidence interval
        notes: Additional notes
    """

    result_id: str
    criterion_id: str
    observed_value: float
    passed: bool
    confidence_interval: tuple[float, float] = (0.0, 0.0)
    notes: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "result_id": self.result_id,
            "criterion_id": self.criterion_id,
            "observed_value": self.observed_value,
            "passed": self.passed,
            "confidence_interval": self.confidence_interval,
            "notes": self.notes,
            "timestamp": self.timestamp,
        }


@dataclass
class InSilicoValidation:
    """In-silico validation specification.

    Attributes:
        validation_id: Unique identifier
        model_type: Type of computational model
        description: Validation description
        simulation_parameters: Parameters for simulation
        synthetic_population_size: Size of synthetic tumor population
        criteria: Validation criteria
        results: Validation results
    """

    validation_id: str
    model_type: ModelSystem
    description: str
    simulation_parameters: dict[str, Any] = field(default_factory=dict)
    synthetic_population_size: int = 1000
    criteria: list[ValidationCriteria] = field(default_factory=list)
    results: list[ValidationResult] = field(default_factory=list)
    status: ValidationStatus = ValidationStatus.NOT_STARTED
    estimated_compute_hours: float = 10.0
    provenance: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "validation_id": self.validation_id,
            "stage": ValidationStage.IN_SILICO.value,
            "model_type": self.model_type.value,
            "description": self.description,
            "simulation_parameters": self.simulation_parameters,
            "synthetic_population_size": self.synthetic_population_size,
            "criteria": [c.to_dict() for c in self.criteria],
            "results": [r.to_dict() for r in self.results],
            "status": self.status.value,
            "estimated_compute_hours": self.estimated_compute_hours,
            "provenance": self.provenance,
        }


@dataclass
class InVitroValidation:
    """In-vitro validation specification.

    Attributes:
        validation_id: Unique identifier
        model_system: In-vitro model system
        cell_lines: Cell lines to use
        conditions: Experimental conditions
        assays: Assays to perform
        criteria: Validation criteria
        ethical_approval: Ethics approval reference
    """

    validation_id: str
    model_system: ModelSystem
    cell_lines: list[str] = field(default_factory=list)
    conditions: list[dict[str, Any]] = field(default_factory=list)
    assays: list[str] = field(default_factory=list)
    criteria: list[ValidationCriteria] = field(default_factory=list)
    results: list[ValidationResult] = field(default_factory=list)
    status: ValidationStatus = ValidationStatus.NOT_STARTED
    ethical_approval: str = ""
    estimated_duration_weeks: int = 8
    estimated_cost_usd: float = 50000.0
    provenance: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "validation_id": self.validation_id,
            "stage": ValidationStage.IN_VITRO.value,
            "model_system": self.model_system.value,
            "cell_lines": self.cell_lines,
            "conditions": self.conditions,
            "assays": self.assays,
            "criteria": [c.to_dict() for c in self.criteria],
            "results": [r.to_dict() for r in self.results],
            "status": self.status.value,
            "ethical_approval": self.ethical_approval,
            "estimated_duration_weeks": self.estimated_duration_weeks,
            "estimated_cost_usd": self.estimated_cost_usd,
            "provenance": self.provenance,
        }


@dataclass
class InVivoValidation:
    """In-vivo validation specification.

    Attributes:
        validation_id: Unique identifier
        model_system: Animal model system
        species: Animal species
        sample_size: Number of animals
        treatment_groups: Treatment group definitions
        endpoints: Study endpoints
        ethical_considerations: Ethical constraints and considerations
        iacuc_approval: IACUC protocol reference
    """

    validation_id: str
    model_system: ModelSystem
    species: str = "Mus musculus"
    sample_size: int = 30
    treatment_groups: list[dict[str, Any]] = field(default_factory=list)
    endpoints: list[str] = field(default_factory=list)
    criteria: list[ValidationCriteria] = field(default_factory=list)
    results: list[ValidationResult] = field(default_factory=list)
    status: ValidationStatus = ValidationStatus.NOT_STARTED
    ethical_considerations: list[str] = field(default_factory=list)
    iacuc_approval: str = ""
    humane_endpoints: list[str] = field(default_factory=list)
    estimated_duration_weeks: int = 16
    estimated_cost_usd: float = 150000.0
    provenance: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "validation_id": self.validation_id,
            "stage": ValidationStage.IN_VIVO.value,
            "model_system": self.model_system.value,
            "species": self.species,
            "sample_size": self.sample_size,
            "treatment_groups": self.treatment_groups,
            "endpoints": self.endpoints,
            "criteria": [c.to_dict() for c in self.criteria],
            "results": [r.to_dict() for r in self.results],
            "status": self.status.value,
            "ethical_considerations": self.ethical_considerations,
            "iacuc_approval": self.iacuc_approval,
            "humane_endpoints": self.humane_endpoints,
            "estimated_duration_weeks": self.estimated_duration_weeks,
            "estimated_cost_usd": self.estimated_cost_usd,
            "provenance": self.provenance,
        }


@dataclass
class ClinicalTranslation:
    """Clinical translation specification.

    Attributes:
        translation_id: Unique identifier
        phase: Clinical phase
        trial_design: Trial design type
        target_enrollment: Target patient enrollment
        primary_endpoints: Primary endpoints
        secondary_endpoints: Secondary endpoints
        biomarkers: Biomarkers for patient selection/monitoring
        regulatory_pathway: Regulatory pathway considerations
    """

    translation_id: str
    phase: ValidationStage
    trial_design: str = "Open-label, single-arm"
    target_enrollment: int = 30
    primary_endpoints: list[str] = field(default_factory=list)
    secondary_endpoints: list[str] = field(default_factory=list)
    biomarkers: list[dict[str, Any]] = field(default_factory=list)
    inclusion_criteria: list[str] = field(default_factory=list)
    exclusion_criteria: list[str] = field(default_factory=list)
    safety_monitoring: list[str] = field(default_factory=list)
    status: ValidationStatus = ValidationStatus.NOT_STARTED
    regulatory_pathway: str = ""
    irb_approval: str = ""
    data_safety_monitoring_board: bool = True
    estimated_duration_months: int = 24
    estimated_cost_usd: float = 5000000.0
    provenance: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "translation_id": self.translation_id,
            "phase": self.phase.value,
            "trial_design": self.trial_design,
            "target_enrollment": self.target_enrollment,
            "primary_endpoints": self.primary_endpoints,
            "secondary_endpoints": self.secondary_endpoints,
            "biomarkers": self.biomarkers,
            "inclusion_criteria": self.inclusion_criteria,
            "exclusion_criteria": self.exclusion_criteria,
            "safety_monitoring": self.safety_monitoring,
            "status": self.status.value,
            "regulatory_pathway": self.regulatory_pathway,
            "irb_approval": self.irb_approval,
            "data_safety_monitoring_board": self.data_safety_monitoring_board,
            "estimated_duration_months": self.estimated_duration_months,
            "estimated_cost_usd": self.estimated_cost_usd,
            "provenance": self.provenance,
        }


class ValidationPipeline:
    """
    Complete validation pipeline from in-silico to clinical.

    Manages the progression of research hypotheses through
    validation stages with appropriate go/no-go criteria.

    RESEARCH USE ONLY - Not for clinical applications.
    """

    RESEARCH_DISCLAIMER = """
    This validation pipeline is for RESEARCH PLANNING ONLY.
    Actual clinical trials require full regulatory approval.
    Not approved for direct clinical implementation.
    """

    def __init__(
        self,
        hypothesis_id: str,
        name: str,
        seed: int = 42,
    ) -> None:
        """Initialize the validation pipeline.

        Args:
            hypothesis_id: Associated hypothesis ID
            name: Pipeline name
            seed: Random seed for reproducibility
        """
        self.hypothesis_id = hypothesis_id
        self.name = name
        self.seed = seed
        self.rng = np.random.RandomState(seed)

        self.in_silico: list[InSilicoValidation] = []
        self.in_vitro: list[InVitroValidation] = []
        self.in_vivo: list[InVivoValidation] = []
        self.clinical: list[ClinicalTranslation] = []

        self.created_at = datetime.now(timezone.utc).isoformat()
        self.go_no_go_decisions: list[dict[str, Any]] = []

        logger.info(f"Initialized ValidationPipeline: {name}")

    def add_in_silico(self, validation: InSilicoValidation) -> None:
        """Add an in-silico validation step."""
        self.in_silico.append(validation)

    def add_in_vitro(self, validation: InVitroValidation) -> None:
        """Add an in-vitro validation step."""
        self.in_vitro.append(validation)

    def add_in_vivo(self, validation: InVivoValidation) -> None:
        """Add an in-vivo validation step."""
        self.in_vivo.append(validation)

    def add_clinical(self, translation: ClinicalTranslation) -> None:
        """Add a clinical translation step."""
        self.clinical.append(translation)

    def evaluate_stage_completion(self, stage: ValidationStage) -> dict[str, Any]:
        """Evaluate completion status of a validation stage.

        Args:
            stage: Stage to evaluate

        Returns:
            Evaluation summary
        """
        if stage == ValidationStage.IN_SILICO:
            validations = self.in_silico
        elif stage == ValidationStage.IN_VITRO:
            validations = self.in_vitro
        elif stage == ValidationStage.IN_VIVO:
            validations = self.in_vivo
        else:
            validations = [c for c in self.clinical if c.phase == stage]

        total = len(validations)
        completed = sum(
            1
            for v in validations
            if v.status in [ValidationStatus.COMPLETED_SUCCESS, ValidationStatus.COMPLETED_FAILURE]
        )
        successful = sum(1 for v in validations if v.status == ValidationStatus.COMPLETED_SUCCESS)

        # Calculate rates with proper handling of edge cases
        completion_rate = completed / total if total > 0 else 0.0
        # success_rate is percentage of completed validations that were successful
        # Returns 0.0 if no validations completed (not misleading - correctly indicates no data)
        success_rate = successful / completed if completed > 0 else 0.0

        return {
            "stage": stage.value,
            "total_validations": total,
            "completed": completed,
            "successful": successful,
            "completion_rate": completion_rate,
            "success_rate": success_rate,
            "can_proceed": total > 0 and successful >= total * 0.7,  # 70% threshold
        }

    def generate_go_no_go_recommendation(
        self, from_stage: ValidationStage, to_stage: ValidationStage
    ) -> dict[str, Any]:
        """Generate go/no-go recommendation for stage transition.

        Args:
            from_stage: Current stage
            to_stage: Target stage

        Returns:
            Recommendation with rationale
        """
        current_eval = self.evaluate_stage_completion(from_stage)

        recommendation = {
            "from_stage": from_stage.value,
            "to_stage": to_stage.value,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "current_stage_evaluation": current_eval,
            "recommendation": "GO" if current_eval["can_proceed"] else "NO-GO",
            "rationale": [],
            "conditions": [],
            "risk_factors": [],
        }

        if current_eval["can_proceed"]:
            recommendation["rationale"].append(
                f"{from_stage.value} validation criteria met "
                f"({current_eval['success_rate'] * 100:.0f}% success rate)"
            )
            recommendation["conditions"].append(
                f"Ensure {to_stage.value} prerequisites are in place"
            )
        else:
            recommendation["rationale"].append(
                f"Insufficient success in {from_stage.value} "
                f"({current_eval['success_rate'] * 100:.0f}% success rate)"
            )
            recommendation["conditions"].append(
                "Address failed validation criteria before proceeding"
            )
            recommendation["risk_factors"].append(
                "Proceeding despite failures may lead to costly late-stage failures"
            )

        # Record decision
        self.go_no_go_decisions.append(recommendation)

        return recommendation

    def estimate_total_timeline(self) -> dict[str, Any]:
        """Estimate total timeline for validation pipeline.

        Returns:
            Timeline estimate with breakdown by stage
        """
        in_silico_weeks = sum(v.estimated_compute_hours / 40 for v in self.in_silico)
        in_vitro_weeks = sum(v.estimated_duration_weeks for v in self.in_vitro)
        in_vivo_weeks = sum(v.estimated_duration_weeks for v in self.in_vivo)
        clinical_months = sum(c.estimated_duration_months for c in self.clinical)

        total_months = (
            in_silico_weeks / 4 + in_vitro_weeks / 4 + in_vivo_weeks / 4 + clinical_months
        )

        return {
            "in_silico_weeks": in_silico_weeks,
            "in_vitro_weeks": in_vitro_weeks,
            "in_vivo_weeks": in_vivo_weeks,
            "clinical_months": clinical_months,
            "total_months": total_months,
            "total_years": total_months / 12,
        }

    def estimate_total_cost(self) -> dict[str, Any]:
        """Estimate total cost for validation pipeline.

        Returns:
            Cost estimate with breakdown by stage
        """
        in_silico_cost = len(self.in_silico) * 10000  # Compute costs
        in_vitro_cost = sum(v.estimated_cost_usd for v in self.in_vitro)
        in_vivo_cost = sum(v.estimated_cost_usd for v in self.in_vivo)
        clinical_cost = sum(c.estimated_cost_usd for c in self.clinical)

        total = in_silico_cost + in_vitro_cost + in_vivo_cost + clinical_cost

        return {
            "in_silico_usd": in_silico_cost,
            "in_vitro_usd": in_vitro_cost,
            "in_vivo_usd": in_vivo_cost,
            "clinical_usd": clinical_cost,
            "total_usd": total,
        }

    def compute_pipeline_hash(self) -> str:
        """Compute unique hash of pipeline contents."""
        pipeline_dict = {
            "hypothesis_id": self.hypothesis_id,
            "name": self.name,
            "in_silico": [v.to_dict() for v in self.in_silico],
            "in_vitro": [v.to_dict() for v in self.in_vitro],
            "in_vivo": [v.to_dict() for v in self.in_vivo],
            "clinical": [c.to_dict() for c in self.clinical],
        }
        pipeline_json = json.dumps(pipeline_dict, sort_keys=True)
        return hashlib.sha256(pipeline_json.encode()).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        """Serialize pipeline to dictionary."""
        return {
            "hypothesis_id": self.hypothesis_id,
            "name": self.name,
            "seed": self.seed,
            "created_at": self.created_at,
            "in_silico": [v.to_dict() for v in self.in_silico],
            "in_vitro": [v.to_dict() for v in self.in_vitro],
            "in_vivo": [v.to_dict() for v in self.in_vivo],
            "clinical": [c.to_dict() for c in self.clinical],
            "go_no_go_decisions": self.go_no_go_decisions,
            "timeline_estimate": self.estimate_total_timeline(),
            "cost_estimate": self.estimate_total_cost(),
            "hash": self.compute_pipeline_hash(),
            "disclaimer": self.RESEARCH_DISCLAIMER,
        }


def create_example_validation_pipeline(hypothesis_id: str) -> ValidationPipeline:
    """Create an example validation pipeline.

    Args:
        hypothesis_id: Associated hypothesis ID

    Returns:
        ValidationPipeline with example steps
    """
    pipeline = ValidationPipeline(
        hypothesis_id=hypothesis_id,
        name=f"Validation_{hypothesis_id}",
        seed=42,
    )

    # In-silico validation
    in_silico = InSilicoValidation(
        validation_id="VS001",
        model_type=ModelSystem.AGENT_BASED_MODEL,
        description=("Agent-based model of tumor growth with adaptive therapy scheduling"),
        simulation_parameters={
            "initial_tumor_cells": 1000000,
            "resistant_fraction": 0.01,
            "growth_rate": 0.1,
            "drug_kill_rate": 0.5,
            "adaptive_threshold": 0.5,
        },
        synthetic_population_size=10000,
        criteria=[
            ValidationCriteria(
                criterion_id="C001",
                description="Time to progression improvement",
                metric="time_to_progression_ratio",
                threshold=1.5,
                direction="greater",
            ),
            ValidationCriteria(
                criterion_id="C002",
                description="Resistant clone suppression",
                metric="resistant_fraction_ratio",
                threshold=0.5,
                direction="less",
            ),
        ],
        estimated_compute_hours=100,
    )
    pipeline.add_in_silico(in_silico)

    # In-vitro validation
    in_vitro = InVitroValidation(
        validation_id="VT001",
        model_system=ModelSystem.PDO,
        cell_lines=["PC9", "HCC827", "H1975"],
        conditions=[
            {"drug": "Osimertinib", "concentration_nM": [10, 100, 1000]},
            {"schedule": "continuous"},
            {"schedule": "adaptive", "threshold": 0.5},
        ],
        assays=["viability", "apoptosis", "resistance_marker_expression"],
        criteria=[
            ValidationCriteria(
                criterion_id="C003",
                description="Adaptive shows improved long-term control",
                metric="day30_viability_ratio",
                threshold=0.7,
                direction="less",
            ),
        ],
        estimated_duration_weeks=12,
        estimated_cost_usd=75000,
    )
    pipeline.add_in_vitro(in_vitro)

    # In-vivo validation
    in_vivo = InVivoValidation(
        validation_id="VV001",
        model_system=ModelSystem.PDX,
        species="Mus musculus",
        sample_size=40,
        treatment_groups=[
            {"name": "Vehicle", "n": 10},
            {"name": "Continuous", "n": 15},
            {"name": "Adaptive", "n": 15},
        ],
        endpoints=[
            "Tumor volume",
            "Progression-free survival",
            "Overall survival",
            "Resistant clone frequency",
        ],
        criteria=[
            ValidationCriteria(
                criterion_id="C004",
                description="PFS improvement with adaptive therapy",
                metric="pfs_hazard_ratio",
                threshold=0.7,
                direction="less",
            ),
        ],
        ethical_considerations=[
            "Minimize animal suffering through humane endpoints",
            "Use minimum necessary sample size",
            "Provide appropriate analgesia",
        ],
        humane_endpoints=[
            "Tumor volume > 2000mm3",
            "Body weight loss > 20%",
            "Severe distress signs",
        ],
        estimated_duration_weeks=20,
        estimated_cost_usd=200000,
    )
    pipeline.add_in_vivo(in_vivo)

    # Clinical translation (Phase I)
    clinical = ClinicalTranslation(
        translation_id="CT001",
        phase=ValidationStage.CLINICAL_PHASE_I,
        trial_design="Open-label, single-arm, adaptive dosing",
        target_enrollment=30,
        primary_endpoints=[
            "Safety and tolerability of adaptive osimertinib dosing",
            "Recommended adaptive dosing schedule",
        ],
        secondary_endpoints=[
            "Progression-free survival",
            "ctDNA dynamics",
            "Quality of life",
        ],
        biomarkers=[
            {
                "name": "ctDNA",
                "purpose": "Treatment decision guidance",
                "threshold": "50% decrease from baseline",
            },
            {
                "name": "T790M mutation",
                "purpose": "Resistance monitoring",
                "method": "Digital PCR",
            },
        ],
        inclusion_criteria=[
            "EGFR-mutant NSCLC (L858R or exon 19 deletion)",
            "Treatment-naive or prior TKI therapy",
            "ECOG PS 0-1",
            "Adequate organ function",
            "Measurable disease per RECIST 1.1",
        ],
        exclusion_criteria=[
            "Prior osimertinib treatment",
            "Symptomatic CNS metastases",
            "Interstitial lung disease",
            "Significant cardiac disease",
        ],
        safety_monitoring=[
            "Weekly labs for first 8 weeks",
            "Adverse event monitoring per NCI-CTCAE v5.0",
            "QTc monitoring",
        ],
        regulatory_pathway="FDA IND pathway",
        data_safety_monitoring_board=True,
        estimated_duration_months=24,
        estimated_cost_usd=3000000,
    )
    pipeline.add_clinical(clinical)

    return pipeline
