"""
Data Schema Models for SEER-Medicare Integration

Defines normalized data models for claims and registry data.

RESEARCH USE ONLY - Not for clinical diagnosis or treatment decisions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Any, Optional


class ClaimSetting(Enum):
    """Medicare claim setting types."""

    INPATIENT = "inpatient"
    OUTPATIENT = "outpatient"
    CARRIER = "carrier"
    HHA = "hha"
    HOSPICE = "hospice"
    DME = "dme"
    PDE = "pde"  # Part D prescription drug event
    UNKNOWN = "unknown"


class CodeSystem(Enum):
    """Medical code systems."""

    ICD9_DX = "icd9_dx"
    ICD9_PX = "icd9_px"
    ICD10_DX = "icd10_dx"
    ICD10_PX = "icd10_px"
    CPT = "cpt"
    HCPCS = "hcpcs"
    NDC = "ndc"
    DRG = "drg"
    REVENUE = "revenue"
    UNKNOWN = "unknown"


class VitalStatus(Enum):
    """Patient vital status."""

    ALIVE = "alive"
    DEAD = "dead"
    UNKNOWN = "unknown"


class CauseOfDeath(Enum):
    """Cause of death classification (SEER-specific)."""

    CANCER = "cancer"
    OTHER_CAUSE = "other_cause"
    UNKNOWN = "unknown"


@dataclass
class ClaimEvent:
    """
    Normalized claim event from Medicare data.

    This is the fundamental unit of healthcare utilization data,
    representing a single procedure, diagnosis, or drug exposure.

    Attributes:
        patient_key: Internal hashed key, stable across run (NOT original ID)
        event_date: Date of service/claim
        setting: Claim setting (inpatient, outpatient, carrier, pde, etc.)
        code_system: Code system used (ICD9/ICD10/CPT/HCPCS/NDC/DRG)
        code: The actual code value
        quantity: Quantity/units if applicable
        cost: Allowed amount or payment if available
        provider_type: Provider specialty code if available
        raw_source: Source file/table name for provenance
        line_number: Line number within claim if multi-line
        claim_id_hash: Hashed claim identifier for grouping
        metadata: Additional metadata dictionary
    """

    patient_key: str
    event_date: date
    setting: ClaimSetting
    code_system: CodeSystem
    code: str
    quantity: Optional[int] = None
    cost: Optional[float] = None
    provider_type: Optional[str] = None
    raw_source: str = ""
    line_number: int = 0
    claim_id_hash: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "patient_key": self.patient_key,
            "event_date": self.event_date.isoformat() if self.event_date else None,
            "setting": self.setting.value,
            "code_system": self.code_system.value,
            "code": self.code,
            "quantity": self.quantity,
            "cost": self.cost,
            "provider_type": self.provider_type,
            "raw_source": self.raw_source,
            "line_number": self.line_number,
            "claim_id_hash": self.claim_id_hash,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ClaimEvent:
        """Deserialize from dictionary."""
        event_date = data.get("event_date")
        if isinstance(event_date, str):
            event_date = date.fromisoformat(event_date)

        return cls(
            patient_key=data["patient_key"],
            event_date=event_date,
            setting=ClaimSetting(data.get("setting", "unknown")),
            code_system=CodeSystem(data.get("code_system", "unknown")),
            code=data["code"],
            quantity=data.get("quantity"),
            cost=data.get("cost"),
            provider_type=data.get("provider_type"),
            raw_source=data.get("raw_source", ""),
            line_number=data.get("line_number", 0),
            claim_id_hash=data.get("claim_id_hash", ""),
            metadata=data.get("metadata", {}),
        )


@dataclass
class RegistryCase:
    """
    Normalized SEER registry case record.

    Represents a cancer diagnosis from SEER with key clinical attributes.

    Attributes:
        patient_key: Internal hashed key, stable across run
        dx_date: Date of cancer diagnosis (index date)
        cancer_site: Primary site code/description
        histology: Histology type code/description
        stage: Stage at diagnosis (AJCC or summary)
        grade: Tumor grade if available
        laterality: Laterality if applicable
        age_at_dx: Age at diagnosis in years
        sex: Patient sex
        race: Race/ethnicity code
        vital_status: Alive/Dead/Unknown
        death_date: Date of death if applicable
        cause_of_death: Cancer vs other cause
        last_followup_date: Last known contact date
        survival_months: Survival time from SEER
        sequence_number: SEER sequence number (primary=0)
        reporting_source: Source of case report
        raw_source: Source file for provenance
        metadata: Additional metadata
    """

    patient_key: str
    dx_date: date
    cancer_site: str
    histology: str = ""
    stage: str = ""
    grade: str = ""
    laterality: str = ""
    age_at_dx: Optional[int] = None
    sex: str = ""
    race: str = ""
    vital_status: VitalStatus = VitalStatus.UNKNOWN
    death_date: Optional[date] = None
    cause_of_death: CauseOfDeath = CauseOfDeath.UNKNOWN
    last_followup_date: Optional[date] = None
    survival_months: Optional[int] = None
    sequence_number: int = 0
    reporting_source: str = ""
    raw_source: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "patient_key": self.patient_key,
            "dx_date": self.dx_date.isoformat() if self.dx_date else None,
            "cancer_site": self.cancer_site,
            "histology": self.histology,
            "stage": self.stage,
            "grade": self.grade,
            "laterality": self.laterality,
            "age_at_dx": self.age_at_dx,
            "sex": self.sex,
            "race": self.race,
            "vital_status": self.vital_status.value,
            "death_date": self.death_date.isoformat() if self.death_date else None,
            "cause_of_death": self.cause_of_death.value,
            "last_followup_date": (
                self.last_followup_date.isoformat() if self.last_followup_date else None
            ),
            "survival_months": self.survival_months,
            "sequence_number": self.sequence_number,
            "reporting_source": self.reporting_source,
            "raw_source": self.raw_source,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RegistryCase:
        """Deserialize from dictionary."""
        dx_date = data.get("dx_date")
        if isinstance(dx_date, str):
            dx_date = date.fromisoformat(dx_date)

        death_date = data.get("death_date")
        if isinstance(death_date, str):
            death_date = date.fromisoformat(death_date)

        last_followup = data.get("last_followup_date")
        if isinstance(last_followup, str):
            last_followup = date.fromisoformat(last_followup)

        return cls(
            patient_key=data["patient_key"],
            dx_date=dx_date,
            cancer_site=data.get("cancer_site", ""),
            histology=data.get("histology", ""),
            stage=data.get("stage", ""),
            grade=data.get("grade", ""),
            laterality=data.get("laterality", ""),
            age_at_dx=data.get("age_at_dx"),
            sex=data.get("sex", ""),
            race=data.get("race", ""),
            vital_status=VitalStatus(data.get("vital_status", "unknown")),
            death_date=death_date,
            cause_of_death=CauseOfDeath(data.get("cause_of_death", "unknown")),
            last_followup_date=last_followup,
            survival_months=data.get("survival_months"),
            sequence_number=data.get("sequence_number", 0),
            reporting_source=data.get("reporting_source", ""),
            raw_source=data.get("raw_source", ""),
            metadata=data.get("metadata", {}),
        )


@dataclass
class TreatmentEvent:
    """
    Normalized treatment event derived from claims.

    Represents a specific treatment (surgery, radiation, chemo, etc.)
    identified from claims data using code mappings.

    Attributes:
        patient_key: Internal hashed key
        event_date: Date of treatment
        treatment_type: Type (surgery, radiation, chemo, immunotherapy, etc.)
        treatment_code: Specific treatment code
        drug_name: Drug name if applicable
        setting: Care setting
        days_from_index: Days from index (diagnosis) date
        line_of_therapy: Estimated line of therapy number
        episode_id: Grouping identifier for treatment episodes
        source_claims: List of source ClaimEvent references
        confidence: Confidence score for treatment classification
        metadata: Additional metadata
    """

    patient_key: str
    event_date: date
    treatment_type: str
    treatment_code: str = ""
    drug_name: str = ""
    setting: ClaimSetting = ClaimSetting.UNKNOWN
    days_from_index: int = 0
    line_of_therapy: int = 1
    episode_id: str = ""
    source_claims: list[str] = field(default_factory=list)
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "patient_key": self.patient_key,
            "event_date": self.event_date.isoformat() if self.event_date else None,
            "treatment_type": self.treatment_type,
            "treatment_code": self.treatment_code,
            "drug_name": self.drug_name,
            "setting": self.setting.value,
            "days_from_index": self.days_from_index,
            "line_of_therapy": self.line_of_therapy,
            "episode_id": self.episode_id,
            "source_claims": self.source_claims,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }


@dataclass
class PatientTimeline:
    """
    Complete treatment timeline for a patient.

    Aggregates registry information and treatment events.

    Attributes:
        patient_key: Internal hashed key
        registry_case: Associated RegistryCase
        treatment_events: List of TreatmentEvent objects
        claim_events: List of raw ClaimEvent objects (if retained)
        index_date: Index date (usually dx_date)
        lookback_start: Start of lookback window
        followup_end: End of followup window
        time_to_first_treatment: Days from index to first treatment
        number_of_lines: Estimated number of therapy lines
        last_treatment_date: Date of last observed treatment
        metadata: Additional metadata
    """

    patient_key: str
    registry_case: Optional[RegistryCase] = None
    treatment_events: list[TreatmentEvent] = field(default_factory=list)
    claim_events: list[ClaimEvent] = field(default_factory=list)
    index_date: Optional[date] = None
    lookback_start: Optional[date] = None
    followup_end: Optional[date] = None
    time_to_first_treatment: Optional[int] = None
    number_of_lines: int = 0
    last_treatment_date: Optional[date] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary (excludes raw claim_events for privacy)."""
        return {
            "patient_key": self.patient_key,
            "registry_case": self.registry_case.to_dict() if self.registry_case else None,
            "treatment_events": [e.to_dict() for e in self.treatment_events],
            "index_date": self.index_date.isoformat() if self.index_date else None,
            "lookback_start": self.lookback_start.isoformat() if self.lookback_start else None,
            "followup_end": self.followup_end.isoformat() if self.followup_end else None,
            "time_to_first_treatment": self.time_to_first_treatment,
            "number_of_lines": self.number_of_lines,
            "last_treatment_date": (
                self.last_treatment_date.isoformat() if self.last_treatment_date else None
            ),
            "n_claim_events": len(self.claim_events),  # Count only, not content
            "metadata": self.metadata,
        }


@dataclass
class CohortDefinition:
    """
    Cohort definition parameters.

    Defines inclusion/exclusion criteria for cohort selection.
    """

    name: str
    cancer_site: Optional[str] = None
    cancer_site_patterns: list[str] = field(default_factory=list)
    histology: Optional[str] = None
    histology_patterns: list[str] = field(default_factory=list)
    stage: Optional[str] = None
    stage_patterns: list[str] = field(default_factory=list)
    diagnosis_year_min: Optional[int] = None
    diagnosis_year_max: Optional[int] = None
    age_min: Optional[int] = None
    age_max: Optional[int] = None
    sex: Optional[str] = None
    require_continuous_enrollment: bool = False
    enrollment_months_pre: int = 0
    enrollment_months_post: int = 0
    exclude_prior_cancer: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "name": self.name,
            "cancer_site": self.cancer_site,
            "cancer_site_patterns": self.cancer_site_patterns,
            "histology": self.histology,
            "histology_patterns": self.histology_patterns,
            "stage": self.stage,
            "stage_patterns": self.stage_patterns,
            "diagnosis_year_min": self.diagnosis_year_min,
            "diagnosis_year_max": self.diagnosis_year_max,
            "age_min": self.age_min,
            "age_max": self.age_max,
            "sex": self.sex,
            "require_continuous_enrollment": self.require_continuous_enrollment,
            "enrollment_months_pre": self.enrollment_months_pre,
            "enrollment_months_post": self.enrollment_months_post,
            "exclude_prior_cancer": self.exclude_prior_cancer,
            "metadata": self.metadata,
        }
