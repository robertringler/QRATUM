"""
Feature Engineering for QRATUM Integration

Engineers features from claims-based timelines for use with
VITRA causal graphs and XENON intervention search.

RESEARCH USE ONLY - Not for clinical diagnosis or treatment decisions.

IMPORTANT LIMITATIONS:
All features derived from claims data are PROXIES, not direct measurements.
- Tumor burden proxy: Based on utilization patterns, NOT actual tumor measurements
- Immune engagement proxy: Based on treatment codes, NOT immune function
- Toxicity proxy: Based on healthcare utilization, NOT direct toxicity assessment
These proxies require validation and should be interpreted with caution.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Optional

from .privacy import PrivacyConfig, SafeLogger
from .schema import ClaimEvent, ClaimSetting, CodeSystem, PatientTimeline

logger = logging.getLogger(__name__)


@dataclass
class BaselineFeatures:
    """Baseline features at index date.

    All features are derived from available data and represent
    PROXIES rather than direct clinical measurements.

    Attributes:
        patient_key: Hashed patient key
        age_bin: Age at diagnosis bin
        sex: Sex code
        race: Race code
        stage: Disease stage
        histology: Histology type
        comorbidity_score: Charlson-like comorbidity proxy (based on dx codes)
        prior_utilization_score: Pre-index healthcare utilization
        prior_inpatient_count: Inpatient admits in lookback
        prior_outpatient_count: Outpatient visits in lookback
        prior_ed_count: ED visits in lookback
    """

    patient_key: str
    age_bin: str = ""
    sex: str = ""
    race: str = ""
    stage: str = ""
    histology: str = ""
    comorbidity_score: float = 0.0
    prior_utilization_score: float = 0.0
    prior_inpatient_count: int = 0
    prior_outpatient_count: int = 0
    prior_ed_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "patient_key": self.patient_key,
            "age_bin": self.age_bin,
            "sex": self.sex,
            "race": self.race,
            "stage": self.stage,
            "histology": self.histology,
            "comorbidity_score": self.comorbidity_score,
            "prior_utilization_score": self.prior_utilization_score,
            "prior_inpatient_count": self.prior_inpatient_count,
            "prior_outpatient_count": self.prior_outpatient_count,
            "prior_ed_count": self.prior_ed_count,
            "metadata": self.metadata,
        }

    def to_vector(self) -> list[float]:
        """Convert to feature vector for modeling."""
        return [
            self.comorbidity_score,
            self.prior_utilization_score,
            float(self.prior_inpatient_count),
            float(self.prior_outpatient_count),
            float(self.prior_ed_count),
        ]


@dataclass
class StateFeatures:
    """Dynamic state features at a point in time.

    These are PROXIES derived from claims patterns:
    - tumor_burden_proxy: NOT actual tumor size
    - immune_engagement_proxy: NOT immune function measurement
    - toxicity_proxy: NOT direct toxicity assessment

    Attributes:
        patient_key: Hashed patient key
        assessment_date: Date of assessment
        days_from_index: Days from diagnosis
        tumor_burden_proxy: Normalized utilization-based proxy (0-1)
        immune_engagement_proxy: Immunotherapy exposure proxy (0-1)
        toxicity_proxy: Healthcare burden proxy (0-1)
        treatment_intensity: Treatment density in recent window
        line_of_therapy: Current line of therapy
        days_since_last_treatment: Days since last treatment
        cumulative_treatment_count: Total treatment events to date
    """

    patient_key: str
    assessment_date: date
    days_from_index: int = 0
    tumor_burden_proxy: float = 0.5
    immune_engagement_proxy: float = 0.0
    toxicity_proxy: float = 0.0
    treatment_intensity: float = 0.0
    line_of_therapy: int = 1
    days_since_last_treatment: Optional[int] = None
    cumulative_treatment_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "patient_key": self.patient_key,
            "assessment_date": self.assessment_date.isoformat() if self.assessment_date else None,
            "days_from_index": self.days_from_index,
            "tumor_burden_proxy": self.tumor_burden_proxy,
            "immune_engagement_proxy": self.immune_engagement_proxy,
            "toxicity_proxy": self.toxicity_proxy,
            "treatment_intensity": self.treatment_intensity,
            "line_of_therapy": self.line_of_therapy,
            "days_since_last_treatment": self.days_since_last_treatment,
            "cumulative_treatment_count": self.cumulative_treatment_count,
            "metadata": self.metadata,
        }

    def to_vector(self) -> list[float]:
        """Convert to feature vector for modeling."""
        return [
            self.tumor_burden_proxy,
            self.immune_engagement_proxy,
            self.toxicity_proxy,
            self.treatment_intensity,
            float(self.line_of_therapy),
            float(self.days_since_last_treatment or 0),
            float(self.cumulative_treatment_count),
        ]


# ICD diagnosis codes associated with comorbidities (Charlson-like proxy)
#
# IMPORTANT:
# - These are EXAMPLE mappings only and REQUIRE validation by domain experts
# - Codes mix ICD-9 (numeric, e.g., "410") and ICD-10 (alphanumeric, e.g., "I21")
# - The ICD-9 to ICD-10 transition occurred Oct 1, 2015 for Medicare
# - Claims before Oct 2015 use ICD-9; claims after use ICD-10
# - These mappings are adapted from Charlson Comorbidity Index literature
# - Users should validate mappings for their specific study population and era
#
# Format: condition_name: [ICD-9 codes..., ICD-10 codes...]
COMORBIDITY_CODE_GROUPS = {
    # ICD-9: 410  |  ICD-10: I21, I22
    "mi": ["410", "I21", "I22"],  # Myocardial infarction
    # ICD-9: 428  |  ICD-10: I50
    "chf": ["428", "I50"],  # Congestive heart failure
    # ICD-9: 443, etc  |  ICD-10: I70, I73
    "pvd": ["443", "I73", "I70"],  # Peripheral vascular disease
    # ICD-9: 430-434  |  ICD-10: I60-I63
    "cvd": ["430", "431", "432", "433", "434", "I60", "I61", "I62", "I63"],
    # ICD-9: 290  |  ICD-10: F00-F03
    "dementia": ["290", "F00", "F01", "F02", "F03"],  # Dementia
    # ICD-9: 490-496  |  ICD-10: J40-J44
    "copd": ["490", "491", "492", "493", "494", "495", "496", "J40", "J41", "J42", "J43", "J44"],
    # ICD-9: 710, 714, 725  |  ICD-10: M05, M06, M32
    "rheumatic": ["710", "714", "725", "M05", "M06", "M32"],  # Rheumatic disease
    # ICD-9: 531-534  |  ICD-10: K25-K28
    "peptic_ulcer": ["531", "532", "533", "534", "K25", "K26", "K27", "K28"],
    # ICD-9: 571  |  ICD-10: K70, K73, K74
    "liver_mild": ["571", "K70", "K73", "K74"],  # Mild liver disease
    # ICD-9: 250  |  ICD-10: E10, E11, E13
    "diabetes": ["250", "E10", "E11", "E13"],  # Diabetes
    # ICD-9: 582, 583, 585, 586  |  ICD-10: N18, N19
    "renal": ["582", "583", "585", "586", "N18", "N19"],  # Renal disease
}


class FeatureEngineer:
    """
    Engineers features from timelines for QRATUM integration.

    Extracts baseline and state features from claims-based timelines.

    IMPORTANT: All derived features are PROXIES based on healthcare
    utilization patterns. They require validation and expert interpretation.
    """

    PROXY_DISCLAIMER = """
    All features are PROXIES derived from claims data:
    - tumor_burden_proxy: Based on utilization, NOT tumor measurements
    - immune_engagement_proxy: Based on treatment codes, NOT immune function
    - toxicity_proxy: Based on healthcare burden, NOT direct toxicity
    These proxies require clinical validation before any interpretation.
    """

    def __init__(
        self,
        privacy_config: Optional[PrivacyConfig] = None,
    ) -> None:
        """Initialize feature engineer."""
        self.privacy_config = privacy_config or PrivacyConfig()
        self.safe_logger = SafeLogger("feature_engineer", self.privacy_config)

    def extract_baseline_features(
        self,
        timeline: PatientTimeline,
    ) -> BaselineFeatures:
        """Extract baseline features at index date.

        Args:
            timeline: PatientTimeline with claims data

        Returns:
            BaselineFeatures at index
        """
        case = timeline.registry_case
        if case is None:
            return BaselineFeatures(patient_key=timeline.patient_key)

        # Age bin
        age_bin = self._get_age_bin(case.age_at_dx) if case.age_at_dx else "Unknown"

        # Comorbidity score from pre-index diagnoses
        pre_index_claims = [
            c for c in timeline.claim_events if c.event_date < (timeline.index_date or date.min)
        ]
        comorbidity_score = self._compute_comorbidity_score(pre_index_claims)

        # Prior utilization
        prior_ip = sum(1 for c in pre_index_claims if c.setting == ClaimSetting.INPATIENT)
        prior_op = sum(1 for c in pre_index_claims if c.setting == ClaimSetting.OUTPATIENT)
        prior_utilization = self._normalize_utilization(prior_ip, prior_op)

        return BaselineFeatures(
            patient_key=timeline.patient_key,
            age_bin=age_bin,
            sex=case.sex,
            race=case.race,
            stage=case.stage,
            histology=case.histology,
            comorbidity_score=comorbidity_score,
            prior_utilization_score=prior_utilization,
            prior_inpatient_count=prior_ip,
            prior_outpatient_count=prior_op,
            prior_ed_count=0,  # Requires ED identification
            metadata={"proxy_disclaimer": self.PROXY_DISCLAIMER},
        )

    def extract_state_features(
        self,
        timeline: PatientTimeline,
        assessment_date: date,
        window_days: int = 30,
    ) -> StateFeatures:
        """Extract state features at a specific date.

        Args:
            timeline: PatientTimeline with claims data
            assessment_date: Date to assess state
            window_days: Window for recent activity

        Returns:
            StateFeatures at assessment date
        """
        if timeline.index_date is None:
            return StateFeatures(
                patient_key=timeline.patient_key,
                assessment_date=assessment_date,
            )

        days_from_index = (assessment_date - timeline.index_date).days
        window_start = assessment_date - timedelta(days=window_days)

        # Claims and events up to assessment date
        claims_to_date = [c for c in timeline.claim_events if c.event_date <= assessment_date]
        events_to_date = [e for e in timeline.treatment_events if e.event_date <= assessment_date]
        recent_claims = [c for c in claims_to_date if c.event_date >= window_start]
        recent_events = [e for e in events_to_date if e.event_date >= window_start]

        # Tumor burden proxy (utilization-based)
        tumor_burden_proxy = self._compute_tumor_burden_proxy(recent_claims)

        # Immune engagement proxy
        immune_proxy = self._compute_immune_engagement_proxy(events_to_date)

        # Toxicity proxy (ED visits, inpatient)
        toxicity_proxy = self._compute_toxicity_proxy(recent_claims)

        # Treatment intensity
        treatment_intensity = len(recent_events) / max(window_days, 1)

        # Current line of therapy
        post_index_events = [e for e in events_to_date if e.days_from_index >= 0]
        current_line = max((e.line_of_therapy for e in post_index_events), default=1)

        # Days since last treatment
        last_treatment = max((e.event_date for e in events_to_date), default=None)
        days_since_last = (assessment_date - last_treatment).days if last_treatment else None

        return StateFeatures(
            patient_key=timeline.patient_key,
            assessment_date=assessment_date,
            days_from_index=days_from_index,
            tumor_burden_proxy=tumor_burden_proxy,
            immune_engagement_proxy=immune_proxy,
            toxicity_proxy=toxicity_proxy,
            treatment_intensity=treatment_intensity,
            line_of_therapy=current_line,
            days_since_last_treatment=days_since_last,
            cumulative_treatment_count=len(post_index_events),
            metadata={"proxy_disclaimer": self.PROXY_DISCLAIMER, "window_days": window_days},
        )

    def _get_age_bin(self, age: int, bin_size: int = 5) -> str:
        """Get age bin label."""
        if age < 65:
            return "<65"
        elif age >= 85:
            return "85+"
        else:
            bin_start = (age // bin_size) * bin_size
            bin_end = bin_start + bin_size - 1
            return f"{bin_start}-{bin_end}"

    def _compute_comorbidity_score(self, claims: list[ClaimEvent]) -> float:
        """Compute Charlson-like comorbidity score from diagnosis codes.

        Args:
            claims: List of claims with diagnosis codes

        Returns:
            Normalized comorbidity score (0-1)
        """
        # Get unique diagnosis codes
        dx_codes = set()
        for claim in claims:
            if claim.code_system in (CodeSystem.ICD9_DX, CodeSystem.ICD10_DX):
                dx_codes.add(claim.code)

        # Count matching comorbidity groups
        matched_groups = 0
        for group_name, prefixes in COMORBIDITY_CODE_GROUPS.items():
            for dx in dx_codes:
                if any(dx.startswith(p) for p in prefixes):
                    matched_groups += 1
                    break

        # Normalize (max groups = 11)
        return min(matched_groups / 11.0, 1.0)

    def _normalize_utilization(
        self,
        inpatient_count: int,
        outpatient_count: int,
    ) -> float:
        """Normalize utilization to 0-1 scale."""
        # Simple normalization based on expected ranges
        ip_score = min(inpatient_count / 5.0, 1.0) * 0.6
        op_score = min(outpatient_count / 20.0, 1.0) * 0.4
        return ip_score + op_score

    def _compute_tumor_burden_proxy(self, claims: list[ClaimEvent]) -> float:
        """Compute tumor burden proxy from recent utilization.

        This is NOT a measurement of actual tumor burden.
        Higher utilization may correlate with disease progression
        but should not be interpreted as tumor size.

        Methodology:
        - Combines inpatient and total claims into a utilization score
        - Inpatient admissions weighted 3x (reflect higher acuity)
        - Score normalized to 0-1 scale using empirical scaling factor (50)
        - These weights are HEURISTIC and require validation

        Scaling assumptions (require validation):
        - IP_WEIGHT=3: Inpatient admits reflect higher disease burden
        - SCALE_FACTOR=50: Assumes ~15 IP + ~5 total in high-burden month
        - These should be calibrated against clinical outcomes

        Args:
            claims: Recent claims

        Returns:
            Normalized proxy score (0-1)
        """
        if not claims:
            return 0.5  # Unknown/baseline

        # Configurable weights (consider making class attributes)
        IP_WEIGHT = 3  # Inpatient admissions weighted higher
        SCALE_FACTOR = 50.0  # Empirical normalization factor

        # Count different claim types
        ip_count = sum(1 for c in claims if c.setting == ClaimSetting.INPATIENT)
        total_count = len(claims)

        # High inpatient utilization suggests higher burden
        score = min((ip_count * IP_WEIGHT + total_count) / SCALE_FACTOR, 1.0)
        return score

    def _compute_immune_engagement_proxy(self, events: list) -> float:
        """Compute immune engagement proxy from treatment history.

        Based on immunotherapy exposure, NOT immune function.

        Args:
            events: Treatment events to date

        Returns:
            Normalized proxy score (0-1)
        """
        io_events = sum(1 for e in events if e.treatment_type == "immunotherapy")
        if io_events == 0:
            return 0.0
        # Scale by number of IO treatments
        return min(io_events / 10.0, 1.0)

    def _compute_toxicity_proxy(self, claims: list[ClaimEvent]) -> float:
        """Compute toxicity proxy from healthcare burden.

        Based on ED visits and inpatient admissions, NOT direct toxicity.

        Args:
            claims: Recent claims

        Returns:
            Normalized proxy score (0-1)
        """
        if not claims:
            return 0.0

        # Count high-acuity encounters
        ip_count = sum(1 for c in claims if c.setting == ClaimSetting.INPATIENT)
        # Could add ED identification if available

        return min(ip_count / 3.0, 1.0)

    def create_qratum_state_dict(
        self,
        baseline: BaselineFeatures,
        state: StateFeatures,
    ) -> dict[str, float]:
        """Create state dictionary for QRATUM CausalOncologyGraph.

        Args:
            baseline: Baseline features
            state: State features

        Returns:
            Dictionary compatible with CausalOncologyGraph
        """
        return {
            "tumor_burden": state.tumor_burden_proxy,
            "immune_engagement": state.immune_engagement_proxy,
            "toxicity_level": state.toxicity_proxy,
            "treatment_intensity": state.treatment_intensity,
            "comorbidity_level": baseline.comorbidity_score,
            "proliferation_rate": state.tumor_burden_proxy * 0.8,  # Simplified proxy
            "EGFR_activity": 0.5,  # Placeholder - requires molecular data
            "KRAS_activity": 0.5,  # Placeholder - requires molecular data
        }
