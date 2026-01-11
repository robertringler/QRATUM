"""
Treatment Timeline Builder

Builds treatment timelines from Medicare claims data,
identifying treatment events and lines of therapy.

RESEARCH USE ONLY - Not for clinical diagnosis or treatment decisions.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Optional

import yaml

from .linkages import LinkedPatient
from .privacy import PrivacyConfig, SafeLogger, suppress_small_counts
from .schema import ClaimEvent, CodeSystem, PatientTimeline, TreatmentEvent

logger = logging.getLogger(__name__)


@dataclass
class CodeMapping:
    """Mapping from medical codes to treatment categories.

    Attributes:
        code: Medical code
        code_system: Code system (CPT, HCPCS, NDC, etc.)
        treatment_type: Treatment category (surgery, radiation, chemo, etc.)
        drug_name: Drug name if applicable
        description: Description of the treatment
        confidence: Confidence in mapping (0.0 to 1.0)
    """

    code: str
    code_system: CodeSystem
    treatment_type: str
    drug_name: str = ""
    description: str = ""
    confidence: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "code": self.code,
            "code_system": self.code_system.value,
            "treatment_type": self.treatment_type,
            "drug_name": self.drug_name,
            "description": self.description,
            "confidence": self.confidence,
        }


class CodeMappingLibrary:
    """
    Library of code mappings for treatment identification.

    Mappings are loaded from configurable YAML/JSON files.
    Users can extend mappings for their specific use case.

    IMPORTANT: These mappings are NOT clinically authoritative.
    They are starting points that require validation by domain experts.
    """

    def __init__(self) -> None:
        """Initialize mapping library."""
        self._mappings: dict[str, CodeMapping] = {}
        self._treatment_types: set[str] = set()

    def add_mapping(self, mapping: CodeMapping) -> None:
        """Add a code mapping."""
        key = f"{mapping.code_system.value}:{mapping.code}"
        self._mappings[key] = mapping
        self._treatment_types.add(mapping.treatment_type)

    def get_mapping(self, code: str, code_system: CodeSystem) -> Optional[CodeMapping]:
        """Get mapping for a code."""
        key = f"{code_system.value}:{code}"
        return self._mappings.get(key)

    def lookup_code(
        self,
        code: str,
        code_system: CodeSystem,
    ) -> Optional[tuple[str, str, float]]:
        """Look up a code and return (treatment_type, drug_name, confidence).

        Args:
            code: Medical code
            code_system: Code system

        Returns:
            Tuple of (treatment_type, drug_name, confidence) or None
        """
        mapping = self.get_mapping(code, code_system)
        if mapping:
            return (mapping.treatment_type, mapping.drug_name, mapping.confidence)
        return None

    def load_from_yaml(self, filepath: Path) -> int:
        """Load mappings from YAML file.

        Expected format:
        mappings:
          - code: "96413"
            code_system: "cpt"
            treatment_type: "chemo_admin"
            drug_name: ""
            description: "Chemo admin IV infusion"

        Args:
            filepath: Path to YAML file

        Returns:
            Number of mappings loaded
        """
        filepath = Path(filepath)
        count = 0

        with open(filepath, "r") as f:
            data = yaml.safe_load(f)

        for item in data.get("mappings", []):
            try:
                code_system = CodeSystem(item.get("code_system", "unknown").lower())
                mapping = CodeMapping(
                    code=str(item["code"]),
                    code_system=code_system,
                    treatment_type=item.get("treatment_type", "unknown"),
                    drug_name=item.get("drug_name", ""),
                    description=item.get("description", ""),
                    confidence=float(item.get("confidence", 1.0)),
                )
                self.add_mapping(mapping)
                count += 1
            except Exception:
                continue

        return count

    def load_from_json(self, filepath: Path) -> int:
        """Load mappings from JSON file.

        Args:
            filepath: Path to JSON file

        Returns:
            Number of mappings loaded
        """
        filepath = Path(filepath)
        count = 0

        with open(filepath, "r") as f:
            data = json.load(f)

        for item in data.get("mappings", []):
            try:
                code_system = CodeSystem(item.get("code_system", "unknown").lower())
                mapping = CodeMapping(
                    code=str(item["code"]),
                    code_system=code_system,
                    treatment_type=item.get("treatment_type", "unknown"),
                    drug_name=item.get("drug_name", ""),
                    description=item.get("description", ""),
                    confidence=float(item.get("confidence", 1.0)),
                )
                self.add_mapping(mapping)
                count += 1
            except Exception:
                continue

        return count

    def get_treatment_types(self) -> set[str]:
        """Get all treatment types in library."""
        return self._treatment_types.copy()

    @staticmethod
    def create_default_library() -> "CodeMappingLibrary":
        """Create a default mapping library with common oncology codes.

        NOTE: These are EXAMPLE mappings for demonstration.
        They are NOT clinically validated and require expert review.

        Returns:
            CodeMappingLibrary with default mappings
        """
        library = CodeMappingLibrary()

        # Chemotherapy administration CPT codes (examples only)
        chemo_admin_cpts = [
            ("96413", "Chemo IV push single"),
            ("96415", "Chemo IV infusion add-on"),
            ("96416", "Chemo IV prolong infusion"),
            ("96417", "Chemo IV infusion each add"),
            ("96401", "Chemo subcut/IM non-hormone"),
            ("96402", "Chemo subcut/IM hormone"),
            ("96409", "Chemo IV push single"),
            ("96411", "Chemo IV push add-on"),
        ]
        for code, desc in chemo_admin_cpts:
            library.add_mapping(
                CodeMapping(
                    code=code,
                    code_system=CodeSystem.CPT,
                    treatment_type="chemo_admin",
                    description=desc,
                    confidence=0.9,
                )
            )

        # Radiation therapy CPT codes (examples only)
        radiation_cpts = [
            ("77385", "IMRT simple"),
            ("77386", "IMRT complex"),
            ("77401", "Radiation treatment delivery superficial"),
            ("77402", "Radiation treatment delivery MV"),
            ("77407", "Radiation treatment intermediate"),
            ("77412", "Radiation treatment complex"),
            ("77427", "Radiation treatment management"),
        ]
        for code, desc in radiation_cpts:
            library.add_mapping(
                CodeMapping(
                    code=code,
                    code_system=CodeSystem.CPT,
                    treatment_type="radiation",
                    description=desc,
                    confidence=0.9,
                )
            )

        # Surgery CPT codes (examples only - lung cancer)
        surgery_cpts = [
            ("32480", "Lobectomy"),
            ("32482", "Bilobectomy"),
            ("32440", "Pneumonectomy total"),
            ("32663", "VATS lobectomy"),
            ("32666", "VATS wedge resection"),
        ]
        for code, desc in surgery_cpts:
            library.add_mapping(
                CodeMapping(
                    code=code,
                    code_system=CodeSystem.CPT,
                    treatment_type="surgery",
                    description=desc,
                    confidence=0.9,
                )
            )

        # HCPCS J-codes for oncology drugs (examples only)
        j_codes = [
            ("J9271", "pembrolizumab", "Pembrolizumab 1mg"),
            ("J9305", "pemetrexed", "Pemetrexed 10mg"),
            ("J9228", "ipilimumab", "Ipilimumab 1mg"),
            ("J9299", "nivolumab", "Nivolumab 1mg"),
            ("J9035", "bevacizumab", "Bevacizumab 10mg"),
            ("J9045", "carboplatin", "Carboplatin 50mg"),
            ("J9267", "paclitaxel", "Paclitaxel 1mg"),
            ("J9060", "cisplatin", "Cisplatin 10mg"),
            ("J9190", "fluorouracil", "Fluorouracil 500mg"),
            ("J9355", "trastuzumab", "Trastuzumab 10mg"),
        ]
        for code, drug, desc in j_codes:
            library.add_mapping(
                CodeMapping(
                    code=code,
                    code_system=CodeSystem.HCPCS,
                    treatment_type="chemo_drug",
                    drug_name=drug,
                    description=desc,
                    confidence=0.85,
                )
            )

        # Immunotherapy HCPCS codes
        io_codes = [
            ("J9271", "pembrolizumab", "Pembrolizumab"),
            ("J9228", "ipilimumab", "Ipilimumab"),
            ("J9299", "nivolumab", "Nivolumab"),
            ("J9173", "durvalumab", "Durvalumab"),
            ("J9023", "atezolizumab", "Atezolizumab"),
        ]
        for code, drug, desc in io_codes:
            library.add_mapping(
                CodeMapping(
                    code=code,
                    code_system=CodeSystem.HCPCS,
                    treatment_type="immunotherapy",
                    drug_name=drug,
                    description=desc,
                    confidence=0.9,
                )
            )

        return library


@dataclass
class TimelineSummary:
    """Summary statistics for treatment timelines.

    Attributes:
        n_patients: Number of patients with timelines
        n_with_any_treatment: Patients with any identified treatment
        time_to_first_treatment_median: Median days to first treatment
        time_to_first_treatment_mean: Mean days to first treatment
        treatment_type_counts: Counts by treatment type
        lines_of_therapy_distribution: Distribution of number of lines
        common_sequences: Most common treatment sequences
    """

    n_patients: int = 0
    n_with_any_treatment: int = 0
    time_to_first_treatment_median: Optional[float] = None
    time_to_first_treatment_mean: Optional[float] = None
    treatment_type_counts: dict[str, int] = field(default_factory=dict)
    lines_of_therapy_distribution: dict[str, int] = field(default_factory=dict)
    common_sequences: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self, min_cell_size: int = 11) -> dict[str, Any]:
        """Serialize with suppression."""
        suppressed = f"<{min_cell_size}"
        n_pts = self.n_patients
        n_treated = self.n_with_any_treatment
        return {
            "n_patients": n_pts if n_pts >= min_cell_size else suppressed,
            "n_with_any_treatment": (
                n_treated if n_treated >= min_cell_size else suppressed
            ),
            "time_to_first_treatment_median": self.time_to_first_treatment_median,
            "time_to_first_treatment_mean": self.time_to_first_treatment_mean,
            "treatment_type_counts": suppress_small_counts(
                self.treatment_type_counts, min_cell_size
            ),
            "lines_of_therapy_distribution": suppress_small_counts(
                self.lines_of_therapy_distribution, min_cell_size
            ),
            "common_sequences": self.common_sequences,  # Already aggregated
        }


class TreatmentTimelineBuilder:
    """
    Builds treatment timelines from claims data.

    Takes claims events and identifies treatment events,
    groups them into episodes, and estimates lines of therapy.

    Usage:
        builder = TreatmentTimelineBuilder(
            code_library=CodeMappingLibrary.create_default_library(),
            lookback_days=365,
            followup_days=1095,
            gap_days_for_new_line=45,
        )

        for patient in cohort:
            timeline = builder.build_timeline(patient, claims)
    """

    def __init__(
        self,
        code_library: Optional[CodeMappingLibrary] = None,
        lookback_days: int = 365,
        followup_days: int = 1095,
        gap_days_for_new_line: int = 45,
        privacy_config: Optional[PrivacyConfig] = None,
    ) -> None:
        """Initialize timeline builder.

        Args:
            code_library: Code mapping library
            lookback_days: Days before index date to include
            followup_days: Days after index date to include
            gap_days_for_new_line: Gap to start new line of therapy
            privacy_config: Privacy configuration
        """
        self.code_library = code_library or CodeMappingLibrary.create_default_library()
        self.lookback_days = lookback_days
        self.followup_days = followup_days
        self.gap_days_for_new_line = gap_days_for_new_line
        self.privacy_config = privacy_config or PrivacyConfig()
        self.safe_logger = SafeLogger("timeline_builder", self.privacy_config)

        self._summary = TimelineSummary()
        self._sequence_counts: dict[str, int] = defaultdict(int)

    def build_timeline(
        self,
        patient: LinkedPatient,
        claims: list[ClaimEvent],
    ) -> PatientTimeline:
        """Build a treatment timeline for a patient.

        Args:
            patient: LinkedPatient with registry case
            claims: List of ClaimEvent objects for this patient

        Returns:
            PatientTimeline with treatment events
        """
        case = patient.registry_case
        if case is None or case.dx_date is None:
            return PatientTimeline(patient_key=patient.patient_key)

        index_date = case.dx_date
        lookback_start = index_date - timedelta(days=self.lookback_days)
        followup_end = index_date + timedelta(days=self.followup_days)

        # Filter claims to window
        window_claims = [
            c for c in claims if lookback_start <= c.event_date <= followup_end
        ]

        # Identify treatment events
        treatment_events = []
        for claim in window_claims:
            event = self._classify_claim(claim, index_date)
            if event:
                treatment_events.append(event)

        # Sort by date
        treatment_events.sort(key=lambda e: e.event_date)

        # Assign lines of therapy
        treatment_events = self._assign_lines_of_therapy(treatment_events)

        # Calculate time to first treatment
        post_index_events = [e for e in treatment_events if e.days_from_index >= 0]
        time_to_first = (
            min(e.days_from_index for e in post_index_events) if post_index_events else None
        )

        # Count lines
        lines = set(e.line_of_therapy for e in treatment_events if e.days_from_index >= 0)
        n_lines = max(lines) if lines else 0

        # Last treatment date
        last_treatment = max(e.event_date for e in treatment_events) if treatment_events else None

        timeline = PatientTimeline(
            patient_key=patient.patient_key,
            registry_case=case,
            treatment_events=treatment_events,
            claim_events=window_claims,
            index_date=index_date,
            lookback_start=lookback_start,
            followup_end=followup_end,
            time_to_first_treatment=time_to_first,
            number_of_lines=n_lines,
            last_treatment_date=last_treatment,
        )

        # Update summary statistics
        self._update_summary(timeline)

        return timeline

    def _classify_claim(
        self,
        claim: ClaimEvent,
        index_date: date,
    ) -> Optional[TreatmentEvent]:
        """Classify a claim as a treatment event.

        Args:
            claim: ClaimEvent to classify
            index_date: Index (diagnosis) date

        Returns:
            TreatmentEvent or None if not a treatment
        """
        lookup = self.code_library.lookup_code(claim.code, claim.code_system)
        if lookup is None:
            return None

        treatment_type, drug_name, confidence = lookup
        days_from_index = (claim.event_date - index_date).days

        return TreatmentEvent(
            patient_key=claim.patient_key,
            event_date=claim.event_date,
            treatment_type=treatment_type,
            treatment_code=claim.code,
            drug_name=drug_name,
            setting=claim.setting,
            days_from_index=days_from_index,
            confidence=confidence,
            source_claims=[claim.claim_id_hash],
        )

    def _assign_lines_of_therapy(
        self,
        events: list[TreatmentEvent],
    ) -> list[TreatmentEvent]:
        """Assign lines of therapy based on treatment gaps.

        Args:
            events: List of treatment events (sorted by date)

        Returns:
            Events with line_of_therapy assigned
        """
        if not events:
            return events

        # Filter to systemic therapy (chemo, immunotherapy, targeted)
        systemic_types = {"chemo_admin", "chemo_drug", "immunotherapy", "targeted"}
        systemic_events = [e for e in events if e.treatment_type in systemic_types]

        if not systemic_events:
            return events

        current_line = 1
        last_systemic_date: Optional[date] = None

        for event in events:
            if event.treatment_type in systemic_types:
                if last_systemic_date:
                    gap = (event.event_date - last_systemic_date).days
                    if gap > self.gap_days_for_new_line:
                        current_line += 1
                event.line_of_therapy = current_line
                last_systemic_date = event.event_date
            else:
                # Non-systemic treatments get line of nearest systemic
                event.line_of_therapy = current_line

        return events

    def _update_summary(self, timeline: PatientTimeline) -> None:
        """Update summary statistics with a timeline."""
        self._summary.n_patients += 1

        if timeline.treatment_events:
            self._summary.n_with_any_treatment += 1

            # Treatment type counts
            for event in timeline.treatment_events:
                if event.days_from_index >= 0:  # Post-index only
                    self._summary.treatment_type_counts[event.treatment_type] = (
                        self._summary.treatment_type_counts.get(event.treatment_type, 0) + 1
                    )

            # Lines of therapy
            lines_str = str(timeline.number_of_lines)
            self._summary.lines_of_therapy_distribution[lines_str] = (
                self._summary.lines_of_therapy_distribution.get(lines_str, 0) + 1
            )

            # Track sequences
            sequence = self._get_treatment_sequence(timeline)
            if sequence:
                self._sequence_counts[sequence] += 1

    def _get_treatment_sequence(self, timeline: PatientTimeline) -> str:
        """Get a simplified treatment sequence string."""
        post_index = [e for e in timeline.treatment_events if e.days_from_index >= 0]
        if not post_index:
            return ""

        # Get unique treatment types in order
        seen = set()
        sequence_parts = []
        for event in post_index:
            if event.treatment_type not in seen:
                seen.add(event.treatment_type)
                sequence_parts.append(event.treatment_type)
                if len(sequence_parts) >= 5:  # Limit sequence length
                    break

        return " -> ".join(sequence_parts)

    def get_summary(self, top_sequences: int = 10) -> TimelineSummary:
        """Get timeline summary statistics.

        Args:
            top_sequences: Number of top sequences to include

        Returns:
            TimelineSummary with aggregated statistics
        """
        # Compute top sequences
        sorted_sequences = sorted(
            self._sequence_counts.items(), key=lambda x: x[1], reverse=True
        )[:top_sequences]

        min_cell = self.privacy_config.min_cell_size
        self._summary.common_sequences = [
            {"sequence": seq, "count": cnt if cnt >= min_cell else f"<{min_cell}"}
            for seq, cnt in sorted_sequences
        ]

        return self._summary

    def get_summary_dict(self) -> dict[str, Any]:
        """Get summary as dictionary with suppression."""
        summary = self.get_summary()
        return summary.to_dict(min_cell_size=self.privacy_config.min_cell_size)
