"""
Data models for genealogical records, sources, and citations.
"""

from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Any, Dict, List, Optional


class RecordType(Enum):
    """Types of genealogical records."""

    BIRTH = "birth"
    BAPTISM = "baptism"
    MARRIAGE = "marriage"
    DEATH = "death"
    BURIAL = "burial"
    CENSUS = "census"
    WILL = "will"
    PROBATE = "probate"
    LAND_DEED = "land_deed"
    COURT_RECORD = "court_record"
    CHURCH_REGISTER = "church_register"
    HERALDS_VISITATION = "heralds_visitation"
    PEERAGE = "peerage"
    ESTATE_RECORD = "estate_record"
    ENTAIL = "entail"
    CHARTER = "charter"
    CHRONICLE = "chronicle"
    OTHER = "other"


class SourceType(Enum):
    """Types of source materials."""

    ORIGINAL_DOCUMENT = "original"
    MICROFILM = "microfilm"
    DIGITAL_IMAGE = "digital_image"
    TRANSCRIPTION = "transcription"
    ABSTRACT = "abstract"
    COMPILATION = "compilation"
    PUBLISHED_WORK = "published_work"
    ONLINE_DATABASE = "online_database"


class EvidenceQuality(Enum):
    """Quality/reliability of evidence."""

    PRIMARY = "primary"  # Original document, contemporary record
    SECONDARY = "secondary"  # Non-original but reliable
    TERTIARY = "tertiary"  # Compiled from other sources
    QUESTIONABLE = "questionable"  # Reliability uncertain


@dataclass
class Source:
    """
    Represents a source document or repository.
    """

    source_id: str
    title: str
    source_type: SourceType

    # Publication/Repository information
    author: Optional[str] = None
    publisher: Optional[str] = None
    publication_date: Optional[date] = None
    publication_place: Optional[str] = None

    # Repository information
    repository: Optional[str] = None
    repository_location: Optional[str] = None
    call_number: Optional[str] = None

    # Digital/Online information
    url: Optional[str] = None
    database: Optional[str] = None
    accessed_date: Optional[date] = None

    # Quality assessment
    reliability: EvidenceQuality = EvidenceQuality.SECONDARY

    notes: str = ""

    def formatted_citation(self) -> str:
        """Generate a formatted citation string."""
        parts = []

        if self.author:
            parts.append(f"{self.author},")

        parts.append(f'"{self.title}"')

        if self.publisher:
            parts.append(f"({self.publisher}")
            if self.publication_date:
                parts[-1] += f", {self.publication_date.year}"
            parts[-1] += ")"

        if self.repository:
            parts.append(f"; {self.repository}")
            if self.call_number:
                parts[-1] += f", {self.call_number}"

        if self.url:
            parts.append(f"; accessed at {self.url}")
            if self.accessed_date:
                parts[-1] += f" on {self.accessed_date}"

        return " ".join(parts)


@dataclass
class Citation:
    """
    Represents a specific citation to a source for a particular fact.
    """

    citation_id: str
    source_id: str

    # Specific location within source
    page: Optional[str] = None
    volume: Optional[str] = None
    line: Optional[str] = None
    entry: Optional[str] = None

    # Image/document specifics
    image_number: Optional[int] = None
    folio: Optional[str] = None

    # What this citation proves
    fact_description: str = ""

    # Direct transcription or abstract
    transcription: str = ""
    abstract: str = ""

    # Quality of this specific citation
    evidence_quality: EvidenceQuality = EvidenceQuality.SECONDARY

    notes: str = ""

    def formatted_citation(self, source: Source) -> str:
        """Generate formatted citation with source information."""
        citation_str = source.formatted_citation()

        location_parts = []
        if self.volume:
            location_parts.append(f"vol. {self.volume}")
        if self.page:
            location_parts.append(f"p. {self.page}")
        if self.line:
            location_parts.append(f"line {self.line}")
        if self.entry:
            location_parts.append(f"entry {self.entry}")
        if self.folio:
            location_parts.append(f"folio {self.folio}")
        if self.image_number:
            location_parts.append(f"image {self.image_number}")

        if location_parts:
            citation_str += ", " + ", ".join(location_parts)

        return citation_str


@dataclass
class GenealogicalRecord:
    """
    Represents a genealogical record with evidence and citations.
    """

    record_id: str
    record_type: RecordType

    # Persons involved
    person_ids: List[str] = field(default_factory=list)
    primary_person_id: Optional[str] = None

    # Event details
    event_date: Optional[date] = None
    event_place: Optional[str] = None

    # Citations supporting this record
    citations: List[str] = field(default_factory=list)  # citation_ids

    # Extracted information
    extracted_facts: Dict[str, Any] = field(default_factory=dict)

    # Proof analysis
    proof_strength: str = "strong"  # strong, moderate, weak, conflicting
    conflicts: List[str] = field(default_factory=list)

    notes: str = ""

    def __str__(self) -> str:
        event_str = f"{self.record_type.value}"
        if self.event_date:
            event_str += f" on {self.event_date}"
        if self.event_place:
            event_str += f" at {self.event_place}"
        return event_str
