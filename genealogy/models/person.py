"""
Data models for genealogical persons and relationships.
"""

from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import List, Optional


class Gender(Enum):
    """Gender classification."""

    MALE = "male"
    FEMALE = "female"
    UNKNOWN = "unknown"


class RelationshipType(Enum):
    """Types of genealogical relationships."""

    PARENT_CHILD = "parent_child"
    MARRIAGE = "marriage"
    SIBLING = "sibling"
    UNKNOWN = "unknown"


@dataclass
class Person:
    """
    Represents an individual in a genealogical study.
    """

    person_id: str
    full_name: str
    given_names: Optional[str] = None
    surname: Optional[str] = None
    maiden_name: Optional[str] = None

    # Vital dates and places
    birth_date: Optional[date] = None
    birth_place: Optional[str] = None
    death_date: Optional[date] = None
    death_place: Optional[str] = None
    baptism_date: Optional[date] = None
    burial_date: Optional[date] = None
    burial_place: Optional[str] = None

    # Additional information
    gender: Gender = Gender.UNKNOWN
    titles: List[str] = field(default_factory=list)
    occupations: List[str] = field(default_factory=list)
    residences: List[str] = field(default_factory=list)

    # Heraldic information
    arms: Optional[str] = None  # Description of coat of arms
    heraldic_entitlement: Optional[str] = None

    # Noble/Royal status
    noble_title: Optional[str] = None
    royal_house: Optional[str] = None

    # Additional notes
    notes: str = ""

    def __str__(self) -> str:
        dates = ""
        if self.birth_date:
            dates = f"b. {self.birth_date.year}"
        if self.death_date:
            dates += f" - d. {self.death_date.year}" if dates else f"d. {self.death_date.year}"

        title_str = f", {self.noble_title}" if self.noble_title else ""
        return f"{self.full_name}{title_str} ({dates})" if dates else f"{self.full_name}{title_str}"


@dataclass
class Relationship:
    """
    Represents a relationship between two persons.
    """

    relationship_id: str
    person1_id: str
    person2_id: str
    relationship_type: RelationshipType

    # Marriage-specific
    marriage_date: Optional[date] = None
    marriage_place: Optional[str] = None
    divorce_date: Optional[date] = None

    # Parent-child specific
    # For PARENT_CHILD: person1 is parent, person2 is child
    biological: bool = True
    legitimacy: bool = True

    # Evidence
    evidence_quality: str = "primary"  # primary, secondary, tertiary
    notes: str = ""

    def __str__(self) -> str:
        return f"{self.person1_id} --[{self.relationship_type.value}]--> {self.person2_id}"
