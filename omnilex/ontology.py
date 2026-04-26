"""Legal domain ontology for QRATUM-OMNILEX.

This module defines the fundamental legal concepts, traditions, domains,
and reasoning frameworks used throughout the OMNILEX system.

Version: 1.0.0
Status: Production
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class LegalTradition(Enum):
    """Major legal traditions recognized by OMNILEX."""

    COMMON_LAW = "common_law"
    CIVIL_LAW = "civil_law"
    RELIGIOUS_ISLAMIC = "religious_islamic"
    RELIGIOUS_JEWISH = "religious_jewish"
    RELIGIOUS_CANON = "religious_canon"
    SOCIALIST = "socialist"
    CUSTOMARY = "customary"
    MIXED = "mixed"
    INTERNATIONAL = "international"
    SUPRANATIONAL = "supranational"


class LegalDomain(Enum):
    """Legal domains supported by OMNILEX."""

    CONSTITUTIONAL = "constitutional"
    CRIMINAL = "criminal"
    CONTRACT = "contract"
    TORT = "tort"
    PROPERTY = "property"
    CORPORATE = "corporate"
    SECURITIES = "securities"
    BANKRUPTCY = "bankruptcy"
    TAX = "tax"
    IP = "ip"
    LABOR = "labor"
    ENVIRONMENTAL = "environmental"
    FAMILY = "family"
    IMMIGRATION = "immigration"
    HUMAN_RIGHTS = "human_rights"
    SPACE_LAW = "space_law"
    CYBER_LAW = "cyber_law"
    AI_REGULATION = "ai_regulation"


class ReasoningFramework(Enum):
    """Legal reasoning frameworks supported by OMNILEX."""

    IRAC = "irac"  # Issue, Rule, Application, Conclusion
    CRAC = "crac"  # Conclusion, Rule, Application, Conclusion
    CREAC = "creac"  # Conclusion, Rule, Explanation, Application, Conclusion
    FIRAC = "firac"  # Facts, Issue, Rule, Application, Conclusion
    TREACC = "treacc"  # Thesis, Rule, Explanation, Application, Counterargument, Conclusion
    STATUTORY_INTERPRETATION = "statutory_interpretation"


class InterpretiveCanon(Enum):
    """Interpretive canons used in legal analysis."""

    PLAIN_MEANING = "plain_meaning"
    EJUSDEM_GENERIS = "ejusdem_generis"
    EXPRESSIO_UNIUS = "expressio_unius"
    NOSCITUR_A_SOCIIS = "noscitur_a_sociis"
    RULE_OF_LENITY = "rule_of_lenity"
    CHEVRON_DEFERENCE = "chevron_deference"
    AUER_DEFERENCE = "auer_deference"
    CONSTITUTIONAL_AVOIDANCE = "constitutional_avoidance"
    ABSURDITY_DOCTRINE = "absurdity_doctrine"
    WHOLE_ACT_RULE = "whole_act_rule"


@dataclass(frozen=True)
class Jurisdiction:
    """Represents a legal jurisdiction.

    Attributes:
        code: ISO 3166-1 + subdivision code (e.g., 'US-CA' for California)
        name: Human-readable jurisdiction name
        tradition: Primary legal tradition of the jurisdiction
        court_hierarchy: Tuple of court names from lowest to highest
    """

    code: str
    name: str
    tradition: LegalTradition
    court_hierarchy: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate jurisdiction data."""
        if not self.code:
            raise ValueError("Jurisdiction code cannot be empty")
        if not self.name:
            raise ValueError("Jurisdiction name cannot be empty")
        if not self.court_hierarchy:
            raise ValueError("Court hierarchy cannot be empty")


# Common jurisdictions
US_FEDERAL = Jurisdiction(
    code="US",
    name="United States (Federal)",
    tradition=LegalTradition.COMMON_LAW,
    court_hierarchy=("District Court", "Circuit Court of Appeals", "Supreme Court"),
)

US_CALIFORNIA = Jurisdiction(
    code="US-CA",
    name="California",
    tradition=LegalTradition.COMMON_LAW,
    court_hierarchy=("Superior Court", "Court of Appeal", "Supreme Court of California"),
)

US_NEW_YORK = Jurisdiction(
    code="US-NY",
    name="New York",
    tradition=LegalTradition.COMMON_LAW,
    court_hierarchy=("Supreme Court", "Appellate Division", "Court of Appeals"),
)

UK = Jurisdiction(
    code="GB",
    name="United Kingdom",
    tradition=LegalTradition.COMMON_LAW,
    court_hierarchy=("County Court", "High Court", "Court of Appeal", "Supreme Court"),
)

GERMANY = Jurisdiction(
    code="DE",
    name="Germany",
    tradition=LegalTradition.CIVIL_LAW,
    court_hierarchy=("Amtsgericht", "Landgericht", "Oberlandesgericht", "Bundesgerichtshof"),
)

# Registry of common jurisdictions
JURISDICTION_REGISTRY = {
    "US": US_FEDERAL,
    "US-CA": US_CALIFORNIA,
    "US-NY": US_NEW_YORK,
    "GB": UK,
    "DE": GERMANY,
}
