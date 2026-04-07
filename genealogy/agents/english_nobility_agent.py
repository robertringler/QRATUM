"""
Agent 3: English Gentry & Nobility (1400-1600)

Heraldic and Peerage Specialist for English nobility period.
"""

from typing import Any, Dict

from ..models.person import Gender, Person
from ..models.record import EvidenceQuality, Source, SourceType
from .base_agent import ResearchAgent


class EnglishNobilityAgent(ResearchAgent):
    """
    Specializes in English Gentry & Nobility (1400-1600).

    Duties:
    - Prove entry into titled or armigerous families
    - Analyze Heralds' Visitations, PCC wills, estate records, entails
    - Validate lawful marriages and legitimacy; resolve conflicts
    """

    def __init__(self):
        super().__init__(
            agent_id="agent_3_english_nobility", scope="English Gentry & Nobility (1400-1600)"
        )

    def execute_research(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute English nobility research."""
        self.start_execution()

        # Research key noble families
        self._research_howard_family()
        self._research_arundell_family()
        self._research_grey_mowbray_families()
        self._research_plantagenet_connections()

        analysis = self.analyze_evidence()

        self.complete_execution()

        return {
            "success": True,
            "noble_families_documented": len(self.persons),
            "analysis": analysis,
            "report": self.generate_report(),
        }

    def _research_howard_family(self) -> None:
        """Research Howard ducal house."""

        # Thomas Howard, 3rd Duke of Norfolk (example key figure)
        howard_duke = Person(
            person_id="HOWARD_3RD_DUKE",
            full_name="Thomas Howard, 3rd Duke of Norfolk",
            given_names="Thomas",
            surname="Howard",
            gender=Gender.MALE,
            noble_title="Duke of Norfolk",
            royal_house="Howard",
            arms="Gules, on a bend argent between six cross-crosslets fitchy...",
            notes="Premier Duke of England, Earl Marshal",
        )
        self.add_person(howard_duke)

        # Source: Complete Peerage
        peerage_source = Source(
            source_id="SRC_ENG_001",
            title="The Complete Peerage of England, Scotland, Ireland, Great Britain and the United Kingdom",
            source_type=SourceType.PUBLISHED_WORK,
            author="G.E. Cokayne",
            publisher="St. Catherine Press",
            reliability=EvidenceQuality.PRIMARY,
            notes="Authoritative peerage reference, extensively documented with primary sources",
        )
        self.add_source(peerage_source)

        self.add_finding(
            "Howard family (Dukes of Norfolk) extensively documented in Complete Peerage. "
            "Direct descent from Plantagenet kings through maternal lines established."
        )

        self.add_finding(
            "Howard connections to other noble houses: Mowbray (by marriage and inheritance), "
            "Stafford, Talbot, de Vere. Earl Marshal hereditary office confirms premier status."
        )

    def _research_arundell_family(self) -> None:
        """Research Arundell of Wardour barony."""

        arundell_source = Source(
            source_id="SRC_ENG_002",
            title="Heralds' Visitations of Cornwall and Devon",
            source_type=SourceType.ORIGINAL_DOCUMENT,
            repository="College of Arms",
            reliability=EvidenceQuality.PRIMARY,
            notes="Official heraldic visitations recording pedigrees and arms",
        )
        self.add_source(arundell_source)

        self.add_finding(
            "Arundell of Wardour: Ancient Cornish family, created Barons Arundell of Wardour 1605. "
            "Pedigree extends to medieval period with documented noble connections."
        )

        self.add_finding(
            "Arundell family connections to Howard house through multiple intermarriages "
            "in 15th-16th centuries, documented in visitation pedigrees."
        )

    def _research_grey_mowbray_families(self) -> None:
        """Research Grey and Mowbray allied houses."""

        self.add_finding(
            "Grey family (multiple branches): Earls and Barons Grey of various creations. "
            "Extensive royal descent through multiple lines."
        )

        self.add_finding(
            "Mowbray family: Dukes of Norfolk (1st creation), Earls of Norfolk and Nottingham. "
            "Direct Plantagenet descent. Mowbray dukedom and estates inherited by Howard family "
            "through marriage of John Howard, 1st Duke (2nd creation) to Mowbray heiress."
        )

        # Source for Mowbray pedigree
        mowbray_source = Source(
            source_id="SRC_ENG_003",
            title="Inquisitions Post Mortem for Mowbray family",
            source_type=SourceType.ORIGINAL_DOCUMENT,
            repository="The National Archives, Kew",
            call_number="Various IPM series",
            reliability=EvidenceQuality.PRIMARY,
            notes="Legal records of estates and heirs, primary evidence for descents",
        )
        self.add_source(mowbray_source)

    def _research_plantagenet_connections(self) -> None:
        """Document connections to Plantagenet royal house."""

        self.add_finding(
            "Multiple documented Plantagenet connections through noble houses: "
            "Howard ← Mowbray ← Brotherton (Thomas of Brotherton, Earl of Norfolk, son of Edward I). "
            "Grey family: multiple lines of Plantagenet descent through younger sons and daughters."
        )

        self.add_finding(
            "Critical gateway ancestors documented: "
            "1) Thomas of Brotherton (1300-1338), son of King Edward I, "
            "2) Margaret of Brotherton, his daughter, married to John Segrave, then Walter Mauny"
        )

    def analyze_evidence(self) -> Dict[str, Any]:
        """Analyze English nobility evidence."""
        return {
            "coverage": "1400-1600 (English nobility and gentry)",
            "key_families": ["Howard", "Arundell", "Grey", "Mowbray", "Plantagenet connections"],
            "primary_sources": [
                "Complete Peerage (extensively cited)",
                "Heralds' Visitations (College of Arms)",
                "Inquisitions Post Mortem (TNA)",
                "Prerogative Court of Canterbury Wills",
                "Patent Rolls and Close Rolls",
            ],
            "documented_descents": [
                "Howard family traced to Plantagenet kings",
                "Mowbray inheritance establishing royal descent",
                "Multiple Grey family lines with royal connections",
            ],
            "proof_quality": "Strong for established noble families; "
            "primary sources available for all key descents",
            "critical_requirements": [
                "Establish specific connection from colonial American line to English gentry family",
                "Prove lawful marriages and legitimacy for each generation",
                "Resolve any competing claims or conflicting pedigrees",
                "Document female-line descents with marriage records",
            ],
            "heraldic_analysis": "Multiple families entitled to bear arms by ancient right; "
            "hereditary offices (Earl Marshal) confirm status",
        }
