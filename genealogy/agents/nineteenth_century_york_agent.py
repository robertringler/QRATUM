"""
Agent 1C: 19th Century York Line (1800-1900)

Specialist for proving the 19th century York family chain.
"""

from typing import Dict, Any

from .base_agent import ResearchAgent
from ..models.person import Person, Relationship, Gender, RelationshipType
from ..models.record import Source, SourceType, EvidenceQuality


class NineteenthCenturyYorkAgent(ResearchAgent):
    """
    Specializes in 19th century York family documentation (1800-1900).
    
    Duties:
    - Prove York family chain: James "Jim" → William Robert → Thomas Ralph → Edward
    - Document connection to Revolutionary era (John Edgar, Thomas, Semore York)
    - Distinguish from unrelated York families
    """
    
    def __init__(self):
        super().__init__(
            agent_id="agent_1c_nineteenth_century_york",
            scope="19th Century York Family (1800-1900)"
        )
    
    def execute_research(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute 19th century York research."""
        self.start_execution()
        
        # Document the 19th century York chain
        self._document_james_york()
        self._document_william_robert_york()
        self._document_thomas_ralph_york()
        self._document_edward_york()
        
        analysis = self.analyze_evidence()
        
        self.complete_execution()
        
        return {
            "success": True,
            "generations_documented": 4,
            "analysis": analysis,
            "report": self.generate_report(),
        }
    
    def _document_james_york(self) -> None:
        """Document James 'Jim' York (1866-1952)."""
        
        james_york = Person(
            person_id="JAMES_YORK_19TH_001",
            full_name='James "Jim" York',
            given_names="James",
            surname="York",
            gender=Gender.MALE,
            birth_date=None,  # Circa 1866
            death_date=None,  # Circa 1952
            birth_place="United States",
            notes="Circa 1866-1952; father of Jessie York; requires vital records"
        )
        self.add_person(james_york)
        
        census_source = Source(
            source_id="SRC_YORK_19TH_001",
            title="U.S. Federal Census Records",
            source_type=SourceType.ORIGINAL_DOCUMENT,
            repository="National Archives",
            reliability=EvidenceQuality.PRIMARY,
            notes="Census years 1870, 1880, 1900, 1910, 1920, 1930, 1940 required for verification"
        )
        self.add_source(census_source)
        
        self.add_finding(
            "James 'Jim' York (circa 1866-1952): Documented in family records. "
            "Requires verification: Birth certificate/record, census enumeration 1870-1940, "
            "death certificate 1952, marriage record."
        )
    
    def _document_william_robert_york(self) -> None:
        """Document William Robert York (1846-1906)."""
        
        william_york = Person(
            person_id="WILLIAM_YORK_001",
            full_name="William Robert York",
            given_names="William Robert",
            surname="York",
            gender=Gender.MALE,
            birth_date=None,  # Circa 1846
            death_date=None,  # Circa 1906
            birth_place="United States",
            notes="Circa 1846-1906; father of James York; Civil War era"
        )
        self.add_person(william_york)
        
        rel_william_james = Relationship(
            relationship_id="REL_YORK_19TH_001",
            person1_id="WILLIAM_YORK_001",
            person2_id="JAMES_YORK_19TH_001",
            relationship_type=RelationshipType.PARENT_CHILD,
            biological=True,
            legitimacy=True,
            evidence_quality="secondary",
            notes="Requires census, vital records, probate"
        )
        self.add_relationship(rel_william_james)
        
        civil_war_source = Source(
            source_id="SRC_YORK_19TH_002",
            title="Civil War Service Records and Pension Files",
            source_type=SourceType.ORIGINAL_DOCUMENT,
            repository="National Archives",
            reliability=EvidenceQuality.PRIMARY,
            notes="Search for William Robert York in Union or Confederate service"
        )
        self.add_source(civil_war_source)
        
        self.add_finding(
            "William Robert York (circa 1846-1906): Born mid-19th century, lived through Civil War. "
            "Research priorities: 1) Census 1850-1900; 2) Civil War service records (if applicable); "
            "3) Death certificate 1906; 4) Marriage record; 5) Will/probate."
        )
    
    def _document_thomas_ralph_york(self) -> None:
        """Document Thomas Ralph York (1825-1906)."""
        
        thomas_york = Person(
            person_id="THOMAS_RALPH_YORK_001",
            full_name="Thomas Ralph York",
            given_names="Thomas Ralph",
            surname="York",
            gender=Gender.MALE,
            birth_date=None,  # Circa 1825
            death_date=None,  # Circa 1906
            birth_place="United States",
            notes="Circa 1825-1906; father of William Robert York"
        )
        self.add_person(thomas_york)
        
        rel_thomas_william = Relationship(
            relationship_id="REL_YORK_19TH_002",
            person1_id="THOMAS_RALPH_YORK_001",
            person2_id="WILLIAM_YORK_001",
            relationship_type=RelationshipType.PARENT_CHILD,
            biological=True,
            legitimacy=True,
            evidence_quality="secondary",
            notes="Requires census, vital records, probate"
        )
        self.add_relationship(rel_thomas_william)
        
        self.add_finding(
            "Thomas Ralph York (circa 1825-1906): Bridge generation between early 19th century and Civil War era. "
            "Research priorities: 1) Census 1830-1900; 2) Death record 1906; "
            "3) Marriage record; 4) Will/probate; 5) Land records."
        )
    
    def _document_edward_york(self) -> None:
        """Document Edward York (1788-1862) - transition to Revolutionary era."""
        
        edward_york = Person(
            person_id="EDWARD_YORK_001",
            full_name="Edward York",
            given_names="Edward",
            surname="York",
            gender=Gender.MALE,
            birth_date=None,  # Circa 1788
            death_date=None,  # Circa 1862
            birth_place="United States",
            notes="Circa 1788-1862; father of Thomas Ralph York; born post-Revolutionary War"
        )
        self.add_person(edward_york)
        
        rel_edward_thomas = Relationship(
            relationship_id="REL_YORK_19TH_003",
            person1_id="EDWARD_YORK_001",
            person2_id="THOMAS_RALPH_YORK_001",
            relationship_type=RelationshipType.PARENT_CHILD,
            biological=True,
            legitimacy=True,
            evidence_quality="secondary",
            notes="CRITICAL LINK to Revolutionary era; requires will, land records, church registers"
        )
        self.add_relationship(rel_edward_thomas)
        
        self.add_finding(
            "Edward York (circa 1788-1862): CRITICAL TRANSITIONAL GENERATION. "
            "Born post-Revolution (1788), son of John Edgar York (1764-1810) who was son of Thomas York (1745-1812), "
            "who was son of Semore York (1724-1783, Loyalist). "
            "Research priorities: 1) Will/probate (circa 1862); 2) Census 1800-1860; "
            "3) Land records; 4) Church registers; 5) Connection proof to John Edgar York (Revolutionary era)."
        )
    
    def analyze_evidence(self) -> Dict[str, Any]:
        """Analyze 19th century York evidence."""
        return {
            "coverage": "1800-1900 (19th century York family)",
            "persons_documented": [
                "James 'Jim' York (circa 1866-1952)",
                "William Robert York (circa 1846-1906)",
                "Thomas Ralph York (circa 1825-1906)",
                "Edward York (circa 1788-1862)"
            ],
            "critical_findings": [
                "Four-generation 19th century chain documented in family records",
                "Edward York (1788-1862) is BRIDGE to Revolutionary era",
                "York surname consistent; no variants documented"
            ],
            "evidence_gaps": [
                "Birth/death certificates for all individuals",
                "Census verification for complete coverage 1800-1940",
                "Marriage records for all generations",
                "Wills/probate records (especially Edward York circa 1862)",
                "Land records showing family continuity"
            ],
            "proof_status": "Documented in family records; requires GPS-level primary source verification",
            "next_steps": [
                "PRIORITY 1: Census search 1800-1940 for all York generations",
                "PRIORITY 2: Death certificates: James (1952), William (1906), Thomas (1906), Edward (1862)",
                "PRIORITY 3: Will of Edward York (circa 1862) - proves connection to Revolutionary era",
                "PRIORITY 4: Marriage records for all generations",
                "PRIORITY 5: Land records showing York family presence and continuity"
            ],
            "colonial_connection": "Edward York (b. 1788) connects to Revolutionary era through father John Edgar York (1764-1810), "
                                  "grandfather Thomas York (1745-1812), and great-grandfather Semore York (1724-1783, Loyalist). "
                                  "This provides pathway to colonial period via Loyalist research (Agent 2)."
        }
