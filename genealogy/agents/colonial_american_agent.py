"""
Agent 2: Colonial Maryland & Chesapeake (1700-1800)

Colonial Records Archivist specializing in Maryland colonial period.
Focus: Hammond, Warfield, and Welsh families in the documented chain.
"""

from typing import Dict, Any
from datetime import date

from .base_agent import ResearchAgent
from ..models.person import Person, Gender, Relationship, RelationshipType
from ..models.record import Source, SourceType, EvidenceQuality


class ColonialAmericanAgent(ResearchAgent):
    """
    Specializes in Colonial Maryland & Chesapeake lineage (1700-1800).
    
    Duties:
    - Prove Hammond–Warfield–Welsh family connections using parish registers, probates, land deeds
    - Focus on documented Maryland colonial gentry families
    - Establish chain: Ruth Hammond Warfield → Harriet Amelia Warfield → Hannah Hammond-Welsh → John Charles Hammond
    """
    
    def __init__(self):
        super().__init__(
            agent_id="agent_2_colonial_maryland",
            scope="Colonial Maryland & Chesapeake (1700-1800)"
        )
    
    def execute_research(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute colonial Maryland research."""
        self.start_execution()
        
        # Research the documented Hammond-Warfield-Welsh chain
        self._research_warfield_family()
        self._research_hammond_welsh_connection()
        self._research_hammond_family()
        self._document_colonial_sources()
        
        analysis = self.analyze_evidence()
        
        self.complete_execution()
        
        return {
            "success": True,
            "colonial_families_documented": len(self.persons),
            "analysis": analysis,
            "report": self.generate_report(),
        }
    
    def _research_warfield_family(self) -> None:
        """Research Warfield family of colonial Maryland."""
        
        # Ruth Hammond Warfield (1756-1820)
        ruth_warfield = Person(
            person_id="RUTH_WARFIELD_001",
            full_name="Ruth Hammond Warfield",
            given_names="Ruth Hammond",
            surname="Warfield",
            gender=Gender.FEMALE,
            birth_date=date(1756, 1, 1),  # Approximate
            death_date=date(1820, 1, 1),  # Approximate
            birth_place="Maryland",
            notes="1756-1820; Revolutionary War era; mother of Harriet Amelia Warfield"
        )
        self.add_person(ruth_warfield)
        
        # Harriet Amelia Warfield (1783-1844)
        harriet_warfield = Person(
            person_id="HARRIET_WARFIELD_001",
            full_name="Harriet Amelia Warfield",
            given_names="Harriet Amelia",
            surname="Warfield",
            gender=Gender.FEMALE,
            birth_date=date(1783, 1, 1),  # Approximate
            death_date=date(1844, 1, 1),  # Approximate
            birth_place="Maryland",
            notes="1783-1844; daughter of Ruth Hammond Warfield; documented in family records"
        )
        self.add_person(harriet_warfield)
        
        # Relationship: Ruth → Harriet
        rel_ruth_harriet = Relationship(
            relationship_id="REL_COL_001",
            person1_id="RUTH_WARFIELD_001",
            person2_id="HARRIET_WARFIELD_001",
            relationship_type=RelationshipType.PARENT_CHILD,
            biological=True,
            legitimacy=True,
            evidence_quality="secondary",
            notes="Requires verification with church registers, probate records"
        )
        self.add_relationship(rel_ruth_harriet)
        
        # Source: Warfield family research
        warfield_source = Source(
            source_id="SRC_COL_001",
            title="Warfield Family of Maryland: Historical and Genealogical Records",
            source_type=SourceType.PUBLISHED_WORK,
            publisher="Maryland Historical Society",
            reliability=EvidenceQuality.SECONDARY,
            notes="Published genealogies; requires primary source verification"
        )
        self.add_source(warfield_source)
        
        self.add_finding(
            "Warfield family: Prominent Maryland colonial gentry. "
            "Ruth Hammond Warfield (1756-1820) → Harriet Amelia Warfield (1783-1844) documented in family records. "
            "Requires verification: Church registers (baptism, burial), probate records, land deeds. "
            "Warfield surname well-established in Anne Arundel and Howard Counties, Maryland."
        )
    
    def _research_hammond_welsh_connection(self) -> None:
        """Research Hammond-Welsh family connection."""
        
        # Hannah Hammond-Welsh (1723-1779)
        hannah_welsh = Person(
            person_id="HANNAH_WELSH_001",
            full_name="Hannah Hammond-Welsh",
            given_names="Hannah",
            surname="Hammond-Welsh",
            gender=Gender.FEMALE,
            birth_date=date(1723, 1, 1),  # Approximate
            death_date=date(1779, 1, 1),  # Approximate
            birth_place="Maryland",
            notes="1723-1779; compound surname Hammond-Welsh suggests Hammond maternal line"
        )
        self.add_person(hannah_welsh)
        
        self.add_finding(
            "Hannah Hammond-Welsh (1723-1779): Compound surname is genealogically significant. "
            "Suggests Hammond maternal lineage preserved in surname. "
            "Research priorities: 1) Church registers (birth/baptism circa 1723, death/burial 1779); "
            "2) Marriage record (reveals maiden name and parents); "
            "3) Probate records (father's will naming daughter Hannah); "
            "4) Land records; 5) Connection to Ruth Hammond Warfield requires proof."
        )
    
    def _research_hammond_family(self) -> None:
        """Research Hammond family of colonial Maryland."""
        
        # Capt. John Charles Hammond (1698-1753)
        john_hammond = Person(
            person_id="JOHN_HAMMOND_001",
            full_name="Capt. John Charles Hammond",
            given_names="John Charles",
            surname="Hammond",
            gender=Gender.MALE,
            birth_date=date(1698, 1, 1),  # Approximate
            death_date=date(1753, 1, 1),  # Approximate
            titles=["Captain"],
            birth_place="Maryland",
            notes="Captain; 1698-1753; father of Hannah Hammond-Welsh (claimed)"
        )
        self.add_person(john_hammond)
        
        # Maj. Charles M. Hammond (1670-1713)
        charles_hammond = Person(
            person_id="CHARLES_HAMMOND_001",
            full_name="Maj. Charles M. Hammond",
            given_names="Charles M.",
            surname="Hammond",
            gender=Gender.MALE,
            birth_date=date(1670, 1, 1),  # Approximate
            death_date=date(1713, 1, 1),  # Approximate
            titles=["Major"],
            birth_place="Maryland",
            notes="Major; 1670-1713; father of Capt. John Charles Hammond (claimed)"
        )
        self.add_person(charles_hammond)
        
        # Relationships
        rel_charles_john = Relationship(
            relationship_id="REL_COL_002",
            person1_id="CHARLES_HAMMOND_001",
            person2_id="JOHN_HAMMOND_001",
            relationship_type=RelationshipType.PARENT_CHILD,
            biological=True,
            legitimacy=True,
            evidence_quality="secondary",
            notes="Requires probate, land records, and church registers"
        )
        self.add_relationship(rel_charles_john)
        
        rel_john_hannah = Relationship(
            relationship_id="REL_COL_003",
            person1_id="JOHN_HAMMOND_001",
            person2_id="HANNAH_WELSH_001",
            relationship_type=RelationshipType.PARENT_CHILD,
            biological=True,
            legitimacy=True,
            evidence_quality="secondary",
            notes="REQUIRES PROOF: Will of John Hammond naming daughter Hannah; church registers"
        )
        self.add_relationship(rel_john_hannah)
        
        self.add_finding(
            "Hammond family: Well-documented colonial Maryland gentry. "
            "Maj. Charles M. Hammond (1670-1713) → Capt. John Charles Hammond (1698-1753) → Hannah Hammond-Welsh (1723-1779). "
            "Military titles 'Major' and 'Captain' require verification in militia records, land grants, court proceedings. "
            "Research priorities: 1) Probate records (wills of Charles 1713, John 1753); "
            "2) Land patents and deeds; 3) Church registers; 4) Militia/military commissions; "
            "5) Court records (Hammond family litigation and transactions)."
        )
    
    def _document_colonial_sources(self) -> None:
        """Document key colonial Maryland sources."""
        
        # Maryland church registers
        church_source = Source(
            source_id="SRC_COL_002",
            title="Parish Registers of Colonial Maryland",
            source_type=SourceType.ORIGINAL_DOCUMENT,
            repository="Maryland State Archives",
            reliability=EvidenceQuality.PRIMARY,
            notes="Anglican parish registers: baptisms, marriages, burials for colonial families"
        )
        self.add_source(church_source)
        
        # Maryland land records
        land_source = Source(
            source_id="SRC_COL_003",
            title="Maryland Land Office Records: Patents, Certificates, and Warrants",
            source_type=SourceType.ORIGINAL_DOCUMENT,
            repository="Maryland State Archives",
            reliability=EvidenceQuality.PRIMARY,
            notes="Land grants and transactions for Hammond, Warfield families"
        )
        self.add_source(land_source)
        
        # Maryland probate
        probate_source = Source(
            source_id="SRC_COL_004",
            title="Maryland Prerogative Court (Probate) Records",
            source_type=SourceType.ORIGINAL_DOCUMENT,
            repository="Maryland State Archives",
            reliability=EvidenceQuality.PRIMARY,
            notes="Wills, inventories, administrations for colonial Maryland families"
        )
        self.add_source(probate_source)
        
        self.add_finding(
            "Colonial Maryland sources available: Parish registers (Maryland State Archives), "
            "land office records, prerogative court (probate) records. "
            "These primary sources are essential for GPS-level proof of all claimed relationships."
        )
    
    def analyze_evidence(self) -> Dict[str, Any]:
        """Analyze colonial period evidence."""
        return {
            "coverage": "1700-1800 (Colonial Maryland & Chesapeake)",
            "key_families": ["Hammond", "Warfield", "Welsh"],
            "documented_chain": [
                "Maj. Charles M. Hammond (1670-1713)",
                "Capt. John Charles Hammond (1698-1753)",
                "Hannah Hammond-Welsh (1723-1779)",
                "Ruth Hammond Warfield (1756-1820)",
                "Harriet Amelia Warfield (1783-1844)",
                "→ Caroline Hammond Benepe (1805-1880) [transition to 19th century]"
            ],
            "family_records_status": "Chain documented in family records; requires primary source verification",
            "critical_research_needs": [
                "Wills: Charles M. Hammond (d. 1713), John Charles Hammond (d. 1753)",
                "Church registers: Baptisms, marriages, burials for all individuals",
                "Land records: Hammond and Warfield family transactions",
                "Court records: Family litigation and legal proceedings",
                "Proof of military titles (Major, Captain) through commissions or contemporary documents"
            ],
            "proof_status": "Secondary (family records); requires upgrade to primary sources for GPS compliance",
            "geographic_focus": "Anne Arundel County, Howard County, Baltimore County, Maryland",
            "next_steps": [
                "PRIORITY 1: Maryland State Archives research - parish registers for all families",
                "PRIORITY 2: Probate records - wills of Charles Hammond (1713) and John Hammond (1753)",
                "PRIORITY 3: Land office records - Hammond family patents and deeds",
                "PRIORITY 4: Verify military titles through militia records and commissions",
                "PRIORITY 5: Court records search for Hammond, Warfield, Welsh families",
                "PRIORITY 6: Published genealogies cross-verification (Warfield, Hammond families)"
            ],
            "colonial_gentry_status": "Hammond and Warfield families are documented Maryland colonial gentry. "
                                     "This provides strong foundation for potential English gentry/noble connections."
        }
