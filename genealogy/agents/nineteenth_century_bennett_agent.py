"""
Agent 1B: 19th Century Bennett Line (1800-1900)

Specialist for proving the 19th century Bennett family chain in Maryland.
"""

from typing import Dict, Any
from datetime import date

from .base_agent import ResearchAgent
from ..models.person import Person, Relationship, Gender, RelationshipType
from ..models.record import (
    GenealogicalRecord,
    Source,
    Citation,
    RecordType,
    SourceType,
    EvidenceQuality,
)


class NineteenthCenturyBennettAgent(ResearchAgent):
    """
    Specializes in 19th century Bennett family documentation (1800-1900).
    
    Duties:
    - Prove Bennett family chain: John E. → Seth E. → Caroline (Benepe)
    - Document name variant Bennett/Benepe
    - Establish connection to Maryland colonial families
    """
    
    def __init__(self):
        super().__init__(
            agent_id="agent_1b_nineteenth_century_bennett",
            scope="19th Century Bennett Family (1800-1900)"
        )
    
    def execute_research(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute 19th century Bennett research."""
        self.start_execution()
        
        # Document the 19th century Bennett chain
        self._document_john_e_bennett()
        self._document_seth_e_bennett()
        self._document_caroline_benepe_bennett()
        self._analyze_bennett_benepe_variant()
        
        analysis = self.analyze_evidence()
        
        self.complete_execution()
        
        return {
            "success": True,
            "generations_documented": 3,
            "analysis": analysis,
            "report": self.generate_report(),
        }
    
    def _document_john_e_bennett(self) -> None:
        """Document John E. Bennett (1876-1964)."""
        
        john_bennett = Person(
            person_id="JOHN_E_BENNETT_001",
            full_name="John E. Bennett",
            given_names="John E.",
            surname="Bennett",
            gender=Gender.MALE,
            birth_date=date(1876, 1, 1),  # Approximate
            death_date=date(1964, 1, 1),  # Approximate
            birth_place="Maryland (requires verification)",
            notes="1876-1964; father of Ralph E. Bennett; requires birth/death certificate"
        )
        self.add_person(john_bennett)
        
        # Source: Census records
        census_source = Source(
            source_id="SRC_19TH_001",
            title="U.S. Federal Census Records, Maryland",
            source_type=SourceType.ORIGINAL_DOCUMENT,
            repository="National Archives",
            reliability=EvidenceQuality.PRIMARY,
            notes="Census years 1880, 1900, 1910, 1920, 1930, 1940 required for verification"
        )
        self.add_source(census_source)
        
        self.add_finding(
            "John E. Bennett (1876-1964): Documented in family records. "
            "Requires verification: Birth certificate (Maryland vital records), "
            "census enumeration 1880-1940, death certificate 1964, "
            "marriage record to wife (name needed)."
        )
    
    def _document_seth_e_bennett(self) -> None:
        """Document Seth E. Bennett (1842-1931)."""
        
        seth_bennett = Person(
            person_id="SETH_E_BENNETT_001",
            full_name="Seth E. Bennett",
            given_names="Seth E.",
            surname="Bennett",
            gender=Gender.MALE,
            birth_date=date(1842, 1, 1),  # Approximate
            death_date=date(1931, 1, 1),  # Approximate
            birth_place="Maryland",
            notes="1842-1931; father of John E. Bennett; Civil War era"
        )
        self.add_person(seth_bennett)
        
        # Relationship: Seth → John
        rel_seth_john = Relationship(
            relationship_id="REL_19TH_001",
            person1_id="SETH_E_BENNETT_001",
            person2_id="JOHN_E_BENNETT_001",
            relationship_type=RelationshipType.PARENT_CHILD,
            biological=True,
            legitimacy=True,
            evidence_quality="secondary",
            notes="Documented in family records; requires census and vital record proof"
        )
        self.add_relationship(rel_seth_john)
        
        # Source: Civil War records
        civil_war_source = Source(
            source_id="SRC_19TH_002",
            title="Civil War Service Records and Pension Files",
            source_type=SourceType.ORIGINAL_DOCUMENT,
            repository="National Archives",
            reliability=EvidenceQuality.PRIMARY,
            notes="Search for Maryland regiments, Seth Bennett"
        )
        self.add_source(civil_war_source)
        
        self.add_finding(
            "Seth E. Bennett (1842-1931): Born during antebellum period, lived through Civil War. "
            "Research priorities: 1) Census 1850, 1860, 1870, 1880, 1900, 1910, 1920, 1930; "
            "2) Civil War service records (Maryland units); 3) Death certificate 1931; "
            "4) Marriage record to Caroline Hammond Benepe."
        )
    
    def _document_caroline_benepe_bennett(self) -> None:
        """Document Caroline Hammond Benepe (Bennett) (1805-1880)."""
        
        caroline = Person(
            person_id="CAROLINE_BENEPE_001",
            full_name="Caroline Hammond Benepe (Bennett)",
            given_names="Caroline Hammond",
            surname="Benepe",
            maiden_name="Unknown",
            gender=Gender.FEMALE,
            birth_date=date(1805, 1, 1),  # Approximate
            death_date=date(1880, 1, 1),  # Approximate
            birth_place="Maryland",
            notes="1805-1880; mother of Seth E. Bennett; surname variant Benepe/Bennett critical"
        )
        self.add_person(caroline)
        
        # Relationship: Caroline → Seth
        rel_caroline_seth = Relationship(
            relationship_id="REL_19TH_002",
            person1_id="CAROLINE_BENEPE_001",
            person2_id="SETH_E_BENNETT_001",
            relationship_type=RelationshipType.PARENT_CHILD,
            biological=True,
            legitimacy=True,
            evidence_quality="secondary",
            notes="CRITICAL LINK: Name variant Benepe/Bennett; requires marriage and maiden name proof"
        )
        self.add_relationship(rel_caroline_seth)
        
        self.add_finding(
            "Caroline Hammond Benepe (1805-1880): CRITICAL GENEALOGICAL LINK. "
            "Middle name 'Hammond' suggests connection to colonial Hammond family. "
            "Surname 'Benepe' is variant or maiden name; relationship to 'Bennett' requires proof. "
            "Research priorities: 1) Death record 1880; 2) Marriage record (maiden name); "
            "3) Census 1850-1880; 4) Probate records; 5) Connection to Hammond family of Maryland."
        )
    
    def _analyze_bennett_benepe_variant(self) -> None:
        """Analyze Bennett/Benepe surname variation."""
        
        self.add_finding(
            "SURNAME ANALYSIS: Bennett vs. Benepe\n"
            "Caroline Hammond Benepe (Bennett) presents critical nomenclature question:\n"
            "- Is 'Benepe' her maiden name? (requires marriage record)\n"
            "- Is 'Benepe' a spelling variant of 'Bennett'? (common in 19th century)\n"
            "- Is 'Bennett' an anglicization of Dutch 'Benepe'?\n"
            "Resolution required through: marriage records, census enumeration consistency, "
            "land records, and contemporary documents."
        )
        
        self.add_finding(
            "HAMMOND CONNECTION: Caroline's middle name 'Hammond' is genealogically significant. "
            "Hammond family is well-documented colonial Maryland gentry with multiple branches. "
            "If Caroline is daughter of a Hammond, this provides direct connection to colonial families. "
            "Priority research: Identify Caroline's parents through marriage record, probate, census."
        )
    
    def analyze_evidence(self) -> Dict[str, Any]:
        """Analyze 19th century evidence."""
        return {
            "coverage": "1800-1900 (19th century Bennett family)",
            "persons_documented": [
                "John E. Bennett (1876-1964)",
                "Seth E. Bennett (1842-1931)",
                "Caroline Hammond Benepe (Bennett) (1805-1880)"
            ],
            "critical_findings": [
                "Three-generation chain documented in family records",
                "Caroline Hammond Benepe: surname variant and Hammond middle name both significant",
                "Hammond connection suggests colonial Maryland gentry lineage"
            ],
            "evidence_gaps": [
                "Birth certificates for all three individuals",
                "Death certificates (especially Caroline 1880, Seth 1931, John 1964)",
                "Marriage records (especially Caroline's - reveals maiden name and parents)",
                "Census verification for all: 1850-1880 (Caroline), 1850-1930 (Seth), 1880-1940 (John)",
                "Caroline's maiden name and parentage - CRITICAL GAP"
            ],
            "proof_status": "Documented in family records but requires GPS-level verification with primary sources",
            "next_steps": [
                "PRIORITY 1: Obtain marriage record for Caroline - reveals maiden name",
                "PRIORITY 2: Search Maryland death records 1880 (Caroline), 1931 (Seth), 1964 (John)",
                "PRIORITY 3: Systematic census search 1850-1940 for all three individuals",
                "PRIORITY 4: If Caroline maiden name is Hammond, research Hammond family genealogies",
                "PRIORITY 5: Research Benepe surname in Maryland records - spelling variant or distinct family?"
            ],
            "colonial_connection": "Caroline's middle name 'Hammond' suggests direct connection to "
                                  "documented colonial Maryland Hammond family, which is key to proving "
                                  "colonial gentry descent and potential noble ancestry."
        }
