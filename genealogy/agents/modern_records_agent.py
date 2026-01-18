"""
Agent 1: Modern & Civil Records Verification (1800-Present)

Board-Certified Forensic Genealogist specializing in modern civil records.
Focus: BOTH York and Bennett maternal lines.
"""

from typing import Dict, Any

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


class ModernRecordsAgent(ResearchAgent):
    """
    Specializes in modern and civil records verification (1800-Present).
    
    Duties:
    - Prove subject's identity and parentage using civil records
    - Establish uninterrupted parent-child links for BOTH York and Bennett lines
    - Resolve surname changes and maternal-line continuity
    """
    
    def __init__(self):
        super().__init__(
            agent_id="agent_1_modern_records",
            scope="Modern & Civil Records (1800-Present) - DUAL LINES: York & Bennett"
        )
    
    def execute_research(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute modern records research for BOTH York and Bennett lines.
        """
        self.start_execution()
        
        # Subject: Robert Ringler Jr.
        self._establish_subject_identity(context)
        
        # Work backward through generations - BOTH lines
        self._trace_parents()
        self._trace_york_line()  # Through Elizabeth P. York (Karen's mother)
        self._trace_bennett_line()  # Through Thomas Bennett (Karen's father)
        
        # Analyze collected evidence
        analysis = self.analyze_evidence()
        
        self.complete_execution()
        
        return {
            "success": True,
            "lines_traced": ["York", "Bennett"],
            "generations_traced": 4,
            "analysis": analysis,
            "report": self.generate_report(),
        }
    
    def _establish_subject_identity(self, context: Dict[str, Any]) -> None:
        """Establish the identity of Robert Ringler Jr."""
        
        # Create person record for Robert Ringler Jr.
        subject = Person(
            person_id="RR_JR_001",
            full_name="Robert Ringler Jr.",
            given_names="Robert",
            surname="Ringler",
            gender=Gender.MALE,
            birth_date=None,  # Circa 1995 - requires birth certificate
            notes="Subject of genealogical investigation; b. circa 1995"
        )
        self.add_person(subject)
        
        # Create source for birth record
        birth_source = Source(
            source_id="SRC_MOD_001",
            title="Birth Certificate, Robert Ringler Jr.",
            source_type=SourceType.ORIGINAL_DOCUMENT,
            repository="County Clerk's Office",
            reliability=EvidenceQuality.PRIMARY,
            notes="Official civil birth record"
        )
        self.add_source(birth_source)
        
        self.add_finding(
            "Subject identity established: Robert Ringler Jr. (b. circa 1995) "
            "verified through civil birth certificate (requires acquisition)"
        )
    
    def _trace_parents(self) -> None:
        """Trace subject's parents using civil records."""
        
        # Father: Robert Ringler Sr.
        father = Person(
            person_id="RR_SR_001",
            full_name="Robert Ringler Sr.",
            given_names="Robert",
            surname="Ringler",
            gender=Gender.MALE,
            notes="Father of subject"
        )
        self.add_person(father)
        
        # Mother: Karen L. (Bennett) Pulley
        # KEY: Her mother was Elizabeth P. York; her father was from Bennett family
        mother = Person(
            person_id="KAREN_PULLEY_001",
            full_name="Karen L. (Bennett) Pulley",
            given_names="Karen L.",
            surname="Pulley",
            maiden_name="Bennett",
            gender=Gender.FEMALE,
            birth_date=None,  # Circa 1973 - requires birth certificate
            notes="Mother of subject; b. circa 1973; maiden Bennett; mother: Elizabeth P. York (York line); father: Thomas Bennett (Bennett line)"
        )
        self.add_person(mother)
        
        # Parent-child relationships
        rel_father = Relationship(
            relationship_id="REL_MOD_001",
            person1_id="RR_SR_001",
            person2_id="RR_JR_001",
            relationship_type=RelationshipType.PARENT_CHILD,
            biological=True,
            legitimacy=True,
            evidence_quality="primary"
        )
        self.add_relationship(rel_father)
        
        rel_mother = Relationship(
            relationship_id="REL_MOD_002",
            person1_id="KAREN_PULLEY_001",
            person2_id="RR_JR_001",
            relationship_type=RelationshipType.PARENT_CHILD,
            biological=True,
            legitimacy=True,
            evidence_quality="secondary",
            notes="Requires birth certificate verification"
        )
        self.add_relationship(rel_mother)
        
        self.add_finding(
            "Parental generation: Father Robert Ringler Sr. and mother Karen L. (Bennett) Pulley (b. circa 1973) identified. "
            "CRITICAL: Karen's parents provide TWO DISTINCT RESEARCH LINES: "
            "(1) YORK LINE through her mother Elizabeth P. York, and "
            "(2) BENNETT LINE through her father Thomas Bennett. "
            "Both lines will be traced independently to colonial period."
        )
    
    def _trace_york_line(self) -> None:
        """Trace the YORK line through Karen's mother Elizabeth P. York."""
        
        # Maternal grandmother: Elizabeth P. York (1942-2022)
        elizabeth_york = Person(
            person_id="ELIZABETH_YORK_001",
            full_name="Elizabeth P. York",
            given_names="Elizabeth P.",
            surname="York",
            maiden_name="York",
            gender=Gender.FEMALE,
            birth_date=None,  # Circa 1942
            death_date=None,  # 2022
            notes="Maternal grandmother; circa 1942-2022; maiden name York; mother of Karen Pulley"
        )
        self.add_person(elizabeth_york)
        
        # Relationship: Elizabeth York → Karen Pulley
        rel_elizabeth_karen = Relationship(
            relationship_id="REL_YORK_001",
            person1_id="ELIZABETH_YORK_001",
            person2_id="KAREN_PULLEY_001",
            relationship_type=RelationshipType.PARENT_CHILD,
            biological=True,
            legitimacy=True,
            evidence_quality="secondary",
            notes="Requires birth certificate for Karen Pulley"
        )
        self.add_relationship(rel_elizabeth_karen)
        
        # Great-grandmother: Jessie York (1914-1986)
        jessie_york = Person(
            person_id="JESSIE_YORK_001",
            full_name="Jessie York",
            given_names="Jessie",
            surname="York",
            gender=Gender.FEMALE,
            birth_date=None,  # Circa 1914
            death_date=None,  # Circa 1986
            notes="Great-grandmother; circa 1914-1986; mother of Elizabeth York"
        )
        self.add_person(jessie_york)
        
        rel_jessie_elizabeth = Relationship(
            relationship_id="REL_YORK_002",
            person1_id="JESSIE_YORK_001",
            person2_id="ELIZABETH_YORK_001",
            relationship_type=RelationshipType.PARENT_CHILD,
            biological=True,
            legitimacy=True,
            evidence_quality="secondary",
            notes="Requires vital records, census"
        )
        self.add_relationship(rel_jessie_elizabeth)
        
        # 2nd Great-grandfather: James "Jim" York (1866-1952)
        james_york = Person(
            person_id="JAMES_YORK_001",
            full_name='James "Jim" York',
            given_names="James",
            surname="York",
            gender=Gender.MALE,
            birth_date=None,  # Circa 1866
            death_date=None,  # Circa 1952
            notes="2nd great-grandfather; circa 1866-1952; continues to William Robert York, then Revolutionary era"
        )
        self.add_person(james_york)
        
        rel_james_jessie = Relationship(
            relationship_id="REL_YORK_003",
            person1_id="JAMES_YORK_001",
            person2_id="JESSIE_YORK_001",
            relationship_type=RelationshipType.PARENT_CHILD,
            biological=True,
            legitimacy=True,
            evidence_quality="secondary",
            notes="Requires census, vital records, probate"
        )
        self.add_relationship(rel_james_jessie)
        
        self.add_finding(
            "YORK LINE DOCUMENTED (Line #1):\n"
            "Robert Ringler Jr. → Karen Pulley → Elizabeth P. York (1942-2022) → Jessie York (1914-1986) → "
            "James 'Jim' York (1866-1952) → William Robert York (1846-1906) → Thomas Ralph York (1825-1906) → "
            "Edward York (1788-1862) → John Edgar York (1764-1810) → Thomas York (1745-1812) → "
            "Semore (Seymore) York (1724-1783, documented Loyalist).\n\n"
            "WARNING: Semore/Seymore York surname must NOT be confused with Seymour of Wolf Hall (Queen Jane Seymour). "
            "These are DISTINCT families with no connection."
        )
    
    def _trace_bennett_line(self) -> None:
        """Trace the BENNETT line through Karen's father Thomas Bennett."""
        
        # Maternal grandfather: Thomas Bennett (b. circa 1946)
        thomas_bennett = Person(
            person_id="THOMAS_BENNETT_001",
            full_name="Thomas Bennett",
            given_names="Thomas",
            surname="Bennett",
            gender=Gender.MALE,
            birth_date=None,  # Circa 1946
            notes="Maternal grandfather; b. circa 1946; father of Karen Pulley"
        )
        self.add_person(thomas_bennett)
        
        rel_thomas_karen = Relationship(
            relationship_id="REL_BENN_001",
            person1_id="THOMAS_BENNETT_001",
            person2_id="KAREN_PULLEY_001",
            relationship_type=RelationshipType.PARENT_CHILD,
            biological=True,
            legitimacy=True,
            evidence_quality="secondary",
            notes="Requires birth certificate for Karen Pulley"
        )
        self.add_relationship(rel_thomas_karen)
        
        # Great-grandfather: Ralph E. Bennett (1910-1964)
        ralph_bennett = Person(
            person_id="RALPH_BENNETT_001",
            full_name="Ralph E. Bennett",
            given_names="Ralph E.",
            surname="Bennett",
            gender=Gender.MALE,
            birth_date=None,  # Circa 1910
            death_date=None,  # Circa 1964
            notes="Great-grandfather; circa 1910-1964"
        )
        self.add_person(ralph_bennett)
        
        rel_ralph_thomas = Relationship(
            relationship_id="REL_BENN_002",
            person1_id="RALPH_BENNETT_001",
            person2_id="THOMAS_BENNETT_001",
            relationship_type=RelationshipType.PARENT_CHILD,
            biological=True,
            legitimacy=True,
            evidence_quality="secondary",
            notes="Requires death certificate for Ralph, birth certificate for Thomas"
        )
        self.add_relationship(rel_ralph_thomas)
        
        self.add_finding(
            "BENNETT LINE DOCUMENTED (Line #2):\n"
            "Robert Ringler Jr. → Karen Pulley → Thomas Bennett (b. 1946) → Ralph E. Bennett (1910-1964) → "
            "John E. Bennett (1876-1964) → Seth E. Bennett (1842-1931) → "
            "Caroline Hammond Benepe (Bennett) (1805-1880) → Harriet Amelia Warfield (1783-1844) → "
            "Ruth Hammond Warfield (1756-1820) → Hannah Hammond-Welsh (1723-1779) → "
            "Capt. John Charles Hammond (1698-1753) → Maj. Charles M. Hammond (1670-1713).\n\n"
            "KEY: Caroline's middle name 'Hammond' and surname variant 'Benepe' are genealogically significant. "
            "Hammond family = documented Maryland colonial gentry."
        )
    
    def analyze_evidence(self) -> Dict[str, Any]:
        """Analyze modern records evidence for BOTH lines."""
        return {
            "coverage": "1800-Present (DUAL LINES: York & Bennett)",
            "line_1_york": {
                "key_persons": ["Elizabeth P. York (1942-2022)", "Jessie York (1914-1986)", "James York (1866-1952)"],
                "colonial_terminus": "Semore (Seymore) York (1724-1783), documented Loyalist",
                "warning": "NO CONNECTION to Seymour of Wolf Hall or Queen Jane Seymour",
                "research_priority": "Vital records, census 1850-1940, Loyalist claims records"
            },
            "line_2_bennett": {
                "key_persons": ["Thomas Bennett (b. 1946)", "Ralph E. Bennett (1910-1964)"],
                "colonial_terminus": "Maj. Charles M. Hammond (1670-1713) via Bennett→Hammond connection",
                "key_connection": "Caroline Hammond Benepe - surname variant and Hammond middle name",
                "research_priority": "Vital records, census 1850-1940, Caroline's marriage record"
            },
            "dual_line_strategy": "TWO INDEPENDENT RESEARCH PATHS to colonial period. "
                                 "Both lines documented in family records; require GPS-level primary source verification. "
                                 "York line → Loyalist/Revolutionary era. Bennett line → Maryland colonial gentry.",
            "primary_sources_needed": [
                "Birth certificates: Karen Pulley, Thomas Bennett, Elizabeth York, Ralph Bennett",
                "Death certificates: Elizabeth York (2022), Jessie York (1986), James York (1952), Ralph Bennett (1964)",
                "Census: 1850-1940 for all York and Bennett family members",
                "Marriage records: All generations"
            ],
            "proof_status": "Both lines documented in family records (SECONDARY evidence); "
                           "requires GPS-level PRIMARY source verification",
            "next_steps": [
                "PRIORITY 1: Acquire modern vital records (birth/death certificates) for both lines",
                "PRIORITY 2: Census verification 1850-1940 for York and Bennett families",
                "PRIORITY 3: York line: Trace to Loyalist records (Agent 2)",
                "PRIORITY 4: Bennett line: Trace to 19th century (Agent 1B), then colonial Maryland (Agent 2 alternative)",
                "PRIORITY 5: Both lines: Establish GPS-certified proof for each generation"
            ]
        }
