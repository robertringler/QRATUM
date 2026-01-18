"""
Agent 6: Heraldic & Legal Context

Heraldic Law and Arms Authority.
"""

from typing import Dict, Any

from .base_agent import ResearchAgent
from ..models.record import Source, SourceType, EvidenceQuality


class HeraldicAgent(ResearchAgent):
    """
    Heraldic Law and Arms Authority.
    
    Duties:
    - Analyze historically inherited arms
    - Distinguish historical entitlement from modern or honorary claims
    - Prepare heraldic analysis consistent with College of Arms standards
    """
    
    def __init__(self):
        super().__init__(
            agent_id="agent_6_heraldic",
            scope="Heraldic Law and Arms Analysis"
        )
    
    def execute_research(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute heraldic analysis."""
        self.start_execution()
        
        # Analyze arms of relevant families
        self._analyze_howard_arms()
        self._analyze_plantagenet_arms()
        self._analyze_heraldic_law()
        self._assess_modern_entitlement()
        
        analysis = self.analyze_evidence()
        
        self.complete_execution()
        
        return {
            "success": True,
            "families_analyzed": len(self.sources),
            "analysis": analysis,
            "report": self.generate_report(),
        }
    
    def _analyze_howard_arms(self) -> None:
        """Analyze Howard family arms and hereditary offices."""
        
        howard_arms_source = Source(
            source_id="SRC_HER_001",
            title="A Complete Guide to Heraldry",
            source_type=SourceType.PUBLISHED_WORK,
            author="Arthur Charles Fox-Davies",
            publisher="T.C. & E.C. Jack",
            reliability=EvidenceQuality.PRIMARY,
            notes="Authoritative heraldic reference"
        )
        self.add_source(howard_arms_source)
        
        self.add_finding(
            "HOWARD FAMILY ARMS:\n"
            "Blazon: Gules, on a bend between six cross-crosslets fitchy argent, "
            "an escutcheon or, charged with a demi-lion rampant pierced through "
            "the mouth by an arrow, within a double tressure flory counter-flory of the first.\n"
            "Crest: On a chapeau gules turned up ermine, a lion statant guardant or, "
            "ducally crowned of the first, gorged with a label of three points argent.\n"
            "Supporters: Two lions rampant guardant or, each gorged with a ducal coronet, "
            "therefrom a chain reflexed over the back azure."
        )
        
        self.add_finding(
            "HEREDITARY OFFICE: Earl Marshal of England - hereditary office held by "
            "Duke of Norfolk (Howard family) since 1672. Office demonstrates premier "
            "nobility status and is heraldic evidence of continuous noble standing."
        )
    
    def _analyze_plantagenet_arms(self) -> None:
        """Analyze Plantagenet royal arms."""
        
        self.add_finding(
            "PLANTAGENET ROYAL ARMS (Edward I period):\n"
            "Blazon: Gules, three lions passant guardant in pale or (Arms of England).\n"
            "Edward I used these arms throughout his reign. His legitimate children "
            "bore royal arms with appropriate marks of cadency."
        )
        
        self.add_finding(
            "THOMAS OF BROTHERTON (son of Edward I):\n"
            "Arms: England (three lions) with a label of three points argent. "
            "As Earl of Norfolk, these arms passed to his descendants with modifications."
        )
    
    def _analyze_heraldic_law(self) -> None:
        """Analyze heraldic law and entitlement principles."""
        
        heraldic_law_source = Source(
            source_id="SRC_HER_002",
            title="The Right to Arms in England",
            source_type=SourceType.PUBLISHED_WORK,
            repository="College of Arms",
            reliability=EvidenceQuality.PRIMARY,
            notes="Official guidance on heraldic entitlement"
        )
        self.add_source(heraldic_law_source)
        
        self.add_finding(
            "HERALDIC LAW PRINCIPLES:\n"
            "1. Arms descend in male line from original grantee or person of immemorial use\n"
            "2. Daughters are heraldic heiresses if no brothers; their descendants may "
            "quarter maternal arms\n"
            "3. Ancient nobility bore arms by prescription (immemorial use)\n"
            "4. American descendants: No automatic right under English heraldic law, "
            "but may demonstrate historical descent from armigerous families"
        )
    
    def _assess_modern_entitlement(self) -> None:
        """Assess modern heraldic entitlement."""
        
        self.add_finding(
            "MODERN AMERICAN CONTEXT:\n"
            "The College of Arms (England) has no jurisdiction over American citizens. "
            "Americans may:\n"
            "1. Demonstrate historical descent from armigerous English/European families "
            "(genealogical interest)\n"
            "2. Petition College of Arms for grant of arms (if willing to pay fees)\n"
            "3. Use ancestor's arms for historical/genealogical display (not legal right)\n"
            "4. Register with American heraldic organizations (voluntary, non-official)"
        )
        
        self.add_finding(
            "CRITICAL DISTINCTION:\n"
            "Historical descent from armigerous nobility ≠ Legal right to bear arms today.\n"
            "Genealogical proof of noble descent is historically significant and demonstrates "
            "ancestry, but does NOT confer automatic heraldic rights under English law, "
            "and no such legal framework exists in the United States."
        )
        
        self.add_finding(
            "FOR ROBERT RINGLER JR.:\n"
            "IF proven descent from Howard or other armigerous families is established:\n"
            "- Historical interest: Yes, significant genealogical achievement\n"
            "- Heraldic display: May display ancestral arms with appropriate 'descendant of' notation\n"
            "- Legal entitlement: No automatic legal right under English heraldic law\n"
            "- American context: No official heraldic authority exists; purely genealogical/historical interest"
        )
    
    def analyze_evidence(self) -> Dict[str, Any]:
        """Analyze heraldic evidence and entitlement."""
        return {
            "scope": "Heraldic law and arms analysis",
            "families_analyzed": ["Howard (Duke of Norfolk)", "Plantagenet", "Mowbray"],
            "hereditary_offices": ["Earl Marshal of England (Howard family)"],
            "heraldic_sources": [
                "A Complete Guide to Heraldry (Fox-Davies)",
                "College of Arms official guidance",
                "Heralds' Visitations",
                "Burke's Peerage and Baronetage"
            ],
            "key_findings": [
                "Howard family: Documented armigerous family with hereditary Earl Marshal office",
                "Plantagenet royal arms: Well-documented for Edward I and legitimate descendants",
                "Thomas of Brotherton: Royal arms with label, proper cadency for king's son",
                "Heraldic descent: Arms pass in male line; female line allows quartering"
            ],
            "modern_entitlement_assessment": {
                "historical_descent": "IF proven, demonstrates ancestry from armigerous nobility",
                "legal_rights_england": "None - No automatic right without English residence/citizenship",
                "legal_rights_usa": "None - No official heraldic authority in United States",
                "appropriate_use": "Historical/genealogical display with proper attribution",
                "inappropriate_use": "Claiming legal entitlement or official status"
            },
            "conclusion": "Heraldic analysis confirms that documented English noble families "
                        "(Howard, Mowbray, etc.) were armigerous by ancient right. IF genealogical "
                        "proof of descent is established, this represents significant historical "
                        "ancestry but does NOT confer modern legal heraldic rights. "
                        "Any use of ancestral arms should be clearly marked as historical descent, "
                        "not personal entitlement.",
            "recommendation": "Focus on genealogical proof of descent rather than heraldic claims. "
                            "Heraldic information is supplementary evidence of noble status in "
                            "historical periods but is not the primary goal of genealogical research."
        }
