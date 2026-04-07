"""
Genealogical research agents package.
"""

from .base_agent import ResearchAgent
from .colonial_american_agent import ColonialAmericanAgent
from .english_nobility_agent import EnglishNobilityAgent
from .heraldic_agent import HeraldicAgent
from .loyalist_revolutionary_agent import LoyalistRevolutionaryAgent
from .medieval_royal_agent import MedievalRoyalAgent
from .modern_records_agent import ModernRecordsAgent
from .name_identity_agent import NameIdentityAgent
from .nineteenth_century_bennett_agent import NineteenthCenturyBennettAgent
from .nineteenth_century_york_agent import NineteenthCenturyYorkAgent
from .proof_analysis_agent import ProofAnalysisAgent
from .seventeenth_century_york_agent import SeventeenthCenturyYorkAgent

__all__ = [
    "ResearchAgent",
    "ModernRecordsAgent",
    "NineteenthCenturyBennettAgent",
    "NineteenthCenturyYorkAgent",
    "LoyalistRevolutionaryAgent",
    "SeventeenthCenturyYorkAgent",
    "ColonialAmericanAgent",
    "NameIdentityAgent",
    "EnglishNobilityAgent",
    "MedievalRoyalAgent",
    "ProofAnalysisAgent",
    "HeraldicAgent",
]
