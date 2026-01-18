"""
Genealogical research agents package.
"""

from .base_agent import ResearchAgent
from .modern_records_agent import ModernRecordsAgent
from .nineteenth_century_bennett_agent import NineteenthCenturyBennettAgent
from .nineteenth_century_york_agent import NineteenthCenturyYorkAgent
from .loyalist_revolutionary_agent import LoyalistRevolutionaryAgent
from .colonial_american_agent import ColonialAmericanAgent
from .name_identity_agent import NameIdentityAgent
from .english_nobility_agent import EnglishNobilityAgent
from .medieval_royal_agent import MedievalRoyalAgent
from .proof_analysis_agent import ProofAnalysisAgent
from .heraldic_agent import HeraldicAgent

__all__ = [
    "ResearchAgent",
    "ModernRecordsAgent",
    "NineteenthCenturyBennettAgent",
    "NineteenthCenturyYorkAgent",
    "LoyalistRevolutionaryAgent",
    "ColonialAmericanAgent",
    "NameIdentityAgent",
    "EnglishNobilityAgent",
    "MedievalRoyalAgent",
    "ProofAnalysisAgent",
    "HeraldicAgent",
]
