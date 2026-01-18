"""
Genealogical research agents package.
"""

from .base_agent import ResearchAgent
from .modern_records_agent import ModernRecordsAgent
from .nineteenth_century_bennett_agent import NineteenthCenturyBennettAgent
from .colonial_american_agent import ColonialAmericanAgent
from .english_nobility_agent import EnglishNobilityAgent
from .medieval_royal_agent import MedievalRoyalAgent
from .proof_analysis_agent import ProofAnalysisAgent
from .heraldic_agent import HeraldicAgent

__all__ = [
    "ResearchAgent",
    "ModernRecordsAgent",
    "NineteenthCenturyBennettAgent",
    "ColonialAmericanAgent",
    "EnglishNobilityAgent",
    "MedievalRoyalAgent",
    "ProofAnalysisAgent",
    "HeraldicAgent",
]
