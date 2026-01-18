"""
QRATUM Genealogical Research Orchestration System

A professional-grade multi-agent system for forensic genealogical research
and royal/noble ancestry verification.
"""

__version__ = "1.0.0"
__author__ = "QRATUM Team"

from .orchestrator import ChiefResearchOrchestrator
from .models.record import GenealogicalRecord, Citation, Source
from .models.person import Person, Relationship

__all__ = [
    "ChiefResearchOrchestrator",
    "GenealogicalRecord",
    "Citation",
    "Source",
    "Person",
    "Relationship",
]
