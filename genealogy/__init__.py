"""
QRATUM Genealogical Research Orchestration System

A professional-grade multi-agent system for forensic genealogical research
and royal/noble ancestry verification.
"""

__version__ = "1.0.0"
__author__ = "QRATUM Team"

from .models.person import Person, Relationship
from .models.record import Citation, GenealogicalRecord, Source
from .orchestrator import ChiefResearchOrchestrator

__all__ = [
    "ChiefResearchOrchestrator",
    "GenealogicalRecord",
    "Citation",
    "Source",
    "Person",
    "Relationship",
]
