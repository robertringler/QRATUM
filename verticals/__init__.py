"""QRATUM Vertical Modules - Specialized AI Services.

This package contains the 14 vertical AI modules for the QRATUM platform.
Each module provides domain-specific capabilities with deterministic execution.
"""

# Import all 14 modules
from verticals.aegis import AEGISModule
from verticals.capra import CAPRAModule
from verticals.ecora import ECORAModule
from verticals.fluxa import FLUXAModule
from verticals.helix import HELIXModule
from verticals.juris import JURISModule
from verticals.logos import LOGOSModule
from verticals.neura import NEURAModule
from verticals.nexus import NEXUSModule
from verticals.sentra import SENTRAModule
from verticals.spectra import SPECTRAModule
from verticals.synthos import SYNTHOSModule
from verticals.teragon import TERAGONModule
from verticals.vitra import VITRAModule

__all__ = [
    "JURISModule",
    "VITRAModule",
    "ECORAModule",
    "CAPRAModule",
    "SENTRAModule",
    "NEURAModule",
    "FLUXAModule",
    "SPECTRAModule",
    "AEGISModule",
    "LOGOSModule",
    "SYNTHOSModule",
    "TERAGONModule",
    "HELIXModule",
    "NEXUSModule",
]

__version__ = "2.0.0"
