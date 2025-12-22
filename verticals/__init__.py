"""QRATUM Vertical Modules - Specialized AI Services.

This package contains the 14 vertical AI modules for the QRATUM platform.
Each module provides domain-specific capabilities with deterministic execution.
"""

# Import all 14 modules
from verticals.capra import CAPRAModule
from verticals.ecora import ECORAModule
from verticals.fluxa import FLUXAModule
from verticals.juris import JURISModule
from verticals.neura import NEURAModule
from verticals.sentra import SENTRAModule
from verticals.vitra import VITRAModule
from verticals.spectra import SPECTRAModule
from verticals.aegis import AEGISModule
from verticals.logos import LOGOSModule
from verticals.synthos import SYNTHOSModule
from verticals.teragon import TERAGONModule
from verticals.helix import HELIXModule
from verticals.nexus import NEXUSModule

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
