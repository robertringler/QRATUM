"""QRATUM Vertical Modules - Specialized AI Services.

This package contains the 14 vertical AI modules for the QRATUM platform.
Each module provides domain-specific capabilities with deterministic execution.
"""

# Import Part 1 modules (7 modules)
from verticals.capra import CAPRAModule
from verticals.ecora import ECORAModule
from verticals.fluxa import FLUXAModule
from verticals.juris import JURISModule
from verticals.neura import NEURAModule
from verticals.sentra import SENTRAModule
from verticals.vitra import VITRAModule

__all__ = [
    "JURISModule",
    "VITRAModule",
    "ECORAModule",
    "CAPRAModule",
    "SENTRAModule",
    "NEURAModule",
    "FLUXAModule",
]

__version__ = "2.0.0"
