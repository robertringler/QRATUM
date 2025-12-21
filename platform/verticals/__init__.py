"""QRATUM Platform Vertical Modules.

Domain-specific AI modules with safety controls and audit trails.
"""

from platform.verticals.capra import CapraModule
from platform.verticals.ecora import EcoraModule
from platform.verticals.fluxa import FluxaModule
from platform.verticals.juris import JurisModule
from platform.verticals.neura import NeuraModule
from platform.verticals.sentra import SentraModule
from platform.verticals.vitra import VitraModule

__all__ = [
    "JurisModule",
    "VitraModule",
    "EcoraModule",
    "CapraModule",
    "SentraModule",
    "NeuraModule",
    "FluxaModule",
]
